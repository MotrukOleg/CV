import cv2
import numpy as np
import math
import os

params = {
    'blur_size': 9, 'blur_sigma': 75, 'adapt_block': 11, 'adapt_c': 2,
    'min_area': 250, 'max_area': 25000, 'match_dist': 80, 'separation_threshold': 0.6
}

class GlobalTracker:
    def __init__(self):
        self.global_transform = np.eye(3, dtype=np.float32)
        self.coins = []
        self.next_id = 1
        self.map_size = 2000
        self.map_center = np.array([1000, 1000])
        self.tracking_lost = False

    def update_camera_motion(self, prev_gray, curr_gray):
        if prev_gray is None: return False

        mask = np.zeros_like(prev_gray)
        h, w = mask.shape
        mask[60:h - 60, 60:w - 60] = 255

        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=20, mask=mask)
        if p0 is None or len(p0) < 10:
            self.tracking_lost = True
            return False

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None,
                                               winSize=(31, 31), maxLevel=3,
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) < 10:
            self.tracking_lost = True
            return False

        m, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)

        if m is None:
            self.tracking_lost = True
            return False

        dx = m[0, 2]
        dy = m[1, 2]
        if math.hypot(dx, dy) > 150:
            self.tracking_lost = True
            return False

        m_inv = cv2.invertAffineTransform(m)
        m_3x3 = np.vstack([m_inv, [0, 0, 1]])
        self.global_transform = self.global_transform @ m_3x3
        self.tracking_lost = False
        return True

    def recover_position_using_coins(self, current_candidates):
        if len(current_candidates) < 1 or len(self.coins) < 1:
            return

        possible_shifts = []

        for cand in current_candidates:
            gx, gy = self.frame_to_global(cand['x'], cand['y'])

            best_coin = None
            min_dist = float('inf')

            for known in self.coins:
                if known['seen_count'] < 10: continue
                dist = math.hypot(gx - known['global_pos'][0], gy - known['global_pos'][1])

                if dist < 200 and dist < min_dist:
                    min_dist = dist
                    best_coin = known

            if best_coin:
                shift_x = best_coin['global_pos'][0] - gx
                shift_y = best_coin['global_pos'][1] - gy
                possible_shifts.append((shift_x, shift_y))

        if len(possible_shifts) > 0:
            shifts_np = np.array(possible_shifts)
            median_shift_x = np.median(shifts_np[:, 0])
            median_shift_y = np.median(shifts_np[:, 1])

            correction = np.eye(3, dtype=np.float32)
            correction[0, 2] = median_shift_x
            correction[1, 2] = median_shift_y

            self.global_transform = correction @ self.global_transform
            print(f"!!! АВАРІЙНЕ ВІДНОВЛЕННЯ: Зсув на ({median_shift_x:.1f}, {median_shift_y:.1f})")

    def frame_to_global(self, x, y):
        vec = np.array([x, y, 1.0])
        g = self.global_transform @ vec
        return g[0], g[1]

    def process_candidates(self, candidates):
        if self.tracking_lost:
            self.recover_position_using_coins(candidates)

        matches_for_drift = []
        for cand in candidates:
            gx, gy = self.frame_to_global(cand['x'], cand['y'])

            best_match = None
            min_dist = float('inf')

            for known in self.coins:
                dist = math.hypot(gx - known['global_pos'][0], gy - known['global_pos'][1])
                if dist < min_dist: min_dist = dist; best_match = known

            if best_match and min_dist < params['match_dist']:
                best_match['seen_count'] += 1
                if best_match['seen_count'] > 5:
                    matches_for_drift.append(((gx, gy), best_match['global_pos']))

                alpha = 0.3 if self.tracking_lost else 0.05

                best_match['global_pos'] = (
                    best_match['global_pos'][0] * (1 - alpha) + gx * alpha,
                    best_match['global_pos'][1] * (1 - alpha) + gy * alpha
                )
                cand['id'] = best_match['id']
                cand['is_new'] = False
            else:
                if not cand['on_border']:
                    self.coins.append(
                        {'id': self.next_id, 'global_pos': (gx, gy), 'radius': cand['r'], 'seen_count': 1})
                    cand['id'] = self.next_id;
                    self.next_id += 1
                else:
                    cand['id'] = -1

        if not self.tracking_lost:
            self.correct_map_drift(matches_for_drift)

    def correct_map_drift(self, matches):
        if not matches: return
        total_dx = 0;
        total_dy = 0
        for (curr, saved) in matches:
            total_dx += saved[0] - curr[0]
            total_dy += saved[1] - curr[1]

        count = len(matches)
        correction = np.eye(3, dtype=np.float32)
        correction[0, 2] = (total_dx / count) * 0.1
        correction[1, 2] = (total_dy / count) * 0.1
        self.global_transform = correction @ self.global_transform

    def draw_map(self):
        map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8) + 30
        count = 0
        for coin in self.coins:
            if coin['seen_count'] > 5:
                count += 1
                gx, gy = coin['global_pos']
                ix, iy = int(gx + self.map_center[0]), int(gy + self.map_center[1])
                if 0 <= ix < self.map_size:
                    cv2.circle(map_img, (ix, iy), int(coin['radius']), (0, 255, 0), 2)
                    cv2.putText(map_img, str(coin['id']), (ix, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        status_color = (0, 0, 255) if self.tracking_lost else (0, 255, 0)
        status_text = "LOST" if self.tracking_lost else "TRACKING"
        cv2.putText(map_img, f"Status: {status_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(map_img, f"Total: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        return cv2.resize(map_img, (600, 600))


def process_video(video_path, index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    tracker = GlobalTracker()
    prev_gray = None
    paused = False

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Map', cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Separation", "Frame", 60, 100, lambda x: None)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                tracker.update_camera_motion(prev_gray, gray)
            prev_gray = gray.copy()

            blur = cv2.bilateralFilter(gray, params['blur_size'], params['blur_sigma'], params['blur_sigma'])
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                           params['adapt_block'], params['adapt_c'])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            clean_mask = np.zeros_like(thresh)
            clean_mask = cv2.erode(clean_mask, kernel, iterations=2)

            cnts_raw, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_raw:
                if cv2.contourArea(c) < 500: continue
                hull = cv2.convexHull(c)
                cv2.drawContours(clean_mask, [hull], -1, 255, -1)

            dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
            sep_thresh = cv2.getTrackbarPos("Separation", "Frame") / 100.0
            if sep_thresh < 0.1: sep_thresh = 0.5
            _, sure_fg = cv2.threshold(dist_transform, sep_thresh * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)

            cnts_peaks, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidates = []
            h, w = frame.shape[:2]
            for c in cnts_peaks:
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                real_r = dist_transform[cY, cX]
                area = np.pi * (real_r ** 2)

                if params['min_area'] < area < params['max_area']:
                    on_border = (cX < 40 or cX > w - 40 or cY < 40 or cY > h - 40)
                    candidates.append({'x': cX, 'y': cY, 'r': real_r, 'on_border': on_border})

            tracker.process_candidates(candidates)

            display = frame.copy()
            for cand in candidates:
                x, y, r = int(cand['x']), int(cand['y']), int(cand['r'])
                color = (0, 255, 0)
                if cand.get('is_new'): color = (0, 0, 255)
                if cand['id'] == -1: color = (100, 100, 100)
                cv2.circle(display, (x, y), r, color, 2)
                if cand['id'] != -1:
                    cv2.putText(display, f"ID:{cand['id']}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if tracker.tracking_lost:
                cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 10)
                cv2.putText(display, "FAST MOTION! RECOVERING...", (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 3)

            cv2.imshow('Frame', display)
            cv2.imshow('Clean',clean_mask)
            cv2.imshow('Thresh',thresh)
            cv2.imshow('Map', tracker.draw_map())

        key = cv2.waitKey(25 if not paused else 0) & 0xFF
        if key == ord('q'):
            return 'quit'
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Result: {len([c for c in tracker.coins if c['seen_count'] > 5])}")

for i in range(10):
    fname = f'Var1/Tests/test_{i}.mp4'
    if os.path.exists(fname): process_video(fname, i)