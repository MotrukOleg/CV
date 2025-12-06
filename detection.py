import cv2
import numpy as np
import math

# --- НАЛАШТУВАННЯ ---
params = {
    'blur_size': 9,
    'blur_sigma': 75,
    'adapt_block': 9,
    'adapt_c': 3,
    'min_area': 1000,
    'max_area': 20000,
    'min_circularity': 0.47,
    'match_dist': 80  # Зменшив дистанцію, бо трекінг тепер точніший
}


class GlobalTracker:
    def __init__(self):
        self.global_transform = np.eye(3, dtype=np.float32)
        self.coins = []
        self.next_id = 1
        self.map_size = 2000
        self.map_center = np.array([1000, 1000])

    def update_camera_motion(self, prev_gray, curr_gray):
        if prev_gray is None: return False

        # Маска: ігноруємо краї, бо там сильні спотворення лінзи
        mask = np.zeros_like(prev_gray)
        h, w = mask.shape
        border = 60
        mask[border:h - border, border:w - border] = 255

        # Більше точок для точності
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=20, mask=mask)
        if p0 is None or len(p0) < 8: return False

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) < 8: return False

        # RANSAC для відсіювання поганих точок
        m, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if m is None: return False

        # Оновлюємо глобальну трансформацію
        m_inv = cv2.invertAffineTransform(m)
        m_3x3 = np.vstack([m_inv, [0, 0, 1]])
        self.global_transform = self.global_transform @ m_3x3
        return True

    def correct_map_drift(self, matches):
        """
        НОВА ФУНКЦІЯ: Коригує глобальну карту, використовуючи відомі монети як якорі.
        matches: список пар (cand_global_pos, saved_coin_pos)
        """
        if not matches: return

        # Рахуємо середній вектор помилки (зсуву)
        total_dx = 0
        total_dy = 0
        count = 0

        for (curr_gx, curr_gy), (saved_gx, saved_gy) in matches:
            # Наскільки поточне бачення відрізняється від пам'яті?
            dx = saved_gx - curr_gx
            dy = saved_gy - curr_gy
            total_dx += dx
            total_dy += dy
            count += 1

        if count == 0: return

        avg_dx = total_dx / count
        avg_dy = total_dy / count

        # Створюємо матрицю корекції (зсув)
        correction_matrix = np.eye(3, dtype=np.float32)

        # Застосовуємо корекцію плавно (Soft update), щоб карту не трясло
        # Коефіцієнт 0.1 означає, що ми виправляємо 10% помилки за кожен кадр
        correction_matrix[0, 2] = avg_dx * 0.1
        correction_matrix[1, 2] = avg_dy * 0.1

        # Оновлюємо глобальну трансформацію
        self.global_transform = correction_matrix @ self.global_transform

    def frame_to_global(self, x, y):
        vec = np.array([x, y, 1.0])
        g = self.global_transform @ vec
        return g[0], g[1]

    def process_candidates(self, candidates):
        matches_for_drift_correction = []

        for cand in candidates:
            gx, gy = self.frame_to_global(cand['x'], cand['y'])

            best_match = None
            min_dist = float('inf')

            for known in self.coins:
                dist = math.hypot(gx - known['global_pos'][0], gy - known['global_pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    best_match = known

            if best_match and min_dist < params['match_dist']:
                # Знайшли стару монету!
                best_match['seen_count'] += 1
                best_match['last_seen_frame'] = cand['frame_num']

                # Додаємо цю пару в список для корекції дрейфу
                # Але тільки якщо ми впевнені в монеті (бачили її > 10 разів)
                if best_match['seen_count'] > 10:
                    matches_for_drift_correction.append(((gx, gy), best_match['global_pos']))

                # Оновлюємо позицію монети (дуже повільно, щоб вона була стабільним якорем)
                alpha = 0.1
                best_match['global_pos'] = (
                    best_match['global_pos'][0] * (1 - alpha) + gx * alpha,
                    best_match['global_pos'][1] * (1 - alpha) + gy * alpha
                )

                cand['id'] = best_match['id']
                cand['is_new'] = False
            else:
                # Нова монета
                if not cand['on_border']:
                    new_coin = {
                        'id': self.next_id,
                        'global_pos': (gx, gy),
                        'radius': cand['r'],
                        'seen_count': 1,
                        'last_seen_frame': cand['frame_num']
                    }
                    self.coins.append(new_coin)
                    cand['id'] = self.next_id
                    cand['is_new'] = True
                    self.next_id += 1
                else:
                    cand['id'] = -1

        # --- ЗАПУСКАЄМО КОРЕКЦІЮ ДРЕЙФУ ---
        self.correct_map_drift(matches_for_drift_correction)

    def draw_map(self):
        map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8) + 30

        # Сітка для орієнтації
        for i in range(0, self.map_size, 200):
            cv2.line(map_img, (0, i), (self.map_size, i), (50, 50, 50), 1)
            cv2.line(map_img, (i, 0), (i, self.map_size), (50, 50, 50), 1)

        count = 0
        for coin in self.coins:
            if coin['seen_count'] > 5:
                count += 1
                gx, gy = coin['global_pos']
                ix = int(gx + self.map_center[0])
                iy = int(gy + self.map_center[1])
                if 0 <= ix < self.map_size and 0 <= iy < self.map_size:
                    cv2.circle(map_img, (ix, iy), int(coin['radius']), (0, 255, 0), 2)
                    # Малюємо ID
                    cv2.putText(map_img, str(coin['id']), (ix - 10, iy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                                4)
                    cv2.putText(map_img, str(coin['id']), (ix - 10, iy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 1)

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

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                tracker.update_camera_motion(prev_gray, gray)
            prev_gray = gray.copy()

            # --- ОБРОБКА ЗОБРАЖЕННЯ (Ваш метод) ---
            blur = cv2.bilateralFilter(gray, params['blur_size'], params['blur_sigma'], params['blur_sigma'])
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, params['adapt_block'], params['adapt_c'])

            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large, iterations=2)
            thresh = cv2.dilate(thresh, kernel_large, iterations=2)

            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidates = []
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)

            for c in cnts:
                hull = cv2.convexHull(c)  # Використовуємо Hull
                area = cv2.contourArea(hull)
                perimeter = cv2.arcLength(hull, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))

                if params['min_area'] < area < params['max_area'] and circularity > params['min_circularity']:
                    ((x, y), r) = cv2.minEnclosingCircle(hull)
                    # Відступ від країв збільшено для надійності
                    on_border = (x < 40 or x > w - 40 or y < 40 or y > h - 40)

                    candidates.append({
                        'x': x, 'y': y, 'r': r,
                        'hull': hull, 'frame_num': frame_num,
                        'on_border': on_border,
                        'area': area  # Для сортування
                    })

            # Сортування: найбільші монети - найстабільніші якорі
            candidates.sort(key=lambda k: k['area'], reverse=True)

            # Локальна дедуплікація
            final_candidates = []
            for cand in candidates:
                is_duplicate = False
                for existing in final_candidates:
                    dist = np.sqrt((cand['x'] - existing['x']) ** 2 + (cand['y'] - existing['y']) ** 2)
                    if dist < existing['r'] * 0.8:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    final_candidates.append(cand)

            # Оновлення трекера
            tracker.process_candidates(final_candidates)

            # Візуалізація
            display = frame.copy()
            for cand in final_candidates:
                x, y, r = int(cand['x']), int(cand['y']), int(cand['r'])

                color = (0, 255, 0)
                if cand.get('is_new'): color = (0, 0, 255)
                if cand['id'] == -1: color = (100, 100, 100)

                cv2.circle(display, (x, y), r, color, 2)
                cv2.drawContours(display, [cand['hull']], -1, (255, 0, 0), 1)
                if cand['id'] != -1:
                    cv2.putText(display, f"ID:{cand['id']}", (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                                2)

            cv2.imshow('thresh', thresh)
            cv2.imshow('Frame', display)
            cv2.imshow('Map', tracker.draw_map())

        key = cv2.waitKey(25 if not paused else 0) & 0xFF
        if key == ord('q'):
            return 'quit'
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            break
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker = GlobalTracker()
            prev_gray = None

    cap.release()
    cv2.destroyAllWindows()

    final_count = len([c for c in tracker.coins if c['seen_count'] > 5])
    print(f"Відео {index}: {final_count} монет.")


# --- ЗАПУСК ---
for i in range(10):
    import os

    fname = f'Var1/Tests/test_{i}.mp4'
    if os.path.exists(fname):
        process_video(fname, i)