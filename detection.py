import cv2
import numpy as np
import math

# --- ПАРАМЕТРИ ---
params = {
    'min_area': 500,  # Трохи збільшив, щоб відсіяти шуми
    'max_area': 15000,
    'min_circularity': 0.7,  # Більш сувора перевірка на коло
    'match_dist_threshold': 80  # Радіус (в пікселях) для злиття монет на глобальній карті
}


class GlobalTracker:
    def __init__(self):
        self.global_transform = np.eye(3, dtype=np.float32)
        self.coins = []
        self.next_id = 1

        # Параметри карти
        self.map_size = 2000
        self.map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.map_center = np.array([self.map_size // 2, self.map_size // 2])

    def update_camera_motion(self, prev_gray, curr_gray):
        if prev_gray is None:
            return False

        # 1. Знаходимо точки (features) на попередньому кадрі
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=30)

        if p0 is None or len(p0) < 8:
            return False

        # 2. Шукаємо ці точки на новому кадрі (Optical Flow)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

        # 3. Відбираємо тільки ті, що знайшлися
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) < 8:
            return False

        # 4. Обчислюємо матрицю трансформації (Translation + Rotation)
        # Ransac відсіює точки, які рухаються неправильно (наприклад, відблиски на монетах)
        m, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if m is None:
            return False

        # Важливо: ми рахуємо рух сцени відносно камери, тому беремо обернену трансформацію
        # Щоб отримати рух камери по карті
        m_inv = cv2.invertAffineTransform(m)

        # Додаємо рядок [0, 0, 1]
        m_3x3 = np.vstack([m_inv, [0, 0, 1]])

        # Акумулюємо трансформацію
        self.global_transform = self.global_transform @ m_3x3
        return True

    def frame_to_global(self, x, y):
        vec = np.array([x, y, 1.0])
        global_vec = self.global_transform @ vec
        return global_vec[0], global_vec[1]

    def global_to_map(self, gx, gy):
        """Переводить глобальні координати в координати зображення карти (з центром)"""
        return int(gx + self.map_center[0]), int(gy + self.map_center[1])

    def process_detections(self, detections):
        for det in detections:
            gx, gy = self.frame_to_global(det['x'], det['y'])

            # Пошук збігів
            best_match = None
            min_dist = float('inf')

            for known_coin in self.coins:
                dist = math.hypot(gx - known_coin['global_pos'][0], gy - known_coin['global_pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    best_match = known_coin

            if best_match and min_dist < params['match_dist_threshold']:
                # Оновлюємо існуючу монету
                best_match['seen_count'] += 1
                # Плавне уточнення координат (Low-pass filter)
                alpha = 0.2
                best_match['global_pos'] = (
                    best_match['global_pos'][0] * (1 - alpha) + gx * alpha,
                    best_match['global_pos'][1] * (1 - alpha) + gy * alpha
                )
                det['id'] = best_match['id']
                det['is_new'] = False
            else:
                # Нова монета
                new_coin = {
                    'id': self.next_id,
                    'global_pos': (gx, gy),
                    'radius': det['r'],
                    'seen_count': 1
                }
                self.coins.append(new_coin)
                det['id'] = self.next_id
                det['is_new'] = True
                self.next_id += 1

    def draw_map(self, frame_w, frame_h):
        self.map_img.fill(20)  # Темно-сірий фон

        # Малюємо сітку
        cx, cy = self.map_center
        cv2.line(self.map_img, (0, cy), (self.map_size, cy), (50, 50, 50), 1)
        cv2.line(self.map_img, (cx, 0), (cx, self.map_size), (50, 50, 50), 1)

        # 1. Малюємо монети
        count = 0
        for coin in self.coins:
            if coin['seen_count'] > 10:  # Фільтр стабільності
                count += 1
                ix, iy = self.global_to_map(*coin['global_pos'])

                if 0 <= ix < self.map_size and 0 <= iy < self.map_size:
                    cv2.circle(self.map_img, (ix, iy), int(coin['radius']), (0, 255, 0), 2)
                    cv2.circle(self.map_img, (ix, iy), 2, (0, 0, 255), -1)
                    cv2.putText(self.map_img, str(coin['id']), (ix + 10, iy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 2. Малюємо поточне поле зору камери (прямокутник)
        # Кути кадру в локальних координатах
        corners = [(0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)]
        map_corners = []
        for x, y in corners:
            gx, gy = self.frame_to_global(x, y)
            map_corners.append(self.global_to_map(gx, gy))

        pts = np.array(map_corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.map_img, [pts], True, (0, 255, 255), 2)

        cv2.putText(self.map_img, f"Total Coins: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # Ресайз для відображення, якщо карта занадто велика
        return cv2.resize(self.map_img, (600, 600))


def process_video(video_path, index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    tracker = GlobalTracker()
    prev_gray = None

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Map", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 720), interpolation=cv2.INTER_CUBIC)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- ВАЖЛИВО: ОНОВЛЕННЯ РУХУ ---
        if prev_gray is not None:
            tracker.update_camera_motion(prev_gray, gray)

        # Зберігаємо поточний кадр як попередній для наступної ітерації
        prev_gray = gray.copy()
        # -------------------------------

        # Обробка зображення
        blur = cv2.bilateralFilter(gray, 9, 33, 33)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 9, 3)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        thresh = cv2.dilate(thresh, kernel_large, iterations=2)

        # Пошук монет
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for c in cnts:
            area = cv2.contourArea(c)
            if params['min_area'] < area < params['max_area']:
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter ** 2))

                if circularity > params['min_circularity']:
                    ((x, y), r) = cv2.minEnclosingCircle(c)
                    # Відсікаємо краї (там часто помилки)
                    if r > 10 and x > 10 and x < w - 10 and y > 10 and y < h - 10:
                        detections.append({'x': x, 'y': y, 'r': r})

        # Оновлення логіки трекера
        tracker.process_detections(detections)

        # Візуалізація результату на кадрі
        for det in detections:
            color = (0, 255, 0) if not det.get('is_new', True) else (0, 0, 255)
            cv2.circle(frame, (int(det['x']), int(det['y'])), int(det['r']), color, 2)
            cv2.putText(frame, f"#{det.get('id', '?')}", (int(det['x']), int(det['y'])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Візуалізація карти
        map_vis = tracker.draw_map(w, h)

        cv2.imshow("Result", frame)
        cv2.imshow("Map", map_vis)
        cv2.imshow("Mask", thresh)

        key = cv2.waitKey(20)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    final_count = len([c for c in tracker.coins if c['seen_count'] > 10])
    print(f"Відео {index}: Унікальних монет: {final_count}")


# ЗАПУСК
print("Start processing...")
for i in range(10):
    # Перевірте, чи існує файл перед запуском
    import os

    fname = f'Var1/Tests/test_{i}.mp4'
    if os.path.exists(fname):
        process_video(fname, i)
    else:
        print(f"File {fname} not found.")