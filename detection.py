import cv2
import numpy as np


def check_video(cap, index):
    paused = False
    tracked_coins = []
    next_coin_id = 0

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640,720), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray, 9, 75, 75)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 4)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large, iterations=2)
            thresh = cv2.dilate(thresh, kernel_large, iterations=2)

            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for c in cnts:
                hull = cv2.convexHull(c)
                area = cv2.contourArea(hull)
                perimeter = cv2.arcLength(hull, True)
                if perimeter == 0: continue

                circularity = 4 * np.pi * (area / (perimeter * perimeter))

                if 200 <area < 20000 and circularity > 0.3:
                    ((x, y), r) = cv2.minEnclosingCircle(hull)
                    candidates.append({'x': x, 'y': y, 'r': r, 'area': area, 'hull': hull})

            candidates.sort(key=lambda k: k['area'], reverse=True)

            final_coins = []
            for cand in candidates:
                is_duplicate = False

                for existing in final_coins:
                    dist = np.sqrt((cand['x'] - existing['x']) ** 2 + (cand['y'] - existing['y']) ** 2)


                    if dist < 50 or dist < existing['r'] * 0.8:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    final_coins.append(cand)

            for coin in final_coins:
                is_new_coin = True

                # Перевіряємо, чи це вже відстежена монета
                for tracked in tracked_coins:
                    dist = np.sqrt((coin['x'] - tracked['x']) ** 2 + (coin['y'] - tracked['y']) ** 2)
                    area_diff = abs(coin['area'] - tracked['area']) / tracked['area']

                    if dist < 100 and area_diff < 10:
                        is_new_coin = False
                        tracked['x'] = coin['x']
                        tracked['y'] = coin['y']
                        tracked['last_seen'] = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        break

                if is_new_coin:
                    coin['id'] = next_coin_id
                    coin['last_seen'] = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    tracked_coins.append(coin)
                    next_coin_id += 1

            for coin in final_coins:
                x, y, r = int(coin['x']), int(coin['y']), int(coin['r'])
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                cv2.drawContours(frame, [hull], -1, (255, 0, 0), 1)
                cv2.putText(frame, f"{area:.3f}", (int(x) - 20, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.putText(frame, f"Coins in frame: {len(final_coins)}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Total unique: {len(tracked_coins)}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Video: {index}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('thresh', thresh)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(25 if not paused else 0) & 0xFF

        if key == ord('q'):
            return 'quit'
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            break
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return 'continue'


for i in range(10):
    cap = cv2.VideoCapture(f"Var1/Tests/test_{i}.mp4")

    if not cap.isOpened():
        print(f"Не вдалося відкрити test_{i}.mp4")
        continue

    result = check_video(cap, i)
    cap.release()

    if result == 'quit':
        break

cv2.destroyAllWindows()