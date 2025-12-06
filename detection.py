import cv2
import numpy as np

params = {
    'blur_d': 9,
    'blur_sigma_color': 75,
    'blur_sigma_space': 75,
    'adapt_block_size': 15,
    'adapt_c': 4,
    'morph_open_iter': 2,
    'morph_close_iter': 2,
    'dilate_iter': 2,
    'min_area': 200,
    'max_area': 20000,
    'min_circularity': 50,
}


def update_param(x):
    pass


def setup_trackbars():
    cv2.namedWindow('Parameters')
    cv2.createTrackbar('Blur D', 'Parameters', params['blur_d'], 25, update_param)
    cv2.createTrackbar('Blur Sigma C', 'Parameters', params['blur_sigma_color'], 200, update_param)
    cv2.createTrackbar('Blur Sigma S', 'Parameters', params['blur_sigma_space'], 200, update_param)
    cv2.createTrackbar('Adapt Block', 'Parameters', params['adapt_block_size'], 51, update_param)
    cv2.createTrackbar('Adapt C', 'Parameters', params['adapt_c'], 20, update_param)
    cv2.createTrackbar('Morph Open', 'Parameters', params['morph_open_iter'], 10, update_param)
    cv2.createTrackbar('Morph Close', 'Parameters', params['morph_close_iter'], 10, update_param)
    cv2.createTrackbar('Dilate', 'Parameters', params['dilate_iter'], 10, update_param)
    cv2.createTrackbar('Min Area', 'Parameters', params['min_area'], 2000, update_param)
    cv2.createTrackbar('Max Area', 'Parameters', params['max_area'], 50000, update_param)
    cv2.createTrackbar('Circularity x100', 'Parameters', params['min_circularity'], 100, update_param)


def get_current_params():
    p = {}
    p['blur_d'] = cv2.getTrackbarPos('Blur D', 'Parameters')
    if p['blur_d'] % 2 == 0:
        p['blur_d'] += 1
    p['blur_d'] = max(1, p['blur_d'])

    p['blur_sigma_color'] = cv2.getTrackbarPos('Blur Sigma C', 'Parameters')
    p['blur_sigma_space'] = cv2.getTrackbarPos('Blur Sigma S', 'Parameters')

    p['adapt_block_size'] = cv2.getTrackbarPos('Adapt Block', 'Parameters')
    if p['adapt_block_size'] % 2 == 0:
        p['adapt_block_size'] += 1
    p['adapt_block_size'] = max(3, p['adapt_block_size'])

    p['adapt_c'] = cv2.getTrackbarPos('Adapt C', 'Parameters')
    p['morph_open_iter'] = cv2.getTrackbarPos('Morph Open', 'Parameters')
    p['morph_close_iter'] = cv2.getTrackbarPos('Morph Close', 'Parameters')
    p['dilate_iter'] = cv2.getTrackbarPos('Dilate', 'Parameters')
    p['min_area'] = cv2.getTrackbarPos('Min Area', 'Parameters')
    p['max_area'] = cv2.getTrackbarPos('Max Area', 'Parameters')
    p['min_circularity'] = cv2.getTrackbarPos('Circularity x100', 'Parameters') / 100.0

    return p


def estimate_camera_motion(prev_gray, curr_gray):
    try:
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            return None

        good_matches = matches[:50]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        return M
    except:
        return None


def transform_point(point, matrix):
    if matrix is None:
        return point
    x, y = point
    transformed = matrix @ np.array([x, y, 1])
    return transformed[0], transformed[1]


def check_video(cap, index, tuning_mode=False):
    paused = False
    tracked_coins = []
    next_coin_id = 0

    prev_gray = None

    AREA_DIFF_THRESHOLD = 0.5
    MIN_OBSERVATIONS = 3
    DUPLICATE_THRESHOLD = 1.2
    MATCH_THRESHOLD = 2.5
    FORGET_FRAMES = 100

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 720), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if tuning_mode:
                p = get_current_params()
            else:
                p = params

            camera_motion = None
            if prev_gray is not None:
                camera_motion = estimate_camera_motion(prev_gray, gray)
            prev_gray = gray.copy()

            if p['blur_sigma_color'] > 0 and p['blur_sigma_space'] > 0:
                blur = cv2.bilateralFilter(gray, p['blur_d'], p['blur_sigma_color'], p['blur_sigma_space'])
            else:
                blur = gray

            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, p['adapt_block_size'], p['adapt_c'])

            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            if p['morph_open_iter'] > 0:
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=p['morph_open_iter'])
            if p['morph_close_iter'] > 0:
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large, iterations=p['morph_close_iter'])
            if p['dilate_iter'] > 0:
                thresh = cv2.dilate(thresh, kernel_large, iterations=p['dilate_iter'])

            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []

            for c in cnts:
                hull = cv2.convexHull(c)
                area = cv2.contourArea(hull)
                perimeter = cv2.arcLength(hull, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * (area / (perimeter * perimeter))

                if p['min_area'] < area < p['max_area'] and circularity > p['min_circularity']:
                    ((x, y), r) = cv2.minEnclosingCircle(hull)
                    candidates.append({
                        'x': x,
                        'y': y,
                        'r': r,
                        'area': area,
                        'hull': hull,
                        'circularity': circularity
                    })

            candidates.sort(key=lambda k: k['circularity'], reverse=True)

            final_coins = []
            for cand in candidates:
                is_duplicate = False
                for existing in final_coins:
                    dist = np.sqrt((cand['x'] - existing['x']) ** 2 + (cand['y'] - existing['y']) ** 2)
                    avg_radius = (cand['r'] + existing['r']) / 2
                    relative_dist = dist / avg_radius if avg_radius > 0 else dist

                    if relative_dist < DUPLICATE_THRESHOLD:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    final_coins.append(cand)

            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if camera_motion is not None:
                for tracked in tracked_coins:
                    try:
                        inv_motion = cv2.invertAffineTransform(camera_motion)
                        tx, ty = transform_point((tracked['x'], tracked['y']), inv_motion)
                        tracked['x'] = tx
                        tracked['y'] = ty
                    except:
                        pass

            matched_tracked_ids = set()

            for coin in final_coins:
                best_match = None
                best_score = float('inf')

                for tracked in tracked_coins:
                    if tracked['id'] in matched_tracked_ids:
                        continue

                    dist = np.sqrt((coin['x'] - tracked['x']) ** 2 + (coin['y'] - tracked['y']) ** 2)
                    area_diff = abs(coin['area'] - tracked['area']) / tracked['area'] if tracked['area'] > 0 else 1

                    avg_radius = (coin['r'] + tracked['r']) / 2
                    relative_dist = dist / avg_radius if avg_radius > 0 else dist

                    if relative_dist < MATCH_THRESHOLD and area_diff < AREA_DIFF_THRESHOLD:
                        score = relative_dist + area_diff * 2
                        if score < best_score:
                            best_score = score
                            best_match = tracked

                if best_match is not None:
                    alpha = 0.3
                    best_match['x'] = alpha * coin['x'] + (1 - alpha) * best_match['x']
                    best_match['y'] = alpha * coin['y'] + (1 - alpha) * best_match['y']
                    best_match['r'] = alpha * coin['r'] + (1 - alpha) * best_match['r']
                    best_match['area'] = alpha * coin['area'] + (1 - alpha) * best_match['area']
                    best_match['last_seen'] = current_frame
                    best_match['count'] += 1
                    best_match['active'] = True
                    matched_tracked_ids.add(best_match['id'])

                    coin['matched_id'] = best_match['id']
                else:
                    is_too_close_to_existing = False
                    for tracked in tracked_coins:
                        dist = np.sqrt((coin['x'] - tracked['x']) ** 2 + (coin['y'] - tracked['y']) ** 2)
                        avg_radius = (coin['r'] + tracked['r']) / 2
                        relative_dist = dist / avg_radius if avg_radius > 0 else dist

                        if relative_dist < DUPLICATE_THRESHOLD * 1.5:
                            area_diff = abs(coin['area'] - tracked['area']) / tracked['area'] if tracked[
                                                                                                     'area'] > 0 else 1
                            if area_diff < AREA_DIFF_THRESHOLD * 1.2:
                                is_too_close_to_existing = True
                                coin['matched_id'] = tracked['id']
                                break

                    if not is_too_close_to_existing:
                        coin['id'] = next_coin_id
                        coin['last_seen'] = current_frame
                        coin['first_seen'] = current_frame
                        coin['count'] = 1
                        coin['active'] = True
                        coin['matched_id'] = next_coin_id
                        tracked_coins.append(coin)
                        next_coin_id += 1

            for tracked in tracked_coins:
                if current_frame - tracked['last_seen'] > 5:
                    tracked['active'] = False

            tracked_coins = [t for t in tracked_coins if current_frame - t['last_seen'] < FORGET_FRAMES]

            display = frame.copy()
            for coin in final_coins:
                x, y, r = int(coin['x']), int(coin['y']), int(coin['r'])

                if 'matched_id' in coin:
                    color = (0, 255, 0)
                else:
                    color = (0, 165, 255)

                cv2.circle(display, (x, y), r, color, 2)
                cv2.circle(display, (x, y), 2, (0, 0, 255), -1)
                cv2.drawContours(display, [coin['hull']], -1, (255, 0, 0), 1)

                if 'matched_id' in coin:
                    cv2.putText(display, f"#{coin['matched_id']}", (x - 20, y - int(r) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(display, f"{int(coin['area'])}", (x - 20, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            confirmed = [t for t in tracked_coins if t['count'] >= MIN_OBSERVATIONS]
            active_confirmed = [t for t in confirmed if t['active']]

            cv2.putText(display, f"In frame: {len(final_coins)}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display, f"Active: {len(active_confirmed)}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display, f"Total unique: {len(confirmed)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(display, f"Video: {index} | Frame: {int(current_frame)}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if tuning_mode:
                cv2.putText(display, "TUNING MODE", (10, 195),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow('1. Original', frame)
        cv2.imshow('2. Blur', blur)
        cv2.imshow('3. Threshold', thresh)
        cv2.imshow('4. Detection', display)

        key = cv2.waitKey(25 if not paused else 0) & 0xFF

        if key == ord('q'):
            return 'quit'
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            break
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracked_coins = []
            next_coin_id = 0
            prev_gray = None
        elif key == ord('s') and tuning_mode:
            current = get_current_params()
            print("\n" + "=" * 60)
            print("ЗБЕРЕЖЕНІ ПАРАМЕТРИ:")
            for k, v in current.items():
                print(f"  '{k}': {v},")
            print("=" * 60 + "\n")

    confirmed = [t for t in tracked_coins if t['count'] >= MIN_OBSERVATIONS]

    print(f"\n{'=' * 60}")
    print(f"Відео {index} - ФІНАЛЬНИЙ РЕЗУЛЬТАТ:")
    print(f"Всього унікальних монет на столі: {len(confirmed)}")
    print(f"\nДеталі по кожній монеті:")
    for coin in sorted(confirmed, key=lambda c: c['id']):
        print(f"  Монета #{coin['id']:2d}: "
              f"спостережень={coin['count']:3d}, "
              f"area={coin['area']:6.0f}, "
              f"pos=({coin['x']:5.1f}, {coin['y']:5.1f})")
    print(f"{'=' * 60}\n")

    return 'continue'


print("=" * 60)
print("Система підрахунку монет з відстеженням")
print("=" * 60)
print("\nВиберіть режим:")
print("  1 - Режим НАЛАШТУВАННЯ (з трекбарами)")
print("  2 - Звичайний режим")
print("=" * 60)

mode = input("Оберіть режим (1 або 2): ").strip()
tuning_mode = (mode == "1")

if tuning_mode:
    setup_trackbars()
    print("\nРежим налаштування активовано!")
    print("\nКерування:")
    print("  Пробіл - пауза/продовження")
    print("  S - зберегти параметри")
    print("  N - наступне відео")
    print("  R - перезапустити відео")
    print("  Q - вихід")
else:
    print("\nЗвичайний режим")
    print("\nКерування:")
    print("  Пробіл - пауза/продовження")
    print("  N - наступне відео")
    print("  R - перезапустити відео")
    print("  Q - вихід")

print("=" * 60 + "\n")

for i in range(10):
    cap = cv2.VideoCapture(f"Var1/Tests/test_{i}.mp4")

    if not cap.isOpened():
        print(f"Не вдалося відкрити test_{i}.mp4")
        continue

    result = check_video(cap, i, tuning_mode)
    cap.release()

    if result == 'quit':
        break

cv2.destroyAllWindows()