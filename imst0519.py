import cv2
import numpy as np
import os

LEFT_DIR = "left"
RIGHT_DIR = "right"
RESULT_DIR = "411285047-4"
os.makedirs(RESULT_DIR, exist_ok=True)
overlap_file = open(os.path.join(RESULT_DIR, "overlap.txt"), "w")
filenames = sorted(os.listdir(LEFT_DIR))

sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))

for name in filenames:
    left_img_path = os.path.join(LEFT_DIR, name)
    right_img_path = os.path.join(RIGHT_DIR, name)

    if not os.path.exists(right_img_path):
        continue

    img1 = cv2.imread(left_img_path)
    img2 = cv2.imread(right_img_path)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        print(f"Not enough matches for {name}")
        continue

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # ➤ 計算變換後右圖的範圍（建立大畫布）
    corners_img2 = np.float32([[0,0], [w2,0], [w2,h2], [0,h2]]).reshape(-1,1,2)
    transformed_corners = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate((transformed_corners, np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    shift = np.array([[-xmin, -ymin]])
    canvas_size = (xmax - xmin, ymax - ymin)
    translation = np.array([[1, 0, shift[0][0]], [0, 1, shift[0][1]], [0, 0, 1]])
    H_shifted = translation @ H

    # ➤ 拼接圖
    result_img = cv2.warpPerspective(img2, H_shifted, canvas_size)
    result_img[shift[0][1]:shift[0][1]+h1, shift[0][0]:shift[0][0]+w1] = img1

    # ➤ warp 白色遮罩來表示右圖實際出現的區域
    right_mask = np.ones((h2, w2), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(right_mask, H_shifted, canvas_size)

    # ➤ 裁切黑邊
    gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    cropped_result = result_img[y:y+h, x:x+w]
    cropped_mask = warped_mask[y:y+h, x:x+w]

    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"No contour found for {name}")
        continue

    # 取最大輪廓（有時可能有雜訊）
    contour = max(contours, key=cv2.contourArea)

    # 擬合四邊形（可調 epsilon 決定簡化程度）
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        print(f"Contour is not a quadrilateral for {name}, got {len(approx)} points")
        continue

    # 排序為：左上、右上、右下、左下（根據 y 再 x 排序）
    approx = approx.reshape(-1, 2)
    sorted_pts = sorted(approx, key=lambda pt: (pt[1], pt[0]))  # sort by y, then x

    # 額外排序成順時針順序（如果你希望明確）
    def order_clockwise(pts):
        center = np.mean(pts, axis=0)
        return sorted(pts, key=lambda pt: -np.arctan2(pt[1]-center[1], pt[0]-center[0]))
    pts_ordered = order_clockwise(sorted_pts)

    line = f"{os.path.splitext(name)[0]}"
    for pt in pts_ordered:
        line += f" {pt[1]} {pt[0]}"  # y x
    line += "\n"

    overlap_file.write(line)

    # ➤ 儲存結果圖
    result_path = os.path.join(RESULT_DIR, name)
    cv2.imwrite(result_path, cropped_result)

overlap_file.close()
print("✅ 全部完成")
