import cv2
import numpy as np
import os

LEFT_DIR = "left"
RIGHT_DIR = "right"
RESULT_DIR = "411285047-2"

# 建立輸出資料夾
os.makedirs(RESULT_DIR, exist_ok=True)

# 建立 overlap.txt
overlap_file = open(os.path.join(RESULT_DIR, "overlap.txt"), "w")

# 取得檔案清單
filenames = sorted(os.listdir(LEFT_DIR))

# 初始化 SIFT 特徵偵測器與 FLANN 匹配器
sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.03)
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))

for name in filenames:
    left_img_path = os.path.join(LEFT_DIR, name)
    right_img_path = os.path.join(RIGHT_DIR, name)

    if not os.path.exists(right_img_path):
        continue

    # 讀取圖像（灰階）
    img1 = cv2.imread(left_img_path)
    img2 = cv2.imread(right_img_path)

    # 特徵偵測
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 特徵點匹配
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) < 4:
        print(f"Not enough matches for {name}")
        continue

    # 擷取匹配點
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 計算 Homography
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)

    # 尺寸資訊
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 將右圖角落轉換到左圖座標
    corners_right = np.float32([[0,0], [w2,0], [w2,h2], [0,h2]]).reshape(-1,1,2)
    transformed_corners = cv2.perspectiveTransform(corners_right, H)

    # 將轉換後的角點拉平成 (N, 2)
    transformed_points = transformed_corners.reshape(-1, 2)

    # 計算 bounding box（x, y, w, h）
    x, y, w, h = cv2.boundingRect(transformed_points)

    # 強制限制在左圖範圍內（避免負數與越界）
    x = max(0, x)
    y = max(0, y)
    x_end = min(w1, x + w)
    y_end = min(h1, y + h)

    # 四角座標（左上、右上、右下、左下）
    tl_x, tl_y = x, y
    tr_x, tr_y = x_end, y
    br_x, br_y = x_end, y_end
    bl_x, bl_y = x, y_end


    # 寫入 overlap.txt
    line = f"{os.path.splitext(name)[0]} {tl_y} {tl_x} {tr_y} {tr_x} {br_y} {br_x} {bl_y} {bl_x}\n"
    overlap_file.write(line)

    # 合成影像
    result_img = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
    result_img[0:h1, 0:w1] = img1
    result_path = os.path.join(RESULT_DIR, name)
    cv2.imwrite(result_path, result_img)

    # 視覺化重疊區塊（除錯用）
    pts = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], np.int32)
    cv2.polylines(result_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(RESULT_DIR, f"debug_{name}"), result_img)

overlap_file.close()
print("處理完成 ✅")
