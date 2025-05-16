import cv2
import numpy as np
import os



LEFT_DIR = "left"
RIGHT_DIR = "right"
RESULT_DIR = "411285047"

# 建立輸出資料夾
os.makedirs(RESULT_DIR, exist_ok=True)

# 建立 overlap.txt
overlap_file = open(os.path.join(RESULT_DIR, "overlap.txt"), "w")

# 取得檔案清單
filenames = sorted(os.listdir(LEFT_DIR))

# 初始化特徵偵測器與匹配器
sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))

for name in filenames:
    left_img_path = os.path.join(LEFT_DIR, name)
    right_img_path = os.path.join(RIGHT_DIR, name)

    if not os.path.exists(right_img_path):
        continue  # 若右圖不存在就跳過

    # 讀圖
    img1 = cv2.imread(left_img_path)
    img2 = cv2.imread(right_img_path)

    # 特徵偵測
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 特徵匹配
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        print(f"Not enough matches for {name}")
        continue

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 計算從右圖到左圖的 homography
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 計算圖像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 計算右圖變形後的角落點（以左圖左上角為原點）
    corners_right = np.float32([[0,0], [w2,0], [w2,h2], [0,h2]]).reshape(-1,1,2)
    transformed_corners = cv2.perspectiveTransform(corners_right, H)

    # 找出重疊區域的角落點
    # 重疊區域的左上角
    overlap_tl_x = max(0, transformed_corners[0][0][0])
    overlap_tl_y = max(0, transformed_corners[0][0][1])
    
    # 重疊區域的右上角
    overlap_tr_x = min(w1, transformed_corners[1][0][0])
    overlap_tr_y = max(0, transformed_corners[1][0][1])
    
    # 重疊區域的右下角
    overlap_br_x = min(w1, transformed_corners[2][0][0])
    overlap_br_y = min(h1, transformed_corners[2][0][1])
    
    # 重疊區域的左下角
    overlap_bl_x = max(0, transformed_corners[3][0][0])
    overlap_bl_y = min(h1, transformed_corners[3][0][1])

    # 儲存結果圖
    result_img = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
    result_img[0:h1, 0:w1] = img1
    result_path = os.path.join(RESULT_DIR, name)
    cv2.imwrite(result_path, result_img)

    # 輸出重疊區域的四個角點座標（左上、右上、右下、左下）
    line = f"{os.path.splitext(name)[0]} {int(overlap_tl_y)} {int(overlap_tl_x)} {int(overlap_tr_y)} {int(overlap_tr_x)} {int(overlap_br_y)} {int(overlap_br_x)} {int(overlap_bl_y)} {int(overlap_bl_x)}\n"
    overlap_file.write(line)

overlap_file.close()
print(f"處理完成～")
