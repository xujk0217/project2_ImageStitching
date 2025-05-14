import cv2
import numpy as np
import os

# 資料夾路徑
left_dir = 'left'
right_dir = 'right'
result_dir = 'result'

# 確保 result 資料夾存在
os.makedirs(result_dir, exist_ok=True)

# 取得左圖檔名清單（假設兩邊名稱一致）
left_files = sorted(os.listdir(left_dir))

# SIFT 建立與配對參數
sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

for filename in left_files:
    left_path = os.path.join(left_dir, filename)
    right_path = os.path.join(right_dir, filename)

    # 確保右圖存在
    if not os.path.exists(right_path):
        print(f'⚠️ 找不到對應的右圖：{right_path}')
        continue

    # 讀取圖像
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    # 灰階處理
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 偵測與計算描述子
    kp1, des1 = sift.detectAndCompute(gray_left, None)
    kp2, des2 = sift.detectAndCompute(gray_right, None)

    # 特徵點配對
    matches = flann.knnMatch(des1, des2, k=2)

    # 使用 Lowe's ratio 篩選好配對點
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 計算 Homography 矩陣
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # warp 右圖到左圖座標系
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]
        result = cv2.warpPerspective(img_right, H, (w1 + w2, max(h1, h2)))
        result[0:h1, 0:w1] = img_left

        # 儲存結果圖像
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(result_dir, f'{base_name}_result.jpg')
        cv2.imwrite(output_path, result)
        print(f'✅ 已輸出：{output_path}')
    else:
        print(f'❌ 配對點不足，跳過：{filename}')
