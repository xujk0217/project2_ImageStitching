import cv2
import numpy as np
import os

LEFT_DIR = "left"
RIGHT_DIR = "right"
RESULT_DIR = "411285047-6"

os.makedirs(RESULT_DIR, exist_ok=True)
overlap_file = open(os.path.join(RESULT_DIR, "overlap.txt"), "w")
filenames = sorted(os.listdir(LEFT_DIR))

# 初始化特徵偵測與匹配器
sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.03)
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))

for name in filenames:
    left_img_path = os.path.join(LEFT_DIR, name)
    right_img_path = os.path.join(RIGHT_DIR, name)

    if not os.path.exists(right_img_path):
        continue

    img1 = cv2.imread(left_img_path)
    img2 = cv2.imread(right_img_path)

    # 特徵偵測與匹配
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) < 4:
        print(f"❌ Not enough matches for {name}")
        continue

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # warp 右圖到新畫布
    result_img = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
    result_img[0:h1, 0:w1] = img1

    # === 使用右圖非黑區製作 mask（關鍵步驟）===
    right_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, right_mask = cv2.threshold(right_gray, 1, 255, cv2.THRESH_BINARY)
    warped_mask = cv2.warpPerspective(right_mask, H, (w1 + w2, max(h1, h2)))

    # 建立左圖的 mask（貼在 (0, 0)）
    left_mask = np.zeros_like(warped_mask, dtype=np.uint8)
    left_mask[0:h1, 0:w1] = 255

    # 與右圖 warp mask 做交集
    overlap_mask = cv2.bitwise_and(warped_mask, left_mask)

    # 找重疊區的有效像素
    ys, xs = np.where(overlap_mask > 0)
    if len(xs) < 4:
        print(f"❌ Too little overlap for {name}")
        continue

    # 取得準確重疊區邊界
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    # 真正的重疊區座標
    tl_x, tl_y = min_x, min_y
    tr_x, tr_y = max_x, min_y
    br_x, br_y = max_x, max_y
    bl_x, bl_y = min_x, max_y


    # 四角座標（左上、右上、右下、左下）
    tl_x, tl_y = min_x, min_y
    tr_x, tr_y = max_x, min_y
    br_x, br_y = max_x, max_y
    bl_x, bl_y = min_x, max_y

    # 輸出 overlap.txt
    line = f"{os.path.splitext(name)[0]} {tl_y} {tl_x} {tr_y} {tr_x} {br_y} {br_x} {bl_y} {bl_x}\n"
    overlap_file.write(line)

    # === 畫出重疊區塊（debug 用）===
    pts = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], np.int32)
    cv2.polylines(result_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 儲存結果圖
    result_path = os.path.join(RESULT_DIR, name)
    cv2.imwrite(result_path, result_img)

    debug_path = os.path.join(RESULT_DIR, f"debug_{name}")
    cv2.imwrite(debug_path, result_img)

overlap_file.close()
print("✅ 全部處理完成")
