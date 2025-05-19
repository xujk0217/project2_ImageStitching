import cv2
import numpy as np
import os

class Image_Stitching():
    def __init__(self):
        self.ratio = 0.85
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size = 800

    def registration(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
        if len(good_points) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            H, _ = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
            return H
        else:
            raise ValueError("Not enough good matches")

    def create_mask(self, img1, img2, version):
        h1 = img1.shape[0]
        w1 = img1.shape[1]
        w2 = img2.shape[1]
        height_panorama = h1
        width_panorama = w1 + w2
        offset = int(self.smoothing_window_size / 2)
        barrier = w1 - offset
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        H = self.registration(img1, img2)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # 1. 計算右圖經 Homography 後的角點
        corners_img2 = np.float32([[0,0], [w2,0], [w2,h2], [0,h2]]).reshape(-1,1,2)
        transformed_corners = cv2.perspectiveTransform(corners_img2, H)
        
        # 2. 包含左圖與右圖後的新 canvas 大小
        all_corners = np.concatenate((transformed_corners, np.float32([[0,0], [w1,0], [w1,h1], [0,h1]]).reshape(-1,1,2)), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

        shift = np.array([[-xmin, -ymin]])
        canvas_size = (xmax - xmin, ymax - ymin)

        # 3. 建立偏移矩陣，讓左圖貼在 (0,0)
        translation = np.array([[1, 0, shift[0][0]],
                                [0, 1, shift[0][1]],
                                [0, 0, 1]])
        H_shifted = translation @ H

        # 4. warp right image + 將左圖貼到 canvas 上
        result = cv2.warpPerspective(img2, H_shifted, canvas_size)
        result[shift[0][1]:shift[0][1]+h1, shift[0][0]:shift[0][0]+w1] = img1

        # 5. 建立右圖的 mask 用來找 overlap 區域
        mask_right = np.ones((h2, w2), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask_right, H_shifted, canvas_size)

        # 6. 裁黑邊
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        cropped = result[y:y+h, x:x+w]
        cropped_mask = warped_mask[y:y+h, x:x+w]

        # 7. 重疊區四角
        ys, xs = np.where(cropped_mask > 0)
        if len(xs) < 4:
            raise ValueError("overlap too small")
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        overlap_coords = [
            (min_y, min_x),
            (min_y, max_x),
            (max_y, max_x),
            (max_y, min_x)
        ]

        return cropped, overlap_coords


def process_folder(left_dir="left", right_dir="right", result_dir="411285047--"):
    os.makedirs(result_dir, exist_ok=True)
    overlap_txt_path = os.path.join(result_dir, "overlap.txt")
    with open(overlap_txt_path, "w") as overlap_file:
        stitcher = Image_Stitching()
        filenames = sorted(os.listdir(left_dir))

        for name in filenames:
            left_path = os.path.join(left_dir, name)
            right_path = os.path.join(right_dir, name)
            output_path = os.path.join(result_dir, name)

            if not os.path.exists(right_path):
                print(f"⚠️ 右圖缺少：{name}，跳過")
                continue

            print(f"🧵 處理：{name}")
            img1 = cv2.imread(left_path)
            img2 = cv2.imread(right_path)

            try:
                result_img, overlap = stitcher.blending(img1, img2)
                cv2.imwrite(output_path, result_img)
                # 輸出一行 overlap 座標
                line = f"{os.path.splitext(name)[0]}"
                for y, x in overlap:
                    line += f" {y} {x}"
                overlap_file.write(line + "\n")
            except Exception as e:
                print(f"❌ 錯誤於 {name}: {e}")


if __name__ == '__main__':
    process_folder()

