import numpy as np
import cv2

def parse_coords(line: str):
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError("格式錯誤：必須是 '001 x1 y1 x2 y2 x3 y3 x4 y4'")
    name = parts[0]
    coords = [(int(parts[i+2]), int(parts[i+1])) for i in range(0, 8, 2)]  # (y, x)
    return name, coords

def compute_overlap_area_ratio(pred_coords, gt_coords, shape=(2000, 2000)):
    pred_mask = np.zeros(shape, dtype=np.uint8)
    gt_mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(pred_mask, [np.array(pred_coords, dtype=np.int32)], 255)
    cv2.fillPoly(gt_mask, [np.array(gt_coords, dtype=np.int32)], 255)

    inter = cv2.bitwise_and(pred_mask, gt_mask)
    inter_area = np.sum(inter > 0)
    gt_area = np.sum(gt_mask > 0)

    return inter_area / gt_area if gt_area > 0 else 0

def compute_accuracy_index(overlap_area_ratio, ssim=1.0):
    return 0.7 * overlap_area_ratio + 0.3 * ssim

# ✨ 使用範例 ✨
pred_line = "002 0 396 0 536 840 536 840 396"  # 預測的座標
gt_line   = "002 0 390 0 536 840 536 840 390"  # 相同應該是1

# 解析
_, pred_coords = parse_coords(pred_line)
_, gt_coords = parse_coords(gt_line)

# 計算 Overlap_Area_Ratio（最大圖大小可調整）
overlap_ratio = compute_overlap_area_ratio(pred_coords, gt_coords)

# 計算 Accuracy_Index（可自行改 ssim 值）
accuracy = compute_accuracy_index(overlap_ratio, ssim=0.9)

# 顯示
print(f"Overlap_Area_Ratio = {overlap_ratio:.4f}")
print(f"Accuracy_Index = {accuracy:.4f}")
