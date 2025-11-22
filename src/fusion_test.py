import cv2
import numpy as np

# 這三個值是用校正檔算出來的
# s  = 0.64  # ≈ fx_tof / fx_rgb
# tx = -31
# ty = -116
s  = 0.63  # ≈ fx_tof / fx_rgb
tx = -26
ty = -108
# 讀圖
rgb = cv2.imread("D:/Collect_test_data_2/G_06_rgb.png")
# depth = cv2.imread("D:/Temp2/G_01_depth_heatmap.png", -1)
# depth = cv2.imread("D:/Collect_test_data/G_03_depth_heatmap.png", -1)  # 只拿來決定畫布大小
depth = cv2.imread("D:/Collect_test_data_2/G_06_raw_depth.png", -1)
h_d, w_d = depth.shape[:2]

# 建一個 2x3 仿射矩陣： [ s  0  tx
#                          0  s  ty ]
M = np.array([[s, 0, tx],
              [0, s, ty]], dtype=np.float32)

# 把 RGB warp 到 depth 的座標系大小
rgb_warp = cv2.warpAffine(rgb, M, (w_d, h_d))

# 看看疊合效果（這裡假設你有做 depth 視覺化）
depth_vis = cv2.convertScaleAbs(depth, alpha=255.0 / np.max(depth))
depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(rgb_warp, 0.7, depth_vis, 0.3, 0)

# cv2.imshow("RGB warped", rgb_warp)
# cv2.imshow("Depth vis", depth_vis)
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
