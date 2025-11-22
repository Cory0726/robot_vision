import cv2
import numpy as np

def mouse_event_handler(event, x, y, flags, param):
    """
    Mouse event callback function
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse click at x={x}, y={y}")
        img = param["img"]
        # Print the white pin on the image
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        cv2.imshow(param["win_name"], img)

def img_mouse_click_xy(img_path):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    win_name = "Click to get (x, y)"
    cv2.namedWindow(win_name)

    params = {"img": img, "win_name": win_name}

    # Set the MouseCallback function
    cv2.setMouseCallback(win_name, mouse_event_handler, params)

    # Display the image
    cv2.imshow(win_name, img)

    print("在圖片視窗上用滑鼠左鍵點擊，就會印出 x, y 座標。按任意鍵關閉視窗。")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rawdepth_to_heatmap(rawdepth):
    gray_img = cv2.normalize(rawdepth, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(255 - gray_img, cv2.COLORMAP_TURBO)
    # heatmap = cv2.applyColorMap(255 - gray_img, cv2.COLORMAP_JET)
    return heatmap

def crop_by_4_points(arr, points):
    """
    Crop a 2D or 3D NumPy array using 4 input points.
    Supports:
      - 2D arrays: (H, W)
      - 3D arrays: (H, W, C)

    :param arr: Input NumPy array (H, W) or (H, W, C)
    :param points: Four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                   Note: x = column index, y = row index
    :return: Cropped array with the same number of channels as input
    """

    # Ensure arr has at least 2 dimensions
    if arr.ndim < 2:
        raise ValueError("Input array must be at least 2D (H, W)")

    pts = np.array(points, dtype=np.int32)

    # Compute bounding box from the 4 points
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])

    # Image shape
    h, w = arr.shape[:2]

    # Boundary check (prevent out-of-range indexing)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)

    # NumPy slicing works for both 2D (H,W) and 3D (H,W,C)
    cropped_arr = arr[y_min:y_max, x_min:x_max]

    return cropped_arr

def visualize_rgb_depth_alignment(color_img, depth, alpha_rgb: float = 0.6, alpha_depth: float = 0.4):
    """
    Visualize the alignment between an RGB image and a depth map (array version, no saving).

    Generates two visualizations:
        1. Heatmap overlay (RGB + depth colormap)
        2. Edge overlay (RGB edges vs. depth edges)

    Args:
        color_img (np.ndarray): BGR image, uint8, shape (H, W, 3)
        depth (np.ndarray): Depth map in millimeters, uint16, shape (H, W)
        alpha_rgb (float): Opacity of RGB layer in overlay (0–1)
        alpha_depth (float): Opacity of depth heatmap in overlay (0–1)

    Returns:
        overlay_heatmap (np.ndarray): RGB + depth colormap visualization (uint8)
        overlay_edges (np.ndarray): RGB-edge vs depth-edge visualization (uint8)
    """

    # Heatmap overlay
    # Normalize depth to 0–255 and apply colormap
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    # Blend RGB and heatmap
    overlay_heatmap = cv2.addWeighted(color_img, alpha_rgb, depth_color, alpha_depth, 0)

    # Edge overlay
    # Compute RGB edges (white)
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    edges_rgb = cv2.Canny(gray, 100, 200)
    edges_rgb_colored = cv2.cvtColor(edges_rgb, cv2.COLOR_GRAY2BGR)

    # Compute depth edges (red)
    depth_abs = cv2.convertScaleAbs(depth, alpha=0.03)
    edges_depth = cv2.Canny(depth_abs, 50, 150)
    edges_depth_colored = cv2.cvtColor(edges_depth, cv2.COLOR_GRAY2BGR)
    edges_depth_colored[:, :, 1:] = 0  # keep only red channel

    # Combine RGB edges (white) and depth edges (red)
    overlay_edges = cv2.addWeighted(edges_rgb_colored, 1, edges_depth_colored, 1, 0)

    return overlay_heatmap, overlay_edges

if __name__ == "__main__":
    img = cv2.imread("D:/temp2/G_01_rgb.png")  # 讀入原始 RGB 圖

    scale = 0.63  # 63%

    # fx = zoom in/out horizontally
    # fy = zoom in/out vertically
    img_resized = cv2.resize(img, None, fx=scale, fy=scale)

    cv2.imwrite("D:/temp2/G_01_rgb_resized.png", img_resized)

    img_mouse_click_xy(img_path="D:/temp2/G_01_rgb_resized.png")