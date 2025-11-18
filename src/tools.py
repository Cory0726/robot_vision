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

def crop_by_4_points(img, points):
    """
    Crop the image by 4 input points

    :param img: Original image (H, W, 3)
    :param points: Four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], Note: x = horizontal axis (column), y = vertical axis (row)
    :return: Cropped image (H, W, 3)
    """
    pts = np.array(points, dtype=np.int32)

    # Compute the bounding box from the 4 points
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])

    # Boundary check to ensure valid crop range
    h, w = img.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)

    # OpenCV indexing: img[row, col] = img[y, x]
    cropped_img = img[y_min:y_max, x_min:x_max]

    return cropped_img

if __name__ == "__main__":
    img_mouse_click_xy(img_path="D:/overlay_heatmap.png")