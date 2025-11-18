import cv2

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

if __name__ == "__main__":
    img_mouse_click_xy(img_path="D:/overlay_heatmap.png")