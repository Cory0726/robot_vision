import os
from pypylon import pylon
import cv2
import basler_cam_init
import numpy as np
from pathlib import Path

def create_rgb_cam_obj():
    """
    Create a RGB camera object by serial number.
    """
    rgb_cam_sn = "24747625"
    rgb_cam = basler_cam_init.create_basler_cam(rgb_cam_sn)
    return rgb_cam

def config_rgb_cam_para(cam: pylon.InstantCamera) -> None:
    """
    Configurate RGB camera (acA1300-75gc) parameter after opening the camera.

    Args:
        camera (pylon.InstantCamera): A RGB camera instance
    """
    # Width and height
    cam.Width.Value = 1280
    cam.Height.Value = 1024
    # Pixel format
    cam.PixelFormat.Value = "BayerBG8"
    # Exposure time (Abs) [us]
    cam.ExposureTimeAbs.Value = 7500
    # Exposure auto
    cam.ExposureAuto.Value = "Off"
    # Gain (Raw)
    cam.GainSelector.Value = "All"
    cam.GainRaw.Value = 136
    # Gain auto
    cam.GainAuto.Value = "Off"
    # Balance white auto
    cam.BalanceWhiteAuto.Value = "Off"

def stream_rgb_img() -> None:
    """
    Streaming the RGB images from basler RGB camera.
    """
    # Initialize the rgb camera
    cam = create_rgb_cam_obj()
    cam.Open()
    config_rgb_cam_para(cam)

    # Start the grabbing of images with strategy
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
    print("Start streaming RGB images ...")
    while cam.IsGrabbing():
        # Get the grab retrieve
        grab_retrieve = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_retrieve.GrabSucceeded():
            bayer_image = grab_retrieve.Array

            # Convert bayer to RGB
            rgb_img = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)
            cv2.imshow("RGB", rgb_img)

            # Read the keyboard keyin
            key = cv2.waitKey(5) & 0xFF
            # Break the loop by pressing q
            if key == ord("q"):
                break
            # Save the image by pressing s
            elif key == ord("s"):
                file_number = 0
                file_path = f"robot_vision_result/rgb_img_by_stream_{file_number:02d}.png"
                while os.path.exists(file_path):
                    file_number += 1
                    file_path = f"robot_vision_result/rgb_img_by_stream_{file_number:02d}.png"
                cv2.imwrite(file_path, rgb_img)
                print(f"Saved: {file_path}")
        
            grab_retrieve.Release()
    cam.StopGrabbing()
    cam.Close()
    cv2.destroyAllWindows

def grab_one_rgb_img():
    # Initialize the RGB camera
    cam = create_rgb_cam_obj()
    cam.Open()
    config_rgb_cam_para(cam)

    # Grab one rgb image
    grab_result = cam.GrabOne(1000)  # timeout: 1 s
    if grab_result.GrabSucceeded():
        bayer_img = grab_result.Array
        rgb_img = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_BG2RGB)  # Convert bayer to RGB
    grab_result.Release()
    cam.Close()
    return rgb_img

def halcon_to_opencv_intrinsics(
    Sx_um=8.30179, Sy_um=8.3,             # pixel size [micrometers]
    f_mm=7.09725,                         # focal length [mm]
    Cx_px=635.69, Cy_px=535.704,          # principal point [pixels]
    K1=5067.98, K2=8.93518e6, K3=-6.00845e10,  # radial distortion [1/m^2, 1/m^4, 1/m^6]
    P1=-0.128108, P2=0.130449,                 # tangential distortion [1/m^2]
):
    """
    Convert HALCON-style camera parameters into OpenCV-style intrinsic matrix and distortion coefficients.
    HALCON expresses distortion in metric units; OpenCV expects normalized coordinates.
    """

    # --- Unit conversion to meters ---
    f_m  = f_mm / 1000.0     # mm → m
    Sx_m = Sx_um * 1e-6      # μm → m
    Sy_m = Sy_um * 1e-6

    # --- Compute focal lengths in pixels ---
    fx = f_m / Sx_m
    fy = f_m / Sy_m
    cx = Cx_px
    cy = Cy_px

    # --- Camera intrinsic matrix ---
    K = np.array([[fx, 0,  cx],
                [0,  fy, cy],
                [0,   0,  1]], dtype=np.float64)

    # --- Convert HALCON distortion coefficients to OpenCV format ---
    # (since HALCON uses metric units, we multiply by f powers)
    k1 = K1 * (f_m**2)
    k2 = K2 * (f_m**4)
    k3 = K3 * (f_m**6)
    p1 = P1 * (f_m**2)
    p2 = P2 * (f_m**2)

    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return K, dist

def undistort_rgb_image(img, alpha=1.0):
    """
    Apply lens undistortion to an RGB image (OpenCV array) and return the corrected image and new camera matrix.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image (e.g., loaded via cv2.imread or from camera stream).
    alpha : float, optional
        Trade-off between cropping and field of view in cv2.getOptimalNewCameraMatrix.
        - alpha = 0 : no black borders, tighter crop.
        - alpha = 1 : keep full FOV, possible black borders.

    Returns
    -------
    undistorted_img : np.ndarray
        The undistorted RGB image.
    newK : np.ndarray
        The new optimal camera matrix.
    """

    # Build OpenCV intrinsics and distortion coefficients
    K, dist = halcon_to_opencv_intrinsics()

    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Input must be a valid image (numpy array).")
    h, w = img.shape[:2]

    # Compute new optimal camera matrix (alpha controls cropping)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=alpha)

    # Undistort image
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, R=None, newCameraMatrix=newK,
                                            size=(w, h), m1type=cv2.CV_32FC1)
    undist_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Alternatively, use remapping for faster processing:
    # undist_img = cv2.undistort(img, K, dist, None, newK)

    # Crop the image based on ROI (optional)
    # x, y, rw, rh = roi
    # if rw > 0 and rh > 0:
    #     undist_img = undist_img[y:y+rh, x:x+rw]


    # Return useful info for debugging
    return undist_img, newK
