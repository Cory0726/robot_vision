import os
from pypylon import pylon
import cv2
import basler_cam_init
import numpy as np

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
