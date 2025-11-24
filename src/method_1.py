import os

import cv2
import numpy as np
import basler_tof_cam_grab
from pypylon import pylon
from tools import rawdepth_to_heatmap

def capture_intensity_and_depth():
    # Initialize the ToF Camera
    cam = basler_tof_cam_grab.create_tof_cam()
    cam.Open()
    basler_tof_cam_grab.config_tof_cam_para(cam)
    # ToF data configuration
    # Open 3d point cloud image
    cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Range")
    cam.GetNodeMap().GetNode("ComponentEnable").SetValue(True)
    cam.GetNodeMap().GetNode("PixelFormat").SetValue("Coord3D_ABC32f")
    # Close intensity image
    cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Intensity")
    cam.GetNodeMap().GetNode("ComponentEnable").SetValue(True)
    cam.GetNodeMap().GetNode("PixelFormat").SetValue("Mono16")
    # Close confidence map
    cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Confidence")
    cam.GetNodeMap().GetNode("ComponentEnable").SetValue(False)
    cam.GetNodeMap().GetNode("PixelFormat").SetValue("Confidence16")

    # Starts the grabbing
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("Start grabbing...")
    while cam.IsGrabbing():
        # Get the grab retrieve
        grab_retrieve = cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

        if grab_retrieve.GrabSucceeded():
            # Get the grab retrieve as data container
            data_container = grab_retrieve.GetDataContainer()
            # Split the container
            data = basler_tof_cam_grab.split_tof_container_data(data_container)

            intensity_image = data["Intensity_Image"]
            raw_depth = basler_tof_cam_grab.pcl_to_rawdepth(data["Point_Cloud"])

            # Display
            cv2.imshow("Intensity_image", intensity_image)
            grab_retrieve.Release()

        # Read the keyboard key-in
        key = cv2.waitKey(5) & 0xFF
        # Break the loop by pressing q
        if key == ord("q"):
            break
        elif key == ord("s"):
            file_number = 0
            intensity_file_path = f"robot_vision_result/M1_{file_number:02d}_intensity_image.png"
            while os.path.exists(intensity_file_path):
                file_number += 1
                intensity_file_path = f"robot_vision_result/M1_{file_number:02d}_intensity_image.png"

            cv2.imwrite(intensity_file_path, intensity_image)
            cv2.imwrite(f"robot_vision_result/M1_{file_number:02d}_raw_depth.png", raw_depth)
            np.save(f"robot_vision_result/M1_{file_number:02d}_raw_depth", raw_depth)
            cv2.imwrite(
                f"robot_vision_result/M1_{file_number:02d}_depth_heatmap.png",
                rawdepth_to_heatmap(raw_depth)
            )

            print(f"Saved complete")

    cam.StopGrabbing()
    cam.Close()
    cv2.destroyAllWindows

if __name__ == "__main__":
    capture_intensity_and_depth()