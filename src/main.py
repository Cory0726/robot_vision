import numpy as np
import os
from pypylon import pylon
import cv2
import basler_rgb_cam_grab
import basler_tof_cam_grab
import trans_3d_to_rgb_frame
import tools


def main():
    cam = basler_rgb_cam_grab.create_rgb_cam_obj()
    cam.Open()
    basler_rgb_cam_grab.config_rgb_cam_para(cam)

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

            # Read the keyboard key in
            key = cv2.waitKey(5) & 0xFF
            # Break the loop by pressing q
            if key == ord("q"):
                break
            # Save the image by pressing s
            elif key == ord("s"):
                # Crop the RGB image
                color_img = tools.crop_by_4_points(
                    arr=rgb_img,
                    points=[(115, 185), (1075, 185), (115, 905), (1075, 905)]  # 720 x 960
                )
                # Get the depth image based on RGB image frame
                pcl = basler_tof_cam_grab.grab_one_point_cloud()
                pcl_color_frame = trans_3d_to_rgb_frame.transform_pcl_to_color_frame(pcl)
                raw_depth, _ = trans_3d_to_rgb_frame.project_depth_to_color_frame(pcl_color_frame, rgb_img)
                raw_depth = tools.crop_by_4_points(
                    arr=raw_depth,
                    points=[(115, 185), (1075, 185), (115, 905), (1075, 905)]   # 720 x 960
                )
                print(f"RGB image shape: {color_img.shape}\n")
                print(f"Raw depth shape: {raw_depth.shape}")
                # Heatmap of depth
                depth_heatmap = tools.rawdepth_to_heatmap(raw_depth)
                # Overlay the color image and depth heatmap
                overlay_heatmap, _ = tools.visualize_rgb_depth_alignment(color_img, depth_heatmap)

                # Save the files
                file_number = 0
                file_path = f"robot_vision_result/G_{file_number:02d}_rgb.png"
                while os.path.exists(file_path):
                    file_number += 1
                    file_path = f"robot_vision_result/G_{file_number:02d}_rgb.png"
                cv2.imwrite(file_path, color_img)
                cv2.imwrite(f"robot_vision_result/G_{file_number:02d}_raw_depth.png", raw_depth)
                np.save(f"robot_vision_result/G_{file_number:02d}_raw_depth", raw_depth)
                cv2.imwrite(f"robot_vision_result/G_{file_number:02d}_depth_heatmap.png", depth_heatmap)
                cv2.imwrite(f"robot_vision_result/G_{file_number:02d}_overlay_heatmap.png", overlay_heatmap)

                print(f"Saved: {file_path}")
                print(f"Saved: robot_vision_result/G_{file_number:02d}_raw_depth.png")
                print(f"Saved: robot_vision_result/G_{file_number:02d}_raw_depth.npy")
                print(f"Saved: robot_vision_result/G_{file_number:02d}_depth_heatmap.png")
                print(f"Saved: robot_vision_result/G_{file_number:02d}_overlay_heatmap.png")

            grab_retrieve.Release()
    cam.StopGrabbing()
    cam.Close()
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()
