import numpy as np
import cv2
import basler_rgb_cam_grab
import basler_tof_cam_grab
import trans_3d_to_rgb_frame
import tools

if __name__ == '__main__':
    color_img = basler_rgb_cam_grab.grab_one_rgb_img()
    pcl = basler_tof_cam_grab.grab_one_point_cloud()

    pcl_color_frame = trans_3d_to_rgb_frame.transform_pcl_to_color_frame(pcl)
    depth_color_frame, _ = trans_3d_to_rgb_frame.project_depth_to_color_frame(pcl, color_img)


    overlay_heatmap, overlay_edges = tools.visualize_rgb_depth_alignment(
        color_img, depth_color_frame
    )
    depth_color_frame_heatmap = basler_tof_cam_grab.rawdepth_to_heatmap(depth_color_frame)
    cv2.imshow("Depth heatmap", depth_color_frame_heatmap)
    cv2.imshow("overlay_heatmap", overlay_heatmap)
    cv2.imwrite("E:/overlay_heatmap.png", overlay_heatmap)
    cv2.imwrite("E:/depth_heatmap.png", depth_color_frame_heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


