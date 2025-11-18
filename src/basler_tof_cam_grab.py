import os
from importlib.resources import read_text

import cv2
import numpy as np
from pypylon import pylon
import basler_cam_init


def create_tof_cam():
    """
    Create a ToF camera object by serial number.
    """
    tof_cam_sn = "24945819"
    tof_cam = basler_cam_init.create_basler_cam(tof_cam_sn)
    return tof_cam

def config_tof_cam_para(cam: pylon.InstantCamera) -> None:
    """
    Configure a ToF camera (Basler blaze-101) parameter after opening the camera.
    """
    print("ToF camera information:")
    # Operating mode: ShortRange: 0 - 1498 mm / LongRange: 0 - 9990 mm
    cam.OperatingMode.Value = "ShortRange"
    # Max depth / Min depth (mm)
    cam.DepthMax.Value = 1498
    cam.DepthMin.Value = 0
    print(
        f"Operating mode: {cam.OperatingMode.Value} / Depth max: {cam.DepthMax.Value} / min: {cam.DepthMin.Value}")
    # Fast mode
    cam.FastMode.Value = True
    # Filter spatial
    cam.FilterSpatial.Value = True
    # Filter temporal
    cam.FilterTemporal.Value = True
    # Filter temporal strength
    if cam.FilterTemporal.Value:
        cam.FilterStrength.Value = 200
    # Outlier removal
    cam.OutlierRemoval.Value = True
    # Confidence Threshold (0 - 65536)
    cam.ConfidenceThreshold.Value = 32
    print(f"Confidence threshold: {cam.ConfidenceThreshold.Value}")
    # Gamma correction
    cam.GammaCorrection.Value = True
    # GenDC (Generic Data Container) is used to transmit multiple types of image data,such as depth,
    # intensity, and confidence, in a single, structured data stream, making it
    # ideal for 3D and multi-modal imaging applications.
    cam.GenDCStreamingMode.Value = "Off"

def config_tof_data_comp(cam: pylon.InstantCamera, data_type: str) -> None:
    """
    Configure a ToF camera data container after opening the camera.
    Args:
        data_type (str): "Intensity_Image" or "Point_Cloud" or "Confidence_Map"
    """
    # Image component selector
    if data_type == "Intensity_Image":
        # Close 3d point cloud image
        cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Range")
        cam.GetNodeMap().GetNode("ComponentEnable").SetValue(False)
        cam.GetNodeMap().GetNode("PixelFormat").SetValue("Coord3D_ABC32f")  # Coord3D_C16 / Coord3D_ABC32f
        # Open intensity image
        cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Intensity")
        cam.GetNodeMap().GetNode("ComponentEnable").SetValue(True)
        cam.GetNodeMap().GetNode("PixelFormat").SetValue("Mono16")
        # Close confidence map
        cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Confidence")
        cam.GetNodeMap().GetNode("ComponentEnable").SetValue(False)
        cam.GetNodeMap().GetNode("PixelFormat").SetValue("Confidence16")
        print("Image selector: Intensity")
    elif data_type == "Point_Cloud":
        # Open 3d point cloud image
        cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Range")
        cam.GetNodeMap().GetNode("ComponentEnable").SetValue(True)
        cam.GetNodeMap().GetNode("PixelFormat").SetValue("Coord3D_ABC32f")
        # Close intensity image
        cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Intensity")
        cam.GetNodeMap().GetNode("ComponentEnable").SetValue(False)
        cam.GetNodeMap().GetNode("PixelFormat").SetValue("Mono16")
        # Close confidence map
        cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Confidence")
        cam.GetNodeMap().GetNode("ComponentEnable").SetValue(False)
        cam.GetNodeMap().GetNode("PixelFormat").SetValue("Confidence16")
        print("Image selector: Point cloud")
    elif data_type == "Confidence_Map":
        # Close 3d point cloud image
        cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Range")
        cam.GetNodeMap().GetNode("ComponentEnable").SetValue(False)
        cam.GetNodeMap().GetNode("PixelFormat").SetValue("Coord3D_ABC32f")
        # Close intensity image
        cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Intensity")
        cam.GetNodeMap().GetNode("ComponentEnable").SetValue(False)
        cam.GetNodeMap().GetNode("PixelFormat").SetValue("Mono16")
        # Open confidence map
        cam.GetNodeMap().GetNode("ComponentSelector").SetValue("Confidence")
        cam.GetNodeMap().GetNode("ComponentEnable").SetValue(True)
        cam.GetNodeMap().GetNode("PixelFormat").SetValue("Confidence16")
        print("Image selector: Confidence Map")
    else:
        print("Wrong data type input of function config_tof_camera_para")

def split_tof_container_data(container) -> dict:
    """
    Split the data component from the grab retrieve data container
    Args:
        container: A grab retrieve as data container

    Returns:
        dict: data_dict{Intensity_Image, Confidence_Map, Point_Cloud}
    """
    data_dict = {
        "Intensity_Image": None,
        "Confidence_Map": None,
        "Point_Cloud": None
    }
    for i in range(container.DataComponentCount):
        data_component = container.GetDataComponent(i)
        if data_component.ComponentType == pylon.ComponentType_Intensity:
            data_dict["Intensity_Image"] = data_component.Array
        elif data_component.ComponentType == pylon.ComponentType_Confidence:
            data_dict["Confidence_Map"] = data_component.Array
        elif data_component.ComponentType == pylon.ComponentType_Range:
            data_dict["Point_Cloud"] = data_component.Array.reshape(data_component.Height, data_component.Width, 3)
        data_component.Release()
    return data_dict

def pcl_to_rawdepth(pcl):
    return pcl[:,:,2]  # Get z data from point cloud

def rawdepth_to_heatmap(rawdepth):
    gray_img = cv2.normalize(rawdepth, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(255 - gray_img, cv2.COLORMAP_TURBO)
    # heatmap = cv2.applyColorMap(255 - gray_img, cv2.COLORMAP_JET)
    return heatmap

def stream_tof_img(img_type: str) -> None:
    cam = create_tof_cam()
    cam.Open()
    config_tof_cam_para(cam)
    if img_type == "Intensity_Image":
        config_tof_data_comp(cam, "Intensity_Image")
    elif img_type == "Confidence_Map":
        config_tof_data_comp(cam, "Confidence_Map")
    elif img_type == "Depth_Image":
        config_tof_data_comp(cam, "Point_Cloud")
    else:
        raise Exception("Not supported image type")

    # Starts the grabbing of data with strategy
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("Start grabbing ...")
    while cam.IsGrabbing():
        # Get the grab retrieve
        grab_retrieve = cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

        if grab_retrieve.GrabSucceeded():
            # Get the grab retrieve as data container
            data_container = grab_retrieve.GetDataContainer()
            # Get the intensity image
            data = split_tof_container_data(data_container)

            if img_type == "Intensity_Image":
                img= data["Intensity_Image"]
                display_title = "Intensity_image"
            elif img_type == "Confidence_Map":
                img = data["Confidence_Map"]
                display_title = "Confidence_map"
            elif img_type == "Depth_Image":
                img = rawdepth_to_heatmap(pcl_to_rawdepth(data["Point_Cloud"]))
                display_title = "Depth_image"
            else:
                raise Exception("Not supported image type")

            # Display
            cv2.imshow(display_title, img)
            grab_retrieve.Release()

        # Read the keyboard keyin
        key = cv2.waitKey(5) & 0xFF
        # Break the loop by pressing q
        if key == ord("q"):
            break
        elif key == ord("s"):
            file_number = 0
            file_path = f"robot_vision_result/{display_title}_{file_number:02d}.png"
            while os.path.exists(file_path):
                file_number += 1
                file_path = f"robot_vision_result/{display_title}_{file_number:02d}.png"
            cv2.imwrite(file_path, img)
            print(f"Saved: {file_path}")

    cam.StopGrabbing()
    cam.Close()
    cv2.destroyAllWindows

def grab_one_point_cloud():
    """
    Grab one point cloud from camera.
    Returns:
        pcl: point cloud (unit : mm)
    """
    cam = create_tof_cam()
    cam.Open()
    config_tof_cam_para(cam)
    config_tof_data_comp(cam, "Point_Cloud")

    # Grab point cloud data
    grab_result = cam.GrabOne(1000)  # timeout: 1s
    assert grab_result.GrabSucceeded(), "Failed to grab depth data"
    cam.Close()
    return split_tof_container_data(grab_result.GetDataContainer())["Point_Cloud"]  # Unit: mm

def grab_one_intensity():
    cam = create_tof_cam()
    cam.Open()
    config_tof_cam_para(cam)
    config_tof_data_comp(cam, "Intensity_Image")

    # Grab point cloud data
    grab_result = cam.GrabOne(1000)  # timeout: 1s
    assert grab_result.GrabSucceeded(), "Failed to grab intensity data"
    cam.Close()
    return split_tof_container_data(grab_result.GetDataContainer())["Intensity_Image"]



def halcon_to_opencv_intrinsics_tof(
    # Pixel size [micrometers]
    Sx_um=8.29572, Sy_um=8.3,
    # Focal length [mm]
    f_mm=4.32481,
    # Principal point [pixels]
    Cx_px=329.512, Cy_px=225.884,
    # HALCON distortion coefficients (units shown in your screenshot)
    # Radial: [1/m^2, 1/m^4, 1/m^6]
    K1=-40.7214, K2=-1.66096e8, K3=2.11721e13,
    # Tangential 2nd order (HALCON labels 1/m^2)
    P1=0.32381, P2=0.538484
):
    """
    Build OpenCV cameraMatrix (in pixels) and distCoeffs from HALCON parameters.
    NOTE:
      - HALCON radial units: K1 [1/m^2], K2 [1/m^4], K3 [1/m^6]
      - HALCON tangential units here: P1, P2 [1/m^2]  <-- from your GUI label
      - OpenCV expects *dimensionless* coefficients in normalized coordinates
        so we convert by multiplying with powers of focal length (in meters):
          k1 = K1 * f^2,  k2 = K2 * f^4,  k3 = K3 * f^6
          p1 = P1 * f^2,  p2 = P2 * f^2
    """

    # --- Units to meters ---
    f_m  = f_mm / 1000.0          # mm -> m
    Sx_m = Sx_um * 1e-6           # um -> m
    Sy_m = Sy_um * 1e-6

    # --- Focal length in pixels ---
    fx = f_m / Sx_m
    fy = f_m / Sy_m
    cx, cy = Cx_px, Cy_px

    cameraMatrix = np.array([[fx, 0,  cx],
                             [0,  fy, cy],
                             [0,  0,   1]], dtype=np.float64)

    # --- HALCON -> OpenCV coefficient conversion (dimensionless) ---
    k1 = K1 * (f_m ** 2)
    k2 = K2 * (f_m ** 4)
    k3 = K3 * (f_m ** 6)
    p1 = P1 * (f_m ** 2)   # IMPORTANT: P1,P2 shown as 1/m^2 in your UI
    p2 = P2 * (f_m ** 2)

    distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return cameraMatrix, distCoeffs

def build_undistort_maps(K, dist, size, alpha=1.0):
    """
    Create undistortion/rectification maps once, then reuse for remapping.
    - size: (width, height) in pixels
    - alpha:
        0.0 -> crop all black borders (smaller FOV, no black edges)
        1.0 -> keep full FOV (same resolution, black edges may appear)
        (0~1 for trade-off)
    Returns: map1, map2, newK, roi
    """
    w, h = size
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, R=None, newCameraMatrix=newK, size=(w, h), m1type=cv2.CV_32FC1
    )
    return map1, map2, newK, roi


def undistort_tof_intensity(img,alpha=1.0):
    """
    Undistort a ToF intensity image (or any 2D image).
    """
    h, w = img.shape[:2]
    K, dist = halcon_to_opencv_intrinsics_tof()
    map1, map2, newK, roi = build_undistort_maps(K, dist, (w, h), alpha=alpha)
    undist_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Alternatively, use remapping for faster processing:
    # undist_img = cv2.undistort(img, K, dist, None, newK)

    # Crop the image based on ROI (optional)
    # x, y, rw, rh = roi
    # if rw > 0 and rh > 0:
    #     undist_img = undist_img[y:y+rh, x:x+rw]

    return undist_img, newK

def undistort_tof_depth(depth,alpha=1.0,):
    """
    Undistort a ToF depth map (e.g., in millimeters).
    """
    depth = depth / 1000.0
    if depth is None:
        raise FileNotFoundError(f"Dept data is not available")
    h, w = depth.shape[:2]

    K, dist = halcon_to_opencv_intrinsics_tof()
    map1, map2, newK, roi = build_undistort_maps(K, dist, (w, h), alpha=alpha)
    # Use NEAREST interpolation to avoid averaging depth values.
    undist_img = cv2.remap(depth, map1, map2, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Crop the image based on ROI (optional)
    # x, y, rw, rh = roi
    # if rw > 0 and rh > 0:
    #     undist_img = undist_img[y:y+rh, x:x+rw]
    undist_img = np.clip(undist_img * 1000.0,0,65535).astype(np.uint16)

    return undist_img, newK
