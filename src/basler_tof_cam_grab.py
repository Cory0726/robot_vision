import os

import cv2
from pypylon import pylon
import basler_cam_init
from tools import rawdepth_to_heatmap

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
    cam.OperatingMode.Value = "LongRange"
    # Exposure time (us)
    cam.ExposureTime.Value = 200.0
    # Max depth / Min depth (mm)
    cam.DepthMax.Value = 1498
    cam.DepthMin.Value = 0
    print(f"Operating mode: {cam.OperatingMode.Value} / Exposure time: {cam.ExposureTime.Value}\n")
    print(f"Depth max: {cam.DepthMax.Value} / min: {cam.DepthMin.Value}")
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
            # Split the container
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

