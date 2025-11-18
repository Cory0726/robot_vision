"""
This sample demonstrates how to calibrate a system consisting of a Basler GigE color camera
and a Basler blaze camera.

After the camera system has been calibrated successfully, start the fusion as the second step.
The sample program uses the data of the color camera to display a colored point cloud. This step
requires the calibration file that was created during calibration. This file contains the
intrinsic camera parameters of the blaze camera and the color camera as well as the values for
the relative rotation and translation between the cameras.
"""

import os
import platform
import traceback

# This is used for reshaping the image buffers.
import numpy as np

# This is used for image processing.
import cv2

# This is used for visualization.
import open3d as o3d

# Use of Harvester to access the camera.
# For more information regarding Harvester, visit the github page:
# https://github.com/genicam/harvesters
from harvesters.core import Harvester


BAYER_FORMATS = {"BayerGR8": cv2.COLOR_BayerGR2BGR,
                 "BayerRG8": cv2.COLOR_BayerRG2BGR,
                 "BayerBG8": cv2.COLOR_BayerBG2BGR,
                 "BayerGB8": cv2.COLOR_BayerGB2BGR}


def find_producer(name):
    """ Helper for the GenTL producers from the environment path.
    """
    paths = os.environ['GENICAM_GENTL64_PATH'].split(os.pathsep)

    if platform.system() == "Linux":
        paths.append('/opt/pylon/lib/gentlproducer/gtl/')

    for path in paths:
        path += os.path.sep + name
        if os.path.exists(path):
            return path
    return ""


class Fusion:
    def __init__(self):
        # Create Harvester instances.
        self.h = Harvester()

        # Location of the Basler blaze GenTL producer.
        if platform.system() == "Windows" or platform.system() == "Linux":
            path_to_blaze_cti = find_producer("ProducerBaslerBlazePylon.cti")
            path_to_gev_cti = find_producer("ProducerGEV.cti")
        else:
            print(f"{platform.system()} is not supported")
            assert False

        # Add producer to Harvester.
        assert os.path.exists(path_to_blaze_cti)
        assert os.path.exists(path_to_gev_cti)

        self.h.add_file(path_to_blaze_cti)
        self.h.add_file(path_to_gev_cti)

        # Update device list.
        self.h.update()

        # Print device list.
        print(self.h.device_info_list)

    rotation = np.zeros((3, 3), np.float32)
    translation = np.zeros((1, 3), np.float32)
    color_camera_matrix = np.zeros((3, 3), np.float32)
    color_dist = np.zeros((3, 5), np.float32)

    def setup_blaze(self):
        """
        Connect and configure the Basler blaze camera (3D ToF).
        """
        dev_info = next(
            (d for d in self.h.device_info_list if str(d.model).startswith('blaze')), None)
        if dev_info is not None:
            self.ia_blaze = self.h.create({"model":dev_info.model, "serial_number":dev_info.serial_number})
            print("Connected to blaze camera: {}".format(
                self.ia_blaze.remote_device.node_map.DeviceSerialNumber.value))
        else:
            print("No blaze camera found.")
            raise RuntimeError

        # Disable depth data.
        self.ia_blaze.remote_device.node_map.ComponentSelector.value = "Range"
        self.ia_blaze.remote_device.node_map.ComponentEnable.value = True
        self.ia_blaze.remote_device.node_map.PixelFormat.value = "Coord3D_ABC32f"

        # Enable the intensity image.
        self.ia_blaze.remote_device.node_map.ComponentSelector.value = "Intensity"
        self.ia_blaze.remote_device.node_map.ComponentEnable.value = True
        self.ia_blaze.remote_device.node_map.PixelFormat.value = "Mono16"

        # Disable the confidence map.
        self.ia_blaze.remote_device.node_map.ComponentSelector.value = "Confidence"
        self.ia_blaze.remote_device.node_map.ComponentEnable.value = False
        self.ia_blaze.remote_device.node_map.PixelFormat.value = "Confidence16"

        # Reduce exposure time to avoid overexposure at close range.
        self.ia_blaze.remote_device.node_map.ExposureTime.value = 1000

        # Switch off gamma correction for accurate corner detection.
        self.ia_blaze.remote_device.node_map.GammaCorrection.value = True

        # Configure the camera for software triggering.
        self.ia_blaze.remote_device.node_map.TriggerMode.value = "On"
        self.ia_blaze.remote_device.node_map.TriggerSource.value = "Software"

        # Disable GenDC. This mode is currently not supported by the python GenICam module.
        self.ia_blaze.remote_device.node_map.GenDCStreamingMode.value = "Off"

        # Start image acquisition.
        self.ia_blaze.start()

    def setup_2Dcamera(self):
        """
        Connect and configure the Basler 2D GigE color camera.
        """
        # Connect to the first available 2D camera. Ignore blaze cameras, which will
        # be enumerated as well.
        dev_info = next(
            (d for d in self.h.device_info_list if 'blaze' not in d.model), None)
        if dev_info is not None:
            self.ia_gev = self.h.create({"serial_number":dev_info.serial_number})
            print("Connected to ace-camera: {}".format(
                self.ia_gev.remote_device.node_map.DeviceID.value))
        else:
            print("No 2D camera found.")
            raise RuntimeError

        # Figure out which 8-bit Bayer pixel format the camera supports.
        # If the camera supports an 8-bit Bayer format, enable the format.
        # Otherwise, exit the program.
        bayer_pattern = next(
            (pf for pf in self.ia_gev.remote_device.node_map.PixelFormat.symbolics if pf in BAYER_FORMATS), None)
        if bayer_pattern is not None:
            self.ia_gev.remote_device.node_map.PixelFormat.value = bayer_pattern
        else:
            print("The camera does not provide Bayer pattern-encoded 8-bit color images.")
            raise RuntimeError

        # Configure the camera for software triggering.
        # Each software trigger will start the acquisition of one single frame.
        self.ia_gev.remote_device.node_map.AcquisitionMode.value = "Continuous"
        self.ia_gev.remote_device.node_map.TriggerSelector.value = "FrameStart"
        self.ia_gev.remote_device.node_map.TriggerMode.value = "On"
        self.ia_gev.remote_device.node_map.TriggerSource.value = "Software"

        # Start image acquisition.
        self.ia_gev.start()

    def close_blaze(self):
        """
        Stop acquisition and disconnect from the blaze camera.
        """
        self.ia_blaze.stop()
        self.ia_blaze.remote_device.node_map.TriggerMode.value = "Off"

        # Disconnect from camera.
        self.ia_blaze.destroy()

    def close_2DCamera(self):
        """
        Stop acquisition and disconnect from the 2D camera.
        """
        self.ia_gev.stop()
        self.ia_gev.remote_device.node_map.TriggerMode.value = "Off"

        # Disconnect from camera.
        self.ia_gev.destroy()

    def close_harvesters(self):
        """
        Release producer files and reset Harvester.
        """
        self.h.reset()

    def get_image_blaze(self):
        """
        Fetch one blaze buffer and return:
           - point cloud as (H, W, 3) float32 (X,Y,Z in meters)
           - intensity as (H, W) uint16
        """
        with self.ia_blaze.fetch() as buffer:
            # Warning: The buffer is only valid in the with statement and will be destroyed
            # when you leave the scope.
            # If you want to use the buffers outside of the with scope, you have to use np.copy()
            # to make a deep copy of the image.

            # Create an alias of the image components:
            pointcloud = buffer.payload.components[0]

            intensity = buffer.payload.components[1]

            # Reshape the depth map into a 2D/3D array:
            # "num_components_per_pixel" depends on the pixel format selected:
            # "Coord3D_ABC32f" = 3
            # "Coord3D_C16" = 1
            _3d = pointcloud.data.reshape(pointcloud.height, pointcloud.width,
                                          int(pointcloud.num_components_per_pixel))

            # Reshape the intensity image into a 2D array:
            _2d_intensity = intensity.data.reshape(
                intensity.height, intensity.width)

            return np.copy(_3d), np.copy(_2d_intensity)

    def get_image_2DCamera(self):
        """
        Fetch one color frame, debayer to BGR uint8 image (H, W, 3).
        """
        with self.ia_gev.fetch() as buffer:
            # Warning: The buffer is only valid in the with statement and will be destroyed
            # when you leave the scope.
            # If you want to use the buffers outside of the with scope, you have to use np.copy()
            # to make a deep copy of the image.

            # Create an alias of the image component:
            image = buffer.payload.components[0]

            # Reshape the image into a 2D array:
            _2d = image.data.reshape(image.height, image.width)

            # Debayer the image to a grayscale image.
            color = cv2.cvtColor(_2d, BAYER_FORMATS[image.data_format])

            return color

    def load_calibration_file(self):
        """
        Load stereo calibration (R, T, color intrinsics, color distortion) from XML.

        The filename is built from the two device identifiers, matching the calibration script.
        """
        # It is assumed that the calibration file contains information about the relative
        # orientation of the cameras to each other and the optical calibration of the
        # color camera.
        # The calibration program can be used to create the file.
        dirname = os.path.dirname(__file__)
        filename = "./calibration/calibration_" + str(self.ia_blaze.remote_device.node_map.DeviceSerialNumber.value) + \
            "_" + str(self.ia_gev.remote_device.node_map.DeviceID.value) + ".xml"
        path = os.path.join(dirname, filename)

        print("Loading the calibration file:", path)
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        if not cv_file.isOpened():
            raise ValueError(
                "For the fusion of the data, calibration must be performed first.\nMake sure that there is a valid calibration with file name '" + path + "'.")

        # Note that we also have to specify the type to retrieve. Otherwise, we will only get a
        # FileNode object back instead of a matrix.
        self.rotation = cv_file.getNode('rotation').mat()
        self.translation = cv_file.getNode("translation").mat()
        self.color_camera_matrix = cv_file.getNode("colorCameraMatrix").mat()
        self.color_dist = cv_file.getNode("colorDistortion").mat()

    def warp_color_to_depth(self, pointcloud, color):
        """
        Project each 3D point (in blaze frame) to the color image and sample color.

        Args:
            pointcloud: (H, W, 3) float32, XYZ in meters.
            color:      (Hc, Wc, 3) uint8, BGR image from the color camera.

        Returns:
            warped: (H, W, 3) float64, per-point BGR sampled from color image.
                    Pixels with invalid depth or out-of-bounds projections remain zero.
        """
        warped = np.zeros(pointcloud.shape, np.float64)

        # Project the 3D points into the color camera.
        pointvec = pointcloud.reshape(640*480, 3)
        img_points = cv2.projectPoints(
            pointvec, self.rotation, self.translation, self.color_camera_matrix, self.color_dist)
        img_points = img_points[0].reshape(480, 640, 2)
        img_points = np.rint(img_points)
        img_points = img_points.astype(int)


        # No depth information available for this pixel or pixel is invalid
        mask_1 = pointcloud[:, :, 2] == 0.0
        mask_2 = (img_points[:, :, 1] < 0) | (img_points[:, :, 1] > color.shape[0]-1)
        mask_3 = (img_points[:, :, 0] < 0) | (img_points[:, :, 0] > color.shape[1]-1)
        mask_sum = (mask_1 | mask_2) | mask_3


        # Determine color values for the points of the point cloud.
        for i in range(warped.shape[0]):
            for j in range(warped.shape[1]):
                if mask_sum[i, j]:
                    continue
                
                u = img_points.item(i, j, 1)
                v = img_points.item(i, j, 0)

                warped[i, j] = color[u, v]

        return warped

    # Open3D callbacks
    def cbStopGrabbing(self, vis):
        self.stopGrabbing = True
        return False

    def cbSavePcd(self, vis):
        self.savePcd = True
        return False

    def run(self):
        """
        Main loop: setup devices, load calib, fuse, visualize, and optionally save .pcd.
        """
        # Set up the cameras.
        self.setup_blaze()
        self.setup_2Dcamera()
        self.load_calibration_file()

        # Set up Open3D.
        # Open3D viewer setup and key bindings
        glfw_key_s = 83  # some envs map "S" to GLFW code 83
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord('q'), self.cbStopGrabbing)  # quit
        self.vis.register_key_callback(glfw_key_s, self.cbSavePcd)  # save .pcd

        self.vis.create_window()

        self.pcd = o3d.geometry.PointCloud()
        self.addedGeometry = False
        self.stopGrabbing = False
        self.savePcd = False
        self.savePcdCnt = 0

        print('')
        print('Fusion of color and depth data')
        print('  - Press "s" in the viewer to save a point cloud as .pcd file')
        print('  - Press "q" in the viewer to exit')
        print('')

        # Grab the images.
        while not self.stopGrabbing:
            # To optimize bandwidth usage, the color camera is triggered first to
            # allow it to already transfer image data while the blaze camera is still internally
            # processing the acquired raw data.
            # Trigger order: color first (transfer), then blaze (process) for bandwidth efficiency.
            self.ia_gev.remote_device.node_map.TriggerSoftware.execute()
            self.ia_blaze.remote_device.node_map.TriggerSoftware.execute()

            pointcloud, intensity = self.get_image_blaze()  # (H,W,3), (H,W)
            color = self.get_image_2DCamera()  # (Hc,Wc,3) BGR
            color_warped = self.warp_color_to_depth(pointcloud, color)  # (H,W,3)

            # Prepare data for display in Open3d viewer.
            # Prepare Open3D geometry (N,3) float64
            self.pcd.points = o3d.utility.Vector3dVector(
                pointcloud.reshape(pointcloud.shape[0] * pointcloud.shape[1], 3).astype(np.float64))

            # The color data must be scaled to the range of 0 to 1 for display with Open3d viewer.
            color_warped = color_warped / 256.0
            self.pcd.colors = o3d.utility.Vector3dVector(
                color_warped.reshape(color_warped.shape[0] * color_warped.shape[1], 3).astype(np.float64))

            # Save .pcd file.
            if self.savePcd:
                o3d.io.write_point_cloud(
                    "Pointcloud_{}.pcd".format(self.savePcdCnt), self.pcd)
                self.savePcdCnt += 1
                self.savePcd = False

            # We only add geometry once. Otherwise, we would create a memory leak.
            if not self.addedGeometry:
                self.vis.add_geometry(self.pcd)

                # Adjust camera position to correct perspective.
                view_control = self.vis.get_view_control()
                view_control.set_front([0, 0, -1])
                view_control.set_up([0, -1, 0])
                view_control.set_zoom(0.5)

                self.addedGeometry = True

            self.vis.update_geometry(self.pcd)

            if not self.vis.poll_events():
                self.stopGrabbing = True
            self.vis.update_renderer()

        self.vis.destroy_window()

        # Close the camera and release the producers.
        self.close_blaze()
        self.close_2DCamera()
        self.close_harvesters()


if __name__ == "__main__":
    """ Run the sample.
    """
    Sample = Fusion()
    try:
        Sample.run()
    except Exception:
        traceback.print_exc()
        Sample.close_harvesters()
