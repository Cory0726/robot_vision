import cv2
import numpy as np

def load_cam_calibration_file():
    """
    Load intrinsic/extrinsic matrices from the calibration XML.

    Returns:
        Kc (3x3), dc (Nx1), Kd (3x3), dd (Nx1), R (3x3), T (3x1)
        where:
          Kc, dc: color intrinsics, distortion
          Kd, dd: blaze (depth) intrinsics, distortion (often zeros)
          R, T:   transform from depth(blaze) frame to color frame
    """
    xml_path = f"./basler_calibration/calibration_24945819_24747625.xml"
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {xml_path}")

    Kc = fs.getNode("colorCameraMatrix").mat()
    dc = fs.getNode("colorDistortion").mat()
    Kd = fs.getNode("blazeCameraMatrix").mat()  # may exist in your XML
    dd = fs.getNode("blazeDistortion").mat()    # may be zeros in your sample
    R  = fs.getNode("rotation").mat()
    T  = fs.getNode("translation").mat()
    fs.release()

    if Kd is None or Kd.size == 0:
        raise ValueError("Missing 'blazeCameraMatrix' in XML (depth intrinsics are required).")

    if dd is None or dd.size == 0:
        dd = np.zeros((1,5), dtype=np.float32)

    # Ensure proper shapes
    Kc = Kc.astype(np.float32)
    dc = dc.astype(np.float32).reshape(-1,1)
    Kd = Kd.astype(np.float32)
    dd = dd.astype(np.float32).reshape(-1,1)
    R  = R.astype(np.float32)
    T  = T.astype(np.float32).reshape(3,1)

    return Kc, dc, Kd, dd, R, T

def warp_depth_with_color(pcl, color_img, interp="nearest"):
    """
    Project organized 3D points (depth frame) into the color camera and sample color.

    Args:
        pcl:     (Hd, Wd, 3) float32, XYZ in depth frame
        R, T:      color <- depth transform (3x3, 3x1)
        Kc, dc:    color intrinsics and distortion
        color_img: (Hc, Wc, 3) uint8 BGR
        interp:    'nearest' or 'bilinear'

    Returns:
        color_on_depth: (Hd, Wd, 3) uint8, BGR on depth grid (zeros where invalid)
        valid_mask:     (Hd, Wd) bool, True if sampled inside color bounds and Z>0
    """

    # Load calibration parameter
    Kc, dc, Kd, dd, R, T = load_cam_calibration_file()

    Hd, Wd, _ = pcl.shape
    Hc, Wc = color_img.shape[:2]

    pts = pcl.reshape(-1, 3)

    # OpenCV can take a 3x3 rotation matrix in place of rvec; cv2 will convert internally
    img_pts, _ = cv2.projectPoints(pts, R, T, Kc, dc)  # -> (N,1,2)
    img_pts = img_pts.reshape(-1, 2)

    if interp == "nearest":
        u = np.rint(img_pts[:,0]).astype(np.int32)  # x (col)
        v = np.rint(img_pts[:,1]).astype(np.int32)  # y (row)
        Z = pts[:,2]
        valid = (Z > 0) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)

        out = np.zeros((Hd*Wd, 3), dtype=np.uint8)
        out[valid] = color_img[v[valid], u[valid]]
        out = out.reshape(Hd, Wd, 3)
        return out, valid.reshape(Hd, Wd)

    elif interp == "bilinear":
        # Bilinear sampling on float coords
        u = img_pts[:,0]
        v = img_pts[:,1]
        Z = pts[:,2]
        # bounds for sampling window
        u0 = np.floor(u).astype(np.int32)
        v0 = np.floor(v).astype(np.int32)
        u1 = u0 + 1
        v1 = v0 + 1

        valid = (Z > 0) & (u0 >= 0) & (v0 >= 0) & (u1 < Wc) & (v1 < Hc)
        out = np.zeros((Hd*Wd, 3), dtype=np.float32)

        # weights
        du = (u - u0).astype(np.float32)
        dv = (v - v0).astype(np.float32)
        w00 = (1-du)*(1-dv)
        w10 = du*(1-dv)
        w01 = (1-du)*dv
        w11 = du*dv

        idx = np.where(valid)[0]
        uu0, vv0 = u0[idx], v0[idx]
        uu1, vv1 = u1[idx], v1[idx]

        c00 = color_img[vv0, uu0].astype(np.float32)
        c10 = color_img[vv0, uu1].astype(np.float32)
        c01 = color_img[vv1, uu0].astype(np.float32)
        c11 = color_img[vv1, uu1].astype(np.float32)

        out[idx] = (c00*w00[idx,None] + c10*w10[idx,None] +
                    c01*w01[idx,None] + c11*w11[idx,None])
        out = np.clip(out, 0, 255).astype(np.uint8).reshape(Hd, Wd, 3)
        return out, valid.reshape(Hd, Wd)

    else:
        raise ValueError("interp must be 'nearest' or 'bilinear'")

def transform_pcl_to_color_frame(pcl):
    """
    Transform an organized point cloud from DEPTH frame to COLOR frame.

    Args:
        pcl: (Hd, Wd, 3) float32, XYZ in depth frame, **millimeters**
        R:      (3,3) rotation,  color <- depth
        T:      (3,1) translation, color <- depth, **millimeters**

    Returns:
        pcl_on_color_frame: (Hd, Wd, 3) float32, XYZ in color frame, **millimeters**
    """
    # Load calibration parameter
    Kc, dc, Kd, dd, R, T = load_cam_calibration_file()
    Hd, Wd, _ = pcl.shape
    pts = pcl.reshape(-1, 3).astype(np.float32)
    Xc = (pts @ R.T) + T.ravel()
    return Xc.reshape(Hd, Wd, 3).astype(np.float32)  # pcl_on_color_frame

def project_depth_to_color_frame(pcl, color_img):
    """
    Directly project the depth camera point cloud (in mm) into the color camera frame,
    and rasterize it into a Z-buffer depth map at the color image resolution.

    Args:
        pcl    : (Hd, Wd, 3) float32
                    3D points (X, Y, Z) in the depth camera frame [millimeters]
        Kc, dc    : color camera intrinsics and distortion coefficients
        R, T      : extrinsic parameters (color <- depth)
                    R: 3x3 rotation matrix
                    T: 3x1 translation vector [millimeters]
        color_img : The color image (output depth map size)

    Returns:
        depth_rgb : (Hc, Wc) uint16
                       dense Z-buffer depth map [millimeters], aligned to color frame
        valid_mask   : (Hc, Wc) bool
                       True for pixels where at least one 3D point projects to it
    """

    # Load calibration parameter
    Kc, dc, Kd, dd, R, T = load_cam_calibration_file()

    # Prepare and flatten input points
    Hc, Wc = color_img[:,:,0].shape
    Hd, Wd, _ = pcl.shape
    pts_d = pcl.reshape(-1, 3).astype(np.float32)  # (N,3), in mm

    # Project all 3D depth points directly into the color image plane
    # cv2.projectPoints applies: [u, v] = Kc * (R * Xd + T) / Z
    # It handles both rotation/translation (extrinsics) and lens distortion (dc)
    img_pts, _ = cv2.projectPoints(pts_d, R, T, Kc, dc)  # → (N,1,2)
    img_pts = img_pts.reshape(-1, 2)

    # Round to nearest integer pixel coordinates
    u = np.rint(img_pts[:, 0]).astype(np.int32)
    v = np.rint(img_pts[:, 1]).astype(np.int32)
    Z = pts_d[:, 2]  # depth values in mm (from depth camera frame)

    # Keep only valid pixels inside the color image bounds
    valid = (Z > 0) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
    u = u[valid]
    v = v[valid]
    Z = Z[valid]

    # Z-buffer rasterization
    # Initialize the depth map with infinity (meaning "no point yet")
    depth = np.full((Hc, Wc), np.inf, dtype=np.float32)

    # For each projected pixel (v, u), keep the smallest Z (closest point)
    # np.minimum.at is an in-place vectorized version of:
    #   for i in range(len(Z)):
    #       depth[v[i], u[i]] = min(depth[v[i], u[i]], Z[i])
    np.minimum.at(depth, (v, u), Z)

    # Pixels that received at least one point will have finite values
    hit = np.isfinite(depth)

    # Convert to 16-bit depth map (mm)
    # Replace infinity with zeros and clip the valid range to [0, 65535]
    raw_depth = np.zeros_like(depth, dtype=np.uint16)
    if hit.any():
        np.clip(depth, 0, 65535, out=depth)
        raw_depth[hit] = depth[hit].astype(np.uint16)

    return raw_depth, hit  # (depth_rgb, valid_mask)
