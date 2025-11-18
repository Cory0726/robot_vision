from pypylon import pylon

def list_basler_devices() -> None:
    """ 
    List all devices which connected to the computer.
    """
    # Build the transport layer factory
    tl_factory = pylon.TlFactory.GetInstance()
    # List the devices
    devices = tl_factory.EnumerateDevices()
    for i, dec in enumerate(devices):
        print(f"[{i}] {dec.GetModelName()} - {dec.GetDeviceClass()}"
            + f"- {dec.GetFullName()} - {dec.GetSerialNumber()}")

def create_basler_cam(serial_number: str) -> pylon.InstantCamera:
    """
    Create a Basler camera instance by serial number.

    :param serial_number: (str), Basler camera serial number.
    :return: (pylon.InstantCamera) Basler camera instance.
    """
    # Get the transport layer factory
    tl_factory = pylon.TlFactory.GetInstance()
    # Set the device information
    device = pylon.DeviceInfo()
    device.SetSerialNumber(serial_number)
    # Create the camera
    cam = pylon.InstantCamera(tl_factory.CreateDevice(device))
    return cam
