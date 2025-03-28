import cv2

try:
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"cv2.ocl.haveOpenCL(): {cv2.ocl.haveOpenCL()}")
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        print(f"cv2.ocl.useOpenCL() after setUseOpenCL(True): {cv2.ocl.useOpenCL()}")

        platforms = []
        get_platforms_func = None
        if hasattr(cv2.ocl, "getPlatforms"):
            get_platforms_func = cv2.ocl.getPlatforms
        elif hasattr(cv2.ocl, "getPlatfoms"):
            get_platforms_func = cv2.ocl.getPlatfoms

        if get_platforms_func:
            try:
                platforms = get_platforms_func()
                print(f"\nNumber of OpenCL platforms found: {len(platforms)}")
                if platforms:
                    print("--- Platforms ---")
                    for i, p in enumerate(platforms):
                        platform_name = (
                            getattr(p, "name", lambda: "N/A")()
                            if callable(getattr(p, "name", None))
                            else getattr(p, "name", "N/A")
                        )
                        platform_vendor = (
                            getattr(p, "vendor", lambda: "N/A")()
                            if callable(getattr(p, "vendor", None))
                            else getattr(p, "vendor", "N/A")
                        )
                        print(
                            f"  [{i}] Name: {platform_name}, Vendor: {platform_vendor}"
                        )
                        devices = []
                        get_device_func = getattr(p, "getDevice", None)
                        if get_device_func and callable(get_device_func):
                            try:
                                devices = get_device_func(cv2.ocl.Device_TYPE_ALL)
                            except cv2.error as e_getdev:
                                print(f"    Error getting devices: {e_getdev}")
                        if devices:
                            for j, d in enumerate(devices):
                                device_name = (
                                    getattr(d, "name", lambda: "N/A")()
                                    if callable(getattr(d, "name", None))
                                    else getattr(d, "name", "N/A")
                                )
                                print(f"    Device [{j}]: {device_name}")
                        else:
                            print("    No devices found for this platform.")
                    print("-----------------")
                else:
                    print("No platforms reported by getPlatforms().")
            except Exception as e_platform_info:
                print(f"Error getting platform info: {e_platform_info}")
        else:
            print("getPlatforms function not found in cv2.ocl")
except Exception as e:
    print(f"An error occurred: {e}")
