from harvesters.core import Harvester
import cv2
import numpy as np
import os
import yaml
import sys
from collections import deque

# Flag for recording the calibration dataset (snapshot by pressing the "s" key)
RECORD_DATASET = False
ARUCO_DETECT = False

# Dictionary for marker detection in the preview
DICTIONARY = cv2.aruco.DICT_4X4_250
aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ============================= Directory paths =================================
script_dir = os.path.dirname(os.path.abspath(__file__))
camera_cti_api_path = os.path.join(script_dir, 'camera_API/MvProducerU3V.cti')
calibration_dataset_path = os.path.join(script_dir, 'calibration/calibration_dataset')
camera_stream_params_path = os.path.join(script_dir, 'config/camera_stream_params.yaml')
# ===============================================================================

# ============================== Camera params ==================================
with open(camera_stream_params_path, 'r') as f:
    camera_config = yaml.safe_load(f)

camera_stream_params = camera_config['camera_stream_params']
PIXEL_FORMAT = camera_stream_params['pixel_format']
WIDTH = camera_stream_params['width']
HEIGHT = camera_stream_params['height']
FPS = camera_stream_params['fps']
AUTO_EXPOSURE = camera_stream_params['auto_exposure']
AUTO_GAIN = camera_stream_params['auto_gain']
BLACK_LEVEL = camera_stream_params['black_level']
# ===============================================================================

# Initializing a queue for storing frame time
frame_times = deque(maxlen=6)


def calculate_avg_fps(start_time, end_time):

    frame_times.append((end_time - start_time) / cv2.getTickFrequency())
    
    return len(frame_times) / sum(frame_times)

def gamma_correction(image, gamma=1.0):

    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    adjusted_image = cv2.LUT(image, table)

    return adjusted_image

def sharpening(image, kernel=15, sigma=20):

    blurred = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    adjusted_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    return adjusted_image


def main():
    
    # Initializing Harvester
    h = Harvester()
    
    # Path to GenTL Producer (MvProducerU3V.cti) for USB3 Vision
    cti_file = camera_cti_api_path
    h.add_file(cti_file)
    h.update()

    # Image name index
    count = 0
    
    # Checking available devices
    if not h.device_info_list:
        print("Camera not found. Check your connection and access to MvProducerU3V.cti.")
        h.reset()
        return
    
    # Displaying information about devices
    for i, device in enumerate(h.device_info_list):
        print(f"Device {i}: {device}")
    
    try:
        # ============================ Initializing the camera ==========================
        ia = h.create(0)
        ia.device.node_map.Width.value = WIDTH
        ia.device.node_map.Height.value = HEIGHT
        ia.device.node_map.PixelFormat.value = PIXEL_FORMAT
        ia.device.node_map.AcquisitionFrameRateEnable.value = True
        ia.device.node_map.AcquisitionFrameRate.value = FPS
        ia.remote_device.node_map.ExposureAuto.value = 'Continuous'
        ia.remote_device.node_map.AutoExposureTimeLowerLimit.value = AUTO_EXPOSURE[0]
        ia.remote_device.node_map.AutoExposureTimeUpperLimit.value = AUTO_EXPOSURE[1]
        ia.remote_device.node_map.GainAuto.value = 'Continuous'
        ia.remote_device.node_map.AutoGainLowerLimit.value = AUTO_GAIN[0]
        ia.remote_device.node_map.AutoGainUpperLimit.value = AUTO_GAIN[1]
        ia.remote_device.node_map.BlackLevelEnable.value = True
        ia.remote_device.node_map.BlackLevel.value = BLACK_LEVEL
        # ===============================================================================

        # Starting a video stream
        ia.start()
        
        cv2.namedWindow("Camera stream", cv2.WINDOW_NORMAL)

        start_timestamp = cv2.getTickCount()

        while True:
            with ia.fetch() as buffer:
                if buffer.payload.components:
                    # Image data extraction
                    img = buffer.payload.components[0].data
                    img = img.reshape(buffer.payload.components[0].height, buffer.payload.components[0].width)

                    # ============================== Image post-processing ============================
                    # Demosaicing
                    aruco_img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR_EA)

                    # Filtering
                    aruco_img = cv2.bilateralFilter(aruco_img, 5, 80, 40)
                    
                    # Gamma corrections
                    aruco_img = gamma_correction(aruco_img, 0.5)

                    # Sharpening
                    # aruco_img = sharpening(aruco_img, kernel=11, sigma=25)
                    # =================================================================================

                    display_img = aruco_img.copy()

                    # =============================== Aruco detection =================================
                    if ARUCO_DETECT:
                        corners, ids, rejected = detector.detectMarkers(display_img)
                        if ids is not None:
                            cv2.aruco.drawDetectedMarkers(display_img, corners, ids, (0, 0, 255))
                    # =================================================================================

                    cv2.imshow("Camera stream", display_img)

                    key = cv2.waitKey(1) & 0xFF

                    # ======================= Image recording for calibration =========================
                    if RECORD_DATASET:
                        if key == ord('s'):
                            filename = os.path.join(calibration_dataset_path, f'img_{count}.png')
                            cv2.imwrite(filename, aruco_img)
                            print(f"Image saved as {filename}")
                            count += 1
                    # =================================================================================

                    # ================================== fps counter ==================================
                    raw_fps = ia.remote_device.node_map.ResultingFrameRate.value
                    end_timestamp = cv2.getTickCount()
                    main_fps = calculate_avg_fps(start_timestamp, end_timestamp)
                    start_timestamp = cv2.getTickCount()
                    print(f"Aruco detection fps: {main_fps:.1f} Hz   RAW stream fps: {raw_fps:.1f} Hz")
                    # =================================================================================

            if key == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc(file=sys.stdout)
    
    finally:
        ia.stop()
        ia.destroy()
        h.reset()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()