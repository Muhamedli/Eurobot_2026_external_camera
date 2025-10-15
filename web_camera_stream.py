import cv2
import os
from collections import deque


# Camera params
camera_params = {
    'width': 2560,
    'height': 1440,
    'fps': 30,
    'format': 'MJPG'
}

# Flag for recording the calibration dataset (snapshot by pressing the "s" key)
RECORD_DATASET = False
DICTIONARY = cv2.aruco.DICT_4X4_250

aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
parameters = cv2.aruco.DetectorParameters()

script_dir = os.path.dirname(os.path.abspath(__file__))
calibration_dataset_path = os.path.join(script_dir, 'calibration/calibration_dataset')

# Initializing the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_params['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_params['height'])
cap.set(cv2.CAP_PROP_FPS, camera_params['fps'])
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_params['format']))

# Image name index
count = 0

# Initializing a queue for storing frame time
frame_times = deque(maxlen=6)


def calculate_avg_fps(start_time, end_time):

    frame_times.append((end_time - start_time) / cv2.getTickFrequency())
    
    return len(frame_times) / sum(frame_times)


start_timestamp = cv2.getTickCount()

while True:

    r, img = cap.read()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(img)

    display_img = img.copy()

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display_img, corners, ids, (0, 0, 255))

    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.imshow("preview", display_img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # ================================== fps counter ==================================
    end_timestamp = cv2.getTickCount()
    main_fps = calculate_avg_fps(start_timestamp, end_timestamp)
    start_timestamp = cv2.getTickCount()
    print(f"Aruco detection fps: {main_fps:.1f} Hz")
    # =================================================================================

    if RECORD_DATASET:
        if key == ord('s'):
            filename = os.path.join(calibration_dataset_path, f'img_{count}.png')
            cv2.imwrite(filename, img)
            print(f"The image is saved as {filename}")
            count += 1
