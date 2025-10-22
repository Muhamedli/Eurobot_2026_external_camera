import cv2
import os
from collections import deque
import numpy as np

# Camera params
camera_params = {
    'width': 3840,
    'height': 2160,
    'fps': 30,
    'format': 'MJPG'
}

# Flag for recording the calibration dataset (snapshot by pressing the "s" key)
RECORD_DATASET = False

# Params for marker detection
ROI_SIZE = 500 # Frame area to search
ROI_CENTER_ARUCO = 6 # ID of ROI-center aruco marker
prev_center = None

# Dictionary for marker detection in the preview
DICTIONARY = cv2.aruco.DICT_4X4_250
aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

script_dir = os.path.dirname(os.path.abspath(__file__))
calibration_dataset_path = os.path.join(script_dir, 'calibration/calibration_dataset')

# Image name index
count = 0

# Initializing the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_params['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_params['height'])
cap.set(cv2.CAP_PROP_FPS, camera_params['fps'])
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_params['format']))

# Initializing a queue for storing frame time
frame_times = deque(maxlen=6)

start_timestamp = cv2.getTickCount()


def calculate_avg_fps(start_time, end_time):

    frame_times.append((end_time - start_time) / cv2.getTickFrequency())
    
    return len(frame_times) / sum(frame_times)

def get_roi(image, center, roi_size):
    h, w = image.shape[:2]
    half_roi = roi_size // 2
     
    x_start = max(0, center[0] - half_roi)
    y_start = max(0, center[1] - half_roi)
    x_end = min(w, center[0] + half_roi)
    y_end = min(h, center[1] + half_roi)
    
    roi = image[y_start:y_end, x_start:x_end]

    return roi, (x_start, y_start, x_end, y_end)

def aruco_detect_in_roi(image, roi_size, roi_center_aruco):
    global prev_center
    
    roi_coords = None
    
    if prev_center is None:
        corners, ids, rejected = detector.detectMarkers(image)
        offset = [0, 0]
        # print("================== Global detect ==================")
    else:
        roi, roi_coords = get_roi(image, prev_center, roi_size)
        offset = roi_coords[:2] 
        corners, ids, rejected = detector.detectMarkers(roi)

    # Displaying ROI
    if roi_coords is not None:
        x_start, y_start, x_end, y_end = roi_coords
        
        cv2.rectangle(
            img=image,
            pt1=(x_start, y_start),
            pt2=(x_end, y_end),
            color=(0, 0, 255),
            thickness=2
        )
    
    top_id = np.where(ids == roi_center_aruco)
    
    if top_id[0].size > 0:
        n_markers = len(corners)
        corners_flat = np.concatenate(corners, axis=0).reshape(-1, 2)
        corners_flat += offset
        corners = corners_flat.reshape(n_markers, 1, 4, 2)

        top_aruco_corners = corners[top_id[0][0]][0]
        center_x = int(np.mean(top_aruco_corners[:, 0]))
        center_y = int(np.mean(top_aruco_corners[:, 1]))
        prev_center = (center_x, center_y)
        # print(prev_center)
    else:
        prev_center = None
        corners, ids, rejected = detector.detectMarkers(image)

    return corners, ids, rejected


while True:

    _, img = cap.read()

    display_img = img.copy()

    corners, ids, rejected = aruco_detect_in_roi(display_img, ROI_SIZE, ROI_CENTER_ARUCO)
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

    # ======================= Image recording for calibration =========================
    if RECORD_DATASET:
        if key == ord('s'):
            filename = os.path.join(calibration_dataset_path, f'img_{count}.png')
            cv2.imwrite(filename, img)
            print(f"The image is saved as {filename}")
            count += 1
    # =================================================================================
