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

# Params for fast marker detection
ROI_SIZE = 1000 # Frame area to search
ROI_CENTER_ARUCO = [6, 127, 126] # ID-list of ROI-center aruco marker
prev_center = None
current_target_aruco = None

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

def aruco_detect_in_roi(image, roi_size, roi_center_aruco_list):
    global prev_center, current_target_aruco
    
    roi_coords = None
    offset = [0, 0]
    
    if prev_center is None or current_target_aruco is None:
        # Глобальная детекция - ищем любой из желательных маркеров
        corners, ids, rejected = detector.detectMarkers(image)
        print("Global detection")
        
        if ids is not None:
            # Ищем первый попавшийся маркер из желательных
            target_mask = np.isin(ids.flatten(), roi_center_aruco_list)
            target_indices = np.flatnonzero(target_mask)
            
            if target_indices.size > 0:
                # Выбираем первый найденный маркер как целевой
                current_target_aruco = ids[target_indices[0]][0]
                # Вычисляем центр для нового целевого маркера
                top_aruco_corners = corners[target_indices[0]][0]
                prev_center = (int(np.mean(top_aruco_corners[:, 0])), 
                              int(np.mean(top_aruco_corners[:, 1])))
            else:
                current_target_aruco = None
                prev_center = None
    else:
        # ROI детекция вокруг текущего целевого маркера
        roi, roi_coords = get_roi(image, prev_center, roi_size)
        offset = roi_coords[:2]
        corners, ids, rejected = detector.detectMarkers(roi)
        
        target_found = False
        if ids is not None:
            # Проверяем наличие текущего целевого маркера в ROI
            current_target_mask = (ids == current_target_aruco).flatten()
            if np.any(current_target_mask):
                target_found = True
            else:
                # Если текущий маркер потерян, ищем другие желательные маркеры в ROI
                alternative_mask = np.isin(ids.flatten(), roi_center_aruco_list)
                alternative_indices = np.flatnonzero(alternative_mask)
                
                if alternative_indices.size > 0:
                    # Переключаемся на альтернативный маркер
                    current_target_aruco = ids[alternative_indices[0]][0]
                    target_found = True
        
        if not target_found:
            # Если в ROI ничего не нашли, переключаемся на глобальную детекцию
            corners, ids, rejected = detector.detectMarkers(image)
            offset = [0, 0]
            roi_coords = None
            print("switching local detection to global detection")
            
            # Пытаемся найти любой желательный маркер
            if ids is not None:
                target_mask = np.isin(ids.flatten(), roi_center_aruco_list)
                target_indices = np.flatnonzero(target_mask)
                
                if target_indices.size > 0:
                    current_target_aruco = ids[target_indices[0]][0]
                    top_aruco_corners = corners[target_indices[0]][0]
                    prev_center = (int(np.mean(top_aruco_corners[:, 0])), 
                                  int(np.mean(top_aruco_corners[:, 1])))
                else:
                    current_target_aruco = None
                    prev_center = None

    # Отрисовка ROI
    if roi_coords is not None:
        x_start, y_start, x_end, y_end = roi_coords
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.putText(image, f'Target: {current_target_aruco}', 
                   (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Обновление позиции центра для текущего целевого маркера
    if ids is not None and current_target_aruco is not None:
        target_mask = (ids == current_target_aruco).flatten()
        target_indices = np.flatnonzero(target_mask)
        
        # Если целевой маркер не найден - сбрасываем состояние
        if target_indices.size == 0:
            prev_center = None
            current_target_aruco = None
            return corners, ids, rejected
        
        # Целевой маркер найден - обновляем позицию
        n_markers = len(corners)
        corners_flat = np.concatenate(corners, axis=0).reshape(-1, 2)
        corners_flat += offset
        corners = corners_flat.reshape(n_markers, 1, 4, 2)
        
        top_aruco_corners = corners[target_indices[0]][0]
        prev_center = (int(np.mean(top_aruco_corners[:, 0])),
                    int(np.mean(top_aruco_corners[:, 1])))

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
