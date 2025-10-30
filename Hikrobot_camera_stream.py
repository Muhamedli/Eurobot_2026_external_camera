import cv2
import numpy as np
import os
import yaml
import cvf
from cvf import Camera, HarvesterCamera
from collections import deque


# Flag for recording the calibration dataset (snapshot by pressing the "s" key)
RECORD_DATASET = False
ARUCO_DETECT = False
FAST_ARUCO_DETECT = False
RECORD_DATASET = False # Flag for recording the calibration dataset (snapshot by pressing the "s" key)
ROBOT_TRACKING = True

# Params for fast marker detection
ROI_SIZE = 1000 # Frame area to search
ROI_CENTER_ARUCO = [6, 126, 127, 74, 75, 76, 77] # ID-list of ROI-center aruco marker
prev_center = None
current_target_aruco = None

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

# ======================== Configs for robot tracking ===========================
team = "yellow"
# team_enemy = "blue"
camera_our = Camera(camera_stream_params_path, team)
# camera_enemy = Camera(camera_stream_params_path, team_enemy)
cap = HarvesterCamera(camera_stream_params_path, camera_cti_api_path)
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
tvec_history = deque(maxlen=10)

def calculate_avg_fps(start_time, end_time):
    frame_times.append((end_time - start_time) / cv2.getTickFrequency())
    
    return len(frame_times) / sum(frame_times)

def sharpening(image, kernel=15, sigma=20):
    blurred = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    adjusted_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    return adjusted_image

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

def main():
    try:
        cv2.namedWindow("Camera stream", cv2.WINDOW_NORMAL)

        start_timestamp = cv2.getTickCount()

        while True:
            key = cv2.waitKey(1) & 0xFF

            ret, aruco_img = cap.harvester_read()

            # ============================== Image post-processing ============================
            # Filtering
            # aruco_img = cv2.bilateralFilter(aruco_img, 5, 80, 40)

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

            # ============================= Fast aruco detection ==============================
            if FAST_ARUCO_DETECT:
                corners, ids, rejected = aruco_detect_in_roi(display_img, ROI_SIZE, ROI_CENTER_ARUCO)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(display_img, corners, ids, (0, 0, 255))
            # =================================================================================

            # ======================= Image recording for calibration =========================
            if RECORD_DATASET:
                if key == ord('s'):
                    filename = os.path.join(calibration_dataset_path, f'img_{count}.png')
                    cv2.imwrite(filename, aruco_img)
                    print(f"Image saved as {filename}")
                    count += 1
            # =================================================================================

            # ================================= Robot tracking ================================
            if ROBOT_TRACKING:
                # Оценка положения камеры (делается один раз)
                if not camera_our.initialize_field_pose(cap, num_frames=20):
                    cv2.destroyAllWindows()
                    exit()

                # camera_enemy.tmatrix_field = camera_our.tmatrix_field
                # camera_enemy.tvec_cam_to_field = camera_our.tvec_cam_to_field
                # camera_enemy.pose_initialized = camera_our.pose_initialized

                # Вызываем быструю функцию отслеживания
                results, results_enemy = None, None
                results = camera_our.fast_robot_tracking(display_img)
                # results_enemy = camera_enemy.fast_robot_tracking(display_img)

                points_robot = np.array([
                    [0.035, 0.035, 0.0], [-0.035, 0.035, 0], [-0.035, -0.035, 0], [0.035, -0.035, 0],
                    [0.05, 0.025, -0.03], [0.05, 0.025, -0.08], [0.05, -0.025, -0.08], [0.05, -0.025, -0.03]
                    ], dtype=np.float32)

                if results is not None:
                    tvec = results[0]
                    # print(f"Our robot: x={tvec[0]:.4f}, y={tvec[1]:.4f}, z={tvec[2]:.4f}")
                    cv2.aruco.drawDetectedMarkers(display_img, results[3], results[4], (0, 0, 255))
                    display_img = camera_our.project_field_and_robot_to_image(display_img, points_robot)
                    means, stds = cvf.calculate_moving_stats(tvec, tvec_history)
                    print(f"x={means[0]:.4f}, y={means[1]:.4f}, z={means[2]:.4f}")
                    print(f"sigma_x={stds[0]:.4f}, sigma_y={stds[1]:.4f}, sigma_z={stds[2]:.4f}")
                    np.set_printoptions(linewidth=300)
                    print("Covariance matrix:\n", results[2])

                # if results_enemy is not None:
                #     tvec_enemy = results_enemy[0]
                #     print(f"Enemy robot: x={tvec_enemy[0]:.3f}, y={tvec_enemy[1]:.3f}, z={tvec_enemy[2]:.3f}")
                #     cv2.aruco.drawDetectedMarkers(display_img, results_enemy[3], results_enemy[4], (0, 0, 255))
            # =================================================================================

            cv2.imshow("Camera stream", display_img)

            # ================================== fps counter ==================================
            raw_fps = cap.get_raw_fps()
            end_timestamp = cv2.getTickCount()
            main_fps = calculate_avg_fps(start_timestamp, end_timestamp)
            start_timestamp = cv2.getTickCount()
            print(f"Aruco detection fps: {main_fps:.1f} Hz   RAW stream fps: {raw_fps:.1f} Hz\n")
            # =================================================================================

            if key == ord('q'):
                break

    except Exception as e:
        print(f"Критическая ошибка: {e}")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
