import cv2
import numpy as np
import os
import yaml
import cvf
from cvf import Camera, HarvesterCamera, PoseFilter
from collections import deque
from scipy.spatial.transform import Rotation as R


ARUCO_DETECT = False
FAST_ARUCO_DETECT = False
RECORD_DATASET = False # Flag for recording the calibration dataset (snapshot by pressing the "s" key)
ROBOT_TRACKING = True
MULTI_ROI = False

# Params for fast marker detection
ROI_SIZE = 1000 # Frame area to search
ROI_CENTER_ARUCO = [2] # ID-list of ROI-center aruco marker
prev_center = None
current_target_aruco = None
# Centers of the zones to check nuts
check_zone_centers = [
    (0.25, 0.45, 0),
    (0.0, -0.2, 0),
    (0.7, -0.2, 0),
    (1.4, -0.2, 0),
    (0.0, -0.9, 0),
    (0.8, -0.9, 0)
]

# Dictionary for marker detection in the preview
DICTIONARY = cv2.aruco.DICT_4X4_100
aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
parameters = cv2.aruco.DetectorParameters()
# parameters.adaptiveThreshWinSizeMin = 10
# parameters.adaptiveThreshWinSizeMax = 60
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ============================= Directory paths =================================
script_dir = os.path.dirname(os.path.abspath(__file__))
camera_cti_api_path = os.path.join(script_dir, 'camera_API/MvProducerU3V.cti')
calibration_dataset_path = os.path.join(script_dir, 'calibration/calibration_dataset')
camera_stream_params_path = os.path.join(script_dir, 'config/camera_stream_params.yaml')
# ===============================================================================

# ======================== Configs for robot tracking ===========================
team = "blue"
# team_enemy = "blue"
camera_our = Camera(camera_stream_params_path, team)
cap = HarvesterCamera(camera_stream_params_path, camera_cti_api_path)
pose_filter = PoseFilter(min_cutoff=1.0, beta=1.0)
# ===============================================================================

# ============================== Camera params ==================================
with open(camera_stream_params_path, 'r') as f:
    camera_config = yaml.safe_load(f)

camera_stream_params = camera_config['camera_stream_params']
PIXEL_FORMAT = camera_stream_params['pixel_format']
RESOLUTION = camera_stream_params['resolution']
FPS = camera_stream_params['fps']
AUTO_EXPOSURE = camera_stream_params['auto_exposure']
AUTO_GAIN = camera_stream_params['auto_gain']
BLACK_LEVEL = camera_stream_params['black_level']
# ===============================================================================

# Initializing a queue for storing frame time
frame_times = deque(maxlen=10)
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
        # Global detection - we search for any of the desired markers
        corners, ids, rejected = detector.detectMarkers(image)
        print("Global detection")
        
        if ids is not None:
            # We are looking for the first marker we come across from the desired ones
            target_mask = np.isin(ids.flatten(), roi_center_aruco_list)
            target_indices = np.flatnonzero(target_mask)
            
            if target_indices.size > 0:
                # We select the first found marker as the target
                current_target_aruco = ids[target_indices[0]][0]
                # Calculate the center for the new target marker
                top_aruco_corners = corners[target_indices[0]][0]
                prev_center = (int(np.mean(top_aruco_corners[:, 0])), 
                              int(np.mean(top_aruco_corners[:, 1])))
            else:
                current_target_aruco = None
                prev_center = None
    else:
        # ROI detection around the current target marker
        roi, roi_coords = get_roi(image, prev_center, roi_size)
        offset = roi_coords[:2]
        corners, ids, rejected = detector.detectMarkers(roi)
        
        target_found = False
        if ids is not None:
            # Checking the presence of the current target marker in the ROI
            current_target_mask = (ids == current_target_aruco).flatten()
            if np.any(current_target_mask):
                target_found = True
            else:
                # If the current marker is lost, we look for other desired markers in the ROI
                alternative_mask = np.isin(ids.flatten(), roi_center_aruco_list)
                alternative_indices = np.flatnonzero(alternative_mask)
                
                if alternative_indices.size > 0:
                    # Switching to an alternative marker
                    current_target_aruco = ids[alternative_indices[0]][0]
                    target_found = True
        
        if not target_found:
            # If nothing is found in the ROI, we switch to global detection
            corners, ids, rejected = detector.detectMarkers(image)
            offset = [0, 0]
            roi_coords = None
            print("switching local detection to global detection")
            
            # We try to find any desired marker
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

    # Drawing ROI
    if roi_coords is not None:
        x_start, y_start, x_end, y_end = roi_coords
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.putText(image, f'Target: {current_target_aruco}', 
                   (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Update the center position for the current target marker
    if ids is not None and current_target_aruco is not None:
        target_mask = (ids == current_target_aruco).flatten()
        target_indices = np.flatnonzero(target_mask)
        
        # If the target marker is not found, we reset the state
        if target_indices.size == 0:
            prev_center = None
            current_target_aruco = None
            return corners, ids, rejected
        
        # Target marker found - update position
        n_markers = len(corners)
        corners_flat = np.concatenate(corners, axis=0).reshape(-1, 2)
        corners_flat += offset
        corners = corners_flat.reshape(n_markers, 1, 4, 2)
        
        top_aruco_corners = corners[target_indices[0]][0]
        prev_center = (int(np.mean(top_aruco_corners[:, 0])),
                    int(np.mean(top_aruco_corners[:, 1])))

    return corners, ids, rejected

def main():
    # try:
        count = 0
        
        cv2.namedWindow("Camera stream", cv2.WINDOW_NORMAL)

        start_timestamp = cv2.getTickCount()

        while True:
            key = cv2.waitKey(1) & 0xFF

            ret, aruco_img = cap.harvester_read()
            # aruco_img = cv2.imread("multi_roi.bmp", cv2.IMREAD_GRAYSCALE)

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
                    cv2.aruco.drawDetectedMarkers(display_img, corners, ids, (255, 255, 255))
            # =================================================================================

            # ============================= Fast aruco detection ==============================
            if FAST_ARUCO_DETECT:
                corners, ids, rejected = aruco_detect_in_roi(display_img, ROI_SIZE, ROI_CENTER_ARUCO)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(display_img, corners, ids, (255, 255, 255))
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
                # Estimating the camera position (done once)
                if not camera_our.pose_initialized:
                    print("Initializing field pose...")
                    finished = camera_our.initialize_field_pose(display_img, num_frames=20)
                    
                    if finished:
                        print("Field initialized successfully! Starting tracking...")
                    else:
                        continue

                # Calling a quick tracking function
                results = camera_our.fast_robot_tracking(display_img)
                # euler = R.from_quat(quat).as_euler("xyz", True)

                if results is not None:
                    tvec, quat, cov, corners, ids = results
                    tvec, quat = pose_filter.one_euro(tvec, quat)
                    # means, stds = cvf.calculate_moving_stats(tvec, tvec_history)

                    # Key points to check
                    points = np.array([
                        [0.0, 0.0, 0.0],
                        [0.05, 0.0, -0.058],
                        [0.0, -0.05, -0.058],
                        [-0.05, 0.0, -0.058],
                        [0.0, 0.05, -0.058],
                        [-0.0474, -0.106, -0.28],
                        [-0.0474, 0.106, -0.28],
                    ], dtype=np.float64)
                    
                    # Displaying points in the robot's coordinate system
                    display_img = camera_our.project_3D_points_from_robot_to_image(display_img, points)
                    # Displaying points in the field coordinate system
                    # display_img = camera_our.project_3D_points_from_filed_to_image(display_img, points)

                    cv2.aruco.drawDetectedMarkers(display_img, results[3], results[4], (255, 255, 255))

                    np.set_printoptions(linewidth=300)
                    print(f"tvec: x={tvec[0]:.4f}, y={tvec[1]:.4f}, z={tvec[2]:.4f}")
                    print(f"quat: x={quat[0]:.4f}, y={quat[1]:.4f}, z={quat[2]:.4f}, w={quat[3]:.4f}")
                    # print(f"euler: x={euler[0]:.4f}, y={euler[1]:.4f}, z={euler[2]:.4f}")
                    # print(f"mean x={means[0]:.4f}, y={means[1]:.4f}, z={means[2]:.4f}")
                    # print(f"sigma_x={stds[0]:.4f}, sigma_y={stds[1]:.4f}, sigma_z={stds[2]:.4f}")
                    print("Covariance matrix:\n", results[2])
            # =================================================================================

            # ================================ MULTI ROI ======================================
            if MULTI_ROI:
                # Estimating the camera position (done once)
                if not camera_our.pose_initialized:
                    print("Initializing field pose...")
                    finished = camera_our.initialize_field_pose(display_img, num_frames=20)
                    
                    if finished:
                        print("Field initialized successfully! Starting tracking...")
                    else:
                        continue
                    
                display_img, updated_scores = camera_our.pantry_checker(team, display_img, check_zone_centers, roi_size=250)
                print(updated_scores)
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

    # except Exception as e:
    #     print(f"Error: {e}")

    # finally:
    #     if cap:
    #         cap.release()
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
