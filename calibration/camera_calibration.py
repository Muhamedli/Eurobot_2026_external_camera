import cv2
import numpy as np
import os
import yaml
from datetime import datetime


# ========================== Calibration configuration ==========================
DISPLAY_DETECTED_MARKERS = False  # True - отображение original img с маркерами
DISPLAY_UNDISTORTED = False  # True - отображение undistorted img для radtan/KB
USE_8_COEFFS = False  # True - 8 коэффициентов radtan, False - 5 коэффициентов
# ===============================================================================

# ============================= Directory paths =================================
script_dir = os.path.dirname(os.path.abspath(__file__))
intrinsics_path = os.path.join(script_dir, 'intrinsics/intrinsics.yaml')
ChArUco_board_path = os.path.join(script_dir, 'config/ChArUco_board.yaml')
calibration_dataset_path = os.path.join(script_dir, 'calibration_dataset')
# ===============================================================================

# =============== Reading the parameters of the calibration board ===============
with open(ChArUco_board_path, 'r') as f:
    board_config = yaml.safe_load(f)

board_params = board_config['ChArUco_board']
SQUARES_X = board_params['width']
SQUARES_Y = board_params['height']
SQUARE_LENGTH = board_params['square_size']
MARKER_LENGTH = board_params['marker_size']
ARUCO_DICT_NAME = board_params['dict_id']
LEGACY_PATTERN = board_params['legacy_pattern']
# ===============================================================================

# ======================= Setting up the marker detector ========================
aruco_dict_map = {
    4: cv2.aruco.DICT_4X4_250,
    5: cv2.aruco.DICT_5X5_250
}

if ARUCO_DICT_NAME not in aruco_dict_map:
    print(f"Unsupported ArUco dictionary: {ARUCO_DICT_NAME}")
    exit(1)

DICTIONARY = cv2.aruco.getPredefinedDictionary(aruco_dict_map[ARUCO_DICT_NAME])

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(DICTIONARY, parameters)
board = cv2.aruco.CharucoBoard([SQUARES_X, SQUARES_Y], SQUARE_LENGTH, MARKER_LENGTH, DICTIONARY)
board.setLegacyPattern(LEGACY_PATTERN)
print(f"ChArUco_board params:\nSQUARES_X={SQUARES_X}\nSQUARES_Y={SQUARES_Y}\nSQUARE_LENGTH={SQUARE_LENGTH}\nMARKER_LENGTH={MARKER_LENGTH}\n")
# ===============================================================================

# ========================================== Loading images for calibration ===========================================
images = [os.path.join(calibration_dataset_path, f) for f in os.listdir(calibration_dataset_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

all_charuco_corners = []
all_charuco_ids = []

for image_path in images:

    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to load image: {image_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    # Adding corners and ids to arrays for calibration
    if ids is not None and len(ids) > 0:
        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 4:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            print(f"{len(charuco_corners)} points found in image: {os.path.basename(image_path)}")
            
    # Preview images with detected markers
    if DISPLAY_DETECTED_MARKERS and ids is not None and len(ids) > 0:
        img_with_markers = img.copy()
        cv2.aruco.drawDetectedMarkers(img_with_markers, corners, ids)
        
        if charuco_corners is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(img_with_markers, charuco_corners, charuco_ids)

        name = f'Detected Markers - {os.path.basename(image_path)}'

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1280, 720)
        cv2.imshow(name, img_with_markers)

        if cv2.waitKey(0) & 0xFF != 255:
            cv2.destroyAllWindows()

if len(all_charuco_corners) == 0:
    print("There are not enough points for calibration. Check the board's images and parameters")
    exit(1)
# =====================================================================================================================

height, width = gray.shape
image_size = (width, height)

# Preparing data for calibration
obj_points = []
img_points = []

for i in range(len(all_charuco_corners)):
    if all_charuco_ids[i] is not None and len(all_charuco_ids[i]) > 0:
        obj_pts = board.getChessboardCorners()
        valid_indices = all_charuco_ids[i].flatten()
        
        if len(valid_indices) > 0:
            # Only those points of the object that correspond to the detected corners are selected
            selected_obj_pts = obj_pts[valid_indices]
            obj_points.append(selected_obj_pts)
            img_points.append(all_charuco_corners[i])

# ========================= Radial-Tangential calibration =======================
if USE_8_COEFFS:
    flags_rad = cv2.CALIB_RATIONAL_MODEL
else:
    flags_rad = cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6

ret_rad, mtx_rad, dist_rad, rvecs_rad, tvecs_rad = cv2.calibrateCamera(
    obj_points, img_points, image_size, None, None, flags=flags_rad
)

print(f"\nRadial-Tangential Model:")
print("Intrinsics matrix:\n", mtx_rad)
print("Distortion coefficients:", dist_rad.ravel())
print("Reprojection error:", ret_rad)
# ===============================================================================

# =========================== Kannala-Brandt calibration ========================
kb_success = False
K_kb = np.eye(3, dtype=np.float64)
D_kb = np.zeros((4, 1), dtype=np.float64)
rvecs_kb = []
tvecs_kb = []

try:
    flags_kb = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | 
                cv2.fisheye.CALIB_CHECK_COND | 
                cv2.fisheye.CALIB_FIX_SKEW)
    
    # Data conversion for fisheye calibration
    obj_points_fisheye = [op.reshape(-1, 1, 3).astype(np.float64) for op in obj_points]
    img_points_fisheye = [ip.reshape(-1, 1, 2).astype(np.float64) for ip in img_points]
    
    ret_kb, K_kb, D_kb, rvecs_kb, tvecs_kb = cv2.fisheye.calibrate(
        obj_points_fisheye, img_points_fisheye, image_size,
        K_kb, D_kb,
        flags=flags_kb,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    
    print("\nKannala-Brandt Model:")
    print("Intrinsics matrix:\n", K_kb)
    print("Distortion coefficients:", D_kb.ravel())
    print("Reprojection error:", ret_kb)
    kb_success = True
except cv2.error as e:
    print("\nError in calibration of Kannala-Brandt model:", str(e))
    print("We continue with the results only for the Radial-Tangential model")
except Exception as e:
    print("\nUnexpected error while calibrating Kannala-Brandt model:", str(e))
# ===============================================================================

# Preview of undistorted images
if DISPLAY_UNDISTORTED:    
    for image_path in images:
        img = cv2.imread(image_path)
        if img is None:
            continue

        # Undistort image for the Radial-Tangentia model
        undist_rad = cv2.undistort(img, mtx_rad, dist_rad)
        name_radtan = f'Undistorted radtan - {os.path.basename(image_path)}'
        cv2.namedWindow(name_radtan, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name_radtan, 640, 480)
        cv2.moveWindow(name_radtan, 100, 200)
        cv2.imshow(name_radtan, undist_rad)
        
        # Undistort image for the Kannala-Brandt model
        if kb_success:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K_kb, D_kb, np.eye(3), K_kb, image_size, cv2.CV_16SC2
            )
            undist_kb = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            name_KB = f'Undistorted KB - {os.path.basename(image_path)}'
            cv2.namedWindow(name_KB, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name_KB, 640, 480)
            cv2.moveWindow(name_KB, 740, 200)
            cv2.imshow(name_KB, undist_kb)

        if cv2.waitKey(0) & 0xFF != 255:
            cv2.destroyAllWindows()

# Saving calibration results in YAML
data = {
    'camera': {
        'description': 'Calibration of camera',
        'date_time': datetime.now().strftime('%Y-%m-%d_%H-%M'),
        'radial_tangential': {
            'intrinsics': mtx_rad.tolist(),
            'distortion_coeffs': dist_rad.ravel().tolist(),
            'reprojection error': ret_rad
        },
        'kannala_brandt': {},
        'resolution': [width, height]
    }
}

# Adding Kannala-Brandt results if calibration is successful
if kb_success:
    data['camera']['kannala_brandt'] = {
        'intrinsics': K_kb.tolist(),
        'distortion_coeffs': D_kb.ravel().tolist(),
        'reprojection error': ret_kb
    }

with open(intrinsics_path, 'w') as file:
    yaml.dump(data, file, default_flow_style=None)

print(f"\nThe calibration results are recorded in {intrinsics_path}")
