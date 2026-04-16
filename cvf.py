import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from harvesters.core import Harvester
from typing import Tuple, Deque, List
from collections import deque
import time


def read_config(config_path):
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def gamma_correction(image, gamma=0.5):
    table = np.array([((i / 255.0) ** gamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
    adjusted_image = cv2.LUT(image, table)

    return adjusted_image

def calculate_moving_stats(
    vec: List[float], 
    history_deque: Deque[List[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Рассчитывает скользящее среднее и СКО для выборки.

    Args:
        vec (List[float]): список с данными: [x, y, z].
        history_deque (Deque): очередь для формирования выборки.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Кортеж из двух массивов: (means, stds)
    """
    
    # Добавляем новый вектор в историю
    history_deque.append(vec)
    data = np.array(history_deque)

    n = len(history_deque)
    
    if n == 0:
        return np.zeros(3), np.zeros(3)

    # Рассчитываем среднее по столбцам
    means = np.mean(data, axis=0)
    
    # Рассчитываем СКО
    if n < 2:
        stds = np.zeros(3)
    else:
        stds = np.std(data, axis=0, ddof=1)
        
    return means, stds

def adjoint_SE3(R, t):
    """Adjoint matrix for SE(3) transform [R,t]."""
    t = t.reshape(3, 1)
    skew = np.array([[0, -t[2,0], t[1,0]],
                     [t[2,0], 0, -t[0,0]],
                     [-t[1,0], t[0,0], 0]])
    Ad = np.zeros((6,6))
    Ad[:3,:3] = R
    Ad[3:,3:] = R
    Ad[3:,:3] = skew @ R
    return Ad

def compute_pose_covariance(object_points, image_points, rvec, tvec,
                            camera_matrix, dist_coeffs, sigma_pix=0.5):
    """
    Оценка ковариации 6D-позы (rvec,tvec) из ошибки репроекции.
    sigma_pix — СКО ошибки обнаружения точек в пикселях.
    """
    # Получаем якобиан проекции по параметрам позы
    _, J_full = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    J = J_full[:, :6]                   # (2N x 6)
    # Простая аппроксимация: Sigma_pix = sigma^2 * I
    H = (J.T @ J) / (sigma_pix**2)
    Sigma_pose = np.linalg.inv(H)
    return Sigma_pose

def transform_covariance_SE3(Sigma_source, R_src_dst, t_src_dst):
    """
    Перенос ковариации SE3 через преобразование T=[R,t].
    """
    Ad = adjoint_SE3(R_src_dst, t_src_dst)
    return Ad @ Sigma_source @ Ad.T



class Camera:
    def __init__(self, config_path: str, team: str, roi_size: int = 1000, aruco_cube: bool = False):
        # Инициализация параметров из конфига
        self.config = read_config(config_path)
        self.camera_matrix = np.array(self.config['camera_stream_params']['radial_tangential']['intrinsics'], dtype=np.float64)
        self.dist_coefs = np.array(self.config['camera_stream_params']['radial_tangential']['distortion_coeffs'], dtype=np.float64)

        # Настройки детектора ArUco
        self.arucoParams = cv2.aruco.DetectorParameters()
        # self.arucoParams.adaptiveThreshWinSizeMin = 10
        # self.arucoParams.adaptiveThreshWinSizeMax = 60
        # self.arucoParams.minMarkerPerimeterRate = 0.04
        # self.arucoParams.maxMarkerPerimeterRate = 0.3
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

        # Параметры для отслеживания в ROI
        self.roi_size = roi_size
        self.prev_center = None
        self.current_target_aruco = None

        # Очередь для расчета СКО
        self.tvec_history = deque(maxlen=3)
        self.yaw_history = deque(maxlen=3)

        # Параметры ковариации
        self.cov = np.zeros((6, 6), dtype=np.float64)
        self.base_pos_cov = 0.0001
        self.base_ang_cov = 0.0012

        # Добавляем оффсет, если установлен aruco-cube
        cube_offset = 0.085 if aruco_cube else 0.0

        # Параметры, зависящие от команды
        if team == "yellow":
            self.colour_range = range(6, 11)
            self.roi_center_aruco_list = [6, 74, 75, 76, 77, 78, 79]
            self.RotSideDict = {
                74: R.from_euler('y', 90, degrees=True).as_matrix(),
                75: R.from_euler('x', 90, degrees=True).as_matrix(),
                76: R.from_euler('y', -90, degrees=True).as_matrix(),
                77: R.from_euler('x', -90, degrees=True).as_matrix(),
                78: R.from_euler('xz', [90, -30], degrees=True).as_matrix(),
                79: R.from_euler('xz', [90, -150], degrees=True).as_matrix(),
            }
            self.TvecSideDict = {
                74: np.array([-0.05, 0.0, 0.058 + cube_offset]),
                75: np.array([0.0, 0.05, 0.058 + cube_offset]),
                76: np.array([0.05, 0.0, 0.058 + cube_offset]),
                77: np.array([0.0, -0.05, 0.058 + cube_offset]),
                78: np.array([0.0474, 0.106, 0.28 + cube_offset]),
                79: np.array([0.0474, -0.106, 0.28 + cube_offset]),
            }
        else: # team == "blue"
            self.colour_range = range(1, 6)
            self.roi_center_aruco_list = [2, 55, 56, 57, 58, 59, 60]
            self.RotSideDict = {
                55: R.from_euler('y', 90, degrees=True).as_matrix(),
                56: R.from_euler('x', 90, degrees=True).as_matrix(),
                57: R.from_euler('y', -90, degrees=True).as_matrix(),
                58: R.from_euler('x', -90, degrees=True).as_matrix(),
                59: R.from_euler('xz', [90, -30], degrees=True).as_matrix(),
                60: R.from_euler('xz', [90, -150], degrees=True).as_matrix(),
            }
            self.TvecSideDict = {
                55: np.array([-0.05, 0.0, 0.058 + cube_offset]),
                56: np.array([0.0, 0.05, 0.058 + cube_offset]),
                57: np.array([0.05, 0.0, 0.058 + cube_offset]),
                58: np.array([0.0, -0.05, 0.058 + cube_offset]),
                59: np.array([0.0474, 0.106, 0.28 + cube_offset]),
                60: np.array([0.0474, -0.106, 0.28 + cube_offset]),
            }
        
        # Координаты маркеров поля
        self.field_markers = {
            20: np.array([-0.9, 0.4, 0.0]),
            21: np.array([0.9, 0.4, 0.0]),
            22: np.array([-0.9, -0.4, 0.0]),
            23: np.array([0.9, -0.4, 0.0])
        }
        
        self.field_marker_size = 0.1

        # Координаты углов маркеров поля
        self.field_marker_obj_pts = {}
        base_pts = np.array([
            [-self.field_marker_size/2, self.field_marker_size/2, 0],
            [self.field_marker_size/2, self.field_marker_size/2, 0],
            [self.field_marker_size/2, -self.field_marker_size/2, 0],
            [-self.field_marker_size/2, -self.field_marker_size/2, 0]
        ], dtype=np.float64)

        for mid, tvec in self.field_markers.items():
            self.field_marker_obj_pts[mid] = base_pts + tvec

        # Состояние инициализации положения камеры
        self.pose_initialized = False
        self.tmatrix_field = None      # Матрица поворота камера2поле
        self.tvec_cam_to_field = None  # Вектор трансляции камера2поле
        self.init_tmatrices = []
        self.init_tvecs = []

        # Транформы поле2робот
        self.robot_tvec = None
        self.robot_rot_matrix = None

        # Очки в зоне при MULTI_ROI отслеживании
        self.zone_states = {}

    def initialize_field_pose(self, img, num_frames=10):
        """
        Принимает один кадр (img).
        Возвращает True, если инициализация ЗАВЕРШЕНА (или уже была завершена).
        Возвращает False, если процесс еще идет.
        """
        # Если уже инициализировано, сразу говорим ОК
        if self.pose_initialized:
            return True
        
        # Детекция маркеров (используем детектор класса)
        corners, ids, _ = self.detector.detectMarkers(img)
        
        if ids is not None:
            ids = list(map(lambda x: x[0], ids))

        # Если нет маркеров поля, выходим, ждем следующий кадр
        if not ids or not any(marker_id in self.field_markers for marker_id in ids):
            return False
        
        field_corners, field_ids = [], []

        for i, marker_id in enumerate(ids):
            if marker_id in self.field_markers:
                field_corners.append(corners[i][0])
                field_ids.append(marker_id)
        
        # Проверка полноты (нужно 3 маркера)
        if len(field_ids) < 2:
            # print(f"Detected {len(field_ids)} markers out of 4") # Можно раскомментировать для отладки
            return False

        object_points, image_points = [], []

        for mid, corners_set in zip(field_ids, field_corners):
            object_points.extend(self.field_marker_obj_pts[mid])
            image_points.extend(corners_set)

        success, rvec, tvec = cv2.solvePnP(
            np.array(object_points), np.array(image_points), 
            self.camera_matrix, self.dist_coefs, flags=cv2.SOLVEPNP_IPPE)
        
        success, rvec, tvec = cv2.solvePnP(
            np.array(object_points), np.array(image_points),
            self.camera_matrix, self.dist_coefs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            self.init_tmatrices.append(cv2.Rodrigues(rvec)[0])
            self.init_tvecs.append(tvec.flatten())
            # print(f"Init progress: {len(self.init_tmatrices)}/{num_frames}")

        # ПРОВЕРКА: Набрали ли достаточно кадров?
        if len(self.init_tmatrices) >= num_frames:
            # === ФИНАЛИЗАЦИЯ (Расчет средних) ===
            self.tvec_cam_to_field = np.mean(self.init_tvecs, axis=0)

            quats = R.from_matrix(self.init_tmatrices).as_quat()
            quats[quats[:, 3] < 0] *= -1
            avg_quat = np.mean(quats, axis=0)
            avg_quat /= np.linalg.norm(avg_quat)
            self.tmatrix_field = R.from_quat(avg_quat).as_matrix()

            self.pose_initialized = True
            
            # Очищаем буферы, они больше не нужны
            self.init_tmatrices = []
            self.init_tvecs = []
            
            # print("Field pose initialization successful!")
            return True

        return False

    def get_roi(self, image, center, roi_size):
        h, w = image.shape[:2]
        half_roi = roi_size // 2

        x_start = max(0, center[0] - half_roi)
        y_start = max(0, center[1] - half_roi)
        x_end = min(w, center[0] + half_roi)
        y_end = min(h, center[1] + half_roi)

        roi = image[y_start:y_end, x_start:x_end]
        
        return roi, (x_start, y_start, x_end, y_end)

    def aruco_detect_in_roi(self, image, roi_size, roi_center_aruco_list):
        """Быстрый поиск маркеров с использованием ROI"""
        roi_coords = None
        offset = [0, 0]
        # Глобальная детекция, если нет цели
        if self.prev_center is None or self.current_target_aruco is None:
            corners, ids, rejected = self.detector.detectMarkers(image)

            if ids is not None:
                target_mask = np.isin(ids.flatten(), roi_center_aruco_list)
                target_indices = np.flatnonzero(target_mask)
                if target_indices.size > 0:
                    self.current_target_aruco = ids[target_indices[0]][0]
                    top_aruco_corners = corners[target_indices[0]][0]
                    self.prev_center = (int(np.mean(top_aruco_corners[:, 0])), int(np.mean(top_aruco_corners[:, 1])))
        else:
            # Локальная детекция в ROI
            roi, roi_coords = self.get_roi(image, self.prev_center, roi_size)
            offset = roi_coords[:2]
            corners, ids, rejected = self.detector.detectMarkers(roi)
            
            target_found = False
            if ids is not None and np.any((ids == self.current_target_aruco)):
                target_found = True
            
            if not target_found:
                # Если цель потеряна, снова переключаемся на глобальный поиск
                corners, ids, rejected = self.detector.detectMarkers(image)
                offset = [0, 0]

                if ids is not None:
                    target_mask = np.isin(ids.flatten(), roi_center_aruco_list)
                    target_indices = np.flatnonzero(target_mask)
                    if target_indices.size > 0:
                        self.current_target_aruco = ids[target_indices[0]][0]
                        top_aruco_corners = corners[target_indices[0]][0]
                        self.prev_center = (int(np.mean(top_aruco_corners[:, 0])), int(np.mean(top_aruco_corners[:, 1])))
                    else: # Если ничего не нашли, сбрасываем состояние
                        self.current_target_aruco = None
                        self.prev_center = None

        # Обновление координат и центра для следующего кадра
        if ids is not None and self.current_target_aruco is not None:
            target_mask = (ids.flatten() == self.current_target_aruco)
            target_indices = np.flatnonzero(target_mask)
            
            if target_indices.size == 0:
                self.prev_center = None
                self.current_target_aruco = None
                return None, None, None # Маркер потерян
            
            n_markers = len(corners)
            corners_flat = np.concatenate(corners, axis=0).reshape(-1, 2).astype(np.float64)
            corners_flat += np.array(offset, dtype=np.float64)
            corners = corners_flat.reshape(n_markers, 1, 4, 2)
            
            top_aruco_corners = corners[target_indices[0]][0]
            self.prev_center = (int(np.mean(top_aruco_corners[:, 0])), int(np.mean(top_aruco_corners[:, 1])))

            return corners, ids.flatten(), rejected
        return None, None, None

    def estimate_robot_pose(self, ids, corners, cov_flag=True):
        robot_corners, robot_ids = [], []
        target_range = self.colour_range

        for i, marker_id in enumerate(ids):
            condition = (marker_id in target_range) or (marker_id in self.RotSideDict)
            if condition:
                robot_corners.append(corners[i][0])
                robot_ids.append(marker_id)

        if not robot_ids:
            return None, None, None

        object_points, image_points = [], []
        for mid, corners_set in zip(robot_ids, robot_corners):
            if 1 <= mid <= 10: marker_length = 0.07
            elif mid in [59, 60, 78, 79]: marker_length = 0.08
            else: marker_length = 0.05
            
            obj_pts = np.array([
                [-marker_length/2, marker_length/2, 0],
                [marker_length/2, marker_length/2, 0],
                [marker_length/2, -marker_length/2, 0],
                [-marker_length/2, -marker_length/2, 0]
            ], dtype=np.float64)
            
            if mid in self.RotSideDict:
                obj_pts = np.dot(obj_pts, self.RotSideDict[mid].T) - self.TvecSideDict[mid]
            object_points.extend(obj_pts)
            image_points.extend(corners_set)
            
        success, rvec, tvec = cv2.solvePnP(
            np.array(object_points), np.array(image_points), 
            self.camera_matrix, self.dist_coefs, flags=cv2.SOLVEPNP_SQPNP)
        
        success, rvec, tvec = cv2.solvePnP(
            np.array(object_points), np.array(image_points),
            self.camera_matrix, self.dist_coefs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success:
            return None, None, None

        tvec_cam = tvec.flatten()
        rot_matrix = cv2.Rodrigues(rvec)[0]

        # Преобразование в систему координат поля, используя вектор камера2поле и матрицу поворота
        self.robot_tvec = np.dot(self.tmatrix_field.T, tvec_cam - self.tvec_cam_to_field)
        self.robot_rot_matrix = np.dot(self.tmatrix_field.T, rot_matrix)
        quat = R.from_matrix(self.robot_rot_matrix).as_quat()

        r_temp = R.from_matrix(self.robot_rot_matrix)
        current_euler = r_temp.as_euler('xyz', degrees=False)

        if cov_flag:
            # Рассчитываем ковариацию в системе координат камеры
            Sigma_cam = compute_pose_covariance(
                object_points=np.array(object_points), 
                image_points=np.array(image_points), 
                rvec=rvec, 
                tvec=tvec,
                camera_matrix=self.camera_matrix, 
                dist_coeffs=self.dist_coefs, 
                sigma_pix=1.0
            )
            
            # Переносим ковариацию в систему координат поля (SE(3) transformation)                
            R_field_to_cam = self.tmatrix_field.T
            t_field_to_cam = - R_field_to_cam @ self.tvec_cam_to_field.reshape(3, 1)
            
            # Применяем формулу переноса ковариации
            cov = transform_covariance_SE3(
                Sigma_cam, 
                R_field_to_cam, 
                t_field_to_cam
            )
        else:
            _, pose_stds = calculate_moving_stats(self.robot_tvec.tolist(), self.tvec_history)
            _, ang_std = calculate_moving_stats(current_euler[1], self.yaw_history)

            if len(self.tvec_history) >= 2:
                k_gain_pose = 5.0
                k_gain_ang = 5.0
                
                self.cov[0, 0] = self.base_pos_cov + (pose_stds[0]**2) * k_gain_pose
                self.cov[1, 1] = self.base_pos_cov + (pose_stds[1]**2) * k_gain_pose
                self.cov[2, 2] = self.base_pos_cov + (pose_stds[2]**2) * k_gain_pose
                
                ang_cov = self.base_ang_cov + (ang_std**2) * k_gain_ang
                
                self.cov[3, 3] = ang_cov
                self.cov[4, 4] = ang_cov
                self.cov[5, 5] = ang_cov
            else:
                self.cov.flat[0:15:7] = self.base_pos_cov
                self.cov.flat[21:36:7] = self.base_ang_cov

        return self.robot_tvec, quat, self.cov

    def fast_robot_tracking(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray] | None:
        """
        Performs fast robot pose tracking.

        This method implements high-performance tracking by detecting ArUco markers
        in a limited region of interest (ROI) rather than across the entire frame.
        This significantly speeds up processing and reduces computational load.

        Requires that `initialize_field_pose()` be called at least once
        before this method can be used.
        
        Args:
            img (np.ndarray): The input video frame on which to detect markers.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray] | None:
                On successful detection, returns a tuple containing:
                    - tvec (np.ndarray): Translation vector (x, y, z) of the
                    robot's pose relative to the field.
                    - quat (np.ndarray): Quaternion (w, x, y, z) representing
                    the robot's rotation relative to the field.
                    - cov (np.ndarray): The 6x6 pose covariance matrix.
                    - corners (list): The list of
                    detected marker corner coordinates.
                    - ids (np.ndarray): The array of IDs for the detected markers.
                
                Returns None if `self.pose_initialized` is False or if no
                markers are found in the ROI.
            """
        if not self.pose_initialized:
            print("Error: Field pose is not initialized. Call initialize_field_pose() first.")
            return None
                
        # Используем быстрый детектор с ROI
        corners, ids, _ = self.aruco_detect_in_roi(img, self.roi_size, self.roi_center_aruco_list)

        # Оценка позы робота, используя найденные маркеры и трансформ к полю
        if ids is not None:
            tvec, quat, cov = self.estimate_robot_pose(ids, corners, cov_flag=False)
            return tvec, quat, cov, corners, ids

        return None

    def project_3D_points_from_robot_to_image(self, image, points_robot):
        """
        Отображение 3D точек в СК робота на изображении.
        """
        # Объединяем вращения: R_cam_robot = R_cam_field @ R_field_robot
        R_cam_robot = self.tmatrix_field @ self.robot_rot_matrix

        # Убедимся, что векторы переноса имеют форму (3, 1) для умножения
        tvec_field_robot_col = np.asarray(self.robot_tvec).reshape(3, 1)
        tvec_cam_field_col = np.asarray(self.tvec_cam_to_field).reshape(3, 1)

        # Объединяем переносы: t_cam_robot = R_cam_field * t_field_robot + t_cam_field
        t_cam_robot = (self.tmatrix_field @ tvec_field_robot_col) + tvec_cam_field_col

        # Проецирование
        # Преобразуем итоговую матрицу вращения R_cam_robot обратно в rvec
        # для использования в cv2.projectPoints
        rvec_cam_robot, _ = cv2.Rodrigues(R_cam_robot)

        # Убедимся, что входные точки имеют правильный тип и форму
        # cv2.projectPoints ожидает (N, 3) или (N, 1, 3) и тип float64
        points_robot_np = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)

        # Проецируем точки, используя объединенное преобразование
        image_points, _ = cv2.projectPoints(
            points_robot_np,
            rvec_cam_robot,
            t_cam_robot,
            self.camera_matrix,
            self.dist_coefs
        )

        image_points = image_points.reshape(-1, 2)

        for point in image_points:
            # Координаты должны быть целыми числами (пикселями) и в виде кортежа (tuple)
            center = tuple(point.astype(int))
            
            # Рисуем круг в этой точке
            cv2.circle(image, center, 1, (255, 0, 0), 2)

        return image

    def project_3D_points_from_filed_to_image(self, image, points, color=(0, 255, 0), radius=1):
        """
        Отображение 3D точек в СК поля на изображении в соответствии
        с rvec и tvec после начальной инициализации камеры.
        """
        tvec = self.tvec_cam_to_field
        rvec, _ = cv2.Rodrigues(self.tmatrix_field)

        # Убедимся, что входные 3D точки имеют правильный тип и форму
        # (cv2.projectPoints ожидает (N, 3) или (N, 1, 3) и тип float64)
        points_field_np = np.asarray(points, dtype=np.float64).reshape(-1, 3)

        # Проецируем 3D точки из СК поля в СК изображения
        image_points, _ = cv2.projectPoints(
            points_field_np,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coefs
        )

        # Рисуем 2D точки на изображении
        image_points = image_points.reshape(-1, 2) # Убираем лишние измерения

        for point in image_points:
            # Координаты пикселей должны быть целыми числами
            center = tuple(point.astype(int))
            
            # Рисуем круг в найденной 2D точке
            cv2.circle(image, center, radius, color, -1)

        return image

    def pantry_checker(self, team_color, image, zones_3d_dict, roi_size=200):
        """
        Осматривает кладовые-ROI и подсчитывает очки.

        Args:
            team_color (str): 'yellow' или 'blue'.
            image (np.ndarray): исходное изображение.
            zones_3d_dict (list): список xyz координат центров зон.
            roi_size (int): размер ROI в пикселях.

        Returns:
            updated_zones (dict): Словарь формата:
                - ключ: xyz координаты ROI области.
                - значение: кол-во очков (our_score, enemy_score).
        """
        MEMORY_LIMIT = 30   # Буфер для памяти маркеров
        DIST_THRESHOLD = 10 # Разрешающая способность для различения маркеров

        if team_color == 'yellow':
            target_id = 47
            opponent_id = 36
        else:  # 'blue'
            target_id = 36
            opponent_id = 47

        rvec, _ = cv2.Rodrigues(self.tmatrix_field)
        tvec = self.tvec_cam_to_field

        updated_zones = {}

        for center_3d_key in zones_3d_dict:
            x, y, z = center_3d_key
            
            points_to_check = [(x, y, z)]
            if abs(x) > 0.0:
                points_to_check.append((-x, y, z))
            
            for pt_3d in points_to_check:
                pt_3d_np = np.array([pt_3d], dtype=np.float64)
                img_pts, _ = cv2.projectPoints(
                    pt_3d_np, rvec, tvec, self.camera_matrix, self.dist_coefs
                )
                center_2d = tuple(img_pts[0][0].astype(int))
                
                roi, coords = self.get_roi(image, center_2d, roi_size)
                x_start, y_start, x_end, y_end = coords

                corners, ids, rejected = self.detector.detectMarkers(roi)
                
                current_detections = []
                
                if ids is not None:
                    ids_flat = ids.flatten()
                    for i, mid in enumerate(ids_flat):
                        c = corners[i][0]
                        cx = int(np.mean(c[:, 0])) + x_start
                        cy = int(np.mean(c[:, 1])) + y_start
                        current_detections.append({'id': mid, 'center': (cx, cy)})

                if pt_3d not in self.zone_states:
                    self.zone_states[pt_3d] = []
                
                tracked_markers = self.zone_states[pt_3d]
                
                # Помечаем все сохраненные маркеры как "не найденные в этом кадре" пока что
                for tm in tracked_markers:
                    tm['updated_this_frame'] = False

                # Пытаемся сопоставить новые детекции с сохраненными
                for det in current_detections:
                    matched = False
                    # Ищем ближайший сохраненный маркер с таким же ID
                    best_dist = float('inf')
                    best_idx = -1
                    
                    for i, tm in enumerate(tracked_markers):
                        if tm['id'] == det['id']:
                            # Считаем расстояние между центрами
                            dist = np.linalg.norm(np.array(det['center']) - np.array(tm['center']))
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = i
                    
                    # Если нашли близкий маркер того же типа
                    if best_idx != -1 and best_dist < DIST_THRESHOLD:
                        # Обновляем его позицию и сбрасываем счетчик потери
                        tracked_markers[best_idx]['center'] = det['center']
                        tracked_markers[best_idx]['lost_frames'] = 0
                        tracked_markers[best_idx]['updated_this_frame'] = True
                        matched = True
                    
                    # Если совпадений нет - это новый маркер, добавляем в память
                    if not matched:
                        tracked_markers.append({
                            'id': det['id'],
                            'center': det['center'],
                            'lost_frames': 0,
                            'updated_this_frame': True
                        })

                # Чистка памяти: удаляем те, которые долго не видели, обновляем счетчики
                # Используем list comprehension для фильтрации
                new_tracked_list = []
                for tm in tracked_markers:
                    if not tm.get('updated_this_frame', False):
                        tm['lost_frames'] += 1
                    
                    # Оставляем только те, что не превысили лимит памяти
                    if tm['lost_frames'] < MEMORY_LIMIT:
                        new_tracked_list.append(tm)
                
                self.zone_states[pt_3d] = new_tracked_list

                my_count = 0
                opponent_count = 0
                
                for tm in new_tracked_list:
                    mid = tm['id']
                    if mid == target_id:
                        my_count += 1
                    elif mid == opponent_id:
                        opponent_count += 1
                
                our_score = my_count * 3
                enemy_score = opponent_count * 3
                
                if my_count > opponent_count:
                    our_score += 5
                if opponent_count > my_count:
                    enemy_score += 5

                updated_zones[pt_3d] = [our_score, enemy_score]

        return updated_zones

    def pantry_checker_dominance(self, team_color, image, zones_3d_dict, roi_size=200):
        """
        Осматривает кладовые-ROI и определяет доминирование.

        Args:
            team_color (str): 'yellow' или 'blue'.
            image (np.ndarray): исходное изображение.
            zones_3d_dict (list): список xyz координат центров зон.
            roi_size (int): размер ROI в пикселях.

        Returns:
            updated_zones (dict): Словарь формата:
                - ключ: xyz координаты ROI области.
                - значение: 0 (наше доминирование), 1 (доминирование противника), -1 (ничья/пусто).
        """
        MEMORY_LIMIT = 30   # Буфер для памяти маркеров
        DIST_THRESHOLD = 10 # Разрешающая способность для различения маркеров

        if team_color == 'yellow':
            target_id = 47
            opponent_id = 36
        else:  # 'blue'
            target_id = 36
            opponent_id = 47

        rvec, _ = cv2.Rodrigues(self.tmatrix_field)
        tvec = self.tvec_cam_to_field

        updated_zones = {}

        for center_3d_key in zones_3d_dict:
            x, y, z = center_3d_key
            
            points_to_check = [(x, y, z)]
            if abs(x) > 0.0:
                points_to_check.append((-x, y, z))
            
            for pt_3d in points_to_check:
                pt_3d_np = np.array([pt_3d], dtype=np.float64)
                img_pts, _ = cv2.projectPoints(
                    pt_3d_np, rvec, tvec, self.camera_matrix, self.dist_coefs
                )
                center_2d = tuple(img_pts[0][0].astype(int))
                
                roi, coords = self.get_roi(image, center_2d, roi_size)
                x_start, y_start, x_end, y_end = coords

                corners, ids, rejected = self.detector.detectMarkers(roi)
                
                current_detections = []

                if ids is not None:
                    ids_flat = ids.flatten()
                    for i, mid in enumerate(ids_flat):
                        c = corners[i][0]
                        cx = int(np.mean(c[:, 0])) + x_start
                        cy = int(np.mean(c[:, 1])) + y_start
                        current_detections.append({'id': mid, 'center': (cx, cy)})

                if pt_3d not in self.zone_states:
                    self.zone_states[pt_3d] = []
                
                tracked_markers = self.zone_states[pt_3d]
                
                for tm in tracked_markers:
                    tm['updated_this_frame'] = False

                for det in current_detections:
                    matched = False
                    best_dist = float('inf')
                    best_idx = -1
                    
                    for i, tm in enumerate(tracked_markers):
                        if tm['id'] == det['id']:
                            dist = np.linalg.norm(np.array(det['center']) - np.array(tm['center']))
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = i
                    
                    if best_idx != -1 and best_dist < DIST_THRESHOLD:
                        tracked_markers[best_idx]['center'] = det['center']
                        tracked_markers[best_idx]['lost_frames'] = 0
                        tracked_markers[best_idx]['updated_this_frame'] = True
                        matched = True
                    
                    if not matched:
                        tracked_markers.append({
                            'id': det['id'],
                            'center': det['center'],
                            'lost_frames': 0,
                            'updated_this_frame': True
                        })

                new_tracked_list = []
                for tm in tracked_markers:
                    if not tm.get('updated_this_frame', False):
                        tm['lost_frames'] += 1
                    
                    if tm['lost_frames'] < MEMORY_LIMIT:
                        new_tracked_list.append(tm)
                
                self.zone_states[pt_3d] = new_tracked_list

                my_count = 0
                opponent_count = 0
                
                for tm in new_tracked_list:
                    mid = tm['id']
                    if mid == target_id:
                        my_count += 1
                    elif mid == opponent_id:
                        opponent_count += 1
                
                if my_count > opponent_count:
                    dominance = 0
                elif opponent_count > my_count:
                    dominance = 1
                else:
                    dominance = -1
                
                updated_zones[pt_3d] = dominance

        return updated_zones



class PoseFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        Implementation of the one_euro filter for robot pose.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def one_euro(self, xyz, xyzw, t=None):
        current_x = np.concatenate((xyz, xyzw))
        
        if t is None:
            t = time.time()

        if self.x_prev is None:
            self.x_prev = current_x
            self.dx_prev = np.zeros_like(current_x)
            self.t_prev = t
            return current_x[:3], current_x[3:]

        dt = t - self.t_prev
        if dt <= 0: return self.x_prev[:3], self.x_prev[3:]

        dot = np.dot(self.x_prev[3:], current_x[3:])
        if dot < 0:
            current_x[3:] *= -1

        a_d = self._smoothing_factor(dt, self.d_cutoff)
        dx = (current_x - self.x_prev) / dt
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        a = self._smoothing_factor(dt, cutoff)
        x_hat = self._exponential_smoothing(a, current_x, self.x_prev)

        q_len = np.linalg.norm(x_hat[3:])
        if q_len > 0:
            x_hat[3:] /= q_len

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat[:3], x_hat[3:]

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev



class HarvesterCamera:
    """Класс-обертка для Harvester, имитирующий интерфейс cv2.VideoCapture"""
    def __init__(self, stream_params_path, cti_path):
        self.h = Harvester()
        self.ia = None
        self.running = False

        # self.gamma_val = 0.5
        # self.lut = self.create_gamma_lut(self.gamma_val)
        
        # Загружаем параметры из YAML
        with open(stream_params_path, 'r') as f:
            camera_config = yaml.safe_load(f)
        params = camera_config['camera_stream_params']
        
        # Сохраняем параметры для использования в .read()
        self.width = params['resolution'][0]
        self.height = params['resolution'][1]
        
        # Инициализируем Harvester
        self.h.add_file(cti_path)
        self.h.update()

        if not self.h.device_info_list:
            self.h.reset()
            raise RuntimeError("Camera not found. Check your connection and access to MvProducerU3V.cti")
        
        for i, device in enumerate(self.h.device_info_list):
            print(f"Device {i}: {device}")
        
        try:
            # Создаем и настраиваем камеру
            self.ia = self.h.create(0)
            self.ia.device.node_map.Width.value = params['resolution'][0]
            self.ia.device.node_map.Height.value = params['resolution'][1]
            self.ia.device.node_map.PixelFormat.value = params['pixel_format']
            self.ia.device.node_map.AcquisitionFrameRateEnable.value = True
            self.ia.device.node_map.AcquisitionFrameRate.value = params['fps']
            
            self.ia.remote_device.node_map.ExposureAuto.value = 'Continuous'
            self.ia.remote_device.node_map.AutoExposureTimeLowerLimit.value = params['auto_exposure'][0]
            self.ia.remote_device.node_map.AutoExposureTimeUpperLimit.value = params['auto_exposure'][1]
            
            self.ia.remote_device.node_map.GainAuto.value = 'Continuous'
            self.ia.remote_device.node_map.AutoGainLowerLimit.value = params['auto_gain'][0]
            self.ia.remote_device.node_map.AutoGainUpperLimit.value = params['auto_gain'][1]
            
            self.ia.remote_device.node_map.BlackLevelEnable.value = True
            self.ia.remote_device.node_map.BlackLevel.value = params['black_level']

            self.ia.remote_device.node_map.GammaEnable.value = True
            self.ia.remote_device.node_map.Gamma.value = params['gamma']
            
            # Запускаем захват
            self.ia.start()
            self.running = True

        except Exception as e:
            self.release()
            raise e
        
    def create_gamma_lut(self, gamma):
        lut = np.arange(256, dtype=np.float16)
        lut = ((lut / 255.0) ** gamma) * 255.0
        return lut.astype("uint8")

    def harvester_read(self):
        """
        Захватывает, обрабатывает и возвращает один кадр.
        Возвращает (ret, frame), как в cv2.VideoCapture.
        """
        if not self.running:
            return False, None

        # try:
        with self.ia.fetch(timeout=2.0) as buffer:
            if buffer.payload.components:
                # Извлечение данных
                img = buffer.payload.components[0].data
                img = img.reshape(self.height, self.width)

                # frame = cv2.LUT(img, self.lut)
                # frame = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_Y422)
                frame = img.copy()

                return True, frame
            else:
                return False, None
        # except Exception as e:
        #     print(f"Error when reading the frame: {e}")
        #     return False, None

    def get_raw_fps(self):
        """Вспомогательный метод для получения текущего FPS с камеры"""
        if self.running:
            return self.ia.remote_device.node_map.ResultingFrameRate.value
        return 0

    def release(self):
        """Останавливает поток и освобождает все ресурсы"""
        if self.running and self.ia:
            self.ia.stop()
            self.running = False
        if self.ia:
            self.ia.destroy()
        if self.h:
            self.h.reset()
        self.ia = None
        self.h = None
        print("The camera is closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
