import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from harvesters.core import Harvester
from typing import Tuple, Deque, List


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
    new_tvec: List[float], 
    history_deque: Deque[List[float]]
) -> Tuple[tuple, tuple]:
    """
    Рассчитывает скользящее среднее и СКО для выборки.

    Эта функция МОДИФИЦИРУЕТ 'history_deque', переданный ей.

    Args:
        new_tvec (List[float]): Новый вектор [x, y, z].
        history_deque (Deque): Deque (из collections) с maxlen=10,
                               хранящий историю векторов.

    Returns:
        Tuple[tuple, tuple]: Кортеж из двух кортежей:
        ((mean_x, mean_y, mean_z), (std_x, std_y, std_z))
    """
    
    # Добавляем новый вектор в историю
    if len(new_tvec) == 3:
        history_deque.append(new_tvec)
    
    # Конвертируем историю (deque из списков) в NumPy массив
    data = np.array(history_deque)
    
    n = len(history_deque)
    
    if n == 0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    # Рассчитываем среднее по столбцам (axis=0)
    means = np.mean(data, axis=0)
    
    # Рассчитываем СКО
    if n < 2:
        # Нельзя рассчитать СКО для 1 элемента (деление на n-1 будет делением на 0)
        stds = np.array([0.0, 0.0, 0.0])
    else:
        # ddof=1 использует N-1 (СКО для *выборки*), а не N
        stds = np.std(data, axis=0, ddof=1)
        
    return tuple(means), tuple(stds)

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
    def __init__(self, config_path: str, team: str, roi_size: int = 1000):
        # Инициализация параметров из конфига
        self.config = read_config(config_path)
        self.camera_matrix = np.array(self.config['camera_stream_params']['radial_tangential']['intrinsics'], dtype=np.float64)
        self.dist_coefs = np.array(self.config['camera_stream_params']['radial_tangential']['distortion_coeffs'], dtype=np.float64)

        # Настройки детектора ArUco
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

        # Параметры для отслеживания в ROI
        self.roi_size = roi_size
        self.prev_center = None
        self.current_target_aruco = None

        # Параметры, зависящие от команды
        if team == "yellow":
            self.colour_range = range(6, 11)
            self.roi_center_aruco_list = [6, 74, 75, 76, 77]
            self.RotSideDict = {
                74: R.from_euler('y', 90, degrees=True).as_matrix(),
                75: R.from_euler('x', 90, degrees=True).as_matrix(),
                76: R.from_euler('y', -90, degrees=True).as_matrix(),
                77: R.from_euler('x', -90, degrees=True).as_matrix()
            }
            self.TvecSideDict = {
                74: np.array([-0.05, 0.0, 0.055]),
                75: np.array([0.0, 0.05, 0.055]),
                76: np.array([0.05, 0.0, 0.055]),
                77: np.array([0.0, -0.05, 0.055])
            }
        else: # team == "blue"
            self.colour_range = range(1, 6)
            self.roi_center_aruco_list = [2, 55, 56, 57, 58]
            self.RotSideDict = {
                55: R.from_euler('y', 90, degrees=True).as_matrix(),
                56: R.from_euler('x', 90, degrees=True).as_matrix(),
                57: R.from_euler('y', -90, degrees=True).as_matrix(),
                58: R.from_euler('x', -90, degrees=True).as_matrix()
            }
            self.TvecSideDict = {
                55: np.array([-0.05, 0.0, 0.055]),
                56: np.array([0.0, 0.05, 0.055]),
                57: np.array([0.05, 0.0, 0.055]),
                58: np.array([0.0, -0.05, 0.055])
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

        # Транформы поле2робот
        self.robot_tvec = None
        self.robot_rot_matrix = None

    def initialize_field_pose(self, cap, num_frames=10):
        """Вычисляет и сохраняет усредненную матрицу перехода от камеры к полю"""
        if not self.pose_initialized:
            print(f"Starting field pose initialization from {num_frames} frames...")
            tmatrices = []
            tvecs = []
            
            while len(tmatrices) < num_frames:

                _, img = cap.harvester_read()

                corners, ids, _ = self.detector.detectMarkers(img)
                if ids is not None:
                    ids = list(map(lambda x: x[0], ids))

                if not ids or not any(marker_id in self.field_markers for marker_id in ids):
                    continue

                field_corners, field_ids = [], []

                for i, marker_id in enumerate(ids):
                    if marker_id in self.field_markers:
                        field_corners.append(corners[i][0])
                        field_ids.append(marker_id)
                
                # Проверка полноты детекции маркеров поля
                if len(field_ids) < 4:
                    print(f"Detected {len(field_ids)} markers out of 4")
                    continue

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
                    tmatrices.append(cv2.Rodrigues(rvec)[0])
                    tvecs.append(tvec.flatten())
                    print(f"Frame {len(tmatrices)}/{num_frames} successfully processed")

            # Усреднение векторов трансляции
            self.tvec_cam_to_field = np.mean(tvecs, axis=0)

            # Получаем кватернионы [x, y, z, w]
            quats = R.from_matrix(tmatrices).as_quat()
            # Выравниваем кватернионы: если w < 0, инвертируем весь кватернион (q -> -q), чтобы получить то же вращение, но с w > 0.
            quats[quats[:, 3] < 0] *= -1
            # Теперь усредняем
            avg_quat = np.mean(quats, axis=0)
            # Нормализуем результат
            avg_quat /= np.linalg.norm(avg_quat)
            # Конвертируем обратно в матрицу
            self.tmatrix_field = R.from_quat(avg_quat).as_matrix()

            self.pose_initialized = True
            print("Field pose initialization successful!")
            print(f"Constant T-Matrix:\n{self.tmatrix_field}")
            print(f"Constant T-Vec: {self.tvec_cam_to_field}")

        return True

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
            cov = np.zeros((6, 6), dtype=np.float64)

        return self.robot_tvec, quat, cov

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
            tvec, quat, cov = self.estimate_robot_pose(ids, corners, cov_flag=True)
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



class HarvesterCamera:
    """Класс-обертка для Harvester, имитирующий интерфейс cv2.VideoCapture"""
    def __init__(self, stream_params_path, cti_path):
        self.h = Harvester()
        self.ia = None
        self.running = False

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
            
            # Запускаем захват
            self.ia.start()
            self.running = True

        except Exception as e:
            self.release()
            raise e

    def harvester_read(self):
        """
        Захватывает, обрабатывает и возвращает один кадр.
        Возвращает (ret, frame), как в cv2.VideoCapture.
        """
        if not self.running:
            return False, None

        try:
            with self.ia.fetch(timeout=2.0) as buffer:
                if buffer.payload.components:
                    # Извлечение данных
                    img = buffer.payload.components[0].data
                    img = img.reshape(self.height, self.width)

                    frame = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR_EA)
                    frame = gamma_correction(frame)
                    
                    return True, frame
                else:
                    return False, None
        except Exception as e:
            print(f"Error when reading the frame: {e}")
            return False, None

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
