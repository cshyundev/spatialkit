import cv2
import numpy as np
import glob
import argparse
import time
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton

class StereoCalibrationApp(QWidget):
    def __init__(self, camera_index_left, camera_index_right, pattern_size, fisheye):
        super().__init__()
        self.camera_index_left = camera_index_left
        self.camera_index_right = camera_index_right
        self.pattern_size = pattern_size
        self.fisheye = fisheye
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Stereo Camera Calibration')
        self.layout = QVBoxLayout()

        self.info_label = QLabel('Move the checkerboard pattern to different angles and distances.')
        self.layout.addWidget(self.info_label)

        self.captured_label = QLabel('Captured Frames: 0')
        self.layout.addWidget(self.captured_label)

        self.diversity_label = QLabel('Diversity Status:')
        self.layout.addWidget(self.diversity_label)

        self.capture_button = QPushButton('Start Capture')
        self.capture_button.clicked.connect(self.start_capture)
        self.layout.addWidget(self.capture_button)

        self.setLayout(self.layout)

    def start_capture(self):
        cap_left = cv2.VideoCapture(self.camera_index_left)
        cap_right = cv2.VideoCapture(self.camera_index_right)
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        centroids = []
        angles = []
        distances = []

        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)

        captured_frames = 0
        start_time = time.time()

        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            if not ret_left or not ret_right:
                break

            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.pattern_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.pattern_size, None)

            if ret_left and ret_right:
                current_time = time.time()
                if current_time - start_time > 1:  # Capture every second
                    if is_diverse(corners_left, centroids, angles, distances):
                        imgpoints_left.append(corners_left)
                        imgpoints_right.append(corners_right)
                        objpoints.append(objp)
                        centroids.append(calculate_centroid(corners_left))
                        angles.append(calculate_rotation(corners_left))
                        distances.append(calculate_distance(corners_left))
                        captured_frames += 1
                        self.captured_label.setText(f'Captured Frames: {captured_frames}')

                        diversity_status = f'Diversity Status:\nCentroids: {len(set(centroids))}\nAngles: {len(set(angles))}\nDistances: {len(set(distances))}'
                        self.diversity_label.setText(diversity_status)

                    start_time = current_time
                    frame_left = cv2.drawChessboardCorners(frame_left, self.pattern_size, corners_left, ret_left)
                    frame_right = cv2.drawChessboardCorners(frame_right, self.pattern_size, corners_right, ret_right)
            
            combined_frame = np.hstack((frame_left, frame_right))
            cv2.imshow('Stereo Frame', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

        if len(objpoints) < 10:
            self.info_label.setText("Not enough frames captured for calibration. Please try again.")
            return

        if self.fisheye:
            K_left, D_left = self.fisheye_calibrate(objpoints, imgpoints_left, gray_left.shape[::-1])
            K_right, D_right = self.fisheye_calibrate(objpoints, imgpoints_right, gray_right.shape[::-1])
        else:
            K_left, D_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
            K_right, D_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

        R, T, E, F = self.stereo_calibrate(objpoints, imgpoints_left, imgpoints_right, K_left, D_left, K_right, D_right, gray_left.shape[::-1])

        self.info_label.setText("Stereo Calibration complete.")
        print("Left Camera matrix (K_left):", K_left)
        print("Left Distortion coefficients (D_left):", D_left)
        print("Right Camera matrix (K_right):", K_right)
        print("Right Distortion coefficients (D_right):", D_right)
        print("Rotation (R):", R)
        print("Translation (T):", T)

    def fisheye_calibrate(self, objpoints, imgpoints, image_shape):
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints, imgpoints, image_shape, K, D,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        )
        return K, D

    def stereo_calibrate(self, objpoints, imgpoints_left, imgpoints_right, K_left, D_left, K_right, D_right, image_shape):
        ret, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right, K_left, D_left, K_right, D_right, image_shape,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        return R, T, E, F

def calculate_centroid(corners):
    x_mean = np.mean(corners[:, 0, 0])
    y_mean = np.mean(corners[:, 0, 1])
    return x_mean, y_mean

def calculate_rotation(corners):
    v1 = corners[0, 0] - corners[pattern_size[0] - 1, 0]
    angle = np.arctan2(v1[1], v1[0])
    return angle

def calculate_distance(corners):
    return np.linalg.norm(corners[0, 0] - corners[pattern_size[0] - 1, 0])

def is_diverse(corners, centroids, angles, distances, threshold=0.1):
    new_centroid = calculate_centroid(corners)
    new_angle = calculate_rotation(corners)
    new_distance = calculate_distance(corners)

    for i in range(len(centroids)):
        if (np.linalg.norm(np.array(new_centroid) - np.array(centroids[i])) < threshold and
            abs(new_angle - angles[i]) < np.radians(10) and
            abs(new_distance - distances[i]) < threshold):
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration')
    parser.add_argument('--pattern_size', type=int, nargs=2, required=True, help='Size of the checkerboard pattern')
    parser.add_argument('--fisheye', action='store_true', help='Use fisheye model')
    parser.add_argument('--camera_index_left', type=int, default=0, help='Index of the left camera')
    parser.add_argument('--camera_index_right', type=int, default=1, help='Index of the right camera')

    args = parser.parse_args()

    pattern_size = tuple(args.pattern_size)

    app = QtWidgets.QApplication([])
    ex = StereoCalibrationApp(args.camera_index_left, args.camera_index_right, pattern_size, args.fisheye)
    ex.show()
    app.exec_()
