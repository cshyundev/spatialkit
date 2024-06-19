import cv2
import numpy as np
import glob
import argparse
import time
import matplotlib.pyplot as plt

def calibrate_camera(images, pattern_size, fisheye=False):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if fisheye:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints, imgpoints, gray.shape[::-1], K, D,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        )
    else:
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return K, D

def calibrate_from_images(image_dir, pattern_size, fisheye):
    images = glob.glob(f'{image_dir}/*.jpg')
    print("Calibration start.")
    K, D = calibrate_camera(images, pattern_size, fisheye)
    print("Calibration complete.")
    print("Camera matrix (K):", K)
    print("Distortion coefficients (D):", D)

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

def start_capture(camera_index, pattern_size, fisheye):
    cap = cv2.VideoCapture(camera_index)
    objpoints = []
    imgpoints = []
    centroids = []
    angles = []
    distances = []

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    captured_frames = 0
    start_time = time.time()

    plt.ion()
    fig, ax = plt.subplots()
    text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.set_axis_off()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            current_time = time.time()
            if current_time - start_time > 1:  # Capture every second
                if is_diverse(corners, centroids, angles, distances):
                    imgpoints.append(corners)
                    objpoints.append(objp)
                    centroids.append(calculate_centroid(corners))
                    angles.append(calculate_rotation(corners))
                    distances.append(calculate_distance(corners))
                    captured_frames += 1

                    text_info.set_text(f'Captured Frames: {captured_frames}\n'
                                       f'Centroids: {len(set(map(tuple, centroids)))}\n'
                                       f'Angles: {len(set(angles))}\n'
                                       f'Distances: {len(set(distances))}')
                    plt.draw()

                start_time = current_time
                frame = cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
            
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    plt.ioff()
    plt.show()

    if len(objpoints) < 10:
        print("Not enough frames captured for calibration. Please try again.")
        return

    if fisheye:
        K, D = fisheye_calibrate(objpoints, imgpoints, gray.shape[::-1])
    else:
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Calibration complete.")
    print("Camera matrix (K):", K)
    print("Distortion coefficients (D):", D)

def fisheye_calibrate(objpoints, imgpoints, image_shape):
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, image_shape, K, D,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    )
    return K, D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mono Camera Calibration')
    parser.add_argument('--pattern_size', type=int, nargs=2, required=True, help='Size of the checkerboard pattern')
    parser.add_argument('--fisheye', action='store_true', help='Use fisheye model')
    parser.add_argument('--camera_index', type=int, default=0, help='Index of the camera')
    parser.add_argument('--image_dir', type=str, help='Directory containing calibration images')

    args = parser.parse_args()

    pattern_size = tuple(args.pattern_size)

    if args.image_dir:
        calibrate_from_images(args.image_dir, pattern_size, args.fisheye)
    else:
        start_capture(args.camera_index, pattern_size, args.fisheye)
