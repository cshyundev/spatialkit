import cv2
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import os.path as osp
from cv_utils import file_utils as fu

def calibrate_camera(images, pattern_size, square_size, fisheye=False):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    objpoints = []
    imgpoints = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    print(f"{len(objpoints)}/{len(images)} images Detected.")
    if len(objpoints) < 10:
        print("Not enough frames captured for calibration. Please try again.")
        return None

    objpoints = [np.array(pts, dtype=np.float32).reshape(-1, 1, 3) for pts in objpoints]
    imgpoints = [np.array(pts, dtype=np.float32).reshape(-1, 1, 2) for pts in imgpoints]

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

def calibrate_stereo_cameras(image_pairs, pattern_size, square_size, fisheye=False):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    for left_image, right_image in image_pairs:
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

        if ret_left and ret_right:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

    print(f"{len(objpoints)}/{len(image_pairs)} pairs Detected.")
    if len(objpoints) < 10:
        print("Not enough pairs captured for calibration. Please try again.")
        return None

    objpoints = [np.array(pts, dtype=np.float32).reshape(-1, 1, 3) for pts in objpoints]
    imgpoints_left = [np.array(pts, dtype=np.float32).reshape(-1, 1, 2) for pts in imgpoints_left]
    imgpoints_right = [np.array(pts, dtype=np.float32).reshape(-1, 1, 2) for pts in imgpoints_right]

    if fisheye:
        K_left = np.zeros((3, 3))
        D_left = np.zeros((4, 1))
        K_right = np.zeros((3, 3))
        D_right = np.zeros((4, 1))
        rms, _, _, _ = cv2.fisheye.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right, K_left, D_left, K_right, D_right, gray_left.shape[::-1],
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        )
        return K_left, D_left, K_right, D_right, rms
    else:
        ret, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right, None, None, None, None, gray_left.shape[::-1]
        )
        return K_left, D_left, K_right, D_right, R, T

def calibrate_from_images(image_dir_left, image_dir_right, pattern_size, square_size, fisheye):
    images_left = fu.read_all_images(image_dir_left)
    images_right = fu.read_all_images(image_dir_right)
    if images_left is None or images_right is None or len(images_left) != len(images_right):
        print("Reading Images Failed or unequal number of images.")
        return
    print("Stereo Calibration start.")
    image_pairs = list(zip(images_left, images_right))
    res = calibrate_stereo_cameras(image_pairs, pattern_size, square_size, fisheye)
    
    if res is not None:
        if fisheye:
            K_left, D_left, K_right, D_right, rms = res
            print("Stereo Calibration complete.")
            print("Left Camera matrix (K_left):", K_left)
            print("Left Distortion coefficients (D_left):", D_left)
            print("Right Camera matrix (K_right):", K_right)
            print("Right Distortion coefficients (D_right):", D_right)
            print("Reprojection error:", rms)
        else:
            K_left, D_left, K_right, D_right, R, T = res
            print("Stereo Calibration complete.")
            print("Left Camera matrix (K_left):", K_left)
            print("Left Distortion coefficients (D_left):", D_left)
            print("Right Camera matrix (K_right):", K_right)
            print("Right Distortion coefficients (D_right):", D_right)
            print("Rotation matrix (R):", R)
            print("Translation vector (T):", T)

def calculate_centroid(corners):
    x_mean = np.mean(corners[:, 0, 0])
    y_mean = np.mean(corners[:, 0, 1])
    return x_mean, y_mean

def calculate_rotation(corners, pattern_size):
    v1 = corners[0, 0] - corners[pattern_size[0] - 1, 0]
    angle = np.arctan2(v1[1], v1[0])
    return angle

def calculate_distance(corners, pattern_size):
    return np.linalg.norm(corners[0, 0] - corners[pattern_size[0] - 1, 0])

def is_diverse(corners, centroids, angles, distances, pattern_size, threshold=0.1):
    new_centroid = calculate_centroid(corners)
    new_angle = calculate_rotation(corners, pattern_size)
    new_distance = calculate_distance(corners, pattern_size)

    for i in range(len(centroids)):
        if (np.linalg.norm(np.array(new_centroid) - np.array(centroids[i])) < threshold and
            abs(new_angle - angles[i]) < np.radians(10) and
            abs(new_distance - distances[i]) < threshold):
            return False
    return True

def start_capture(camera_index_left, camera_index_right, pattern_size, square_size, fisheye):
    cap_left = cv2.VideoCapture(camera_index_left)
    cap_right = cv2.VideoCapture(camera_index_right)
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    centroids_left = []
    angles_left = []
    distances_left = []
    centroids_right = []
    angles_right = []
    distances_right = []

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    captured_frames = 0
    start_time = time.time()

    plt.ion()
    fig, axs = plt.subplots(1, 2)
    text_info = fig.text(0.02, 0.95, '', transform=fig.transAxes)
    for ax in axs:
        ax.set_axis_off()

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        if not ret_left or not ret_right:
            break

        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

        if ret_left and ret_right:
            current_time = time.time()
            if current_time - start_time > 1:  # Capture every second
                if (is_diverse(corners_left, centroids_left, angles_left, distances_left, pattern_size) and
                    is_diverse(corners_right, centroids_right, angles_right, distances_right, pattern_size)):
                    
                    imgpoints_left.append(corners_left)
                    imgpoints_right.append(corners_right)
                    objpoints.append(objp)
                    centroids_left.append(calculate_centroid(corners_left))
                    angles_left.append(calculate_rotation(corners_left, pattern_size))
                    distances_left.append(calculate_distance(corners_left, pattern_size))
                    centroids_right.append(calculate_centroid(corners_right))
                    angles_right.append(calculate_rotation(corners_right, pattern_size))
                    distances_right.append(calculate_distance(corners_right, pattern_size))
                    captured_frames += 1

                    # text_info.set_text(f'Captured Frames: {captured_frames}\n',
                    #                    f'Left Centroids: {len(set(map(tuple, centroids_left)))}\n',
                    #                    f'Right Centroids: {len(set(map(tuple, centroids_right)))}\n'
                    #                    f'Left Angles: {len(set(angles_left))}\n'
                    #                    f'Right Angles: {len(set(angles_right)))}\n'
                    #                    f'Left Distances: {len(set(distances_left))}\n'
                    #                    f'Right Distances: {len(set(distances_right)))}')
                    plt.draw()

                start_time = current_time
                frame_left = cv2.drawChessboardCorners(frame_left, pattern_size, corners_left, ret_left)
                frame_right = cv2.drawChessboardCorners(frame_right, pattern_size, corners_right, ret_right)
            
            axs[0].imshow(cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
            axs[1].imshow(cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB))
            plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    plt.ioff()
    plt.show()

    if len(objpoints) < 10:
        print("Not enough frames captured for calibration. Please try again.")
        return

    if fisheye:
        K_left, D_left, K_right, D_right, rms = fisheye_calibrate(objpoints, imgpoints_left, imgpoints_right, gray_left.shape[::-1])
        print("Calibration complete.")
        print("Left Camera matrix (K_left):", K_left)
        print("Left Distortion coefficients (D_left):", D_left)
        print("Right Camera matrix (K_right):", K_right)
        print("Right Distortion coefficients (D_right):", D_right)
        print("Reprojection error:", rms)
    else:
        ret, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right, None, None, None, None, gray_left.shape[::-1]
        )
        print("Calibration complete.")
        print("Left Camera matrix (K_left):", K_left)
        print("Left Distortion coefficients (D_left):", D_left)
        print("Right Camera matrix (K_right):", K_right)
        print("Right Distortion coefficients (D_right):", D_right)
        print("Rotation matrix (R):", R)
        print("Translation vector (T):", T)

def fisheye_calibrate(objpoints, imgpoints_left, imgpoints_right, image_shape):
    K_left = np.zeros((3, 3))
    D_left = np.zeros((4, 1))
    K_right = np.zeros((3, 3))
    D_right = np.zeros((4, 1))
    rms, _, _, _ = cv2.fisheye.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, K_left, D_left, K_right, D_right, image_shape,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    )
    return K_left, D_left, K_right, D_right, rms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration')
    parser.add_argument('--pattern_size', type=int, nargs=2, required=True, help='Size of the checkerboard pattern')
    parser.add_argument('--square_size', type=float, required=True, help='Size of a square in your defined unit (meters or millimeters)')
    parser.add_argument('--fisheye', action='store_true', help='Use fisheye model')
    parser.add_argument('--camera_index_left', type=int, required=True, help='Index of the left camera')
    parser.add_argument('--camera_index_right', type=int, required=True, help='Index of the right camera')
    parser.add_argument('--image_dir_left', type=str, help='Directory containing left camera images')
    parser.add_argument('--image_dir_right', type=str, help='Directory containing right camera images')

    args = parser.parse_args()

    pattern_size = tuple(args.pattern_size)
    square_size = args.square_size

    if args.image_dir_left and args.image_dir_right:
        if osp.exists(args.image_dir_left) and osp.exists(args.image_dir_right):
            calibrate_from_images(args.image_dir_left, args.image_dir_right, pattern_size, square_size, args.fisheye)
        else:
            print("Image folders are not found.")
    else:
        start_capture(args.camera_index_left, args.camera_index_right, pattern_size, square_size, args.fisheye)
