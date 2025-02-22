"""
Module Name: parser.py

Description:
This module provides functions for parsing multi-view stereo (MVS) datasets.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

License: MIT LICENSE
"""

import re
from glob import glob

# from typing import Dict, Any
import os.path as osp
import numpy as np
from ...utils.io import *
from ...geom.pose import *

# import cv2 as cv

# def parse_monosdf_dataset(dataset_path:str,
#                           center_crop_type:str="center_crop_for_replica",
#                           cam_type:str="PERSPECTIVE"):
#     """
#     Parse  monosdf dataset to dictionary
#     Adapted https://github.com/autonomousvision/monosdf/blob/main/code/datasets/scene_dataset.py
#     """
#     dataset_dict = {}

#     dataset_dict["root"] = dataset_path
#     def glob_data(data_dir):
#             data_paths = []
#             data_paths.extend(glob(data_dir))
#             data_paths = sorted(data_paths)
#             return data_paths

#     image_paths = glob_data(osp.join('{0}'.format(dataset_path), "*_rgb.png"))

#     n_frames = len(image_paths)
#     camera_dict = np.load(osp.join(dataset_path,"cameras.npz"))
#     scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_frames)]
#     world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_frames)]

#     frames = []

#     for frame_id in range(n_frames):
#         world_mat = world_mats[frame_id]
#         scale_mat = scale_mats[frame_id]
#         projection_mat = world_mat @ scale_mat

#         out = cv.decomposeProjectionMatrix(projection_mat[:3,:4])
#         K = out[0]
#         R = out[1]
#         t = out[2]
#         K = K/K[2,2]
#         pose = np.eye(4, dtype=np.float32)
#         pose[:3, :3] = R.transpose()
#         pose[:3,3] = (t[:3] / t[3])[:,0]

#         if center_crop_type == 'center_crop_for_replica':
#             scale = 384 / 680
#             offset = (1200 - 680 ) * 0.5
#             K[0, 2] -= offset
#             K[:2, :] *= scale
#         elif center_crop_type == 'center_crop_for_tnt':
#             scale = 384 / 540
#             offset = (960 - 540) * 0.5
#             K[0, 2] -= offset
#             K[:2, :] *= scale
#         elif center_crop_type == 'center_crop_for_dtu':
#             scale = 384 / 1200
#             offset = (1600 - 1200) * 0.5
#             K[0, 2] -= offset
#             K[:2, :] *= scale
#         elif center_crop_type == 'padded_for_dtu':
#             scale = 384 / 1200
#             offset = 0
#             K[0, 2] -= offset
#             K[:2, :] *= scale
#         elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
#             pass
#         else:
#             raise NotImplementedError

#         image_size = read_image(image_paths[frame_id]).shape[:2]
#         K = K.tolist()
#         cam = {
#             "cam_type": cam_type,
#             "image_size": image_size,
#             "focal_length": [K[0][0], K[1][1]],
#             "principal_point": [K[0][2], K[1][2]],
#             "skew": K[0][1],
#             "radial": [0.,0.,0.],
#             "tangential": [0.,0.]
#         }
#         frame = {
#             "frame_id": frame_id,
#             "cam": cam,
#             "camtoworld": pose.tolist(),
#             "timestamp": 0.,
#             "image": f"{frame_id:06d}_rgb.png",
#             "depth": f"{frame_id:06d}_depth.npy",
#             "normal": f"{frame_id:06d}_normal.npy",
#             # "dump_depth": f"{frame_id:06d}_pred_depth.npy" # for debug
#         }
#         frames.append(frame)

#     dataset_dict["frames"] = frames
#     meta_data = read_json(osp.join(dataset_path,"meta_data.json"))
#     near = meta_data["scene_box"]["near"]
#     far = meta_data["scene_box"]["far"]

#     dataset_dict["near"] = near
#     dataset_dict["far"] = far

#     return dataset_dict


def parse_scannetv1_dataset(dataset_path: str):
    """
    Parse Scannet v1 dataset to dictionary.

    Args:
        dataset_path (str): Path to the Scannet v1 dataset.

    Returns:
        dataset (dict): Dictionary containing parsed dataset information, including frames, image size, depth size, and calibration data.

    Details:
    - dataset["frames"]: List of dictionaries, each containing:
        - "id" (int): Frame identifier.
        - "color" (str): Path to the color image file.
        - "depth" (str): Path to the depth image file.
        - "pose" (str): Path to the pose file.
        - "timestamp" (int): Timestamp of the frame (default is 0).
    - dataset["image_size"]: color image size [width, height].
    - dataset["image_calib"]: 3x3 numpy array containing the intrinsic calibration matrix for the color camera.
    - dataset["depth_size"]: depth map size [width, height].
    - dataset["depth_calib"]: 3x3 numpy array containing the intrinsic calibration matrix for the depth camera.
    - dataset["depth_shift"]: the depth shift value to convert depth units (mm -> m).
    - dataset["m_calibrationColorExtrinsic"]: 4x4 numpy array containing the extrinsic calibration matrix for the color camera.
    - dataset["m_calibrationDepthIntrinsic"]: 4x4 numpy array containing the extrinsic calibration matrix for the depth camera.
    """
    dataset = {}

    def glob_data(pattern):
        data_paths = glob(pattern)
        data_paths = sorted(data_paths)
        return data_paths

    image_paths = glob_data(osp.join(dataset_path, "frame-*.color.jpg"))
    depth_paths = glob_data(osp.join(dataset_path, "frame-*.depth.pgm"))
    pose_paths = glob_data(osp.join(dataset_path, "frame-*.pose.txt"))

    if (
        len(image_paths) != len(depth_paths)
        or len(image_paths) != len(pose_paths)
        or len(depth_paths) != len(pose_paths)
    ):
        if len(image_paths) != len(depth_paths):
            LOG_ERROR("The number of images and depths must be same.")
        if len(image_paths) != len(pose_paths):
            LOG_ERROR("The number of images and poses must be same.")
        if len(depth_paths) != len(pose_paths):
            LOG_ERROR("The number of depths and poses must be same.")
        return None

    def parse_txt(file_path):
        data = {}
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                key, value = line.split(" = ")
                value = value.strip()
                # Check if the value is a list of numbers (for matrices)
                if " " in value:
                    if value == "StructureSensor (calibrated)":
                        continue  # unused value
                    numbers = list(map(float, value.split()))
                    if len(numbers) == 16:  # 4x4 matrix
                        value = np.array(numbers).reshape((4, 4))
                    elif len(numbers) == 9:  # 3x3 matrix
                        value = np.array(numbers).reshape((3, 3))
                    else:
                        value = np.array(numbers)
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                data[key] = value
        return data

    def extract_id(file_name: str) -> int:
        pattern = re.compile(r"frame-(\d+)\.color\.jpg")

        # 정규 표현식으로 숫자 추출
        match = pattern.search(file_name)
        return int(match.group(1))

    # parse intrinsic information
    info = parse_txt(osp.join(dataset_path, "_info.txt"))

    frames = []

    for paths in zip(image_paths, depth_paths, pose_paths):
        image_path, depth_path, pose_path = paths
        id = extract_id(image_path)

        frame = {
            "id": id,
            "color": image_path,
            "depth": depth_path,
            "pose": pose_path,
            "timestamp": 0,
        }
        frames.append(frame)

    dataset["frames"] = frames
    dataset["image_size"] = [info["m_colorWidth"], info["m_colorHeight"]]
    dataset["image_calib"] = info["m_calibrationColorIntrinsic"]
    dataset["depth_size"] = [info["m_depthWidth"], info["m_depthHeight"]]
    dataset["depth_calib"] = info["m_calibrationDepthIntrinsic"]
    dataset["depth_shift"] = float(info["m_depthShift"])
    dataset["m_calibrationColorExtrinsic"] = info["m_calibrationColorExtrinsic"]
    dataset["m_calibrationDepthExtrinsic"] = info["m_calibrationDepthExtrinsic"]

    return dataset
