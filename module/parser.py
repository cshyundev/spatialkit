from glob import glob  
from typing import Dict, Any
import os.path as osp
import numpy as np
from .file_utils import *
import cv2 as cv

def parse_monosdf_dataset(dataset_path:str,
                          center_crop_type,
                          cam_type:str, margin:float):
    """
    Parse  monosdf dataset to dictionary
    Adapted https://github.com/autonomousvision/monosdf/blob/main/code/datasets/scene_dataset.py
    """
    dataset_dict = {}

    dataset_dict["root"] = dataset_path
    dataset_dict["is_mono"] = True
    dataset_dict["margin"] = margin
    def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
    image_paths = glob_data(osp.join('{0}'.format(dataset_path), "*_rgb.png"))
    # depth_paths = glob_data(osp.join('{0}'.format(dataset_path), "*_depth.npy"))
    # normal_paths = glob_data(osp.join('{0}'.format(dataset_path), "*_normal.npy"))

    n_frames = len(image_paths)
    camera_dict = np.load(osp.join(dataset_path,"cameras.npz"))
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_frames)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_frames)]
    
    frames = []

    for frame_id in range(n_frames):
        world_mat = world_mats[frame_id]
        scale_mat = scale_mats[frame_id]
        projection_mat = world_mat @ scale_mat
        
        out = cv.decomposeProjectionMatrix(projection_mat[:3,:4])
        K = out[0]
        R = out[1]
        t = out[2]
        K = K/K[2,2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3,3] = (t[:3] / t[3])[:,0]

        if center_crop_type == 'center_crop_for_replica':
            scale = 384 / 680
            offset = (1200 - 680 ) * 0.5
            K[0, 2] -= offset
            K[:2, :] *= scale
        elif center_crop_type == 'center_crop_for_tnt':
            scale = 384 / 540
            offset = (960 - 540) * 0.5
            K[0, 2] -= offset
            K[:2, :] *= scale
        elif center_crop_type == 'center_crop_for_dtu':
            scale = 384 / 1200
            offset = (1600 - 1200) * 0.5
            K[0, 2] -= offset
            K[:2, :] *= scale
        elif center_crop_type == 'padded_for_dtu':
            scale = 384 / 1200
            offset = 0
            K[0, 2] -= offset
            K[:2, :] *= scale
        elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
            pass
        else:
            raise NotImplementedError

        image_size = read_image(image_paths[frame_id]).shape[:2]
        K = K.tolist()
        cam = {
            "cam_type": cam_type,
            "image_size": image_size,
            "focal_length": [K[0][0], K[1][1]],
            "principal_point": [K[0][2], K[1][2]],
            "skew": K[0][1],
            "radial": [0.,0.,0.],
            "tangential": [0.,0.]
        }
        frame = {
            "frame_id": frame_id,
            "cam": cam,
            "camtoworld": pose.tolist(),
            "timestamp": 0.,
            "image": f"{frame_id:06d}_rgb.png",
            "depth": f"{frame_id:06d}_depth.npy",
            "normal": f"{frame_id:06d}_normal.npy",
            "dump_depth": f"dump_depth_{frame_id:06d}.npy" # for debug
        }
        frames.append(frame)

    dataset_dict["frames"] = frames

    return dataset_dict

if __name__ == "__main__":
    replica_path = "/home/sehyun/replica/scan1"
    dict = parse_monosdf_dataset(replica_path, "center_crop_for_replica","PINHOLE", 1.)
    write_json("/home/sehyun/nerf-project/dataset/monosdf_replica.json", dict)