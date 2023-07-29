from glob import glob  
from typing import Dict, Any
import os.path as osp
import numpy as np

def parse_monosdf_dataset(dataset_path:str, cam_type:str):
    """
    Parse  monosdf dataset to dictionary
    """

    dataset_dict = {}

    dataset_dict["root"] = dataset_path

    def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
    image_paths = glob_data(osp.join('{0}'.format(dataset_dict), "*_rgb.png"))
    depth_paths = glob_data(osp.join('{0}'.format(dataset_dict), "*_depth.npy"))
    normal_paths = glob_data(osp.join('{0}'.format(dataset_dict), "*_normal.npy"))

    n_frames = len(image_paths)
    camera_dict = np.load(osp.join(dataset_path,"cameras.npz"))
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_frames)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_frames)]

    for frame_id in range(n_frames):
        world_mat = world_mats[0]
        scale_mat = scale_mats[0]
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
        self.pose_all.append(torch.from_numpy(pose).float())


    
    return dataset_dict