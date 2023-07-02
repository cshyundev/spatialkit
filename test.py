import numpy as np
from module.plot import *
from module.geometry import MultiView
from module.pose import Pose
from module.file_utils import read_json
import os.path as osp

def main():
    dataset_path = "/home/sehyun/workspace/Replica/scan1"
    meta_dict = read_json(osp.join(dataset_path, "meta_data.json"))
    
    multiviews = MultiView.from_meta_data(dataset_path, meta_dict)
    
    pose = multiviews.relative_pose(0,1)
    print(pose.t)
    print(pose.rot_mat())
    

if __name__ == "__main__":
    main() 