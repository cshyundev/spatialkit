import numpy as np
from module.plot import *
from module.geometry import MultiView
from module.pose import Pose
from module.file_utils import read_json, write_image
import os.path as osp

def main():
    dataset_path = "/home/sehyun/workspace/Replica/scan1"
    meta_dict = read_json(osp.join(dataset_path, "meta_data.json"))
    
    multiviews = MultiView.from_meta_data(dataset_path, meta_dict)
    
    pts2d  = multiviews.choice_points(0,3,1)
    image = multiviews.draw_epipolar_line(0,3,pts2d)
    write_image(image, "./epipolar.png")
    
if __name__ == "__main__":
    main() 