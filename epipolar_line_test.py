import os.path as osp
import numpy as np
from src.pose import Pose
from src.camera import *
from src.geometry import MultiView
from src.parser import parse_monosdf_dataset
from src.plot import show_image
from absl import app

def main(unused_args):
    """
    Epipolar Line Test to Validate That a Pose is Correct

    Currently available camera models
        1. Pinhole Camera (no distortion)

    To conduct the test, you need:
        1. A pair of images.
        2. Intrinsic Matrix (Assume the images were taken from the same camera)
        3. Poses for each image (Camera to World).

    Test Process
        1. Set image path and the necessary information.
        2. Select 3 points.
        3. Show the result.
    """   

    ###################################################
    # Fill the value below!!!
    
    # Set Parameters
    # 1.base folder and image name
    base_folder_path:str = ""
    left_image_name:str  = "" 
    right_image_name:str = ""

    # 2. Intrinsic Matrix Format
    # [[fx,skew, cx],
    # [0., fy,   cy],
    # [0., 0.,   1.]]

    image_size = [0,0] # width and height
    K = [[0., 0., 0.],
        [0., 0.,  0.],
        [0., 0.,  1.]]
    dist = [0., 0., 0., 0., 0.] # distortion paremeter k1,k2,p1,p2,k3


    # 3. poses for each images
    # Format 4*3 matrix
    #  [[r11, r12, r13, tx],
    #  [r21, r22, r23, ty],
    #  [r31, r32, r33, tz]]

    left_pose = [[0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.]]

    right_pose = [[0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.]]

    ###################################################
    # Do not Modify Codes!!!
    image_path = [osp.join(base_folder_path,left_image_name), 
                  osp.join(base_folder_path,right_image_name)]  

    cameras = [PinholeCamera.from_K(K,image_size,dist),
               PinholeCamera.from_K(K,image_size,dist)]

    poses = [Pose.from_mat(np.array(left_pose,dtype=np.float32)),
             Pose.from_mat(np.array(right_pose,dtype=np.float32))]

    multiview = MultiView(image_path,cameras, poses)    
    ###################################################
    # Select 4 points. You can change the number of points.
    n_pts = 4
    left_pt2d = multiview.choice_points(0,1,n_pts)
    image = multiview.draw_epipolar_line(idx1=0,idx2=1,left_pt2d=left_pt2d)
    show_image(image)
 

if __name__ == "__main__":
    app.run(main)