import os
import os.path as osp
from cv_utils.core.geometry.camera import *
# from cv_utils.vis.image_utils import show_image
from cv_utils.utils import read_image, write_image
from absl import app
import time


def main(argv):
    """
    Undistort Image

    To conduct the test, you need:
        1. A directory path saved images.
        2. Camera Type(Perspective Camera or Opencv Fisheye Camera)
        3. Intrinsic information and distortion parameters 
    
    Test Process
        1. Set image directory path and the necessary information.
        2. Undistort images
        3. show or save undistorted images
    """   

    ###################################################
    # Fill the value below!!!
    
    # Set Parameters
    # 1.base folder and image name
    dataset_folder_path:str = ""
    output_path = ""
    os.makedirs(output_path,exist_ok=False)
    # 2. Intrinsic Matrix Format
    # [[fx,skew, cx],
    # [0., fy,   cy],
    # [0., 0.,   1.]]

    cam_type = "" # "PRESPECTIVE" or "OPENCV" (fisheye)

    image_size = (0,0) # width and height
    fx,fy = (0.,0.) # fx,fy
    cx,cy = (0.,0.) # cx,cy
    skew: float = 0. # skew

    K = [[fx,skew,cx],
        [0., fy,  cy],
        [0., 0.,  1.]]

    """
    Distortion parameters 
    PERSPECTIVE: k1,k2,p1,p2,k3
    FISHEYE: k1,k2,k3,k4
    """
    dist = []
    ###################################################


    if cam_type == "PERSPECTIVE":
        assert(len(dist) == 5), "Number of distortion parameters must be 5 for perspective camera"
        cam = PerpectiveCamera.from_K(K,image_size,dist)
    elif cam_type == "FISHEYE":
        assert(len(dist) == 4), "Number of distortion parameters must be 4 for fisheye camera"
        cam = OpenCVCamera.from_K_D(K,image_size,dist)
    else:
        print("Unkown Camera Type.")
        return 

    if osp.exists(dataset_folder_path) is False:
        print(f"Unkown directory path: {dataset_folder_path}")
        return
    
    image_list = os.listdir(dataset_folder_path)
    total_images = len(image_list)
    
    start_time = time.time()
    last_report_time = start_time

    for i, image_name in enumerate(image_list):
        image = read_image(osp.join(dataset_folder_path,image_name))
        undistort_image = cam.undistort_image(image)
        write_image(undistort_image, osp.join(output_path,image_name))
        
        current_time = time.time()
        if current_time - last_report_time >= 10:  # Update every 10 seconds
            elapsed_time = current_time - start_time
            remaining_time = elapsed_time / (i + 1) * (total_images - i - 1)
            print(f"Processed {i + 1}/{total_images} images.")
            print(f"Estimated Time Remaining: {remaining_time:.2f} seconds")
            last_report_time = current_time
    return

if __name__ == "__main__":
    app.run(main)