import numpy as np
from cv_utils.utils.io import read_yaml, read_image
from cv_utils.geom import Transform, Rotation
from cv_utils.geom.geom_utils  import compute_fundamental_matrix_from_essential, compute_essential_matrix_from_pose
from cv_utils.geom.rotation import RotType
from cv_utils.geom.camera import PerspectiveCamera, OpenCVFisheyeCamera, RadialCamera
from cv_utils.common.logger import *
from cv_utils.utils.point_selector import DoubleImagesPointSelector
from cv_utils.utils.vis import *
from cv_utils import mvs 
import absl.app as app


def main(unused_args):
    """
    Epipolar Line Test to Validate That a Pose and is Correct

    Currently available camera models
        1. Pinhole Camera (no distortion)

    To conduct the test, you need:
        1. A pair of images.
        2. Intrinsic Matrices
        3. Poses for each image (Camera to World).

    Fill above information into ./images_info.yaml
        
    Test Process
        1. Set image path and the necessary information.
        2. Select several points(default number is 4).
        3. Show the result.
    """

    scannet_datamanager = mvs.ScannetV1Manager("/home/sehyun/datasets/scans/scene0000_00/outputs")

    id1 = 0
    id2 = 150
    num_points = 3

    image1 = scannet_datamanager.get_image(id1) 
    image2 = scannet_datamanager.get_image(id2) 

    cam1:RadialCamera = scannet_datamanager.get_cam(id1)
    cam2:RadialCamera = scannet_datamanager.get_cam(id2)

    tf1:Transform = scannet_datamanager.get_pose(id1)
    tf2:Transform = scannet_datamanager.get_pose(id2)
    tf = tf2.inverse() * tf1  # compute relative transform: cam1 to cam2

    # undistort image
    image1 = cam1.undistort_image(image1)
    image2 = cam2.undistort_image(image2)

    point_selector = DoubleImagesPointSelector(image1, image2, num_points)
    point_selector.connect()
    pts1, pts2 = point_selector.get_points()

    ## draw dots and epipolar lines

    for pt1,pt2 in zip(pts1, pts2):
        rgb = tuple(np.random.randint(0,255,3).tolist())
        E = compute_essential_matrix_from_pose(rel_p=tf)
        F = compute_fundamental_matrix_from_essential(K1=cam1.K,K2=cam2.K,E=E)
        pt_homo = np.array((pt1[0],pt1[1],1.))
        image1 = draw_circle(image1,pt1,1,rgb,2)
        image2 = draw_line_by_line(image2,tuple((F@pt_homo).tolist()),rgb,1)
        image2 = draw_circle(image2,pt2,1,rgb,2)

    show_two_images(image1,image2)

if __name__ == "__main__":
    app.run(main)