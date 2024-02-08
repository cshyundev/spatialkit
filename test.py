from src.rotation import *
from src.hybrid_operations import *
from src.hybrid_math import *
from src.file_utils import *
from src.camera import PinholeCamera, Camera
from src.geometry import *
from src.pose import interpolate_pose, Pose
import numpy as np
import absl.app as app
from cv2 import undistortPoints
from numpy.testing import assert_almost_equal
from src.camera import PinholeCamera

def main(unused_args):
    
    test_cam = PinholeCamera.from_K(
    K = [[2960.5102118128075/2., 0., 2874.4394903402335/2.],
         [0.,2960.5102118128075/2.,  4314.472094388692/2.],
         [0., 0., 1.]],
    image_size = [5784//2,8660//2],
    dist=[
        -0.03823023528744702,
        0.011062009972943147,
        -0.0004975289812637742,
        0.0002160237193391678,
        -0.0009125422794027956
      ]
    )

    uv = test_cam.make_pixel_grid()
    undist_uv = test_cam.undistort_pixel(uv,False)
    dist_uv = test_cam.distort_pixel(undist_uv,False)
    # res = norm_l2(uv-dist_uv,dim=0,keepdim=False)
    res = norm_l2(uv-dist_uv,dim=0,keepdim=False)

    print(f"max error: {res.max()}")
    print(f"min error: {res.min()}")
    print(f"mean error:{res.mean()}")

    # compare to opencv function
    



    return None

if __name__ == "__main__":
    app.run(main)