import unittest
from src import camera
from src.hybrid_math import norm_l2  
from numpy.testing import assert_almost_equal

class TestCameraClass(unittest.TestCase):
    def test_reprojection_error(self):
        # https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/EyefulTower/apartment/index.html
        test_cam = camera.PinholeCamera.from_K(
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
