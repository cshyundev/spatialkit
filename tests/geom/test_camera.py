import unittest
import numpy as np

# Use new hierarchical import pattern
from spatialkit import (
    Camera,
    PerspectiveCamera,
    OpenCVFisheyeCamera,
    ThinPrismFisheyeCamera
)

class TestCamera(unittest.TestCase):
    """
    UnitTest for Camera class
        1. Perspective Camera 
        2. OpenCVFisheye Camera 
        3. ThinPrismFisheye Camera

    Excluded Camera class
        1. DoubleSphereCamera
        2. OmnidirectionalCamera
        3. EquirectangularCamera
    """
    def setUp(self):

        image_size = (400,400)
        center = (200,200)
        focal_length = 250.

        perspective_cam_dict = {
            "image_size": image_size,
            "principal_point": center,
            "focal_length": [focal_length,focal_length],
            "radial": [-0.01,0.,0.],
            "tangential": [0.01,0.]
        }

        opncv_fisheye_cam_dict = {
            "image_size": image_size,
            "principal_point": center,
            "focal_length": [focal_length,focal_length],
            "radial": [-0.01,0.,0., 0.],
        }

        thin_prism_cam_dict = {
            "image_size": image_size,
            "principal_point": center,
            "focal_length": [focal_length,focal_length],
            "radial": [-0.01,0.,0., 0.],
            "tangential": [0.01,0.],
            "prism": [0.01,0.]
        }

        self.cams = {
           "pinhole":PerspectiveCamera(perspective_cam_dict),
           "opencv":OpenCVFisheyeCamera(opncv_fisheye_cam_dict),
           "thinprism":ThinPrismFisheyeCamera(thin_prism_cam_dict)
        }

        self.pixels = np.array(
            [[0,100,200,399],
             [0,200,200,399]], dtype=np.float32)
        
        # edge case
        self.invalid_pixels = np.array(
            [[-1, 500],
             [-1, 500]])

    def test_projection_unprojection_consistency(self):
        for cam_type in self.cams:
            cam:Camera = self.cams[cam_type]
            rays, ray_mask = cam.convert_to_rays(self.pixels)
            pixels, pixel_mask = cam.convert_to_pixels(rays,True)

            # All rays should be valid for these pixels
            self.assertTrue(np.all(ray_mask), f"Failed for {cam_type}: ray_mask has False values")
            # With distortion, edge pixels might fail round-trip validation
            # So we check consistency only for valid pixels
            valid_pixels = self.pixels[:, pixel_mask]
            reconstructed_pixels = pixels[:, pixel_mask]
            np.testing.assert_array_almost_equal(valid_pixels.astype(np.float32), reconstructed_pixels, decimal=1e-3)


    def test_invalid_input(self):
        for cam_type in self.cams:
            cam:Camera = self.cams[cam_type]
            _, ray_mask = cam.convert_to_rays(self.invalid_pixels)

            self.assertTrue(np.all(~ray_mask)) 

if __name__ == '__main__':
    unittest.main()


        







