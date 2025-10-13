import unittest
import numpy as np
import torch

# Use new hierarchical import pattern
from cv_utils import OpenCVFisheyeCamera, ThinPrismFisheyeCamera
from cv_utils.exceptions import InvalidCameraParameterError, InvalidShapeError


class TestOpenCVFisheyeCamera(unittest.TestCase):
    """
    Comprehensive unit tests for OpenCVFisheyeCamera class.

    Tests cover:
    1. Initialization from cam_dict
    2. Factory method (from_K_D)
    3. Projection and unprojection
    4. Distortion and undistortion
    5. Image warping
    6. Edge cases and error handling
    7. NumPy and PyTorch compatibility
    """

    def setUp(self):
        """Set up test fixtures for OpenCVFisheyeCamera tests."""
        self.image_size = (640, 480)
        self.width, self.height = self.image_size
        self.center = (320.0, 240.0)
        self.focal_length = (300.0, 300.0)

        # Camera with no distortion
        self.cam_dict_no_dist = {
            "image_size": self.image_size,
            "principal_point": self.center,
            "focal_length": self.focal_length,
            "radial": [0.0, 0.0, 0.0, 0.0]
        }

        # Camera with fisheye distortion (moderate distortion to avoid edge artifacts)
        self.cam_dict_fisheye = {
            "image_size": self.image_size,
            "principal_point": self.center,
            "focal_length": self.focal_length,
            "radial": [-0.15, 0.05, -0.01, 0.001]
        }

        # Test pixels (avoid extreme edges for fisheye cameras)
        self.test_pixels = np.array([
            [320, 200, 440, 120, 520],
            [240, 150, 330, 120, 360]
        ], dtype=np.float32)

        self.invalid_pixels = np.array([
            [-10, 650, 320],
            [-10, 490, 240]
        ], dtype=np.float32)

    def test_initialization_no_distortion(self):
        """Test camera initialization without distortion."""
        cam = OpenCVFisheyeCamera(self.cam_dict_no_dist)

        self.assertEqual(cam.width, self.width)
        self.assertEqual(cam.height, self.height)
        self.assertEqual(cam.fx, self.focal_length[0])
        self.assertEqual(cam.fy, self.focal_length[1])
        self.assertFalse(cam.has_distortion())

    def test_initialization_with_distortion(self):
        """Test camera initialization with fisheye distortion."""
        cam = OpenCVFisheyeCamera(self.cam_dict_fisheye)

        self.assertTrue(cam.has_distortion())
        self.assertEqual(len(cam.radial_params), 4)

        # Check distortion coefficients
        dist_coeffs = cam.dist_coeffs
        np.testing.assert_array_almost_equal(dist_coeffs, self.cam_dict_fisheye["radial"])

    def test_from_K_D_factory(self):
        """Test from_K_D factory method."""
        K = [
            [300.0, 0.0, 320.0],
            [0.0, 300.0, 240.0],
            [0.0, 0.0, 1.0]
        ]
        D = [-0.3, 0.1, -0.05, 0.01]

        cam = OpenCVFisheyeCamera.from_K_D(K, self.image_size, D)

        self.assertEqual(cam.fx, 300.0)
        self.assertEqual(cam.fy, 300.0)
        self.assertEqual(cam.cx, 320.0)
        self.assertEqual(cam.cy, 240.0)
        self.assertTrue(cam.has_distortion())
        np.testing.assert_array_almost_equal(cam.dist_coeffs, D)

    def test_convert_to_rays_no_distortion(self):
        """Test pixel to ray conversion without distortion."""
        cam = OpenCVFisheyeCamera(self.cam_dict_no_dist)
        rays, mask = cam.convert_to_rays(self.test_pixels)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)
        self.assertEqual(rays.shape[1], self.test_pixels.shape[1])

        # All test pixels should be valid
        self.assertTrue(np.all(mask))

        # Rays should be normalized
        norms = np.linalg.norm(rays, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones(rays.shape[1]), decimal=5)

    def test_convert_to_rays_with_distortion(self):
        """Test pixel to ray conversion with fisheye distortion."""
        cam = OpenCVFisheyeCamera(self.cam_dict_fisheye)
        rays, mask = cam.convert_to_rays(self.test_pixels)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)
        self.assertEqual(rays.shape[1], self.test_pixels.shape[1])

        # Rays should be normalized
        norms = np.linalg.norm(rays, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones(rays.shape[1]), decimal=5)

    def test_convert_to_rays_z_fixed(self):
        """Test pixel to ray conversion with z_fixed=True."""
        cam = OpenCVFisheyeCamera(self.cam_dict_no_dist)
        rays, mask = cam.convert_to_rays(self.test_pixels, z_fixed=True)

        # Z component should be 1
        np.testing.assert_array_almost_equal(rays[2, :], np.ones(rays.shape[1]))

    def test_projection_unprojection_consistency_numpy(self):
        """Test round-trip consistency with NumPy arrays."""
        cam = OpenCVFisheyeCamera(self.cam_dict_fisheye)

        rays, ray_mask = cam.convert_to_rays(self.test_pixels)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        self.assertTrue(np.all(ray_mask))
        self.assertTrue(np.all(pixel_mask))
        # Allow some tolerance for fisheye distortion
        np.testing.assert_array_almost_equal(self.test_pixels, pixels, decimal=0)

    def test_projection_unprojection_consistency_torch(self):
        """Test round-trip consistency with PyTorch tensors."""
        cam = OpenCVFisheyeCamera(self.cam_dict_fisheye)

        test_pixels_torch = torch.from_numpy(self.test_pixels)
        rays, ray_mask = cam.convert_to_rays(test_pixels_torch)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Results should be torch tensors
        self.assertTrue(torch.is_tensor(rays))
        self.assertTrue(torch.is_tensor(pixels))

        # Check consistency
        self.assertTrue(torch.all(ray_mask))
        self.assertTrue(torch.all(pixel_mask))
        torch.testing.assert_close(test_pixels_torch, pixels, atol=1.0, rtol=1e-2)

    def test_distort_undistort_consistency(self):
        """Test that distort and undistort are inverse operations."""
        cam = OpenCVFisheyeCamera(self.cam_dict_fisheye)

        distorted = cam.distort_pixel(self.test_pixels, out_subpixel=True)
        undistorted = cam.undistort_pixel(distorted, out_subpixel=True)

        np.testing.assert_array_almost_equal(undistorted, self.test_pixels, decimal=0)

    def test_undistort_image_with_distortion(self):
        """Test image undistortion."""
        cam = OpenCVFisheyeCamera(self.cam_dict_fisheye)

        # Create test image
        test_image = np.random.rand(self.height, self.width, 3).astype(np.float32)
        undistorted = cam.undistort_image(test_image)

        # Check output shape matches input
        self.assertEqual(undistorted.shape, test_image.shape)

    def test_undistort_image_wrong_size(self):
        """Test that undistort_image raises error for wrong image size."""
        cam = OpenCVFisheyeCamera(self.cam_dict_fisheye)

        # Create image with wrong size
        wrong_image = np.random.rand(200, 300, 3).astype(np.float32)

        with self.assertRaises(InvalidShapeError):
            cam.undistort_image(wrong_image)

    def test_export_cam_dict(self):
        """Test camera dictionary export."""
        cam = OpenCVFisheyeCamera(self.cam_dict_fisheye)
        exported = cam.export_cam_dict()

        # Check exported dictionary contains all necessary keys
        self.assertIn("cam_type", exported)
        self.assertIn("image_size", exported)
        self.assertIn("focal_length", exported)
        self.assertIn("principal_point", exported)
        self.assertIn("dist_coeffs", exported)


class TestThinPrismFisheyeCamera(unittest.TestCase):
    """
    Comprehensive unit tests for ThinPrismFisheyeCamera class.

    Tests cover:
    1. Initialization from cam_dict
    2. Factory methods (from_K_D, from_params)
    3. Projection and unprojection
    4. Distortion and undistortion
    5. Image warping
    6. Edge cases and error handling
    7. NumPy and PyTorch compatibility
    """

    def setUp(self):
        """Set up test fixtures for ThinPrismFisheyeCamera tests."""
        self.image_size = (640, 480)
        self.width, self.height = self.image_size
        self.center = (320.0, 240.0)
        self.focal_length = (300.0, 300.0)

        # Camera with no distortion
        self.cam_dict_no_dist = {
            "image_size": self.image_size,
            "principal_point": self.center,
            "focal_length": self.focal_length,
            "radial": [0.0, 0.0, 0.0, 0.0],
            "tangential": [0.0, 0.0],
            "prism": [0.0, 0.0]
        }

        # Camera with full distortion
        self.cam_dict_full_dist = {
            "image_size": self.image_size,
            "principal_point": self.center,
            "focal_length": self.focal_length,
            "radial": [-0.2, 0.08, -0.03, 0.01],
            "tangential": [0.01, -0.01],
            "prism": [0.001, -0.001]
        }

        # Test pixels
        self.test_pixels = np.array([
            [320, 150, 490, 100, 540],
            [240, 100, 380, 80, 400]
        ], dtype=np.float32)

        self.invalid_pixels = np.array([
            [-10, 650, 320],
            [-10, 490, 240]
        ], dtype=np.float32)

    def test_initialization_no_distortion(self):
        """Test camera initialization without distortion."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_no_dist)

        self.assertEqual(cam.width, self.width)
        self.assertEqual(cam.height, self.height)
        self.assertEqual(cam.fx, self.focal_length[0])
        self.assertEqual(cam.fy, self.focal_length[1])
        self.assertFalse(cam.has_distortion())

    def test_initialization_with_distortion(self):
        """Test camera initialization with full distortion."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)

        self.assertTrue(cam.has_distortion())
        self.assertEqual(len(cam.radial_params), 4)
        self.assertEqual(len(cam.tangential_params), 2)
        self.assertEqual(len(cam.prism_params), 2)

        # Check distortion coefficients order: k1, k2, p1, p2, k3, k4, sx1, sy1
        dist_coeffs = cam.dist_coeffs
        self.assertEqual(len(dist_coeffs), 8)

    def test_from_K_D_factory(self):
        """Test from_K_D factory method."""
        K = [
            [300.0, 0.0, 320.0],
            [0.0, 300.0, 240.0],
            [0.0, 0.0, 1.0]
        ]
        # k1, k2, p1, p2, k3, k4, sx1, sy1
        D = [-0.2, 0.08, 0.01, -0.01, -0.03, 0.01, 0.001, -0.001]

        cam = ThinPrismFisheyeCamera.from_K_D(K, self.image_size, D)

        self.assertEqual(cam.fx, 300.0)
        self.assertEqual(cam.fy, 300.0)
        self.assertEqual(cam.cx, 320.0)
        self.assertEqual(cam.cy, 240.0)
        self.assertTrue(cam.has_distortion())

    def test_from_K_D_invalid_params(self):
        """Test from_K_D with invalid number of distortion parameters."""
        K = [[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]]
        D = [-0.2, 0.08]  # Only 2 parameters, need 8

        with self.assertRaises(InvalidCameraParameterError):
            ThinPrismFisheyeCamera.from_K_D(K, self.image_size, D)

    def test_from_params_factory(self):
        """Test from_params factory method."""
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
        params = [300.0, 300.0, 320.0, 240.0,
                  -0.2, 0.08, 0.01, -0.01, -0.03, 0.01, 0.001, -0.001]

        cam = ThinPrismFisheyeCamera.from_params(self.image_size, params)

        self.assertEqual(cam.fx, 300.0)
        self.assertEqual(cam.fy, 300.0)
        self.assertEqual(cam.cx, 320.0)
        self.assertEqual(cam.cy, 240.0)
        self.assertTrue(cam.has_distortion())

    def test_from_params_invalid_count(self):
        """Test from_params with invalid number of parameters."""
        params = [300.0, 300.0, 320.0]  # Only 3 parameters, need 12

        with self.assertRaises(InvalidCameraParameterError):
            ThinPrismFisheyeCamera.from_params(self.image_size, params)

    def test_convert_to_rays_no_distortion(self):
        """Test pixel to ray conversion without distortion."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_no_dist)
        rays, mask = cam.convert_to_rays(self.test_pixels)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)
        self.assertEqual(rays.shape[1], self.test_pixels.shape[1])

        # All test pixels should be valid
        self.assertTrue(np.all(mask))

        # Rays should be normalized
        norms = np.linalg.norm(rays, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones(rays.shape[1]), decimal=5)

    def test_convert_to_rays_with_distortion(self):
        """Test pixel to ray conversion with full distortion."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)
        rays, mask = cam.convert_to_rays(self.test_pixels)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)
        self.assertEqual(rays.shape[1], self.test_pixels.shape[1])

        # Rays should be normalized
        norms = np.linalg.norm(rays, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones(rays.shape[1]), decimal=5)

    def test_convert_to_rays_z_fixed(self):
        """Test pixel to ray conversion with z_fixed=True."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_no_dist)
        rays, mask = cam.convert_to_rays(self.test_pixels, z_fixed=True)

        # Z component should be 1
        np.testing.assert_array_almost_equal(rays[2, :], np.ones(rays.shape[1]))

    def test_projection_unprojection_consistency_numpy(self):
        """Test round-trip consistency with NumPy arrays."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)

        rays, ray_mask = cam.convert_to_rays(self.test_pixels)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        self.assertTrue(np.all(ray_mask))
        self.assertTrue(np.all(pixel_mask))
        # Allow tolerance for thin prism distortion
        np.testing.assert_array_almost_equal(self.test_pixels, pixels, decimal=0)

    def test_projection_unprojection_consistency_torch(self):
        """Test round-trip consistency with PyTorch tensors."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)

        test_pixels_torch = torch.from_numpy(self.test_pixels)
        rays, ray_mask = cam.convert_to_rays(test_pixels_torch)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Results should be torch tensors
        self.assertTrue(torch.is_tensor(rays))
        self.assertTrue(torch.is_tensor(pixels))

        # Check consistency
        self.assertTrue(torch.all(ray_mask))
        self.assertTrue(torch.all(pixel_mask))
        torch.testing.assert_close(test_pixels_torch, pixels, atol=1.0, rtol=1e-2)

    def test_distort_undistort_consistency(self):
        """Test that distort and undistort are inverse operations."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)

        distorted = cam.distort_pixel(self.test_pixels, out_subpixel=True)
        undistorted = cam.undistort_pixel(distorted, out_subpixel=True)

        np.testing.assert_array_almost_equal(undistorted, self.test_pixels, decimal=0)

    def test_distort_pixel_with_distortion(self):
        """Test pixel distortion with full distortion model."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)
        distorted = cam.distort_pixel(self.test_pixels, out_subpixel=True)

        # Distorted pixels should differ from original
        self.assertFalse(np.allclose(distorted, self.test_pixels))

    def test_undistort_image_with_distortion(self):
        """Test image undistortion."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)

        # Create test image
        test_image = np.random.rand(self.height, self.width, 3).astype(np.float32)
        undistorted = cam.undistort_image(test_image)

        # Check output shape matches input
        self.assertEqual(undistorted.shape, test_image.shape)

    def test_distort_image_with_distortion(self):
        """Test image distortion."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)

        # Create test image
        test_image = np.random.rand(self.height, self.width, 3).astype(np.float32)
        distorted = cam.distort_image(test_image)

        # Check output shape matches input
        self.assertEqual(distorted.shape, test_image.shape)

    def test_undistort_image_wrong_size(self):
        """Test that undistort_image raises error for wrong image size."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)

        # Create image with wrong size
        wrong_image = np.random.rand(200, 300, 3).astype(np.float32)

        with self.assertRaises(InvalidShapeError):
            cam.undistort_image(wrong_image)

    def test_distort_image_wrong_size(self):
        """Test that distort_image raises error for wrong image size."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)

        # Create image with wrong size
        wrong_image = np.random.rand(200, 300, 3).astype(np.float32)

        with self.assertRaises(InvalidShapeError):
            cam.distort_image(wrong_image)

    def test_fov_property(self):
        """Test FOV calculation."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_no_dist)
        fov_x, fov_y = cam.fov

        # FOV should be positive
        self.assertGreater(fov_x, 0)
        self.assertGreater(fov_y, 0)

    def test_export_cam_dict(self):
        """Test camera dictionary export."""
        cam = ThinPrismFisheyeCamera(self.cam_dict_full_dist)
        exported = cam.export_cam_dict()

        # Check exported dictionary contains all necessary keys
        self.assertIn("cam_type", exported)
        self.assertIn("image_size", exported)
        self.assertIn("focal_length", exported)
        self.assertIn("principal_point", exported)
        self.assertIn("dist_coeffs", exported)

        # Check exported values
        self.assertEqual(exported["image_size"], (self.width, self.height))


if __name__ == '__main__':
    unittest.main()
