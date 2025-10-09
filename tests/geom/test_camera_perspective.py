import unittest
import numpy as np
import torch

# Use new hierarchical import pattern
from cv_utils import PerspectiveCamera
from cv_utils.exceptions import InvalidShapeError


class TestPerspectiveCamera(unittest.TestCase):
    """
    Comprehensive unit tests for PerspectiveCamera class.

    Tests cover:
    1. Initialization from cam_dict
    2. Factory methods (from_K, from_fov)
    3. Projection and unprojection (convert_to_rays, convert_to_pixels)
    4. Distortion and undistortion (distort_pixel, undistort_pixel)
    5. Image warping (distort_image, undistort_image)
    6. Edge cases and error handling
    7. NumPy and PyTorch compatibility
    """

    def setUp(self):
        """Set up test fixtures for PerspectiveCamera tests."""
        self.image_size = (640, 480)
        self.width, self.height = self.image_size
        self.center = (320.0, 240.0)
        self.focal_length = (500.0, 500.0)

        # Camera with no distortion
        self.cam_dict_no_dist = {
            "image_size": self.image_size,
            "principal_point": self.center,
            "focal_length": self.focal_length,
            "radial": [0.0, 0.0, 0.0],
            "tangential": [0.0, 0.0]
        }

        # Camera with radial distortion
        self.cam_dict_radial = {
            "image_size": self.image_size,
            "principal_point": self.center,
            "focal_length": self.focal_length,
            "radial": [-0.2, 0.05, 0.01],
            "tangential": [0.0, 0.0]
        }

        # Camera with radial and tangential distortion
        self.cam_dict_full_dist = {
            "image_size": self.image_size,
            "principal_point": self.center,
            "focal_length": self.focal_length,
            "radial": [-0.2, 0.05, 0.01],
            "tangential": [0.01, -0.01]
        }

        # Test pixels
        self.test_pixels = np.array([
            [320, 100, 540, 10, 630],
            [240, 50, 430, 10, 470]
        ], dtype=np.float32)

        # Invalid pixels (outside image bounds)
        self.invalid_pixels = np.array([
            [-10, 650, 320],
            [-10, 490, 240]
        ], dtype=np.float32)

    def test_initialization_no_distortion(self):
        """Test camera initialization without distortion."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)

        self.assertEqual(cam.width, self.width)
        self.assertEqual(cam.height, self.height)
        self.assertEqual(cam.fx, self.focal_length[0])
        self.assertEqual(cam.fy, self.focal_length[1])
        self.assertEqual(cam.cx, self.center[0])
        self.assertEqual(cam.cy, self.center[1])
        self.assertFalse(cam.has_distortion())

        # Check K matrix
        K = cam.K
        self.assertEqual(K[0, 0], self.focal_length[0])
        self.assertEqual(K[1, 1], self.focal_length[1])
        self.assertEqual(K[0, 2], self.center[0])
        self.assertEqual(K[1, 2], self.center[1])

    def test_initialization_with_distortion(self):
        """Test camera initialization with distortion."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        self.assertTrue(cam.has_distortion())
        self.assertEqual(len(cam.radial_params), 3)
        self.assertEqual(len(cam.tangential_params), 2)

        # Check distortion coefficients
        dist_coeffs = cam.dist_coeffs
        expected = np.array([-0.2, 0.05, 0.01, -0.01, 0.01])
        np.testing.assert_array_almost_equal(dist_coeffs, expected)

    def test_from_K_no_distortion(self):
        """Test from_K factory method without distortion."""
        K = [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ]

        cam = PerspectiveCamera.from_K(K, self.image_size)

        self.assertEqual(cam.fx, 500.0)
        self.assertEqual(cam.fy, 500.0)
        self.assertEqual(cam.cx, 320.0)
        self.assertEqual(cam.cy, 240.0)
        self.assertFalse(cam.has_distortion())

    def test_from_K_with_distortion(self):
        """Test from_K factory method with distortion."""
        K = [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ]
        dist_coeffs = [-0.2, 0.05, 0.01, -0.01, 0.001]

        cam = PerspectiveCamera.from_K(K, self.image_size, dist_coeffs)

        self.assertTrue(cam.has_distortion())
        self.assertAlmostEqual(cam.radial_params[0], -0.2)
        self.assertAlmostEqual(cam.radial_params[1], 0.05)
        self.assertAlmostEqual(cam.radial_params[2], 0.001)
        self.assertAlmostEqual(cam.tangential_params[0], 0.01)
        self.assertAlmostEqual(cam.tangential_params[1], -0.01)

    def test_from_fov_single_value(self):
        """Test from_fov factory method with single FOV value."""
        fov = 60.0  # degrees
        cam = PerspectiveCamera.from_fov(self.image_size, fov)

        self.assertFalse(cam.has_distortion())
        self.assertGreater(cam.fx, 0)
        self.assertGreater(cam.fy, 0)

        # Check that principal point is at center
        self.assertAlmostEqual(cam.cx, (self.width - 1) / 2.0)
        self.assertAlmostEqual(cam.cy, (self.height - 1) / 2.0)

    def test_from_fov_list_values(self):
        """Test from_fov factory method with separate FOV values."""
        fov = [60.0, 45.0]  # different horizontal and vertical FOV
        cam = PerspectiveCamera.from_fov(self.image_size, fov)

        self.assertFalse(cam.has_distortion())
        self.assertNotEqual(cam.fx, cam.fy)  # Different FOVs should give different focal lengths

    def test_convert_to_rays_no_distortion(self):
        """Test pixel to ray conversion without distortion."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        rays, mask = cam.convert_to_rays(self.test_pixels)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)
        self.assertEqual(rays.shape[1], self.test_pixels.shape[1])
        self.assertEqual(mask.shape[0], self.test_pixels.shape[1])

        # All test pixels should be valid
        self.assertTrue(np.all(mask))

        # Rays should be normalized
        norms = np.linalg.norm(rays, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones(rays.shape[1]), decimal=5)

    def test_convert_to_rays_with_distortion(self):
        """Test pixel to ray conversion with distortion."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)
        rays, mask = cam.convert_to_rays(self.test_pixels)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)
        self.assertEqual(rays.shape[1], self.test_pixels.shape[1])

        # Rays should be normalized
        norms = np.linalg.norm(rays, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones(rays.shape[1]), decimal=5)

    def test_convert_to_rays_z_fixed(self):
        """Test pixel to ray conversion with z_fixed=True."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        rays, mask = cam.convert_to_rays(self.test_pixels, z_fixed=True)

        # Z component should be 1
        np.testing.assert_array_almost_equal(rays[2, :], np.ones(rays.shape[1]))

    def test_convert_to_rays_invalid_pixels(self):
        """Test pixel to ray conversion with invalid pixels."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        rays, mask = cam.convert_to_rays(self.invalid_pixels)

        # Invalid pixels should be masked out
        self.assertFalse(np.all(mask))
        self.assertTrue(np.any(~mask))

    def test_convert_to_pixels_no_distortion(self):
        """Test ray to pixel conversion without distortion."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)

        # Generate rays from known pixels
        rays, _ = cam.convert_to_rays(self.test_pixels)

        # Project back to pixels
        pixels, mask = cam.convert_to_pixels(rays)

        # Should get back original pixels (approximately)
        self.assertTrue(np.all(mask))
        np.testing.assert_array_almost_equal(pixels.astype(np.float32), self.test_pixels, decimal=1)

    def test_convert_to_pixels_with_distortion(self):
        """Test ray to pixel conversion with distortion."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        # Generate rays from known pixels
        rays, _ = cam.convert_to_rays(self.test_pixels)

        # Project back to pixels
        pixels, mask = cam.convert_to_pixels(rays)

        # Should get back original pixels (approximately)
        # With high distortion, allow up to 2 pixels difference
        self.assertTrue(np.all(mask))
        np.testing.assert_array_almost_equal(pixels.astype(np.float32), self.test_pixels, decimal=0)

    def test_convert_to_pixels_subpixel(self):
        """Test ray to pixel conversion with subpixel accuracy."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        rays, _ = cam.convert_to_rays(self.test_pixels)

        pixels, mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Should get float pixels
        self.assertEqual(pixels.dtype, np.float32)

    def test_projection_unprojection_consistency_numpy(self):
        """Test round-trip consistency: pixels -> rays -> pixels (NumPy)."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        rays, ray_mask = cam.convert_to_rays(self.test_pixels)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        self.assertTrue(np.all(ray_mask))
        self.assertTrue(np.all(pixel_mask))
        np.testing.assert_array_almost_equal(self.test_pixels, pixels, decimal=1)

    def test_projection_unprojection_consistency_torch(self):
        """Test round-trip consistency: pixels -> rays -> pixels (PyTorch)."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        test_pixels_torch = torch.from_numpy(self.test_pixels)
        rays, ray_mask = cam.convert_to_rays(test_pixels_torch)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Results should be torch tensors
        self.assertTrue(torch.is_tensor(rays))
        self.assertTrue(torch.is_tensor(pixels))

        # Check consistency
        self.assertTrue(torch.all(ray_mask))
        self.assertTrue(torch.all(pixel_mask))
        torch.testing.assert_close(test_pixels_torch, pixels, atol=0.1, rtol=1e-3)

    def test_distort_pixel_no_distortion(self):
        """Test pixel distortion when no distortion parameters."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        distorted = cam.distort_pixel(self.test_pixels)

        # Should return unchanged pixels
        np.testing.assert_array_equal(distorted, self.test_pixels)

    def test_distort_pixel_with_distortion(self):
        """Test pixel distortion with distortion parameters."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)
        distorted = cam.distort_pixel(self.test_pixels, out_subpixel=True)

        # Distorted pixels should differ from original
        self.assertFalse(np.allclose(distorted, self.test_pixels))

    def test_undistort_pixel_no_distortion(self):
        """Test pixel undistortion when no distortion parameters."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        undistorted = cam.undistort_pixel(self.test_pixels)

        # Should return unchanged pixels
        np.testing.assert_array_equal(undistorted, self.test_pixels)

    def test_undistort_pixel_with_distortion(self):
        """Test pixel undistortion with distortion parameters."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        # First distort, then undistort
        distorted = cam.distort_pixel(self.test_pixels, out_subpixel=True)
        undistorted = cam.undistort_pixel(distorted, out_subpixel=True)

        # Should get back original pixels
        np.testing.assert_array_almost_equal(undistorted, self.test_pixels, decimal=1)

    def test_distort_undistort_consistency(self):
        """Test that distort and undistort are inverse operations."""
        cam = PerspectiveCamera(self.cam_dict_radial)

        distorted = cam.distort_pixel(self.test_pixels, out_subpixel=True)
        undistorted = cam.undistort_pixel(distorted, out_subpixel=True)

        np.testing.assert_array_almost_equal(undistorted, self.test_pixels, decimal=0)

    def test_undistort_image_no_distortion(self):
        """Test image undistortion when no distortion parameters."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)

        # Create test image
        test_image = np.random.rand(self.height, self.width, 3).astype(np.float32)
        undistorted = cam.undistort_image(test_image)

        # Should return unchanged image
        np.testing.assert_array_equal(undistorted, test_image)

    def test_undistort_image_with_distortion(self):
        """Test image undistortion with distortion parameters."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        # Create test image
        test_image = np.random.rand(self.height, self.width, 3).astype(np.float32)
        undistorted = cam.undistort_image(test_image)

        # Check output shape matches input
        self.assertEqual(undistorted.shape, test_image.shape)

    def test_undistort_image_grayscale(self):
        """Test image undistortion with grayscale image."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        # Create grayscale test image
        test_image = np.random.rand(self.height, self.width).astype(np.float32)
        undistorted = cam.undistort_image(test_image)

        # Check output shape matches input
        self.assertEqual(undistorted.shape, test_image.shape)

    def test_undistort_image_wrong_size(self):
        """Test that undistort_image raises error for wrong image size."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        # Create image with wrong size
        wrong_image = np.random.rand(200, 300, 3).astype(np.float32)

        with self.assertRaises(InvalidShapeError):
            cam.undistort_image(wrong_image)

    def test_distort_image_with_distortion(self):
        """Test image distortion with distortion parameters."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        # Create test image
        test_image = np.random.rand(self.height, self.width, 3).astype(np.float32)
        distorted = cam.distort_image(test_image)

        # Check output shape matches input
        self.assertEqual(distorted.shape, test_image.shape)

    def test_distort_image_wrong_size(self):
        """Test that distort_image raises error for wrong image size."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)

        # Create image with wrong size
        wrong_image = np.random.rand(200, 300, 3).astype(np.float32)

        with self.assertRaises(InvalidShapeError):
            cam.distort_image(wrong_image)

    def test_fov_property(self):
        """Test FOV calculation."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        fov_x, fov_y = cam.fov

        # FOV should be positive
        self.assertGreater(fov_x, 0)
        self.assertGreater(fov_y, 0)

    def test_K_matrix_property(self):
        """Test K matrix property."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        K = cam.K

        # Check K matrix structure
        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(K[0, 0], cam.fx)
        self.assertEqual(K[1, 1], cam.fy)
        self.assertEqual(K[0, 2], cam.cx)
        self.assertEqual(K[1, 2], cam.cy)
        self.assertEqual(K[2, 2], 1.0)

    def test_inv_K_matrix_property(self):
        """Test inverse K matrix property."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        K = cam.K
        inv_K = cam.inv_K

        # K * inv_K should be identity
        identity = K @ inv_K
        np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=5)

    def test_export_cam_dict(self):
        """Test camera dictionary export."""
        cam = PerspectiveCamera(self.cam_dict_full_dist)
        exported = cam.export_cam_dict()

        # Check exported dictionary contains all necessary keys
        self.assertIn("cam_type", exported)
        self.assertIn("image_size", exported)
        self.assertIn("focal_length", exported)
        self.assertIn("principal_point", exported)
        self.assertIn("dist_coeffs", exported)

        # Check values
        self.assertEqual(exported["image_size"], (self.width, self.height))
        self.assertEqual(exported["focal_length"], self.focal_length)
        self.assertEqual(exported["principal_point"], self.center)

    def test_make_pixel_grid(self):
        """Test pixel grid generation."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        grid = cam.make_pixel_grid()

        # Check shape
        self.assertEqual(grid.shape[0], 2)
        self.assertEqual(grid.shape[1], self.width * self.height)

        # Check value ranges
        self.assertGreaterEqual(grid[0].min(), 0)
        self.assertLess(grid[0].max(), self.width)
        self.assertGreaterEqual(grid[1].min(), 0)
        self.assertLess(grid[1].max(), self.height)

    def test_mask_property(self):
        """Test mask property."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        mask = cam.mask

        # Check shape
        self.assertEqual(mask.shape, (self.height, self.width))

        # Default mask should be all True
        self.assertTrue(np.all(mask))

    def test_hw_property(self):
        """Test height-width property."""
        cam = PerspectiveCamera(self.cam_dict_no_dist)
        hw = cam.hw

        self.assertEqual(hw, (self.height, self.width))


if __name__ == '__main__':
    unittest.main()
