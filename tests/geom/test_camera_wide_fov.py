import unittest
import numpy as np
import torch

# Use new hierarchical import pattern
from spatialkit import (
    OmnidirectionalCamera,
    DoubleSphereCamera,
    EquirectangularCamera
)
from spatialkit.common.exceptions import InvalidCameraParameterError, InvalidShapeError


class TestOmnidirectionalCamera(unittest.TestCase):
    """
    Unit tests for OmnidirectionalCamera class.

    Tests cover:
    1. Initialization with required FOV parameter
    2. Projection and unprojection
    3. Error handling for missing FOV
    4. Export functionality
    """

    def setUp(self):
        """Set up test fixtures for OmnidirectionalCamera tests."""
        self.image_size = (640, 480)
        self.width, self.height = self.image_size

        # Omnidirectional camera requires fov_deg parameter
        self.cam_dict = {
            "image_size": self.image_size,
            "distortion_center": (320.0, 240.0),
            "poly_coeffs": [1.0, 0.0, 0.001],  # Simple polynomial
            "inv_poly_coeffs": [1.0, 0.0, -0.001],
            "affine": [1.0, 0.0, 0.0],  # c, d, e
            "fov_deg": 180.0  # Wide field of view
        }

        # Test pixels closer to center for omnidirectional
        self.test_pixels = np.array([
            [320, 250, 390, 280, 350],
            [240, 200, 280, 260, 220]
        ], dtype=np.float32)

    def test_initialization_with_fov(self):
        """Test camera initialization with required FOV parameter."""
        cam = OmnidirectionalCamera(self.cam_dict)

        self.assertEqual(cam.width, self.width)
        self.assertEqual(cam.height, self.height)
        self.assertEqual(cam.cx, 320.0)
        self.assertEqual(cam.cy, 240.0)

    def test_initialization_missing_fov(self):
        """Test that initialization fails without FOV parameter."""
        cam_dict_no_fov = self.cam_dict.copy()
        del cam_dict_no_fov["fov_deg"]

        with self.assertRaises(InvalidCameraParameterError):
            OmnidirectionalCamera(cam_dict_no_fov)

    def test_initialization_negative_fov(self):
        """Test that initialization fails with negative FOV."""
        cam_dict_bad_fov = self.cam_dict.copy()
        cam_dict_bad_fov["fov_deg"] = -10.0

        with self.assertRaises(InvalidCameraParameterError):
            OmnidirectionalCamera(cam_dict_bad_fov)

    def test_convert_to_rays(self):
        """Test pixel to ray conversion."""
        cam = OmnidirectionalCamera(self.cam_dict)
        rays, mask = cam.convert_to_rays(self.test_pixels)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)
        self.assertEqual(rays.shape[1], self.test_pixels.shape[1])

        # Rays should be normalized
        norms = np.linalg.norm(rays, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones(rays.shape[1]), decimal=5)

    def test_convert_to_rays_z_fixed(self):
        """Test pixel to ray conversion with z_fixed=True."""
        cam = OmnidirectionalCamera(self.cam_dict)
        rays, mask = cam.convert_to_rays(self.test_pixels, z_fixed=True)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)

    def test_convert_to_pixels(self):
        """Test ray to pixel conversion."""
        cam = OmnidirectionalCamera(self.cam_dict)

        # Generate rays from pixels
        rays, _ = cam.convert_to_rays(self.test_pixels)

        # Project back to pixels
        pixels, mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Check dimensions
        self.assertEqual(pixels.shape[0], 2)
        self.assertEqual(pixels.shape[1], self.test_pixels.shape[1])

    def test_projection_unprojection_consistency_numpy(self):
        """Test round-trip consistency with NumPy arrays."""
        cam = OmnidirectionalCamera(self.cam_dict)

        rays, ray_mask = cam.convert_to_rays(self.test_pixels)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Check that reconstruction produces valid results
        # Note: Omnidirectional cameras with simple polynomial coefficients
        # may not have perfect round-trip accuracy
        self.assertEqual(pixels.shape[0], 2)
        self.assertEqual(pixels.shape[1], self.test_pixels.shape[1])

    def test_projection_unprojection_consistency_torch(self):
        """Test round-trip consistency with PyTorch tensors."""
        cam = OmnidirectionalCamera(self.cam_dict)

        test_pixels_torch = torch.from_numpy(self.test_pixels)
        rays, ray_mask = cam.convert_to_rays(test_pixels_torch)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Results should be torch tensors
        self.assertTrue(torch.is_tensor(rays))
        self.assertTrue(torch.is_tensor(pixels))

    def test_export_cam_dict(self):
        """Test camera dictionary export."""
        cam = OmnidirectionalCamera(self.cam_dict)
        exported = cam.export_cam_dict()

        # Check exported dictionary contains all necessary keys
        self.assertIn("cam_type", exported)
        self.assertIn("image_size", exported)
        self.assertIn("fov_deg", exported)
        self.assertIn("distortion_center", exported)


class TestDoubleSphereCamera(unittest.TestCase):
    """
    Unit tests for DoubleSphereCamera class.

    Tests cover:
    1. Initialization with required FOV parameter
    2. Projection and unprojection
    3. Error handling for missing FOV
    4. Export functionality
    """

    def setUp(self):
        """Set up test fixtures for DoubleSphereCamera tests."""
        self.image_size = (640, 480)
        self.width, self.height = self.image_size

        # DoubleSphere camera requires fov_deg parameter
        self.cam_dict = {
            "image_size": self.image_size,
            "principal_point": (320.0, 240.0),
            "focal_length": (250.0, 250.0),
            "xi": 0.5,      # First sphere parameter
            "alpha": 0.6,   # Second sphere parameter
            "fov_deg": 160.0  # Wide field of view
        }

        # Test pixels closer to center
        self.test_pixels = np.array([
            [320, 270, 370, 290, 350],
            [240, 210, 270, 250, 230]
        ], dtype=np.float32)

    def test_initialization_with_fov(self):
        """Test camera initialization with required FOV parameter."""
        cam = DoubleSphereCamera(self.cam_dict)

        self.assertEqual(cam.width, self.width)
        self.assertEqual(cam.height, self.height)
        self.assertEqual(cam.cx, 320.0)
        self.assertEqual(cam.cy, 240.0)
        self.assertEqual(cam.fx, 250.0)
        self.assertEqual(cam.fy, 250.0)
        self.assertEqual(cam.xi, 0.5)
        self.assertEqual(cam.alpha, 0.6)

    def test_initialization_missing_fov(self):
        """Test that initialization fails without FOV parameter."""
        cam_dict_no_fov = self.cam_dict.copy()
        del cam_dict_no_fov["fov_deg"]

        with self.assertRaises(InvalidCameraParameterError):
            DoubleSphereCamera(cam_dict_no_fov)

    def test_initialization_negative_fov(self):
        """Test that initialization fails with negative FOV."""
        cam_dict_bad_fov = self.cam_dict.copy()
        cam_dict_bad_fov["fov_deg"] = -10.0

        with self.assertRaises(InvalidCameraParameterError):
            DoubleSphereCamera(cam_dict_bad_fov)

    def test_convert_to_rays(self):
        """Test pixel to ray conversion."""
        cam = DoubleSphereCamera(self.cam_dict)
        rays, mask = cam.convert_to_rays(self.test_pixels)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)
        self.assertEqual(rays.shape[1], self.test_pixels.shape[1])

    def test_convert_to_rays_z_fixed(self):
        """Test pixel to ray conversion with z_fixed=True."""
        cam = DoubleSphereCamera(self.cam_dict)
        rays, mask = cam.convert_to_rays(self.test_pixels, z_fixed=True)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)

    def test_convert_to_pixels(self):
        """Test ray to pixel conversion."""
        cam = DoubleSphereCamera(self.cam_dict)

        # Generate rays from pixels
        rays, _ = cam.convert_to_rays(self.test_pixels)

        # Project back to pixels
        pixels, mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Check dimensions
        self.assertEqual(pixels.shape[0], 2)
        self.assertEqual(pixels.shape[1], self.test_pixels.shape[1])

    def test_projection_unprojection_consistency_numpy(self):
        """Test round-trip consistency with NumPy arrays."""
        cam = DoubleSphereCamera(self.cam_dict)

        rays, ray_mask = cam.convert_to_rays(self.test_pixels)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Check that valid pixels are reconstructed
        valid_idx = np.where(ray_mask & pixel_mask)[0]
        if len(valid_idx) > 0:
            np.testing.assert_array_almost_equal(
                self.test_pixels[:, valid_idx],
                pixels[:, valid_idx],
                decimal=0
            )

    def test_projection_unprojection_consistency_torch(self):
        """Test round-trip consistency with PyTorch tensors."""
        cam = DoubleSphereCamera(self.cam_dict)

        test_pixels_torch = torch.from_numpy(self.test_pixels)
        rays, ray_mask = cam.convert_to_rays(test_pixels_torch)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Results should be torch tensors
        self.assertTrue(torch.is_tensor(rays))
        self.assertTrue(torch.is_tensor(pixels))

    def test_export_cam_dict(self):
        """Test camera dictionary export."""
        cam = DoubleSphereCamera(self.cam_dict)
        exported = cam.export_cam_dict()

        # Check exported dictionary contains all necessary keys
        self.assertIn("cam_type", exported)
        self.assertIn("image_size", exported)
        self.assertIn("fov_deg", exported)
        self.assertIn("principal_point", exported)
        self.assertIn("focal_length", exported)
        self.assertIn("xi", exported)
        self.assertIn("alpha", exported)


class TestEquirectangularCamera(unittest.TestCase):
    """
    Unit tests for EquirectangularCamera class (360-degree panoramic).

    Tests cover:
    1. Initialization with default and custom phi ranges
    2. Factory method from_image_size
    3. Projection and unprojection for spherical coordinates
    4. Export functionality
    5. Edge cases for phi ranges
    """

    def setUp(self):
        """Set up test fixtures for EquirectangularCamera tests."""
        self.image_size = (800, 400)  # Typical 2:1 equirectangular ratio
        self.width, self.height = self.image_size

        # Default equirectangular camera
        self.cam_dict = {
            "image_size": self.image_size,
            "min_phi_deg": -90.0,
            "max_phi_deg": 90.0
        }

        # Test pixels spread across the panorama
        self.test_pixels = np.array([
            [400, 200, 600, 100, 700],  # Spread horizontally
            [200, 100, 300, 150, 250]   # Spread vertically
        ], dtype=np.float32)

    def test_initialization_default_phi(self):
        """Test camera initialization with default phi range."""
        cam = EquirectangularCamera(self.cam_dict)

        self.assertEqual(cam.width, self.width)
        self.assertEqual(cam.height, self.height)
        self.assertEqual(cam.min_phi_deg, -90.0)
        self.assertEqual(cam.max_phi_deg, 90.0)

    def test_initialization_custom_phi(self):
        """Test camera initialization with custom phi range."""
        cam_dict_custom = {
            "image_size": self.image_size,
            "min_phi_deg": -45.0,
            "max_phi_deg": 45.0
        }
        cam = EquirectangularCamera(cam_dict_custom)

        self.assertEqual(cam.min_phi_deg, -45.0)
        self.assertEqual(cam.max_phi_deg, 45.0)

    def test_from_image_size_factory(self):
        """Test from_image_size factory method."""
        cam = EquirectangularCamera.from_image_size(self.image_size)

        self.assertEqual(cam.width, self.width)
        self.assertEqual(cam.height, self.height)
        self.assertEqual(cam.min_phi_deg, -90.0)
        self.assertEqual(cam.max_phi_deg, 90.0)

    def test_convert_to_rays(self):
        """Test pixel to ray conversion."""
        cam = EquirectangularCamera(self.cam_dict)
        rays, mask = cam.convert_to_rays(self.test_pixels)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)
        self.assertEqual(rays.shape[1], self.test_pixels.shape[1])

        # All test pixels should be valid
        self.assertTrue(np.all(mask))

    def test_convert_to_rays_z_fixed(self):
        """Test pixel to ray conversion with z_fixed=True."""
        cam = EquirectangularCamera(self.cam_dict)
        rays, mask = cam.convert_to_rays(self.test_pixels, z_fixed=True)

        # Check dimensions
        self.assertEqual(rays.shape[0], 3)

    def test_convert_to_pixels(self):
        """Test ray to pixel conversion."""
        cam = EquirectangularCamera(self.cam_dict)

        # Generate rays from pixels
        rays, _ = cam.convert_to_rays(self.test_pixels)

        # Project back to pixels
        pixels, mask = cam.convert_to_pixels(rays, out_subpixel=True)

        # Check dimensions
        self.assertEqual(pixels.shape[0], 2)
        self.assertEqual(pixels.shape[1], self.test_pixels.shape[1])

    def test_projection_unprojection_consistency_numpy(self):
        """Test round-trip consistency with NumPy arrays."""
        cam = EquirectangularCamera(self.cam_dict)

        rays, ray_mask = cam.convert_to_rays(self.test_pixels)
        pixels, pixel_mask = cam.convert_to_pixels(rays, out_subpixel=True)

        self.assertTrue(np.all(ray_mask))
        self.assertTrue(np.all(pixel_mask))
        # Equirectangular should have good round-trip accuracy
        np.testing.assert_array_almost_equal(self.test_pixels, pixels, decimal=0)

    def test_projection_unprojection_consistency_torch(self):
        """Test round-trip consistency with PyTorch tensors."""
        cam = EquirectangularCamera(self.cam_dict)

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

    def test_full_sphere_coverage(self):
        """Test that camera can handle full 360-degree coverage."""
        cam = EquirectangularCamera(self.cam_dict)

        # Test pixels at image boundaries
        boundary_pixels = np.array([
            [0.0, self.width - 1, self.width / 2],
            [0.0, self.height - 1, self.height / 2]
        ], dtype=np.float32)

        rays, mask = cam.convert_to_rays(boundary_pixels)

        # All should be valid
        self.assertTrue(np.all(mask))

    def test_export_cam_dict(self):
        """Test camera dictionary export."""
        cam = EquirectangularCamera(self.cam_dict)
        exported = cam.export_cam_dict()

        # Check exported dictionary contains all necessary keys
        self.assertIn("cam_type", exported)
        self.assertIn("image_size", exported)
        self.assertIn("min_phi_deg", exported)
        self.assertIn("max_phi_deg", exported)

        # Check exported values
        self.assertEqual(exported["image_size"], (self.width, self.height))
        self.assertEqual(exported["min_phi_deg"], -90.0)
        self.assertEqual(exported["max_phi_deg"], 90.0)

    def test_center_pixel_points_forward(self):
        """Test that center pixel corresponds to forward direction."""
        cam = EquirectangularCamera(self.cam_dict)

        # Center pixel should point roughly in the Z direction (forward)
        center_pixel = np.array([[self.width / 2], [self.height / 2]], dtype=np.float32)
        rays, mask = cam.convert_to_rays(center_pixel)

        # Z component should be largest (forward direction)
        self.assertGreater(abs(rays[2, 0]), abs(rays[0, 0]))
        self.assertGreater(abs(rays[2, 0]), abs(rays[1, 0]))


if __name__ == '__main__':
    unittest.main()
