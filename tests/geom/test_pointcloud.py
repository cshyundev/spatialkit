import unittest
import numpy as np

import spatialkit
from spatialkit.camera import PerspectiveCamera
from spatialkit.geom.pointcloud import (
    depth_to_normal_map,
    depth_to_normal_map_fast,
    down_sample_point_cloud,
    compute_point_cloud_normals
)
from spatialkit.common.exceptions import (
    InvalidDimensionError,
    InvalidArgumentError,
    InvalidShapeError,
)


def create_test_camera(width=640, height=480, fov_deg=60):
    """Create a simple perspective camera for testing."""
    fov_rad = np.deg2rad(fov_deg)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = (height / 2.0) / np.tan(fov_rad / 2.0)

    cam_dict = {
        "image_size": (width, height),
        "principal_point": (width / 2.0, height / 2.0),
        "focal_length": (fx, fy),
        "radial": [0.0, 0.0, 0.0],
        "tangential": [0.0, 0.0]
    }
    return PerspectiveCamera(cam_dict)


class TestDepthToNormalMap(unittest.TestCase):
    """Test depth_to_normal_map function (SVD-based)."""

    @classmethod
    def setUpClass(cls):
        cls.cam = create_test_camera(width=64, height=48)
        cls.H, cls.W = cls.cam.hw

    def test_flat_plane_perpendicular(self):
        """Test flat plane perpendicular to camera."""
        depth = np.ones((self.H, self.W), dtype=np.float32) * 2.0
        normals = depth_to_normal_map(depth, self.cam, patch_size=3)

        # Check shape
        self.assertEqual(normals.shape, (self.H, self.W, 3))

        # Check center normal points towards camera [0, 0, -1]
        center_normal = normals[self.H // 2, self.W // 2]
        np.testing.assert_allclose(center_normal, [0, 0, -1], atol=0.1)

    def test_tilted_plane(self):
        """Test tilted plane detection."""
        y, x = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
        depth = 2.0 + 0.02 * (y - self.H / 2)
        depth = depth.astype(np.float32)

        normals = depth_to_normal_map(depth, self.cam, patch_size=3)
        center_normal = normals[self.H // 2, self.W // 2]

        # Y-component should be non-zero for tilted plane
        self.assertGreater(abs(center_normal[1]), 0.1)

    def test_invalid_depth_regions(self):
        """Test handling of invalid depth regions."""
        depth = np.ones((self.H, self.W), dtype=np.float32) * 2.0
        depth[20:30, 30:40] = 0.0  # Invalid region

        normals = depth_to_normal_map(depth, self.cam, patch_size=3)
        invalid_normals = normals[20:30, 30:40]

        # Most invalid pixels should have zero normals
        zero_count = np.sum(np.all(invalid_normals == 0, axis=-1))
        total = invalid_normals.shape[0] * invalid_normals.shape[1]
        self.assertGreater(zero_count, total * 0.5)

    def test_curvature_threshold(self):
        """Test curvature threshold filtering."""
        depth = np.ones((self.H, self.W), dtype=np.float32) * 2.0
        depth[self.H // 2:, :] = 5.0  # Step discontinuity

        # Strict vs loose threshold
        normals_strict = depth_to_normal_map(depth, self.cam, patch_size=3, curvature_threshold=0.05)
        normals_loose = depth_to_normal_map(depth, self.cam, patch_size=3, curvature_threshold=0.2)

        strict_valid = np.sum(np.any(normals_strict != 0, axis=-1))
        loose_valid = np.sum(np.any(normals_loose != 0, axis=-1))

        # Stricter threshold should have fewer or equal valid normals
        self.assertLessEqual(strict_valid, loose_valid)

    def test_different_patch_sizes(self):
        """Test different patch sizes."""
        depth = np.ones((self.H, self.W), dtype=np.float32) * 2.0

        normals_3 = depth_to_normal_map(depth, self.cam, patch_size=3)
        normals_5 = depth_to_normal_map(depth, self.cam, patch_size=5)

        self.assertEqual(normals_3.shape, (self.H, self.W, 3))
        self.assertEqual(normals_5.shape, (self.H, self.W, 3))

    def test_dtype_preservation(self):
        """Test dtype preservation."""
        for dtype in [np.float32, np.float64]:
            depth = np.ones((self.H, self.W), dtype=dtype) * 2.0
            normals = depth_to_normal_map(depth, self.cam, patch_size=3)
            self.assertEqual(normals.dtype, dtype)

    def test_unit_normals(self):
        """Test that all valid normals are unit vectors."""
        depth = np.ones((self.H, self.W), dtype=np.float32) * 2.0
        normals = depth_to_normal_map(depth, self.cam, patch_size=3)

        valid_normals = normals[np.any(normals != 0, axis=-1)]
        norms = np.linalg.norm(valid_normals, axis=-1)

        # All non-zero normals should be unit vectors
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_camera_orientation(self):
        """Test that normals point towards camera."""
        depth = np.ones((self.H, self.W), dtype=np.float32) * 2.0
        normals = depth_to_normal_map(depth, self.cam, patch_size=3)

        valid_normals = normals[np.any(normals != 0, axis=-1)]
        z_components = valid_normals[:, 2]

        # Most normals should have negative Z (pointing towards camera)
        negative_count = np.sum(z_components < 0)
        self.assertGreater(negative_count, len(z_components) * 0.9)

    def test_invalid_depth_shape(self):
        """Test error when depth shape doesn't match camera."""
        depth = np.ones((100, 100), dtype=np.float32) * 2.0

        with self.assertRaises(InvalidDimensionError):
            depth_to_normal_map(depth, self.cam, patch_size=3)

    def test_invalid_map_type(self):
        """Test error with invalid map_type."""
        depth = np.ones((self.H, self.W), dtype=np.float32) * 2.0

        with self.assertRaises(InvalidArgumentError):
            depth_to_normal_map(depth, self.cam, map_type="INVALID")

    def test_invalid_patch_size(self):
        """Test error with invalid patch_size."""
        depth = np.ones((self.H, self.W), dtype=np.float32) * 2.0

        # Even patch size
        with self.assertRaises(InvalidArgumentError):
            depth_to_normal_map(depth, self.cam, patch_size=4)

        # Too small
        with self.assertRaises(InvalidArgumentError):
            depth_to_normal_map(depth, self.cam, patch_size=1)

        # Too large
        with self.assertRaises(InvalidArgumentError):
            depth_to_normal_map(depth, self.cam, patch_size=100)


class TestDepthToNormalMapFast(unittest.TestCase):
    """Test depth_to_normal_map_fast function (gradient-based)."""

    def test_flat_plane(self):
        """Test flat plane perpendicular to camera."""
        depth = np.ones((48, 64), dtype=np.float32) * 2.0
        normals = depth_to_normal_map_fast(depth, depth_threshold=0.1)

        # Check shape
        self.assertEqual(normals.shape, (48, 64, 3))

        # Check center normal
        center_normal = normals[24, 32]
        np.testing.assert_allclose(center_normal, [0, 0, -1], atol=0.1)

    def test_depth_discontinuity(self):
        """Test handling of depth discontinuities."""
        depth = np.ones((48, 64), dtype=np.float32) * 2.0
        depth[24:, :] = 5.0  # Step discontinuity

        normals = depth_to_normal_map_fast(depth, depth_threshold=0.1)

        # Check that boundary region has some invalid normals
        boundary_normals = normals[22:26, 32]
        zero_count = np.sum(np.all(boundary_normals == 0, axis=-1))
        self.assertGreater(zero_count, 0)

    def test_dtype_preservation(self):
        """Test dtype preservation."""
        for dtype in [np.float32, np.float64]:
            depth = np.ones((48, 64), dtype=dtype) * 2.0
            normals = depth_to_normal_map_fast(depth)
            self.assertEqual(normals.dtype, dtype)

    def test_unit_normals(self):
        """Test that all valid normals are unit vectors."""
        depth = np.ones((48, 64), dtype=np.float32) * 2.0
        normals = depth_to_normal_map_fast(depth)

        valid_normals = normals[np.any(normals != 0, axis=-1)]
        norms = np.linalg.norm(valid_normals, axis=-1)

        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_camera_orientation(self):
        """Test that normals point towards camera."""
        depth = np.ones((48, 64), dtype=np.float32) * 2.0
        normals = depth_to_normal_map_fast(depth)

        valid_normals = normals[np.any(normals != 0, axis=-1)]
        z_components = valid_normals[:, 2]

        # All normals should have negative Z
        negative_count = np.sum(z_components < 0)
        self.assertGreater(negative_count, len(z_components) * 0.9)


class TestDepthToNormalComparison(unittest.TestCase):
    """Test comparison between SVD and fast methods."""

    def test_both_methods_same_orientation(self):
        """Test that both methods produce same orientation."""
        cam = create_test_camera(width=64, height=48)
        depth = np.ones((48, 64), dtype=np.float32) * 2.0

        normals_svd = depth_to_normal_map(depth, cam, patch_size=3)
        normals_fast = depth_to_normal_map_fast(depth, depth_threshold=0.1)

        # Both should point towards camera
        self.assertLess(normals_svd[24, 32, 2], 0)
        self.assertLess(normals_fast[24, 32, 2], 0)

        # Should be similar for flat plane
        diff = np.mean(np.abs(normals_svd - normals_fast))
        self.assertLess(diff, 0.1)


class TestDownSamplePointCloud(unittest.TestCase):
    """Test down_sample_point_cloud function."""

    def test_basic_downsampling(self):
        """Test basic voxel downsampling."""
        # Create dense point cloud
        pcd = np.random.rand(1000, 3).astype(np.float32)
        downsampled = down_sample_point_cloud(pcd, voxel_size=0.1)

        # Downsampled should have fewer points
        self.assertLess(downsampled.shape[0], pcd.shape[0])
        self.assertEqual(downsampled.shape[1], 3)

    def test_wrong_shape_error(self):
        """Test that [3,N] input raises error (not auto-transposed)."""
        pcd = np.random.rand(3, 1000).astype(np.float32)

        with self.assertRaises(InvalidDimensionError):
            down_sample_point_cloud(pcd, voxel_size=0.1)

    def test_dtype_preservation(self):
        """Test dtype preservation."""
        for dtype in [np.float32, np.float64]:
            pcd = np.random.rand(100, 3).astype(dtype)
            downsampled = down_sample_point_cloud(pcd, voxel_size=0.1)
            self.assertEqual(downsampled.dtype, dtype)

    def test_invalid_voxel_size(self):
        """Test error with invalid voxel_size."""
        pcd = np.random.rand(100, 3).astype(np.float32)

        with self.assertRaises(InvalidArgumentError):
            down_sample_point_cloud(pcd, voxel_size=0.0)

        with self.assertRaises(InvalidArgumentError):
            down_sample_point_cloud(pcd, voxel_size=-0.1)

    def test_invalid_shape(self):
        """Test error with invalid point cloud shape."""
        # 1D array
        pcd_1d = np.random.rand(100)
        with self.assertRaises(InvalidDimensionError):
            down_sample_point_cloud(pcd_1d, voxel_size=0.1)

        # Wrong number of coordinates
        pcd_wrong = np.random.rand(100, 2).astype(np.float32)
        with self.assertRaises(InvalidDimensionError):
            down_sample_point_cloud(pcd_wrong, voxel_size=0.1)


class TestComputePointCloudNormals(unittest.TestCase):
    """Test compute_point_cloud_normals function."""

    def test_basic_normal_estimation(self):
        """Test basic normal estimation from point cloud."""
        # Create planar point cloud
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        xx, yy = np.meshgrid(x, y)
        z = np.zeros_like(xx)  # Flat plane at z=0
        pcd = np.stack([xx.ravel(), yy.ravel(), z.ravel()], axis=1).astype(np.float32)

        normals = compute_point_cloud_normals(pcd, search_param="knn", k=10)

        # Check shape
        self.assertEqual(normals.shape, pcd.shape)

        # For a flat plane, normals should be close to [0, 0, ±1]
        # Check that most normals have small x,y components
        xy_norms = np.linalg.norm(normals[:, :2], axis=1)
        small_xy = np.sum(xy_norms < 0.3)
        self.assertGreater(small_xy, len(normals) * 0.8)

    def test_wrong_shape_error(self):
        """Test that [3,N] input raises error (not auto-transposed)."""
        pcd = np.random.rand(3, 100).astype(np.float32)

        with self.assertRaises(InvalidShapeError):
            compute_point_cloud_normals(pcd, search_param="knn", k=10)

    def test_dtype_preservation(self):
        """Test dtype preservation."""
        for dtype in [np.float32, np.float64]:
            pcd = np.random.rand(100, 3).astype(dtype)
            normals = compute_point_cloud_normals(pcd, search_param="knn", k=10)
            self.assertEqual(normals.dtype, dtype)

    def test_unit_normals(self):
        """Test that normals are unit vectors."""
        pcd = np.random.rand(100, 3).astype(np.float32)
        normals = compute_point_cloud_normals(pcd, search_param="knn", k=10)

        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_invalid_search_param(self):
        """Test error with invalid search_param."""
        pcd = np.random.rand(100, 3).astype(np.float32)

        with self.assertRaises((InvalidArgumentError, ValueError)):
            compute_point_cloud_normals(pcd, search_param="invalid")

    def test_radius_search(self):
        """Test radius search method."""
        pcd = np.random.rand(100, 3).astype(np.float32)
        normals = compute_point_cloud_normals(pcd, search_param="radius", radius=0.1, max_nn=30)

        # Check shape and unit vectors
        self.assertEqual(normals.shape, (100, 3))
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
