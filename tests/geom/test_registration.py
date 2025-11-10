"""
Tests for registration.py - ICP (Iterative Closest Point) registration algorithms.
"""
import unittest
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

# Use new hierarchical import pattern
import spatialkit
from spatialkit.geom import Transform, Rotation, Pose


@unittest.skipIf(not OPEN3D_AVAILABLE, "Open3D not available")
class TestICPValidation(unittest.TestCase):
    """Tests for ICP input validation."""

    def test_validate_point_cloud_numpy(self):
        """Test that numpy arrays are accepted."""
        from spatialkit.geom.registration import _validate_point_cloud

        # Valid [N, 3] format
        points = np.random.rand(100, 3).astype(np.float32)
        result = _validate_point_cloud(points)
        self.assertEqual(result.shape, (100, 3))

    def test_validate_point_cloud_transpose(self):
        """Test auto-transpose from [3, N] to [N, 3]."""
        from spatialkit.geom.registration import _validate_point_cloud

        # [3, N] format should be transposed to [N, 3]
        points = np.random.rand(3, 100).astype(np.float32)
        result = _validate_point_cloud(points)
        self.assertEqual(result.shape, (100, 3))

    def test_validate_point_cloud_invalid_type(self):
        """Test that non-numpy input raises TypeError."""
        from spatialkit.geom.registration import _validate_point_cloud

        points_list = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(TypeError):
            _validate_point_cloud(points_list)

    def test_validate_point_cloud_invalid_dimension(self):
        """Test that non-2D input raises InvalidDimensionError."""
        from spatialkit.geom.registration import _validate_point_cloud

        points_1d = np.array([1, 2, 3])
        with self.assertRaises(spatialkit.InvalidDimensionError):
            _validate_point_cloud(points_1d)

    def test_validate_point_cloud_invalid_shape(self):
        """Test that arrays without 3 coordinates raise InvalidShapeError."""
        from spatialkit.geom.registration import _validate_point_cloud

        # 2 coordinates instead of 3
        points = np.random.rand(100, 2)
        with self.assertRaises(spatialkit.InvalidShapeError):
            _validate_point_cloud(points)


@unittest.skipIf(not OPEN3D_AVAILABLE, "Open3D not available")
class TestTransformConversion(unittest.TestCase):
    """Tests for transform conversion utilities."""

    def test_transform_to_matrix_none(self):
        """Test that None returns identity matrix."""
        from spatialkit.geom.registration import _transform_to_matrix

        result = _transform_to_matrix(None)
        expected = np.eye(4, dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_transform_to_matrix_from_transform(self):
        """Test conversion from Transform object."""
        from spatialkit.geom.registration import _transform_to_matrix

        R = Rotation.from_rpy(np.array([0.1, 0.2, 0.3]))
        t = np.array([1.0, 2.0, 3.0])
        transform = Transform(t, R)

        result = _transform_to_matrix(transform)

        self.assertEqual(result.shape, (4, 4))
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result[:3, 3], t, atol=1e-6)

    def test_transform_to_matrix_from_pose(self):
        """Test conversion from Pose object."""
        from spatialkit.geom.registration import _transform_to_matrix

        R = Rotation.from_rpy(np.array([0.1, 0.2, 0.3]))
        t = np.array([1.0, 2.0, 3.0])
        pose = Pose(t, R)

        result = _transform_to_matrix(pose)

        self.assertEqual(result.shape, (4, 4))
        self.assertEqual(result.dtype, np.float64)

    def test_transform_to_matrix_from_numpy(self):
        """Test conversion from numpy array."""
        from spatialkit.geom.registration import _transform_to_matrix

        mat = np.eye(4, dtype=np.float32)
        result = _transform_to_matrix(mat)

        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, np.eye(4, dtype=np.float64))

    def test_transform_to_matrix_invalid_shape(self):
        """Test that non-4x4 matrix raises InvalidShapeError."""
        from spatialkit.geom.registration import _transform_to_matrix

        mat_3x3 = np.eye(3)
        with self.assertRaises(spatialkit.InvalidShapeError):
            _transform_to_matrix(mat_3x3)

    def test_transform_to_matrix_invalid_type(self):
        """Test that invalid type raises TypeError."""
        from spatialkit.geom.registration import _transform_to_matrix

        with self.assertRaises(TypeError):
            _transform_to_matrix("invalid")


@unittest.skipIf(not OPEN3D_AVAILABLE, "Open3D not available")
class TestICPPointToPoint(unittest.TestCase):
    """Tests for ICP point-to-point registration."""

    def setUp(self):
        """Create test point clouds."""
        # Create source point cloud
        np.random.seed(42)
        self.source = np.random.rand(100, 3).astype(np.float32)

        # Apply known transformation to create target
        R = Rotation.from_rpy(np.array([0.1, 0.05, 0.15]))
        t = np.array([0.5, 0.3, 0.1])
        self.gt_transform = Transform(t, R)

        # Transform source to get target
        self.target = self.gt_transform.apply_pts3d(self.source.T).T

    def test_icp_perfect_alignment(self):
        """Test ICP with perfect initial alignment."""
        from spatialkit.geom.registration import icp_point_to_point

        # Start from ground truth transform
        result = icp_point_to_point(
            self.source,
            self.target,
            init_transform=self.gt_transform,
            max_correspondence_distance=0.1,
            max_iterations=10
        )

        # Check return structure
        self.assertIn("transformation", result)
        self.assertIn("fitness", result)
        self.assertIn("rmse", result)
        self.assertIn("correspondences", result)
        self.assertIn("converged", result)

        # Check types
        self.assertIsInstance(result["transformation"], Transform)
        self.assertIsInstance(result["fitness"], float)
        self.assertIsInstance(result["rmse"], float)
        self.assertIsInstance(result["correspondences"], int)
        self.assertIsInstance(result["converged"], bool)

        # RMSE should be very small for perfect alignment
        self.assertLess(result["rmse"], 1e-5)

    def test_icp_identity_init(self):
        """Test ICP with identity initialization."""
        from spatialkit.geom.registration import icp_point_to_point

        result = icp_point_to_point(
            self.source,
            self.target,
            init_transform=None,  # Identity
            max_correspondence_distance=0.5,
            max_iterations=50
        )

        # Should converge to reasonable alignment
        self.assertGreater(result["fitness"], 0.5)
        self.assertLess(result["rmse"], 0.5)

    def test_icp_transpose_input(self):
        """Test ICP with [3, N] format input."""
        from spatialkit.geom.registration import icp_point_to_point

        # Transpose to [3, N] format
        source_t = self.source.T
        target_t = self.target.T

        result = icp_point_to_point(
            source_t,
            target_t,
            max_correspondence_distance=0.1,
            max_iterations=50
        )

        self.assertIsInstance(result["transformation"], Transform)

    def test_icp_convergence_criteria(self):
        """Test ICP with different convergence criteria."""
        from spatialkit.geom.registration import icp_point_to_point

        result = icp_point_to_point(
            self.source,
            self.target,
            max_correspondence_distance=0.1,
            max_iterations=100,
            relative_fitness=1e-8,
            relative_rmse=1e-8
        )

        self.assertTrue(result["converged"])


@unittest.skipIf(not OPEN3D_AVAILABLE, "Open3D not available")
class TestICPPointToPlane(unittest.TestCase):
    """Tests for ICP point-to-plane registration."""

    def setUp(self):
        """Create test point clouds."""
        np.random.seed(42)
        self.source = np.random.rand(200, 3).astype(np.float32)

        R = Rotation.from_rpy(np.array([0.05, 0.05, 0.05]))
        t = np.array([0.1, 0.1, 0.1])
        self.gt_transform = Transform(t, R)

        self.target = self.gt_transform.apply_pts3d(self.source.T).T

    def test_icp_point_to_plane_basic(self):
        """Test basic point-to-plane ICP."""
        from spatialkit.geom.registration import icp_point_to_plane

        result = icp_point_to_plane(
            self.source,
            self.target,
            max_correspondence_distance=0.5,
            max_iterations=50,
            normal_radius=0.1,
            normal_max_nn=30
        )

        self.assertIsInstance(result["transformation"], Transform)
        self.assertGreater(result["fitness"], 0.5)

    def test_icp_point_to_plane_custom_normals(self):
        """Test point-to-plane ICP with custom normal parameters."""
        from spatialkit.geom.registration import icp_point_to_plane

        result = icp_point_to_plane(
            self.source,
            self.target,
            max_correspondence_distance=0.3,
            normal_radius=0.05,  # Smaller radius
            normal_max_nn=20     # Fewer neighbors
        )

        self.assertIsInstance(result["transformation"], Transform)


@unittest.skipIf(not OPEN3D_AVAILABLE, "Open3D not available")
class TestICPGeneralized(unittest.TestCase):
    """Tests for generalized ICP."""

    def setUp(self):
        """Create test point clouds with noise."""
        np.random.seed(42)
        # More points for robust normal estimation
        self.source = np.random.rand(300, 3).astype(np.float32)

        R = Rotation.from_rpy(np.array([0.1, 0.1, 0.1]))
        t = np.array([0.2, 0.2, 0.2])
        self.gt_transform = Transform(t, R)

        # Add small noise to target
        noise = np.random.randn(300, 3).astype(np.float32) * 0.01
        self.target = self.gt_transform.apply_pts3d(self.source.T).T + noise

    def test_icp_generalized_basic(self):
        """Test basic generalized ICP."""
        from spatialkit.geom.registration import icp_generalized

        result = icp_generalized(
            self.source,
            self.target,
            max_correspondence_distance=0.5,
            max_iterations=50
        )

        self.assertIsInstance(result["transformation"], Transform)
        self.assertGreater(result["fitness"], 0.5)

    def test_icp_generalized_with_init(self):
        """Test generalized ICP with initial guess."""
        from spatialkit.geom.registration import icp_generalized

        # Use approximate initial guess
        R_init = Rotation.from_rpy(np.array([0.08, 0.08, 0.08]))
        t_init = np.array([0.15, 0.15, 0.15])
        init_transform = Transform(t_init, R_init)

        result = icp_generalized(
            self.source,
            self.target,
            init_transform=init_transform,
            max_correspondence_distance=0.3,
            max_iterations=30
        )

        self.assertIsInstance(result["transformation"], Transform)


@unittest.skipIf(not OPEN3D_AVAILABLE, "Open3D not available")
class TestICPGeneral(unittest.TestCase):
    """Tests for general ICP function with method selection."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.source = np.random.rand(150, 3).astype(np.float32)

        R = Rotation.from_rpy(np.array([0.1, 0.1, 0.1]))
        t = np.array([0.3, 0.2, 0.1])
        self.gt_transform = Transform(t, R)

        self.target = self.gt_transform.apply_pts3d(self.source.T).T

    def test_icp_method_point_to_point(self):
        """Test ICP with point_to_point method."""
        from spatialkit.geom.registration import icp

        result = icp(
            self.source,
            self.target,
            estimation_method="point_to_point",
            max_correspondence_distance=0.5
        )

        self.assertIsInstance(result["transformation"], Transform)

    def test_icp_method_point_to_plane(self):
        """Test ICP with point_to_plane method."""
        from spatialkit.geom.registration import icp

        result = icp(
            self.source,
            self.target,
            estimation_method="point_to_plane",
            max_correspondence_distance=0.5
        )

        self.assertIsInstance(result["transformation"], Transform)

    def test_icp_method_generalized(self):
        """Test ICP with generalized method."""
        from spatialkit.geom.registration import icp

        result = icp(
            self.source,
            self.target,
            estimation_method="generalized",
            max_correspondence_distance=0.5
        )

        self.assertIsInstance(result["transformation"], Transform)

    def test_icp_invalid_method(self):
        """Test that invalid method raises ValueError."""
        from spatialkit.geom.registration import icp

        with self.assertRaises(ValueError):
            icp(
                self.source,
                self.target,
                estimation_method="invalid_method"
            )


@unittest.skipIf(not OPEN3D_AVAILABLE, "Open3D not available")
class TestICPRobustness(unittest.TestCase):
    """Tests for ICP robustness with challenging scenarios."""

    def test_icp_with_outliers(self):
        """Test ICP with outlier points."""
        from spatialkit.geom.registration import icp_point_to_point

        np.random.seed(42)
        source = np.random.rand(100, 3).astype(np.float32)

        R = Rotation.from_rpy(np.array([0.1, 0.1, 0.1]))
        t = np.array([0.2, 0.2, 0.2])
        gt_transform = Transform(t, R)

        target = gt_transform.apply_pts3d(source.T).T

        # Add outliers to target
        outliers = np.random.rand(20, 3).astype(np.float32) * 5
        target_with_outliers = np.vstack([target, outliers])

        result = icp_point_to_point(
            source,
            target_with_outliers,
            max_correspondence_distance=0.5,  # Reject outliers
            max_iterations=50
        )

        # Should still get reasonable alignment
        self.assertGreater(result["fitness"], 0.3)

    def test_icp_partial_overlap(self):
        """Test ICP with partial overlap between clouds."""
        from spatialkit.geom.registration import icp_point_to_point

        np.random.seed(42)
        source = np.random.rand(100, 3).astype(np.float32)

        R = Rotation.from_rpy(np.array([0.05, 0.05, 0.05]))
        t = np.array([0.1, 0.1, 0.1])
        gt_transform = Transform(t, R)

        # Use only half of the points for target
        target = gt_transform.apply_pts3d(source[:50].T).T

        result = icp_point_to_point(
            source,
            target,
            max_correspondence_distance=0.5,
            max_iterations=50
        )

        # With random point clouds and max_correspondence_distance=0.5,
        # all source points can find correspondences due to dense distribution
        # Fitness should be high (all points matched), RMSE reasonable
        self.assertGreater(result["fitness"], 0.8)
        self.assertLess(result["rmse"], 0.3)

    def test_icp_small_point_clouds(self):
        """Test ICP with very small point clouds."""
        from spatialkit.geom.registration import icp_point_to_point

        np.random.seed(42)
        source = np.random.rand(10, 3).astype(np.float32)
        target = source + 0.01  # Very slight offset

        result = icp_point_to_point(
            source,
            target,
            max_correspondence_distance=0.1,
            max_iterations=20
        )

        self.assertIsInstance(result["transformation"], Transform)


@unittest.skipIf(not OPEN3D_AVAILABLE, "Open3D not available")
class TestICPPerformance(unittest.TestCase):
    """Performance and stress tests for ICP."""

    def test_icp_large_point_cloud(self):
        """Test ICP with large point cloud (1000+ points)."""
        from spatialkit.geom.registration import icp_point_to_point

        np.random.seed(42)
        source = np.random.rand(1000, 3).astype(np.float32)

        R = Rotation.from_rpy(np.array([0.1, 0.1, 0.1]))
        t = np.array([0.2, 0.2, 0.2])
        gt_transform = Transform(t, R)

        target = gt_transform.apply_pts3d(source.T).T

        result = icp_point_to_point(
            source,
            target,
            max_correspondence_distance=0.5,
            max_iterations=50
        )

        self.assertGreater(result["fitness"], 0.9)
        self.assertLess(result["rmse"], 0.1)

    def test_icp_max_iterations_limit(self):
        """Test that ICP respects max_iterations limit."""
        from spatialkit.geom.registration import icp_point_to_point

        np.random.seed(42)
        source = np.random.rand(50, 3).astype(np.float32)
        target = np.random.rand(50, 3).astype(np.float32)  # Random, won't align

        # Very low max_iterations should terminate quickly
        result = icp_point_to_point(
            source,
            target,
            max_correspondence_distance=0.5,
            max_iterations=5
        )

        self.assertIsInstance(result["transformation"], Transform)


class TestICPImportError(unittest.TestCase):
    """Test behavior when Open3D is not available."""

    @unittest.skipIf(OPEN3D_AVAILABLE, "Skip when Open3D is available")
    def test_icp_without_open3d(self):
        """Test that ICP raises ImportError when Open3D is not available."""
        from spatialkit.geom.registration import icp

        source = np.random.rand(10, 3)
        target = np.random.rand(10, 3)

        with self.assertRaises(ImportError):
            icp(source, target)


if __name__ == '__main__':
    unittest.main()
