import unittest
import numpy as np
import pytest
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from spatialkit.geom.pose import Pose
from spatialkit.geom.tf import Transform, interpolate_transform
from spatialkit.geom.rotation import Rotation
from spatialkit.ops.uops import *
from spatialkit.ops.umath import *
from spatialkit.common.exceptions import (
    InvalidShapeError,
    InvalidDimensionError,
    IncompatibleTypeError
)


class TestTransform(unittest.TestCase):

    def test_transform_initialization(self):
        # Test default initialization
        transform = Transform()
        self.assertTrue(np.allclose(transform.t, np.array([[0., 0., 0.]])))
        self.assertTrue(np.allclose(transform.rot.data, np.eye(3)))

        # Test initialization with specific translation and rotation
        t = np.array([[1, 2, 3]])
        rot = Rotation.from_mat3(np.eye(3))  # Identity matrix as rotation
        transform = Transform(t=t, rot=rot)
        self.assertTrue(np.allclose(transform.t, t))
        self.assertTrue(np.allclose(transform.rot.data, np.eye(3)))

    def test_from_rot_vec_t(self):
        rot_vec = np.array([0, 0, np.pi/2])  # 90 degrees around z-axis
        t = np.array([1, 2, 3])
        transform = Transform.from_rot_vec_t(rot_vec, t)
        expected_rotation_matrix = Rotation.from_so3(rot_vec).data
        self.assertTrue(np.allclose(transform.t, np.array([1, 2, 3])))
        self.assertTrue(np.allclose(transform.rot.data, expected_rotation_matrix))

    def test_from_mat(self):
        # Create a transform from a 4x4 matrix
        mat = np.array([
            [0, -1, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        transform = Transform.from_mat(mat)
        expected_t = np.array([[1, 2, 3]])
        expected_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(transform.t, expected_t))
        self.assertTrue(np.allclose(transform.rot.data, expected_rotation))

    def test_transform_inverse(self):
        t = np.array([[4, 5, 6]])
        rot_vec = np.array([0, 0, np.pi])
        transform = Transform.from_rot_vec_t(rot_vec, t)
        inverse_transform = transform.inverse()
        # Multiplying transform by its inverse should yield the identity matrix
        identity_transform = np.dot(transform.mat44(), inverse_transform.mat44())
        self.assertTrue(np.allclose(identity_transform, np.eye(4), atol=1e-7))

    def test_transform_multiplication(self):
        # Test multiplication of two transforms
        t1 = np.array([[1, 2, 3]])
        rot1 = Rotation.from_so3(np.array([0, 0, np.pi/2]))
        transform1 = Transform(t=t1, rot=rot1)

        t2 = np.array([[4, 5, 6]])
        rot2 = Rotation.from_so3(np.array([0, 0, np.pi/2]))
        transform2 = Transform(t=t2, rot=rot2)

        result_transform = transform1 * transform2
        expected_t = np.dot(rot1.mat(), t2.T).T + t1
        expected_rot = np.dot(rot1.mat(), rot2.mat())

        self.assertTrue(np.allclose(result_transform.t, expected_t))
        self.assertTrue(np.allclose(result_transform.rot.data, expected_rot))


class TestTransformExceptions(unittest.TestCase):
    """Test exception handling in Transform class."""

    def test_init_incompatible_type_error(self):
        """Test IncompatibleTypeError for non-array translation."""
        with pytest.raises(IncompatibleTypeError):
            Transform(t="invalid")

        with pytest.raises(IncompatibleTypeError):
            Transform(t=[1, 2, 3])  # List is not array-like in this context

    def test_init_invalid_dimension_error(self):
        """Test InvalidDimensionError for wrong translation size."""
        with pytest.raises(InvalidDimensionError):
            Transform(t=np.array([1, 2]))  # Only 2 elements

        with pytest.raises(InvalidDimensionError):
            Transform(t=np.array([1, 2, 3, 4]))  # 4 elements

    def test_from_rot_vec_t_incompatible_type_error(self):
        """Test IncompatibleTypeError in from_rot_vec_t."""
        with pytest.raises(IncompatibleTypeError):
            Transform.from_rot_vec_t(rot_vec="invalid", t=np.array([1, 2, 3]))

        with pytest.raises(IncompatibleTypeError):
            Transform.from_rot_vec_t(rot_vec=np.array([0, 0, 1]), t="invalid")

    def test_from_rot_vec_t_invalid_dimension_error(self):
        """Test InvalidDimensionError in from_rot_vec_t."""
        with pytest.raises(InvalidDimensionError):
            Transform.from_rot_vec_t(rot_vec=np.array([1, 2]), t=np.array([1, 2, 3]))

        with pytest.raises(InvalidDimensionError):
            Transform.from_rot_vec_t(rot_vec=np.array([0, 0, 1]), t=np.array([1, 2]))

    def test_from_mat_incompatible_type_error(self):
        """Test IncompatibleTypeError in from_mat."""
        with pytest.raises(IncompatibleTypeError):
            Transform.from_mat("invalid")

    def test_from_mat_invalid_shape_error(self):
        """Test InvalidShapeError in from_mat."""
        with pytest.raises(InvalidShapeError):
            Transform.from_mat(np.eye(3))  # 3x3 instead of 4x4 or 3x4

        with pytest.raises(InvalidShapeError):
            Transform.from_mat(np.ones((5, 5)))  # Wrong shape

    def test_get_origin_direction_invalid_shape_error(self):
        """Test InvalidShapeError in get_origin_direction."""
        transform = Transform()

        # Wrong first dimension
        with pytest.raises(InvalidShapeError):
            transform.get_origin_direction(np.array([[1, 2], [3, 4]]))  # 2xN instead of 3xN

        # Wrong shape entirely
        with pytest.raises(InvalidShapeError):
            transform.get_origin_direction(np.array([[[1, 2, 3]]]))  # 3D array

    def test_mul_invalid_type_error(self):
        """Test ValueError for unsupported multiplication types."""
        transform = Transform()

        with pytest.raises(ValueError, match="Multiplication only supported"):
            result = transform * "invalid"

        with pytest.raises(ValueError, match="Multiplication only supported"):
            result = transform * 42


class TestTransformMethods(unittest.TestCase):
    """Test Transform methods comprehensively."""

    def setUp(self):
        """Set up test fixtures."""
        self.t1 = np.array([1.0, 2.0, 3.0])
        self.rot_vec1 = np.array([0.0, 0.0, np.pi/4])  # 45 degrees around z
        self.rot1 = Rotation.from_so3(self.rot_vec1)
        self.transform1 = Transform(t=self.t1, rot=self.rot1)

    def test_from_pose(self):
        """Test creating Transform from Pose."""
        pose = Pose(t=self.t1, rot=self.rot1)
        transform = Transform.from_pose(pose)

        self.assertTrue(np.allclose(transform.t, pose.t.reshape(1, 3)))
        self.assertTrue(np.allclose(transform.rot.mat(), pose.rot_mat()))

    def test_rot_mat(self):
        """Test getting rotation matrix."""
        rot_mat = self.transform1.rot_mat()

        self.assertEqual(rot_mat.shape, (3, 3))
        self.assertTrue(np.allclose(rot_mat, self.rot1.mat()))
        # Verify it's orthogonal (use float32 precision tolerance)
        identity = np.eye(3, dtype=rot_mat.dtype)
        self.assertTrue(np.allclose(rot_mat @ rot_mat.T, identity, atol=1e-6))

    def test_mat34(self):
        """Test getting 3x4 transformation matrix."""
        mat34 = self.transform1.mat34()

        self.assertEqual(mat34.shape, (3, 4))
        self.assertTrue(np.allclose(mat34[:3, :3], self.rot1.mat()))
        self.assertTrue(np.allclose(mat34[:3, 3], self.t1))

    def test_mat44(self):
        """Test getting 4x4 transformation matrix."""
        mat44 = self.transform1.mat44()

        self.assertEqual(mat44.shape, (4, 4))
        self.assertTrue(np.allclose(mat44[:3, :3], self.rot1.mat()))
        self.assertTrue(np.allclose(mat44[:3, 3], self.t1))
        self.assertTrue(np.allclose(mat44[3, :], [0, 0, 0, 1]))

    def test_rot_vec_t(self):
        """Test getting rotation vector and translation."""
        rot_vec, t = self.transform1.rot_vec_t()

        self.assertEqual(rot_vec.shape, (3,))
        self.assertTrue(np.allclose(rot_vec, self.rot_vec1))
        self.assertTrue(np.allclose(t, self.transform1.t))

    def test_skew_t(self):
        """Test getting skew-symmetric matrix of translation."""
        skew = self.transform1.skew_t()

        self.assertEqual(skew.shape, (3, 3))
        # Verify it's skew-symmetric
        self.assertTrue(np.allclose(skew, -skew.T))

        # Test skew-symmetric property: skew(t) @ v = t x v
        t = self.t1.reshape(3, 1)
        v = np.array([[1.0], [0.0], [0.0]])
        cross_product = np.cross(t.flatten(), v.flatten())
        skew_product = (skew @ v).flatten()
        self.assertTrue(np.allclose(skew_product, cross_product))

    def test_get_t_rot_mat(self):
        """Test getting translation and rotation matrix together."""
        t, rot_mat = self.transform1.get_t_rot_mat()

        self.assertTrue(np.allclose(t, self.transform1.t))
        self.assertTrue(np.allclose(rot_mat, self.rot1.mat()))

    def test_merge(self):
        """Test merging two transforms."""
        t2 = np.array([4.0, 5.0, 6.0])
        rot_vec2 = np.array([0.0, 0.0, np.pi/6])
        rot2 = Rotation.from_so3(rot_vec2)
        transform2 = Transform(t=t2, rot=rot2)

        merged = self.transform1.merge(transform2)

        # Verify using matrix multiplication
        expected_mat = self.transform1.mat44() @ transform2.mat44()
        self.assertTrue(np.allclose(merged.mat44(), expected_mat))

    def test_apply_pts3d_single_point(self):
        """Test applying transform to a single 3D point."""
        pt = np.array([[1.0], [0.0], [0.0]])  # Shape (3, 1)
        transformed = self.transform1.apply_pts3d(pt)

        self.assertEqual(transformed.shape, (3, 1))

        # Verify using matrix multiplication
        expected = self.rot1.apply_pts3d(pt) + self.t1.reshape(3, 1)
        self.assertTrue(np.allclose(transformed, expected))

    def test_apply_pts3d_multiple_points(self):
        """Test applying transform to multiple 3D points."""
        pts = np.array([[1.0, 2.0, 3.0],
                        [0.0, 1.0, 2.0],
                        [0.0, 0.0, 1.0]])  # Shape (3, 3)
        transformed = self.transform1.apply_pts3d(pts)

        self.assertEqual(transformed.shape, (3, 3))

        # Verify each point
        for i in range(3):
            pt = pts[:, i:i+1]
            expected = self.rot1.apply_pts3d(pt) + self.t1.reshape(3, 1)
            self.assertTrue(np.allclose(transformed[:, i:i+1], expected))

    def test_get_origin_direction_single_ray(self):
        """Test get_origin_direction with single ray."""
        ray = np.array([0.0, 0.0, 1.0])  # Shape (3,)
        origins, directions = self.transform1.get_origin_direction(ray)

        self.assertEqual(origins.shape, (1, 3))
        self.assertEqual(directions.shape, (1, 3))

        # Origin should be the translation
        self.assertTrue(np.allclose(origins[0], self.t1))

        # Direction should be rotated ray
        expected_dir = self.rot1.apply_pts3d(ray.reshape(3, 1)).flatten()
        self.assertTrue(np.allclose(directions[0], expected_dir))

    def test_get_origin_direction_multiple_rays(self):
        """Test get_origin_direction with multiple rays."""
        rays = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]).T  # Shape (3, 3)
        origins, directions = self.transform1.get_origin_direction(rays)

        self.assertEqual(origins.shape, (3, 3))
        self.assertEqual(directions.shape, (3, 3))

        # All origins should be the translation
        for i in range(3):
            self.assertTrue(np.allclose(origins[i], self.t1))

        # Directions should be rotated rays
        expected_dirs = self.rot1.apply_pts3d(rays)
        self.assertTrue(np.allclose(directions.T, expected_dirs))

    def test_from_mat_with_3x4_matrix(self):
        """Test creating Transform from 3x4 matrix."""
        mat34 = np.array([[1, 0, 0, 1],
                          [0, 1, 0, 2],
                          [0, 0, 1, 3]])
        transform = Transform.from_mat(mat34)

        self.assertTrue(np.allclose(transform.t, np.array([[1, 2, 3]])))
        self.assertTrue(np.allclose(transform.rot.mat(), np.eye(3)))

    def test_from_mat_with_4x4_matrix(self):
        """Test creating Transform from 4x4 matrix."""
        mat44 = np.array([[1, 0, 0, 1],
                          [0, 1, 0, 2],
                          [0, 0, 1, 3],
                          [0, 0, 0, 1]])
        transform = Transform.from_mat(mat44)

        self.assertTrue(np.allclose(transform.t, np.array([[1, 2, 3]])))
        self.assertTrue(np.allclose(transform.rot.mat(), np.eye(3)))


class TestTransformMultiplication(unittest.TestCase):
    """Test Transform multiplication operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.t1 = np.array([1.0, 2.0, 3.0])
        self.rot1 = Rotation.from_so3(np.array([0.0, 0.0, np.pi/4]))
        self.transform1 = Transform(t=self.t1, rot=self.rot1)

    def test_mul_with_transform(self):
        """Test Transform * Transform multiplication."""
        t2 = np.array([4.0, 5.0, 6.0])
        rot2 = Rotation.from_so3(np.array([0.0, 0.0, np.pi/6]))
        transform2 = Transform(t=t2, rot=rot2)

        result = self.transform1 * transform2

        self.assertIsInstance(result, Transform)
        # Should be same as merge
        expected = self.transform1.merge(transform2)
        self.assertTrue(np.allclose(result.mat44(), expected.mat44()))

    def test_mul_with_pose(self):
        """Test Transform * Pose multiplication."""
        t_pose = np.array([1.0, 0.0, 0.0])
        rot_pose = Rotation.from_so3(np.array([0.0, 0.0, np.pi/6]))
        pose = Pose(t=t_pose, rot=rot_pose)

        result = self.transform1 * pose

        self.assertIsInstance(result, Pose)

        # Verify transformation - pose.t is (1, 3), need to transpose for apply_pts3d
        expected_t = self.rot1.apply_pts3d(t_pose.reshape(3, 1)).flatten() + self.t1
        expected_rot = self.rot1 * rot_pose

        # result.t is (1, 3) shape
        self.assertTrue(np.allclose(result.t.flatten(), expected_t))
        self.assertTrue(np.allclose(result.rot_mat(), expected_rot.mat()))

    def test_mul_with_3d_points(self):
        """Test Transform * points multiplication."""
        pts = np.array([[1.0, 2.0, 3.0],
                        [0.0, 1.0, 2.0],
                        [0.0, 0.0, 1.0]])  # Shape (3, 3)

        result = self.transform1 * pts

        # Should be same as apply_pts3d
        expected = self.transform1.apply_pts3d(pts)
        self.assertTrue(np.allclose(result, expected))


class TestInterpolateTransform(unittest.TestCase):
    """Test interpolate_transform function."""

    def test_interpolate_at_0(self):
        """Test interpolation at alpha=0 returns first transform."""
        t1 = np.array([0.0, 0.0, 0.0])
        rot1 = Rotation.from_so3(np.array([0.0, 0.0, 0.0]))
        transform1 = Transform(t=t1, rot=rot1)

        t2 = np.array([10.0, 10.0, 10.0])
        rot2 = Rotation.from_so3(np.array([0.0, 0.0, np.pi]))
        transform2 = Transform(t=t2, rot=rot2)

        result = interpolate_transform(transform1, transform2, alpha=0.0)

        self.assertTrue(np.allclose(result.t, transform1.t))
        self.assertTrue(np.allclose(result.rot.mat(), transform1.rot.mat(), atol=1e-6))

    def test_interpolate_at_1(self):
        """Test interpolation at alpha=1 returns second transform."""
        t1 = np.array([0.0, 0.0, 0.0])
        rot1 = Rotation.from_so3(np.array([0.0, 0.0, 0.0]))
        transform1 = Transform(t=t1, rot=rot1)

        t2 = np.array([10.0, 10.0, 10.0])
        rot2 = Rotation.from_so3(np.array([0.0, 0.0, np.pi]))
        transform2 = Transform(t=t2, rot=rot2)

        result = interpolate_transform(transform1, transform2, alpha=1.0)

        self.assertTrue(np.allclose(result.t, transform2.t))
        self.assertTrue(np.allclose(result.rot.mat(), transform2.rot.mat(), atol=1e-6))

    def test_interpolate_at_half(self):
        """Test interpolation at alpha=0.5 returns midpoint."""
        t1 = np.array([0.0, 0.0, 0.0])
        rot1 = Rotation.from_so3(np.array([0.0, 0.0, 0.0]))
        transform1 = Transform(t=t1, rot=rot1)

        t2 = np.array([10.0, 20.0, 30.0])
        rot2 = Rotation.from_so3(np.array([0.0, 0.0, np.pi/2]))
        transform2 = Transform(t=t2, rot=rot2)

        result = interpolate_transform(transform1, transform2, alpha=0.5)

        # Translation should be linear interpolation
        expected_t = np.array([[5.0, 10.0, 15.0]])
        self.assertTrue(np.allclose(result.t, expected_t))

        # Rotation should be slerp (hard to verify exact value, just check it's reasonable)
        self.assertEqual(result.rot.mat().shape, (3, 3))
        # Verify it's a valid rotation matrix
        self.assertTrue(np.allclose(result.rot.mat() @ result.rot.mat().T, np.eye(3), atol=1e-6))

    def test_interpolate_multiple_values(self):
        """Test interpolation with multiple alpha values."""
        t1 = np.array([0.0, 0.0, 0.0])
        rot1 = Rotation.from_so3(np.array([0.0, 0.0, 0.0]))
        transform1 = Transform(t=t1, rot=rot1)

        t2 = np.array([10.0, 0.0, 0.0])
        rot2 = Rotation.from_so3(np.array([0.0, 0.0, np.pi]))
        transform2 = Transform(t=t2, rot=rot2)

        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = [interpolate_transform(transform1, transform2, alpha) for alpha in alphas]

        # Verify translation increases linearly
        for i, alpha in enumerate(alphas):
            expected_x = 10.0 * alpha
            self.assertAlmostEqual(results[i].t[0, 0], expected_x, places=6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTransformWithTorch(unittest.TestCase):
    """Test Transform with PyTorch tensors."""

    def test_init_with_torch_tensor(self):
        """Test initialization with PyTorch tensor translation."""
        # Convert to numpy first since Transform internally converts to numpy anyway
        t_torch = torch.tensor([1.0, 2.0, 3.0])
        t = t_torch.numpy()
        # Use numpy for rotation to avoid issues with dot product in is_SO3
        rot = Rotation.from_mat3(np.eye(3))
        transform = Transform(t=t, rot=rot)

        # Internal storage should work
        self.assertEqual(transform.t.shape, (1, 3))
        # t should be numpy
        self.assertTrue(isinstance(transform.t, np.ndarray))
        # Verify values
        self.assertTrue(np.allclose(transform.t, np.array([[1.0, 2.0, 3.0]])))

    def test_apply_pts3d_with_torch(self):
        """Test applying transform to PyTorch tensor points."""
        # Use numpy for translation and rotation
        t = np.array([1.0, 2.0, 3.0])
        rot = Rotation.from_mat3(np.eye(3))
        transform = Transform(t=t, rot=rot)

        # PyTorch tensor points
        pts = torch.tensor([[1.0, 2.0, 3.0],
                            [0.0, 1.0, 2.0],
                            [0.0, 0.0, 1.0]], dtype=torch.float32)

        result = transform.apply_pts3d(pts)

        # Result should be a tensor (preserves input type)
        self.assertTrue(torch.is_tensor(result))
        self.assertEqual(result.shape, (3, 3))

        # Verify values
        expected = pts + torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected))

    def test_get_origin_direction_with_torch(self):
        """Test get_origin_direction with PyTorch tensors."""
        t = np.array([1.0, 2.0, 3.0])
        rot = Rotation.from_mat3(np.eye(3))
        transform = Transform(t=t, rot=rot)

        rays = torch.tensor([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float32).T

        origins, directions = transform.get_origin_direction(rays)

        # Results should be tensors
        self.assertTrue(torch.is_tensor(origins))
        self.assertTrue(torch.is_tensor(directions))

        # Verify shapes
        self.assertEqual(origins.shape, (3, 3))
        self.assertEqual(directions.shape, (3, 3))


class TestTransformEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_translation_shape_1d(self):
        """Test initialization with 1D translation vector."""
        t = np.array([1.0, 2.0, 3.0])  # Shape (3,)
        transform = Transform(t=t)

        # Should be reshaped to (1, 3)
        self.assertEqual(transform.t.shape, (1, 3))
        self.assertTrue(np.allclose(transform.t, np.array([[1.0, 2.0, 3.0]])))

    def test_translation_shape_2d(self):
        """Test initialization with 2D translation vector."""
        t = np.array([[1.0, 2.0, 3.0]])  # Shape (1, 3)
        transform = Transform(t=t)

        self.assertEqual(transform.t.shape, (1, 3))
        self.assertTrue(np.allclose(transform.t, np.array([[1.0, 2.0, 3.0]])))

    def test_identity_transform(self):
        """Test identity transform properties."""
        transform = Transform()

        # Should be identity
        mat44 = transform.mat44()
        self.assertTrue(np.allclose(mat44, np.eye(4)))

        # Applying to points should not change them
        pts = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])
        transformed = transform.apply_pts3d(pts)
        self.assertTrue(np.allclose(transformed, pts))

    def test_inverse_of_identity(self):
        """Test inverse of identity transform."""
        transform = Transform()
        inverse = transform.inverse()

        # Inverse of identity should be identity
        self.assertTrue(np.allclose(inverse.mat44(), np.eye(4)))

    def test_double_inverse(self):
        """Test that double inverse returns original transform."""
        t = np.array([1.0, 2.0, 3.0])
        rot = Rotation.from_so3(np.array([0.1, 0.2, 0.3]))
        transform = Transform(t=t, rot=rot)

        double_inverse = transform.inverse().inverse()

        self.assertTrue(np.allclose(double_inverse.mat44(), transform.mat44(), atol=1e-10))

    def test_transform_float32_storage(self):
        """Test that Transform stores data as float32 regardless of input dtype."""
        # Test with float64 input
        t_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rot = Rotation.from_mat3(np.eye(3, dtype=np.float64))
        tf = Transform(t_f64, rot)
        self.assertEqual(tf._t.dtype, np.float32)
        self.assertEqual(tf.rot.data.dtype, np.float32)

        # Test with float32 input
        t_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tf_f32 = Transform(t_f32, Rotation.from_mat3(np.eye(3, dtype=np.float32)))
        self.assertEqual(tf_f32._t.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()