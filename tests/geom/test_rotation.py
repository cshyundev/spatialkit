import unittest
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

# Use new hierarchical import pattern
from cv_utils import Rotation, RotType
from cv_utils.geom.rotation import is_SO3, slerp

class TestRotation(unittest.TestCase):

    def test_rotation_from_so3(self):
        so3 = np.array([0.1, 0.2, 0.3])
        rotation = Rotation.from_so3(so3)
        self.assertTrue(is_SO3(rotation.mat()))

    def test_rotation_from_quat_xyzw(self):
        quat = np.array([0.707, 0, 0, 0.707])
        quat = quat / np.linalg.norm(quat)  # Normalize to unit quaternion
        rotation = Rotation.from_quat_xyzw(quat)
        self.assertTrue(is_SO3(rotation.mat()))

    def test_rotation_from_quat_wxyz(self):
        quat = np.array([0.707, 0, 0, 0.707])
        quat = quat / np.linalg.norm(quat)  # Normalize to unit quaternion
        rotation = Rotation.from_quat_wxyz(quat)
        self.assertTrue(is_SO3(rotation.mat()))

    def test_rotation_from_rpy(self):
        rpy = np.array([0.1, 0.2, 0.3])
        rotation = Rotation.from_rpy(rpy)
        self.assertTrue(is_SO3(rotation.mat()))

    def test_quat_to_SO3_and_back(self):
        quat_xyzw = np.array([0.707, 0, 0, 0.707])
        quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)  # Normalize to unit quaternion
        rotation = Rotation.from_quat_xyzw(quat_xyzw)
        quat_wxyz_back = rotation.quat()  # Returns wxyz format
        # Convert xyzw to wxyz for comparison
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        np.testing.assert_almost_equal(quat_wxyz, quat_wxyz_back, decimal=5)

    def test_rpy_to_SO3_and_back(self):
        rpy = np.array([0.1, 0.2, 0.3])
        rotation = Rotation.from_rpy(rpy)
        rpy_back = rotation.rpy()
        np.testing.assert_almost_equal(rpy, rpy_back, decimal=5)

    def test_SO3_multiplication(self):
        rpy1 = np.array([0.1, 0.2, 0.3])
        rpy2 = np.array([0.4, 0.5, 0.6])
        rotation1 = Rotation.from_rpy(rpy1)
        rotation2 = Rotation.from_rpy(rpy2)
        result_rotation = rotation1 * rotation2
        expected_rotation = rotation1.dot(rotation2)
        np.testing.assert_almost_equal(result_rotation.mat(), expected_rotation.mat(), decimal=5)

    def test_inverse(self):
        rpy = np.array([0.1, 0.2, 0.3])
        rotation = Rotation.from_rpy(rpy)
        inverse_rotation = rotation.inverse()
        identity = rotation * inverse_rotation
        np.testing.assert_almost_equal(identity.mat(), np.eye(3), decimal=5)

    def test_slerp(self):
        rpy1 = np.array([0.1, 0.2, 0.3])
        rpy2 = np.array([0.4, 0.5, 0.6])
        rotation1 = Rotation.from_rpy(rpy1)
        rotation2 = Rotation.from_rpy(rpy2)
        result_rotation = slerp(rotation1, rotation2, 0.5)
        rpy_middle = (rpy1 + rpy2) / 2
        np.testing.assert_almost_equal(result_rotation.rpy(), rpy_middle, decimal=1)

    def test_with_scipy_rotation(self):
        rpy = np.array([0.1, 0.2, 0.3])
        
        # Test RPY to SO3 and back
        rotation = Rotation.from_rpy(rpy)
        scipy_rotation = SciPyRotation.from_euler('xyz', rpy)
        
        np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=5)
        np.testing.assert_almost_equal(rotation.rpy(), scipy_rotation.as_euler('xyz'), decimal=5)
        
        # Test Quaternion to SO3 and back
        quat_xyzw = scipy_rotation.as_quat() # SciPy uses [x, y, z, w]
        rotation = Rotation.from_quat_xyzw(quat_xyzw)
        np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=5)
        # Convert xyzw to wxyz for comparison with rotation.quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        np.testing.assert_almost_equal(rotation.quat(), quat_wxyz, decimal=5)

        # Test SO3 to so3 and back
        so3 = scipy_rotation.as_rotvec()
        rotation = Rotation.from_so3(so3)
        np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=5)
        np.testing.assert_almost_equal(rotation.so3(), so3, decimal=5)

    def test_random_rotations(self):
        for _ in range(10):
            rpy = np.random.uniform(-np.pi, np.pi, 3)
            rotation = Rotation.from_rpy(rpy)
            scipy_rotation = SciPyRotation.from_euler('xyz', rpy)
            # Use decimal=4 for float32 precision (Rotation now uses float32 internally)
            np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=4)
            np.testing.assert_almost_equal(rotation.rpy(), scipy_rotation.as_euler('xyz'), decimal=4)

            quat_xyzw = scipy_rotation.as_quat()
            rotation = Rotation.from_quat_xyzw(quat_xyzw)
            np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=4)
            # Convert xyzw to wxyz for comparison with rotation.quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            # Handle quaternion sign ambiguity: both q and -q represent the same rotation
            quat_back = rotation.quat()
            if np.dot(quat_back, quat_wxyz) < 0:
                quat_wxyz = -quat_wxyz
            np.testing.assert_almost_equal(quat_back, quat_wxyz, decimal=4)

            so3 = scipy_rotation.as_rotvec()
            rotation = Rotation.from_so3(so3)
            np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=4)
            np.testing.assert_almost_equal(rotation.so3(), so3, decimal=4)

    def test_rotation_float32_storage(self):
        """Test that Rotation stores data as float32 regardless of input dtype."""
        # Test with float64 input
        rot_f64 = Rotation.from_mat3(np.eye(3, dtype=np.float64))
        self.assertEqual(rot_f64.data.dtype, np.float32)

        # Test with float32 input
        rot_f32 = Rotation.from_mat3(np.eye(3, dtype=np.float32))
        self.assertEqual(rot_f32.data.dtype, np.float32)

        # Test with quaternion input (float64)
        quat_f64 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        rot_quat = Rotation.from_quat_wxyz(quat_f64)
        self.assertEqual(rot_quat.data.dtype, np.float32)

        # Test with RPY input (float64)
        rpy_f64 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        rot_rpy = Rotation.from_rpy(rpy_f64)
        self.assertEqual(rot_rpy.data.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()