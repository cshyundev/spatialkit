import unittest
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation
from cv_utils.core.geom.rotation import *

class TestRotation(unittest.TestCase):

    def test_rotation_from_so3(self):
        so3 = np.array([0.1, 0.2, 0.3])
        rotation = Rotation.from_so3(so3)
        self.assertTrue(is_SO3(rotation.mat()))

    def test_rotation_from_quat_xyzw(self):
        quat = np.array([0.707, 0, 0, 0.707])
        rotation = Rotation.from_quat_xyzw(quat)
        self.assertTrue(is_SO3(rotation.mat()))

    def test_rotation_from_quat_wxyz(self):
        quat = np.array([0.707, 0, 0, 0.707])
        rotation = Rotation.from_quat_wxyz(quat)
        self.assertTrue(is_SO3(rotation.mat()))

    def test_rotation_from_rpy(self):
        rpy = np.array([0.1, 0.2, 0.3])
        rotation = Rotation.from_rpy(rpy)
        self.assertTrue(is_SO3(rotation.mat()))

    def test_quat_to_SO3_and_back(self):
        quat = np.array([0.707, 0, 0, 0.707])
        rotation = Rotation.from_quat_xyzw(quat)
        quat_back = rotation.quat()
        np.testing.assert_almost_equal(quat, quat_back, decimal=5)

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
        quat = scipy_rotation.as_quat() # SciPy uses [x, y, z, w]
        rotation = Rotation.from_quat_xyzw(quat)
        np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=5)
        np.testing.assert_almost_equal(rotation.quat(), quat, decimal=5)

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
            np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=5)
            np.testing.assert_almost_equal(rotation.rpy(), scipy_rotation.as_euler('xyz'), decimal=5)
            
            quat = scipy_rotation.as_quat()
            rotation = Rotation.from_quat_xyzw(quat)
            np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=5)
            np.testing.assert_almost_equal(rotation.quat(), quat, decimal=5)

            so3 = scipy_rotation.as_rotvec()
            rotation = Rotation.from_so3(so3)
            np.testing.assert_almost_equal(rotation.mat(), scipy_rotation.as_matrix(), decimal=5)
            np.testing.assert_almost_equal(rotation.so3(), so3, decimal=5)

if __name__ == '__main__':
    unittest.main()