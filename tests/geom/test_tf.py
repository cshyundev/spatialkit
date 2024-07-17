import unittest
import numpy as np
from cv_utils.core.geom.pose import Pose
from cv_utils.core.geom.tf import Transform
from cv_utils.core.geom.rotation import Rotation
from cv_utils.core.ops.uops import *
from cv_utils.core.ops.umath import *

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

if __name__ == '__main__':
    unittest.main()