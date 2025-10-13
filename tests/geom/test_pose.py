import unittest
import numpy as np
from cv_utils.geom.pose import SE3_to_se3, se3_to_SE3, Pose
from cv_utils.geom.rotation import Rotation
from cv_utils.ops.uops import *
from cv_utils.ops.umath import *


class TestPose(unittest.TestCase):

    def test_SE3_to_se3_conversion(self):
        # Create a sample SE3 matrix
        R = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
        t = np.array([1, 2, 3])
        SE3 = np.eye(4)
        SE3[:3, :3] = R
        SE3[:3, 3] = t

        # Convert to se3
        se3 = SE3_to_se3(SE3)
        expected_so3 = np.array([0, 0, np.pi/2])  # Example rotation vector for 90 deg about z
        expected_se3 = np.concatenate([expected_so3, t])

        # Verify the se3 vector is correct
        np.testing.assert_array_almost_equal(se3, expected_se3, decimal=5)

    def test_se3_to_SE3_conversion(self):
        # Create a sample se3 vector
        so3 = np.array([0, 0, np.pi/2])
        t = np.array([1, 2, 3])
        se3 = np.concatenate([so3, t])

        # Convert to SE3
        SE3 = se3_to_SE3(se3)
        expected_R = np.array([[0, -1, 0],
                               [1, 0, 0],
                               [0, 0, 1]])
        expected_SE3 = np.eye(4)
        expected_SE3[:3, :3] = expected_R
        expected_SE3[:3, 3] = t

        # Verify the SE3 matrix is correct
        np.testing.assert_array_almost_equal(SE3, expected_SE3, decimal=5)

    def test_pose_initialization(self):
        # Test default initialization
        default_pose = Pose()
        np.testing.assert_array_almost_equal(default_pose.t, np.array([[0., 0., 0.]]))
        np.testing.assert_array_equal(default_pose.rot.mat(), np.eye(3))

        # Test initialization with specific translation and rotation
        translation = np.array([[1, 2, 3]])
        rotation = Rotation.from_mat3(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
        custom_pose = Pose(t=translation, rot=rotation)
        np.testing.assert_array_equal(custom_pose.t, translation)
        np.testing.assert_array_equal(custom_pose.rot.mat(), rotation.mat())

    def test_pose_from_rot_vec_t(self):
        # Create a Pose from rotation vector and translation
        rot_vec = np.array([0, 0, np.pi/2])
        t = np.array([1, 2, 3])
        pose = Pose.from_rot_vec_t(rot_vec, t)
        expected_rotation = Rotation.from_so3(rot_vec)

        np.testing.assert_array_almost_equal(pose.t, t.reshape(1,3))
        np.testing.assert_array_almost_equal(pose.rot.mat(), expected_rotation.mat())

    def test_pose_from_mat(self):
        # Create a Pose from a 4x4 transformation matrix
        mat = np.array([[0, -1, 0, 1],
                        [1, 0, 0, 2],
                        [0, 0, 1, 3],
                        [0, 0, 0, 1]])
        pose = Pose.from_mat(mat)
        expected_translation = np.array([[1, 2, 3]])
        expected_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        np.testing.assert_array_almost_equal(pose.t, expected_translation)
        np.testing.assert_array_almost_equal(pose.rot.mat(), expected_rotation)

    def test_pose_operations(self):
        # Test methods like mat34, mat44, and inverse
        t = np.array([1, 2, 3])
        rot_vec = np.array([0, 0, np.pi/2])
        pose = Pose.from_rot_vec_t(rot_vec, t)

        # mat34 and mat44
        mat34_expected = np.array([[0, -1, 0, 1],
                                   [1, 0, 0, 2],
                                   [0, 0, 1, 3]])
        mat44_expected = np.array([[0, -1, 0, 1],
                                   [1, 0, 0, 2],
                                   [0, 0, 1, 3],
                                   [0, 0, 0, 1]])

        np.testing.assert_array_almost_equal(pose.mat34(), mat34_expected)
        np.testing.assert_array_almost_equal(pose.mat44(), mat44_expected)

        # Inverse
        pose_inv = pose.inverse()
        # Test the inverse by confirming that multiplying pose by its inverse yields the identity matrix
        R_inv = pose_inv.rot.mat()
        t_inv = transpose2d(pose_inv.t)
        R = pose.rot.mat()
        t = transpose2d(pose.t)

        # Compute the full 4x4 matrix multiplication
        pose_mat = concat([concat([R, t], 1), np.array([[0, 0, 0, 1]])], 0)
        pose_inv_mat = concat([concat([R_inv, t_inv], 1), np.array([[0, 0, 0, 1]])], 0)
        identity_mat = np.dot(pose_mat, pose_inv_mat)

        np.testing.assert_array_almost_equal(identity_mat, np.eye(4), decimal=5)

    def test_pose_float32_storage(self):
        """Test that Pose stores data as float32 regardless of input dtype."""
        # Test with float64 input
        t_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rot = Rotation.from_mat3(np.eye(3, dtype=np.float64))
        pose = Pose(t_f64, rot)
        self.assertEqual(pose._t.dtype, np.float32)
        self.assertEqual(pose.rot.data.dtype, np.float32)

        # Test with float32 input
        t_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pose_f32 = Pose(t_f32, Rotation.from_mat3(np.eye(3, dtype=np.float32)))
        self.assertEqual(pose_f32._t.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()