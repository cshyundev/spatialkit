"""
Module Name: pose.py

Description:
This module provides a Pose class that supports various representations and transformations of 3D poses.
It is commonly used in 3D Vision, Robotics, and related fields to convert between different pose representations.

Pose Types:
- SE3: Special Euclidean group in 3D.
- se3: Lie algebra of SE3.
- NONE: No Pose Type.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

License: MIT LICENSE
"""
import numpy as np
from .rotation import Rotation, SO3_to_so3, slerp, so3_to_SO3
from ..ops.uops import *
from ..ops.umath import *


def SE3_to_se3(SE3: ArrayLike) -> ArrayLike:
    """
    Convert SE3 matrix to se3 vector.
    Args:
        SE3: (4,4), float, SE3 transformation matrix
    Returns:
        se3: (6,), float, se3 vector
    """
    R = SE3[:3,:3]
    t = SE3[:3,3]
    so3 = SO3_to_so3(R)    
    se3 = concat([so3, t], dim=0)
    return se3

def se3_to_SE3(se3: ArrayLike) -> ArrayLike:
    """
    Convert se3 vector to SE3 matrix.
    Args:
        se3: (6,), float, se3 vector (3 for so3 and 3 for translation)
    Returns:
        SE3: (4,4), float, SE3 transformation matrix
    """
    assert se3.shape == (6,), 'se3 vector must be of shape (6,)'

    so3 = se3[:3]
    t = se3[3:]

    R = so3_to_SO3(so3)
    SE3 = eye(4,se3)
    SE3[:3, :3] = R
    SE3[:3, 3] = t

    return SE3

class Pose:
    """
        Pose Class representing a 3D pose with position(x,y,z) and orientation.
        This class encapsulates both the translation and rotation components of a pose.

        Attributes:
            t (np.ndarray): Translation vector of shape (1, 3)
            rot (Rotation): Rotation instance represented by a Rotation object
    """
    def __init__(self, t:Optional[ArrayLike]=None, rot:Optional[Rotation]=None):
        """
            Initialization Pose Instance
            Args:
                t (ArrayLike, [3,] or [1,3]): translation vector
                Rotation: Rotation Instance
        """
        if t is None: t = np.array([0.,0.,0.])
        if rot is None: rot = Rotation.from_mat3(np.eye(3))

        assert (is_array(t)), ('translation must be array type(Tensor or Numpy).')
        assert(t.size == 3), ('size of translation must be 3.')

        t = t.reshape(1,3)
        
        self._t:ArrayLike = convert_numpy(t)
        self._rot:Rotation = rot      
    
    @property
    def t(self) -> np.ndarray:
        return self._t
    
    @staticmethod
    def from_rot_vec_t(rot_vec: ArrayLike, t:ArrayLike) -> 'Pose':
        assert(is_array(rot_vec)), 'rotation vector must be ArrayLike type(Tensor or Numpy)'
        assert(is_array(t)), 'translation vector must be ArrayLike type(Tensor or Numpy)'
        assert(rot_vec.shape[-1] == 3), 'Invalid Shape. rotation vector must be (3,)'
        rot = Rotation.from_so3(rot_vec)
        return Pose(t,rot)

    @staticmethod
    def from_mat(mat4: ArrayLike) -> 'Pose':
        assert is_array(mat4), ('mat4 must be array type(Tensor or Numpy)')
        assert((mat4.shape[-1] == 4) and (mat4.shape[-2] == 3 or mat4.shape[-2] == 4)), ('Invalid Shape. The shape of mat4 must be (3,4) or (4,4)')
        return Pose(mat4[0:3,3], Rotation.from_mat3(mat4[:3,:3]))
        
    def rot_mat(self) -> np.ndarray:
        return self._rot.mat()
    
    def mat34(self) -> np.ndarray:
        return concat([self.rot_mat(), transpose2d(self.t)], dim=1)
    
    def mat44(self) -> np.ndarray:
        mat34 = self.mat34()
        last = np.array([0., 0., 0., 1.]).reshape(1,4)
        return concat([mat34, last], dim=0)
    
    def rot_vec_t(self) -> Tuple[np.ndarray,np.ndarray]:
        return self._rot.so3(), self.t
    
    def skew_t(self) -> np.ndarray:
        return vec3_to_skew(self.t)
    
    def get_t_rot_mat(self)-> Tuple[np.ndarray,np.ndarray]:
        return self.t, self.rot_mat()
    
    def inverse(self) -> 'Pose':
        R_inv = self._rot.inverse()
        t_inv = -R_inv.apply_pts3d(transpose2d(self.t))
        return Pose(transpose2d(t_inv), R_inv)

def interpolate_pose(pose1:Pose,pose2:Pose,t:float) -> Pose:
    """
        Interpolate Pose using Lerp and SLerp.
        
        Args:
            pose1 (Pose): start Pose
            pose2 (Pose): end Pose
            t (float): interpolation parameter
        
        Return:
            Interpolated Pose

        Details:
        - translation: linear interpolation(Lerp)
        - rotation: spherical linear interpolation(Slerp)
    """
    r = slerp(pose1._rot,pose2._rot,t)
    trans1, trans2 = pose1.t, pose2.t
    trans = trans1 * (1.- t) + trans2 * t
    return Pose(t=trans,rot=r)

def relative_pose(pose1:Pose, pose2:Pose) -> Pose:
    """
        Calculate the relative pose from pose1 to pose2.
        
        Args:
            pose1(Pose): the first pose (reference pose)
            pose2(Pose): the second pose
        
        Return:
            Pose, the relative pose from pose1 to pose2
    """
    pose1_inv = pose1.inverse()
    
    rel_rot = pose1_inv._rot * pose2._rot
    rel_t = pose1_inv.t + pose1_inv._rot.apply_pts3d(pose2.t)
    
    return Pose(rel_t, rel_rot)    