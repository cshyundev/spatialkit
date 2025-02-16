"""
Module Name: tf.py

Description:
This module provides a Transformation class that supports various operations for 3D transformations.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

License: MIT LICENSE
"""
import numpy as np
from .rotation import Rotation, slerp
from .pose import Pose
from ..ops.uops import *
from ..ops.uops import ArrayLike
from ..ops.umath import *


class Transform:
    """
        Transform Class representing a 3D transformation with position and orientation.
        This class encapsulates both the translation and rotation components of a transformation.

        Attributes:
            t (np.ndarray, [1,3] or [3,]): Translation vector
            rot (Rotation): Rotation instance
    """
    def __init__(self, t:Optional[ArrayLike]=None, rot:Optional[Rotation]=None):
        if t is None: t = np.array([0., 0., 0.])
        if rot is None: rot = Rotation.from_mat3(np.eye(3))

        assert is_array(t), 'Translation must be an array type (Tensor or Numpy).'
        assert t.size == 3, 'Size of translation must be 3.'

        t = t.reshape(1,3)
        
        self.t: ArrayLike = t
        self.rot: Rotation = rot
        
    @staticmethod
    def from_rot_vec_t(rot_vec: ArrayLike, t: ArrayLike) -> 'Transform':
        """
            Create a Transform from a rotation vector and translation vector.
            
            Args:
                rot_vec (ArrayLike, [3,]): Rotation vector
                t (ArrayLike, [3,]): translation vector
            
            Return:
                Transform: Transform instance
        """
        assert is_array(rot_vec), 'Rotation vector must be ArrayLike type (Tensor or Numpy)'
        assert is_array(t), 'Translation vector must be ArrayLike type (Tensor or Numpy)'
        assert rot_vec.size == 3, 'Invalid Shape. Rotation vector must be 3 size'
        assert t.size == 3, 'Invalid Shape. Translation vector must be 3 size'
        rot = Rotation.from_so3(rot_vec.reshape(-1,))
        return Transform(t, rot)

    @staticmethod
    def from_mat(mat4: ArrayLike) -> 'Transform':
        """
            Create a Transform from a 4x4 transformation matrix.
            
            Args:
                mat4 (ArrayLike, [4,4] or [3,4]): transformation matrix
            
            Return:
                Transform: Transform instance
        """
        assert is_array(mat4), 'mat4 must be array type (Tensor or Numpy)'
        assert mat4.shape == (4, 4) or mat4.shape == (3, 4), 'Invalid Shape. The shape of mat4 must be (4,4) or (3,4)'
        return Transform(mat4[:3, 3], Rotation.from_mat3(mat4[:3, :3]))
    
    def from_pose(pose: Pose) -> 'Transform':
        return Transform(pose.t,Rotation.from_mat3(pose.rot_mat())) 

    def rot_mat(self) -> np.ndarray:
        """
        Get the rotation matrix of the transform.
        
        Return:
            rot_mat (np.ndarray, [3,3]): Rotation matrix
        """
        return self.rot.mat()
    
    def mat34(self) -> np.ndarray:
        """
            Get the 3x4 transformation matrix of the transform.
            
            Return:
                tf_mat (np.ndarray, [3,4]): Transformation matrix
        """
        return concat([self.rot_mat(), transpose2d(self.t)], dim=1)
    
    def mat44(self) -> np.ndarray:
        """
            Get the 4x4 transformation matrix of the transform.

            Return:
                tf_mat (np.ndarray, [4,4]): Transformation matrix
        """
        mat34 = self.mat34()
        last = np.array([0., 0., 0., 1.]).reshape(1, 4)
        return concat([mat34, last], dim=0)
    
    def rot_vec_t(self) -> Tuple[np.ndarray,np.ndarray]:
        """
            Get the rotation vector and translation vector of the transform.
            Returns:
                rot_vec (np.ndarray, [3,]): Rotation vector
                t (np.ndarray, [3,]): Translation vector
        """
        return self.rot.so3(), self.t
     
    def skew_t(self) -> np.ndarray:
        """
            Get the skew-symmetric matrix of the translation vector.
            
            Return:
                skew_mat(np.ndarray, [3,3]): Skew-symmetric matrix
            
            Details:
            - t = [tx,ty,tz]
            - skew_x = |0 -tz  ty|
                       |tz  0 -tx|
                       |-ty tx  0|
        """
        return vec3_to_skew(self.t)
    
    def get_t_rot_mat(self) -> Tuple[np.ndarray,np.ndarray]:
        """
            Get the translation vector and rotation matrix of the transform.
            
            Returns:
                t (np.ndarray, [3,]): Translation vector
                rot_mat (np.ndarray, [3,3]): Rotation vector
        """
        return self.t, self.rot_mat()
    
    def inverse(self) -> 'Transform':
        """
            Get the inverse of the current transform.
            
            Return:
                Transform: Inverse transform
        """
        R_inv = self.rot.inverse()
        t_inv = -R_inv.apply_pts3d(self.t.T)
        return Transform(t_inv.T, R_inv)
    
    def get_origin_direction(self, rays: ArrayLike):
        """
            Get the origin and direction vectors from rays (local coordinates).
            
            Args:
                rays (ArrayLike, [3,N]): camera rays from origin
            
            Returns:
                origins (ArrayLike, [N,3]): origin in world coordinates
                directions (ArrayLike, [N,3]): unit direction vector in world coordinates
        """
        assert len(rays.shape) <= 2 and rays.shape[0] == 3, "Invalid Shape. Ray's shape must be (3, n) or (3)"
        if len(rays.shape) == 2:
            n_rays = rays.shape[1]
        else:
            rays = expand_dim(rays, 1)
            n_rays = 1
        origin = np.tile(convert_numpy(self.t),(n_rays, 1))
        origin = convert_array(origin, rays)
        direction = transpose2d(self.rot.apply_pts3d(rays)).reshape((-1, 3))
        return origin, direction

    def merge(self, transform: 'Transform') -> 'Transform':
        """
            Merge the current transform with another transform.
            
            Args:
                transform (Transform):  another transform to merge with
            
            Return:
                Transform: Merged transform
        """
        mat4 = self.mat44() @ transform.mat44()
        t = mat4[:3, 3]
        rot = Rotation.from_mat3(mat4[:3, :3])
        return Transform(t, rot)
    
    def apply_pts3d(self, pts3d: ArrayLike) -> ArrayLike:
        """
            Apply the transform to 3D points.
            
            Args:
                pts3d (ArrayLike, [3,N]): 3D points
            
            Return:
                pts3d (ArrayLike, [3,N]) : Transformed 3D points
        """
        t = transpose2d(convert_array(self.t, pts3d))
        return self.rot.apply_pts3d(pts3d) + t

    def __mul__(self, other:Any):
        """
            Define the multiplication operation for Transform, Pose, and 3D points.

            Arg:
                other (Any): The object to multiply with. Can be one of the following:
                    - Transform: An instance of the Transform class.
                    - Pose: An instance of the Pose class.
                    - ArrayLike: A 3D points array.

            Return:
                Transform, Pose, or ArrayLike: Result of the multiplication.

            Raises:
                ValueError: If the multiplication is attempted with an unsupported type.

            Details:
            - If other is a Transform, the result is a merged Transform.
            - If other is a Pose, the result is a new Pose with combined translation and rotation.
            - If other is a 3D points array, the result is the application of the transformation to the 3D points.
    """
        if isinstance(other, Transform):
            return self.merge(other)
        elif isinstance(other, Pose):
            t_new = self.rot.apply_pts3d(other.t) + self.t
            r_new = self.rot * other._rot
            return Pose(t_new, r_new)
        elif is_array(other):
            return self.apply_pts3d(other)
        else:
            raise ValueError("Multiplication only supported with Transform, Pose or 3D points.")
    
def interpolate_transform(t1: Transform, t2: Transform, alpha: float) -> Transform:
    """
        Interpolate between two transforms using linear interpolation (Lerp) and spherical linear interpolation (Slerp).
        
        Args:
            t1 (Transform): start transform
            t2 (Transform): end transform
            alpha  (float): interpolation parameter
        
        Return:
            Transform: Interpolated transform
    """
    r = slerp(t1.rot, t2.rot, alpha)
    trans1, trans2 = t1.t, t2.t
    trans = trans1 * (1. - alpha) + trans2 * alpha
    return Transform(t=trans, rot=r) 