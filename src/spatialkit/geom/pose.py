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

from typing import Optional, Tuple
import numpy as np

from .rotation import Rotation, SO3_to_so3, slerp, so3_to_SO3
from ..ops.uops import *
from ..ops.umath import *
from ..common.exceptions import InvalidShapeError, InvalidDimensionError, IncompatibleTypeError


def SE3_to_se3(SE3: ArrayLike) -> ArrayLike:
    """
    Convert SE3 matrix to se3 vector.
    Args:
        SE3: (4,4), float, SE3 transformation matrix
    Returns:
        se3: (6,), float, se3 vector
    """
    R = SE3[:3, :3]
    t = SE3[:3, 3]
    so3 = SO3_to_so3(R)
    se3 = concat([so3, t], dim=0)
    return se3


def se3_to_SE3(se3: ArrayLike) -> ArrayLike:
    """
    Convert se3 vector to SE3 matrix.
    
    Args:
        se3 (ArrayLike): se3 vector (6,) with first 3 elements as so3 and last 3 as translation.
        
    Returns:
        ArrayLike: SE3 transformation matrix (4,4).
        
    Raises:
        InvalidShapeError: If se3 vector is not of shape (6,).
    """
    if se3.shape != (6,):
        raise InvalidShapeError(
            f"se3 vector must have shape (6,), got {se3.shape}. "
            f"Expected format: [so3_x, so3_y, so3_z, t_x, t_y, t_z]."
        )

    so3 = se3[:3]
    t = se3[3:]

    R = so3_to_SO3(so3)
    SE3 = eye(4, se3)
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

    def __init__(self, t: Optional[ArrayLike] = None, rot: Optional[Rotation] = None):
        """
        Initialize Pose Instance.
        
        Args:
            t (ArrayLike, optional): Translation vector of shape (3,) or (1,3). Defaults to zero vector.
            rot (Rotation, optional): Rotation instance. Defaults to identity rotation.
            
        Raises:
            IncompatibleTypeError: If translation is not array-like.
            InvalidDimensionError: If translation size is not 3.
        """
        if t is None:
            t = np.array([0.0, 0.0, 0.0])
        if rot is None:
            rot = Rotation.from_mat3(np.eye(3))

        if not is_array(t):
            raise IncompatibleTypeError(
                f"Translation must be array-like (numpy array or tensor), got {type(t)}. "
                f"Please provide a valid array-like object."
            )
        if t.size != 3:
            raise InvalidDimensionError(
                f"Translation vector must have exactly 3 elements, got {t.size}. "
                f"Expected shape: (3,) or (1,3)."
            )

        t = t.reshape(1, 3)

        self._t: ArrayLike = convert_numpy(t)
        # Enforce float32 for consistent precision and memory efficiency
        self._t = self._t.astype(np.float32)
        self._rot: Rotation = rot

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def rot(self) -> Rotation:
        return self._rot

    @staticmethod
    def from_rot_vec_t(rot_vec: ArrayLike, t: ArrayLike) -> "Pose":
        """
        Create a Pose from a rotation vector (axis-angle) and translation.

        Args:
            rot_vec (ArrayLike, (3,)): Axis-angle rotation vector where the direction
                represents the rotation axis and the magnitude represents
                the rotation angle in radians.
            t (ArrayLike, (3,) or (1, 3)): Translation vector representing
                the position in 3D space [x, y, z].

        Returns:
            Pose: New Pose instance with the specified rotation and translation.

        Raises:
            IncompatibleTypeError: If rot_vec or t is not array-like.
            InvalidShapeError: If rotation vector doesn't have shape (3,).

        Example:
            >>> import numpy as np
            >>> rot_vec = np.array([0.0, 0.0, np.pi/2])  # 90 deg around z-axis
            >>> t = np.array([1.0, 2.0, 3.0])
            >>> pose = Pose.from_rot_vec_t(rot_vec, t)
        """
        if not is_array(rot_vec):
            raise IncompatibleTypeError(
                f"Rotation vector must be array-like (numpy array or tensor), got {type(rot_vec)}. "
                f"Please provide a valid array-like object."
            )
        if not is_array(t):
            raise IncompatibleTypeError(
                f"Translation vector must be array-like (numpy array or tensor), got {type(t)}. "
                f"Please provide a valid array-like object."
            )
        if rot_vec.shape[-1] != 3:
            raise InvalidShapeError(
                f"Rotation vector must have last dimension of size 3, got shape {rot_vec.shape}. "
                f"Expected format: (..., 3)."
            )
        rot = Rotation.from_so3(rot_vec)
        return Pose(t, rot)

    @staticmethod
    def from_mat(mat4: ArrayLike) -> "Pose":
        """
        Create a Pose from a transformation matrix.

        Args:
            mat4 (ArrayLike, (3, 4) or (4, 4)): Transformation matrix where
                the upper-left 3x3 block is the rotation matrix (SO3) and
                the first 3 elements of the last column are the translation vector.
                For (4, 4) matrices, the last row should be [0, 0, 0, 1].

        Returns:
            Pose: Pose instance created from the transformation matrix.

        Raises:
            IncompatibleTypeError: If mat4 is not array-like (numpy array or tensor).
            InvalidShapeError: If mat4 shape is not (3, 4) or (4, 4).

        Example:
            >>> import numpy as np
            >>> mat = np.eye(4)
            >>> mat[:3, 3] = [1.0, 2.0, 3.0]  # set translation
            >>> pose = Pose.from_mat(mat)
        """
        if not is_array(mat4):
            raise IncompatibleTypeError(
                f"mat4 must be array type (Tensor or Numpy), got {type(mat4)}. "
                f"Please provide a valid numpy array or PyTorch tensor."
            )
        if not ((mat4.shape[-1] == 4) and (mat4.shape[-2] == 3 or mat4.shape[-2] == 4)):
            raise InvalidShapeError(
                f"Invalid pose matrix shape. Expected (3,4) or (4,4), got {mat4.shape}. "
                f"Please provide a valid transformation matrix."
            )
        return Pose(mat4[0:3, 3], Rotation.from_mat3(mat4[:3, :3]))

    def rot_mat(self) -> np.ndarray:
        return self._rot.mat()

    def mat34(self) -> np.ndarray:
        return concat([self.rot_mat(), transpose2d(self.t)], dim=1)

    def mat44(self) -> np.ndarray:
        mat34 = self.mat34()
        last = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)
        return concat([mat34, last], dim=0)

    def rot_vec_t(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._rot.so3(), self.t

    def skew_t(self) -> np.ndarray:
        return vec3_to_skew(self.t)

    def get_t_rot_mat(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.t, self.rot_mat()

    def inverse(self) -> "Pose":
        R_inv = self._rot.inverse()
        t_inv = -R_inv.apply_pts3d(transpose2d(self.t))
        return Pose(transpose2d(t_inv), R_inv)

    def __repr__(self) -> str:
        """
        Return a verbose string representation of the Pose.

        Returns:
            str: Multi-line string showing translation and rotation matrix.
        """
        t = self._t.flatten()
        mat = self._rot.data
        lines = [
            "Pose(",
            f"  t=[{t[0]:8.4f}, {t[1]:8.4f}, {t[2]:8.4f}]",
            f"  R=[{mat[0,0]:8.4f}, {mat[0,1]:8.4f}, {mat[0,2]:8.4f}]",
            f"    [{mat[1,0]:8.4f}, {mat[1,1]:8.4f}, {mat[1,2]:8.4f}]",
            f"    [{mat[2,0]:8.4f}, {mat[2,1]:8.4f}, {mat[2,2]:8.4f}]",
            ")",
        ]
        return "\n".join(lines)


def interpolate_pose(pose1: Pose, pose2: Pose, t: float) -> Pose:
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
    r = slerp(pose1.rot, pose2.rot, t)
    trans1, trans2 = pose1.t, pose2.t
    trans = trans1 * (1.0 - t) + trans2 * t
    return Pose(t=trans, rot=r)


def relative_pose(pose1: Pose, pose2: Pose) -> Pose:
    """
    Calculate the relative pose from pose1 to pose2.

    Args:
        pose1(Pose): the first pose (reference pose)
        pose2(Pose): the second pose

    Return:
        Pose, the relative pose from pose1 to pose2
    """
    pose1_inv = pose1.inverse()

    rel_rot = pose1_inv.rot * pose2.rot
    rel_t = pose1_inv.t + pose1_inv.rot.apply_pts3d(pose2.t)

    return Pose(rel_t, rel_rot)
