"""
Module Name: tf.py

Description:
This module provides a Transformation class that supports various operations for 3D transformations.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

License: MIT LICENSE
"""

from typing import Optional, Tuple, Any
import numpy as np
from .rotation import Rotation, slerp
from .pose import Pose
from ..ops.uops import *
from ..ops.uops import ArrayLike
from ..ops.umath import *
from ..common.exceptions import (
    InvalidShapeError,
    InvalidDimensionError,
    IncompatibleTypeError,
)


class Transform:
    """
    Transform Class representing a 3D transformation with position and orientation.
    This class encapsulates both the translation and rotation components of a transformation.

    Attributes:
        t (np.ndarray, (1, 3)): Translation vector.
        rot (Rotation): Rotation instance.
    """

    def __init__(self, t: Optional[ArrayLike] = None, rot: Optional[Rotation] = None):
        """
        Initialize Transform instance.
        
        Args:
            t (ArrayLike, optional): Translation vector. Defaults to zero vector.
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

        # Convert to numpy and enforce float32 for consistent precision and memory efficiency
        from ..ops.uops import convert_numpy
        self._t = convert_numpy(t).astype(np.float32)
        self._rot = rot

    @property
    def t(self) -> ArrayLike:
        return self._t

    @property
    def rot(self) -> Rotation:
        return self._rot

    @staticmethod
    def from_rot_vec_t(rot_vec: ArrayLike, t: ArrayLike) -> "Transform":
        """
        Create a Transform from a rotation vector (axis-angle) and translation.

        Args:
            rot_vec (ArrayLike, (3,)): Axis-angle rotation vector where the direction
                represents the rotation axis and the magnitude represents
                the rotation angle in radians.
            t (ArrayLike, (3,) or (1, 3)): Translation vector representing
                the position in 3D space [x, y, z].

        Returns:
            Transform: Transform instance with the specified rotation and translation.

        Raises:
            IncompatibleTypeError: If rot_vec or t is not array-like.
            InvalidDimensionError: If rot_vec or t doesn't have exactly 3 elements.

        Example:
            >>> import numpy as np
            >>> rot_vec = np.array([0.0, 0.0, np.pi/2])  # 90 deg around z-axis
            >>> t = np.array([1.0, 2.0, 3.0])
            >>> tf = Transform.from_rot_vec_t(rot_vec, t)
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
        if rot_vec.size != 3:
            raise InvalidDimensionError(
                f"Rotation vector must have exactly 3 elements, got {rot_vec.size}. "
                f"Expected shape: (3,)."
            )
        if t.size != 3:
            raise InvalidDimensionError(
                f"Translation vector must have exactly 3 elements, got {t.size}. "
                f"Expected shape: (3,)."
            )
        rot = Rotation.from_so3(
            rot_vec.reshape(
                -1,
            )
        )
        return Transform(t, rot)

    @staticmethod
    def from_mat(mat4: ArrayLike) -> "Transform":
        """
        Create a Transform from a transformation matrix.

        Args:
            mat4 (ArrayLike, (3, 4) or (4, 4)): Transformation matrix where
                the upper-left 3x3 block is the rotation matrix (SO3) and
                the first 3 elements of the last column are the translation vector.
                For (4, 4) matrices, the last row should be [0, 0, 0, 1].

        Returns:
            Transform: Transform instance created from the transformation matrix.

        Raises:
            IncompatibleTypeError: If mat4 is not array-like (numpy array or tensor).
            InvalidShapeError: If mat4 shape is not (3, 4) or (4, 4).

        Example:
            >>> import numpy as np
            >>> mat = np.eye(4)
            >>> mat[:3, 3] = [1.0, 2.0, 3.0]  # set translation
            >>> tf = Transform.from_mat(mat)
        """
        if not is_array(mat4):
            raise IncompatibleTypeError(
                f"Transformation matrix must be array-like (numpy array or tensor), got {type(mat4)}. "
                f"Please provide a valid array-like object."
            )
        if mat4.shape not in [(4, 4), (3, 4)]:
            raise InvalidShapeError(
                f"Transformation matrix must have shape (4,4) or (3,4), got {mat4.shape}. "
                f"Please provide a valid transformation matrix."
            )
        return Transform(mat4[:3, 3], Rotation.from_mat3(mat4[:3, :3]))

    @staticmethod
    def from_pose(pose: Pose) -> "Transform":
        """
        Create a Transform from a Pose instance.

        Args:
            pose (Pose): Pose instance containing translation and rotation.

        Returns:
            Transform: Transform instance with the same translation and rotation as the pose.

        Example:
            >>> import numpy as np
            >>> pose = Pose(np.array([1.0, 2.0, 3.0]), Rotation.from_mat3(np.eye(3)))
            >>> tf = Transform.from_pose(pose)
        """
        return Transform(pose.t, Rotation.from_mat3(pose.rot_mat()))

    def rot_mat(self) -> np.ndarray:
        """
        Get the rotation matrix of the transform.

        Returns:
            np.ndarray, (3, 3): Rotation matrix (SO3).
        """
        return self.rot.mat()

    def mat34(self) -> np.ndarray:
        """
        Get the 3x4 transformation matrix of the transform.

        Returns:
            np.ndarray, (3, 4): Transformation matrix [R | t].
        """
        return concat([self.rot_mat(), transpose2d(self.t)], dim=1)

    def mat44(self) -> np.ndarray:
        """
        Get the 4x4 transformation matrix of the transform.

        Returns:
            np.ndarray, (4, 4): Homogeneous transformation matrix [[R | t], [0 0 0 1]].
        """
        mat34 = self.mat34()
        last = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)
        return concat([mat34, last], dim=0)

    def rot_vec_t(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the rotation vector and translation vector of the transform.

        Returns:
            tuple: A tuple containing:
                - rot_vec (np.ndarray, (3,)): Axis-angle rotation vector.
                - t (np.ndarray, (1, 3)): Translation vector.
        """
        return self.rot.so3(), self.t

    def skew_t(self) -> np.ndarray:
        """
        Get the skew-symmetric matrix of the translation vector.

        Returns:
            np.ndarray, (3, 3): Skew-symmetric matrix of translation.

        Note:
            For t = [tx, ty, tz], the skew-symmetric matrix is:
            [[  0, -tz,  ty],
             [ tz,   0, -tx],
             [-ty,  tx,   0]]
        """
        return vec3_to_skew(self.t)

    def get_t_rot_mat(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the translation vector and rotation matrix of the transform.

        Returns:
            tuple: A tuple containing:
                - t (np.ndarray, (1, 3)): Translation vector.
                - rot_mat (np.ndarray, (3, 3)): Rotation matrix (SO3).
        """
        return self.t, self.rot_mat()

    def inverse(self) -> "Transform":
        """
        Get the inverse of the current transform.

        Returns:
            Transform: Inverse transform
        """
        R_inv = self.rot.inverse()
        t_inv = -R_inv.apply_pts3d(self.t.T)
        return Transform(t_inv.T, R_inv)

    def get_origin_direction(self, rays: ArrayLike):
        """
        Get the origin and direction vectors from rays (local coordinates).

        Args:
            rays (ArrayLike, (3, N) or (3,)): Camera rays from origin in local coordinates.

        Returns:
            tuple: A tuple containing:
                - origins (ArrayLike, (N, 3)): Origin in world coordinates.
                - directions (ArrayLike, (N, 3)): Unit direction vector in world coordinates.

        Raises:
            InvalidShapeError: If rays don't have shape (3, N) or (3,).
        """
        if not (len(rays.shape) <= 2 and rays.shape[0] == 3):
            raise InvalidShapeError(
                f"Rays must have shape (3,N) or (3,), got {rays.shape}. "
                f"Please provide valid camera ray vectors."
            )
        if len(rays.shape) == 2:
            n_rays = rays.shape[1]
        else:
            rays = expand_dim(rays, 1)
            n_rays = 1
        origin = np.tile(convert_numpy(self.t), (n_rays, 1))
        origin = convert_array(origin, rays)
        direction = transpose2d(self.rot.apply_pts3d(rays)).reshape((-1, 3))
        return origin, direction

    def merge(self, transform: "Transform") -> "Transform":
        """
        Merge the current transform with another transform.

        Args:
            transform (Transform):  another transform to merge with

        Returns:
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
            pts3d (ArrayLike, (3, N)): 3D points where each column is a point [x, y, z].

        Returns:
            ArrayLike, (3, N): Transformed 3D points.
        """
        t = transpose2d(convert_array(self.t, pts3d))
        return self.rot.apply_pts3d(pts3d) + t

    def __mul__(self, other: Any):
        """
        Define the multiplication operation for Transform, Pose, and 3D points.

        Args:
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
            # other.t is (1, 3), need to transpose for apply_pts3d which expects (3, N)
            t_new = self.rot.apply_pts3d(transpose2d(other.t)) + transpose2d(self.t)
            r_new = self.rot * other._rot
            return Pose(transpose2d(t_new), r_new)
        elif is_array(other):
            return self.apply_pts3d(other)
        else:
            raise ValueError(
                "Multiplication only supported with Transform, Pose or 3D points."
            )

    def __repr__(self) -> str:
        """
        Return a verbose string representation of the Transform.

        Returns:
            str: Multi-line string showing translation and rotation matrix.
        """
        t = self._t.flatten()
        mat = self._rot.data
        lines = [
            "Transform(",
            f"  t=[{t[0]:8.4f}, {t[1]:8.4f}, {t[2]:8.4f}]",
            f"  R=[{mat[0,0]:8.4f}, {mat[0,1]:8.4f}, {mat[0,2]:8.4f}]",
            f"    [{mat[1,0]:8.4f}, {mat[1,1]:8.4f}, {mat[1,2]:8.4f}]",
            f"    [{mat[2,0]:8.4f}, {mat[2,1]:8.4f}, {mat[2,2]:8.4f}]",
            ")",
        ]
        return "\n".join(lines)


def interpolate_transform(t1: Transform, t2: Transform, alpha: float) -> Transform:
    """
    Interpolate between two transforms using linear interpolation (Lerp) and spherical linear interpolation (Slerp).

    Args:
        t1 (Transform): start transform
        t2 (Transform): end transform
        alpha  (float): interpolation parameter

    Returns:
        Transform: Interpolated transform
    """
    r = slerp(t1.rot, t2.rot, alpha)
    trans1, trans2 = t1.t, t2.t
    trans = trans1 * (1.0 - alpha) + trans2 * alpha
    return Transform(t=trans, rot=r)
