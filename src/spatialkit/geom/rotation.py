"""
Module Name: rotation.py

Description:
This module provides a Rotation class that supports various representations
and transformations of 3D rotations. 

Rotation Types:
- SO3: Special Orthogonal group in 3D.
- so3: Lie algebra of SO3. 
- Quaternion: Four-dimensional unit quaternions (w, x, y, z).
- Roll Pitch Yaw: Three angles, corresponding to rotations around the x, y, and z axes.
- NONE: No Rotation Type, used as a placeholder or default.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

License: MIT LICENSE
"""

from enum import Enum
from typing import Any, Union
import numpy as np

from ..ops.uops import *
from ..ops.umath import *
from ..common.constant import PI, ROTATION_SO3_THRESHOLD
from ..common.exceptions import (
    InvalidShapeError,
    InvalidDimensionError,
    GeometryError,
    IncompatibleTypeError,
)


class RotType(Enum):
    """
    Enumeration of rotation representation types.

    This enum defines the various ways a 3D rotation can be represented,
    including matrix forms (SO3, so3), quaternions (XYZW, WXYZ ordering),
    and Euler angles (Roll-Pitch-Yaw).

    Attributes:
        SO3: Special Orthogonal group - 3x3 rotation matrix
        so3: Lie algebra of SO(3) - 3D rotation vector
        QUAT_XYZW: Quaternion with [x, y, z, w] ordering
        QUAT_WXYZ: Quaternion with [w, x, y, z] ordering
        RPY: Roll-Pitch-Yaw Euler angles
        NONE: No rotation type specified
    """

    SO3 = ("SO3", "SO(3): Special orthogonal group")
    so3 = ("so3", "so(3): Lie Algebra of SO(3)")
    QUAT_XYZW = ("QUAT_XYZW", "Quaternion with 'xyzw' ordering")
    QUAT_WXYZ = ("QUAT_WXYZ", "Quaternion with 'wxyz' ordering")
    RPY = ("RPY", "Roll-Pitch-Yaw (Euler Angles)")
    NONE = ("NONE", "NoneType")

    @staticmethod
    def from_string(type_str: str) -> "RotType":
        """
        Create a RotType from a string representation.

        Args:
            type_str (str): String representation of rotation type.
                Valid values: "SO3", "so3", "QUAT_XYZW", "QUAT_WXYZ", "RPY".

        Returns:
            RotType: Corresponding rotation type enum value.
                Returns RotType.NONE if the string is not recognized.

        Example:
            >>> rot_type = RotType.from_string("SO3")
            >>> rot_type == RotType.SO3
            True
        """
        if type_str == "SO3":
            return RotType.SO3
        if type_str == "so3":
            return RotType.so3
        if type_str == "QUAT_XYZW":
            return RotType.QUAT_XYZW
        if type_str == "QUAT_WXYZ":
            return RotType.QUAT_WXYZ
        if type_str == "RPY":
            return RotType.RPY
        return RotType.NONE


def is_SO3(x: ArrayLike) -> bool:
    """
    Check given rotation array's type is either SO3 or not.
    1. The shape of array is (3,3)
    2. x.T * x = I
    3. Det(x) = 1
    """
    shape = x.shape
    if shape != (3, 3):
        return False  # invalid shape

    return allclose(
        dot(transpose2d(x), x), eye(3, x), atol=ROTATION_SO3_THRESHOLD
    ) and isclose(determinant(x), 1.0, atol=1e-3)


def is_so3(x: ArrayLike) -> bool:
    """
    Check given rotation array's type is either so3 or not.
    1. The shape of array is (3,)
    """
    shape = x.shape
    return shape == (3,)


def is_quat(x: ArrayLike) -> bool:
    """
    Check given rotation array's type is either quaternion or not.
    1. The shape of array is (4,)
    2. ||x|| = 1
    """
    shape = x.shape
    if shape != (4,):
        return False  # invalid shape
    return isclose(norm(x), 1.0, atol=1e-3)


def is_rpy(x: ArrayLike) -> bool:
    """
    Check given rotation array's type is either RPY or not.
    1. The shape of array is (3,)
    """
    shape = x.shape
    if shape != (3,):
        return False  # invalid shape
    return True


def so3_to_SO3(so3: ArrayLike) -> ArrayLike:
    """
    Transform so3 to Rotation Matrix(SO3).

    Args:
        so3 (ArrayLike): so3 vector of shape (3,).

    Returns:
        ArrayLike: Rotation Matrix of shape (3,3).
        
    Raises:
        InvalidShapeError: If so3 is not of shape (3,).
    """
    if not is_so3(so3):
        raise InvalidShapeError(
            f"so3 vector must have shape (3,), got {so3.shape}. "
            f"Please provide a valid so3 vector."
        )
    theta = sqrt(so3[0] ** 2 + so3[1] ** 2 + so3[2] ** 2)
    vec = so3 / (theta + 1e-15)
    skew_vec = vec3_to_skew(vec)
    return exponential_map(skew_vec * theta)


def quat_to_SO3(quat: ArrayLike, is_xyzw: bool) -> ArrayLike:
    """
    Transform Quaternion to Rotation Matrix(SO3).

    Args:
        quat (ArrayLike): Quaternion of shape (4,).
        is_xyzw (bool): If True, quaternion order is (x,y,z,w). If False, order is (w,x,y,z).

    Returns:
        ArrayLike: Rotation Matrix of shape (3,3).
        
    Raises:
        GeometryError: If quaternion is not valid.
    """
    if not is_quat(quat):
        raise GeometryError(
            f"Input quaternion {quat} does not satisfy quaternion properties. "
            f"Quaternion must be a unit vector of length 4."
        )

    if is_xyzw:  # real part last
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    else:
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    # Compute the elements of the rotation matrix
    m00 = 1 - 2 * (y**2 + z**2)
    m01 = 2 * (x * y + z * w)
    m02 = 2 * (x * z - y * w)

    m10 = 2 * (x * y - z * w)
    m11 = 1 - 2 * (x**2 + z**2)
    m12 = 2 * (y * z + x * w)

    m20 = 2 * (x * z + y * w)
    m21 = 2 * (y * z - x * w)
    m22 = 1 - 2 * (x**2 + y**2)

    # Create the rotation matrices for each quaternion
    rotation_matrix = stack(
        [
            stack([m00, m01, m02], dim=0),
            stack([m10, m11, m12], dim=0),
            stack([m20, m21, m22], dim=0),
        ],
        dim=1,
    )

    return rotation_matrix


def rpy_to_SO3(rpy: ArrayLike) -> ArrayLike:
    """
    Transform RPY (Roll, Pitch, Yaw) to Rotation Matrix (SO3).

    Args:
        rpy (ArrayLike): RPY angles of shape (3,).

    Returns:
        ArrayLike: Rotation Matrix of shape (3,3).
        
    Raises:
        InvalidShapeError: If RPY is not of shape (3,).
    """
    if not is_rpy(rpy):
        raise InvalidShapeError(
            f"RPY angles must have shape (3,), got {rpy.shape}. "
            f"Expected format: [roll, pitch, yaw]."
        )
    roll, pitch, yaw = rpy

    # Calculate rotation matrix components
    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)

    m00 = cy * cp
    m01 = cy * sp * sr - sy * cr
    m02 = cy * sp * cr + sy * sr

    m10 = sy * cp
    m11 = sy * sp * sr + cy * cr
    m12 = sy * sp * cr - cy * sr

    m20 = -sp
    m21 = cp * sr
    m22 = cp * cr

    # Create the rotation matrix
    rotation_matrix = convert_array(
        [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]], rpy
    )
    return rotation_matrix


def SO3_to_so3(SO3: ArrayLike) -> ArrayLike:
    """
    Transform SO3 Matrix to so3.

    Args:
        SO3 (ArrayLike): SO3 Matrix of shape (3,3).

    Returns:
        ArrayLike: so3 vector of shape (3,).
        
    Raises:
        GeometryError: If input is not a valid SO3 matrix.
    """
    if not is_SO3(SO3):
        raise GeometryError(
            f"Input matrix does not satisfy SO3 properties (orthogonal matrix with determinant 1). "
            f"Matrix shape: {SO3.shape}, determinant: {determinant(SO3) if SO3.shape == (3,3) else 'invalid'}."
        )

    theta = arccos((trace(SO3) - 1.0) * 0.5)
    vec = (
        0.5
        / sin(theta)
        * stack(
            [SO3[2, 1] - SO3[1, 2], SO3[0, 2] - SO3[2, 0], SO3[1, 0] - SO3[0, 1]], 0
        )
    )
    return theta * vec


def SO3_to_quat(SO3: ArrayLike, is_xyzw: bool) -> ArrayLike:
    """
    Transform SO3 matrix to quaternion.

    Args:
        SO3 (ArrayLike): SO3 Matrix of shape (3,3).
        is_xyzw (bool): If True, returns quaternion in (x,y,z,w) format. If False, (w,x,y,z) format.

    Returns:
        ArrayLike: Quaternion of shape (4,).
        
    Raises:
        GeometryError: If input is not a valid SO3 matrix.
    """
    if not is_SO3(SO3):
        raise GeometryError(
            f"Input matrix does not satisfy SO3 properties (orthogonal matrix with determinant 1). "
            f"Matrix shape: {SO3.shape}, determinant: {determinant(SO3) if SO3.shape == (3,3) else 'invalid'}."
        )
    # Extract the elements of the rotation matrix
    r11, r12, r13 = SO3[0, 0], SO3[0, 1], SO3[0, 2]
    r21, r22, r23 = SO3[1, 0], SO3[1, 1], SO3[1, 2]
    r31, r32, r33 = SO3[2, 0], SO3[2, 1], SO3[2, 2]

    # Calculate the quaternion components
    t = r11 + r22 + r33
    if t > 0:
        S = sqrt(t + 1.0) * 2
        w = 0.25 * S
        x = (r32 - r23) / S
        y = (r13 - r31) / S
        z = (r21 - r12) / S
    elif (r11 > r22) and (r11 > r33):
        S = sqrt(1.0 + r11 - r22 - r33) * 2
        w = (r32 - r23) / S
        x = 0.25 * S
        y = (r12 + r21) / S
        z = (r13 + r31) / S
    elif r22 > r33:
        S = sqrt(1.0 + r22 - r11 - r33) * 2
        w = (r13 - r31) / S
        x = (r12 + r21) / S
        y = 0.25 * S
        z = (r23 + r32) / S
    else:
        S = sqrt(1.0 + r33 - r11 - r22) * 2
        w = (r21 - r12) / S
        x = (r13 + r31) / S
        y = (r23 + r32) / S
        z = 0.25 * S

    # Stack scalars into an array
    quat = np.array([float(w), float(x), float(y), float(z)])
    return quat


def SO3_to_rpy(SO3: ArrayLike) -> ArrayLike:
    """
    Transform Rotation Matrix (SO3) to RPY (Roll, Pitch, Yaw).

    Args:
        SO3 (ArrayLike): Rotation Matrix of shape (3,3).

    Returns:
        ArrayLike: RPY angles of shape (3,).
        
    Raises:
        GeometryError: If input is not a valid SO3 matrix.
    """
    if not is_SO3(SO3):
        raise GeometryError(
            f"Input matrix does not satisfy SO3 properties (orthogonal matrix with determinant 1). "
            f"Matrix shape: {SO3.shape}, determinant: {determinant(SO3) if SO3.shape == (3,3) else 'invalid'}."
        )

    # Extract rotation matrix components
    m00, m01, m02 = SO3[0, 0], SO3[0, 1], SO3[0, 2]
    m10 = SO3[1, 0]
    m20, m21, m22 = SO3[2, 0], SO3[2, 1], SO3[2, 2]

    # Calculate RPY angles
    if m20 not in [1.0, -1.0]:
        pitch = -arcsin(m20)
        roll = arctan2(m21 / cos(pitch), m22 / cos(pitch))
        yaw = arctan2(m10 / cos(pitch), m00 / cos(pitch))
    else:
        yaw = 0
        if m20 == -1:
            pitch = PI / 2
            roll = yaw + arctan2(m01, m02)
        else:
            pitch = -PI / 2
            roll = -yaw + arctan2(-m01, -m02)

    # Stack scalars into an array
    rpy = np.array([float(roll), float(pitch), float(yaw)])
    return rpy


class Rotation:
    """
    Rotation Class that supports various rotation representations.
    
    Supported Types:
    - SO3: Rotation matrix of shape (3, 3)
    - so3: Axis angle representation of shape (3,)
    - Quaternion: Unit quaternion of shape (4,)
    - RPY: Roll-Pitch-Yaw angles of shape (3,)
    
    Default representation is SO3 (3x3 rotation matrix).
    """

    def __init__(self, data: ArrayLike, rot_type: RotType):
        """
        Initialize Rotation instance.
        
        Args:
            data (ArrayLike): Rotation data in specified format.
            rot_type (RotType): Type of rotation representation.
            
        Raises:
            IncompatibleTypeError: If data is not array-like.
            InvalidDimensionError: If data has invalid dimensions.
            GeometryError: If data doesn't satisfy rotation properties.
        """
        if not is_array(data):
            raise IncompatibleTypeError(
                f"Rotation data must be array-like (numpy array or tensor), got {type(data)}. "
                f"Please provide a valid array-like object."
            )
        if len(data.shape) >= 3:
            raise InvalidDimensionError(
                f"Rotation data must be 1D or 2D array, got {len(data.shape)}D with shape {data.shape}. "
                f"Please provide valid rotation data."
            )
        if rot_type not in RotType:
            raise GeometryError(
                f"Invalid rotation type {rot_type}. "
                f"Supported types: {[t.value[0] for t in RotType]}."
            )

        if rot_type == RotType.SO3:
            if not is_SO3(data):
                raise GeometryError(
                    f"Input does not satisfy SO3 properties (orthogonal matrix with determinant 1). "
                    f"Matrix shape: {data.shape}, determinant: {determinant(data) if data.shape == (3,3) else 'invalid'}."
                )
            self.data = data
        elif rot_type == RotType.so3:
            if not is_so3(data):
                raise GeometryError(
                    f"Input does not satisfy so3 properties. Expected shape (3,), got {data.shape}."
                )
            self.data = so3_to_SO3(data)
        elif rot_type == RotType.QUAT_XYZW or rot_type == RotType.QUAT_WXYZ:
            if not is_quat(data):
                raise GeometryError(
                    f"Input does not satisfy quaternion properties (unit vector of length 4). "
                    f"Shape: {data.shape}, norm: {norm(data) if data.shape == (4,) else 'invalid'}."
                )
            self.data = quat_to_SO3(data, rot_type == RotType.QUAT_XYZW)
        elif rot_type == RotType.RPY:
            if not is_rpy(data):
                raise GeometryError(
                    f"Input does not satisfy RPY properties. Expected shape (3,), got {data.shape}."
                )
            self.data = rpy_to_SO3(data)
        self.data = convert_numpy(self.data)
        # Enforce float32 for consistent precision and memory efficiency
        self.data = self.data.astype(np.float32)

    # constructor
    @staticmethod
    def from_mat3(mat3: ArrayLike) -> "Rotation":
        """
        Create a Rotation from a 3x3 rotation matrix (SO3).

        Args:
            mat3 (ArrayLike, (3, 3)): Rotation matrix satisfying SO3 properties
                (orthogonal matrix with determinant 1).

        Returns:
            Rotation: Rotation instance initialized from the matrix.

        Raises:
            GeometryError: If mat3 is not a valid SO3 matrix.

        Example:
            >>> import numpy as np
            >>> mat = np.eye(3)
            >>> rot = Rotation.from_mat3(mat)
        """
        return Rotation(mat3, RotType.SO3)

    @staticmethod
    def from_so3(so3: ArrayLike) -> "Rotation":
        """
        Create a Rotation from an axis-angle (so3) vector.

        Args:
            so3 (ArrayLike, (3,)): Axis-angle vector where the direction
                represents the rotation axis and the magnitude represents
                the rotation angle in radians.

        Returns:
            Rotation: Rotation instance initialized from the so3 vector.

        Raises:
            GeometryError: If so3 is not a valid axis-angle vector.

        Example:
            >>> import numpy as np
            >>> so3_vec = np.array([0.0, 0.0, np.pi/2])  # 90 deg around z-axis
            >>> rot = Rotation.from_so3(so3_vec)
        """
        return Rotation(so3, RotType.so3)

    @staticmethod
    def from_quat_xyzw(quat: ArrayLike) -> "Rotation":
        """
        Create a Rotation from a quaternion in (x, y, z, w) format.

        Args:
            quat (ArrayLike, (4,)): Unit quaternion with imaginary parts first,
                real part last [x, y, z, w]. Must have unit norm.

        Returns:
            Rotation: Rotation instance initialized from the quaternion.

        Raises:
            GeometryError: If quat is not a valid unit quaternion.

        Example:
            >>> import numpy as np
            >>> quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity rotation
            >>> rot = Rotation.from_quat_xyzw(quat)
        """
        return Rotation(quat, RotType.QUAT_XYZW)

    @staticmethod
    def from_quat_wxyz(quat: ArrayLike) -> "Rotation":
        """
        Create a Rotation from a quaternion in (w, x, y, z) format.

        Args:
            quat (ArrayLike, (4,)): Unit quaternion with real part first,
                imaginary parts last [w, x, y, z]. Must have unit norm.

        Returns:
            Rotation: Rotation instance initialized from the quaternion.

        Raises:
            GeometryError: If quat is not a valid unit quaternion.

        Example:
            >>> import numpy as np
            >>> quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
            >>> rot = Rotation.from_quat_wxyz(quat)
        """
        return Rotation(quat, RotType.QUAT_WXYZ)

    @staticmethod
    def from_rpy(rpy: ArrayLike) -> "Rotation":
        """
        Create a Rotation from Roll-Pitch-Yaw (Euler) angles.

        Args:
            rpy (ArrayLike, (3,)): Euler angles [roll, pitch, yaw] in radians.
                Roll is rotation around x-axis, pitch around y-axis,
                yaw around z-axis.

        Returns:
            Rotation: Rotation instance initialized from RPY angles.

        Raises:
            GeometryError: If rpy is not a valid RPY vector.

        Example:
            >>> import numpy as np
            >>> rpy = np.array([0.0, 0.0, np.pi/2])  # 90 deg yaw
            >>> rot = Rotation.from_rpy(rpy)
        """
        return Rotation(rpy, RotType.RPY)

    def mat(self) -> np.ndarray:
        """
        Get the rotation matrix (SO3) representation.
        
        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        return self.data

    def so3(self) -> np.ndarray:
        """
        Get the so3 (axis-angle) representation.
        
        Returns:
            np.ndarray: so3 vector of shape (3,).
        """
        return SO3_to_so3(self.data)

    def quat(self) -> np.ndarray:
        """
        Get the quaternion representation in (w,x,y,z) format.
        
        Returns:
            np.ndarray: Quaternion of shape (4,).
        """
        return SO3_to_quat(self.data, False)

    def rpy(self) -> np.ndarray:
        """
        Get the Roll-Pitch-Yaw representation.
        
        Returns:
            np.ndarray: RPY angles of shape (3,).
        """
        return SO3_to_rpy(self.data)

    def apply_pts3d(self, pts3d: ArrayLike) -> ArrayLike:
        """
        Apply rotation to 3D points.
        
        Args:
            pts3d (ArrayLike): 3D points of shape (3, N).
            
        Returns:
            ArrayLike: Rotated points of shape (3, N).
            
        Raises:
            InvalidShapeError: If pts3d doesn't have correct shape.
        """
        if pts3d.shape[0] != 3:
            raise InvalidShapeError(
                f"Points must have shape (3, N), got {pts3d.shape}. "
                f"Please provide 3D points with x, y, z coordinates as the first dimension."
            )
        mat = self.mat()
        if is_tensor(pts3d):
            mat = convert_tensor(mat, pts3d)
        pts3d = matmul(mat, pts3d)  # [3,3] * [3,n] = [3,n]
        return pts3d  # [3,n]

    def inverse_mat(self) -> ArrayLike:
        """
        Get the inverse rotation matrix.
        
        Returns:
            ArrayLike: Inverse rotation matrix (transpose of original).
        """
        return transpose2d(self.data)

    def inverse(self) -> "Rotation":
        """
        Get the inverse rotation.

        Returns:
            Rotation: Inverse rotation instance.
        """
        return Rotation.from_mat3(self.inverse_mat())

    def dot(self, rot: "Rotation") -> "Rotation":
        """
        Compose this rotation with another rotation (public method).

        Args:
            rot (Rotation): Another rotation to compose with.

        Returns:
            Rotation: Composed rotation.
        """
        return self._dot(rot)

    def _dot(self, rot: "Rotation") -> "Rotation":
        """
        Compose this rotation with another rotation.
        
        Args:
            rot (Rotation): Another rotation to compose with.
            
        Returns:
            Rotation: Composed rotation.
        """
        rot1_mat = self.mat()
        rot2_mat = rot.mat()
        rot2_mat = convert_array(rot2_mat, rot1_mat)
        rot_mat = matmul(rot1_mat, rot2_mat)
        return Rotation.from_mat3(rot_mat)

    def __mul__(self, other: Any) -> Union["Rotation", ArrayLike]:
        """
        Multiplication operator for rotation composition or point transformation.

        Args:
            other: Either another Rotation for composition or ArrayLike for point transformation.

        Returns:
            Union[Rotation, ArrayLike]: Composed rotation or transformed points.

        Raises:
            IncompatibleTypeError: If other is not a supported type.
        """
        if isinstance(other, Rotation):
            return self._dot(other)
        if is_array(other):
            return self.apply_pts3d(other)
        raise IncompatibleTypeError(
            f"Unsupported data type {type(other)} for multiplication with Rotation. "
            f"Supported types: Rotation (for composition) or ArrayLike (for point transformation)."
        )

    def __repr__(self) -> str:
        """
        Return a verbose string representation of the Rotation.

        Returns:
            str: Multi-line string showing the SO3 rotation matrix.
        """
        mat = self.data
        lines = [
            "Rotation(",
            f"  SO3=[{mat[0,0]:8.4f}, {mat[0,1]:8.4f}, {mat[0,2]:8.4f}]",
            f"      [{mat[1,0]:8.4f}, {mat[1,1]:8.4f}, {mat[1,2]:8.4f}]",
            f"      [{mat[2,0]:8.4f}, {mat[2,1]:8.4f}, {mat[2,2]:8.4f}]",
            ")",
        ]
        return "\n".join(lines)


def slerp(r1: Rotation, r2: Rotation, t: float):
    """
    Spherical Linear Interpolation between two Rotations.

    Algorithm:
    1. Transform Rotations to unit quaternions q1, q2
    2. Compute angle "w" between two quaternions: w = cos^-1(q1*q2)
    3. Compute slerp(q1,q2,t) = sin((1-t)*w)/sin(w)*q1 + sin(tw)/sin(w)*q2

    Args:
        r1 (Rotation): First rotation instance.
        r2 (Rotation): Second rotation instance.
        t (float): Interpolation parameter, must be between 0 and 1.

    Returns:
        Rotation: Interpolated rotation.
        
    Raises:
        InvalidDimensionError: If t is not between 0 and 1.
    """
    if not (0.0 <= t <= 1.0):
        raise InvalidDimensionError(
            f"Interpolation parameter t must be between 0 and 1, got {t}. "
            f"Please provide a valid interpolation parameter."
        )

    if t == 1.0:
        return r2
    if t == 0.0:
        return r1

    q1, q2 = r1.quat(), r2.quat()  # (w,x,y,z)
    cos_omega = matmul(q1, q2)
    if (
        cos_omega < 0.0
    ):  # If negative dot product, negate one of the quaternions to take shorter arc
        q1 = -q1
        cos_omega = -cos_omega
    if (
        cos_omega > 0.9999
    ):  # If the quaternions are very close, use linear interpolation
        q = normalize(q1 * (1 - t) + q2 * t, dim=0)
    else:
        omega = arccos(cos_omega)
        sin_omega = sin(omega)
        scale1 = sin((1 - t) * omega) / sin_omega
        scale2 = sin(t * omega) / sin_omega
        q = normalize(scale1 * q1 + scale2 * q2, dim=0)
    return Rotation.from_quat_wxyz(q)


__all__ = [
    # Rotation type enum
    "RotType",
    # Main rotation class
    "Rotation",
    # Type checking functions
    "is_SO3",
    "is_so3",
    "is_quat",
    "is_rpy",
    # Conversion to SO3
    "so3_to_SO3",
    "quat_to_SO3",
    "rpy_to_SO3",
    # Conversion from SO3
    "SO3_to_so3",
    "SO3_to_quat",
    "SO3_to_rpy",
    # Interpolation
    "slerp",
]
