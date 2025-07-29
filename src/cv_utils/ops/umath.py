"""
Module Name: umath.py

Description: 
Unified Math (umath) module provides a unified interface for common mathematical operations that can be performed using both Numpy and Torch. 
This module helps to write agnostic code that can handle both Numpy arrays and Torch tensors seamlessly.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1-alpha

License: MIT LICENSE

Usage:
>>> import numpy as np
>>> import torch
>>> from cv_utils import umath

>>> np_array = np.array([1., 2., 3., 4.])
>>> torch_tensor = torch.tensor([1., 2., 3., 4.])

>>> umath.mean(np_array)
2.5
>>> umath.mean(torch_tensor)
tensor(2.5000)
"""

from scipy.linalg import expm
import torch

from .uops import *
from ..common.constant import EPSILON
from ..exceptions import (
    InvalidDimensionError,
    InvalidShapeError,
    IncompatibleTypeError,
    NumericalError,
    InvalidCoordinateError
)


# Basic Mathematical Operations
def abs(x: ArrayLike) -> ArrayLike:
    """
    Compute the absolute value of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Absolute value of each element.
    """
    if is_tensor(x):
        return torch.abs(x)
    return np.abs(x)


def sqrt(x: ArrayLike) -> ArrayLike:
    """
    Compute the square root of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Square root of each element.
    """
    if is_tensor(x):
        return torch.sqrt(x)
    return np.sqrt(x)


def mean(x: ArrayLike, dim: Optional[int] = None) -> ArrayLike:
    """
    Compute the mean of elements along the specified dimension.

    Args:
        x (ArrayLike): Input array or tensor.
        dim (Optional[int]): Dimension along which to compute the mean.

    Returns:
        ArrayLike: Mean of elements.
        
    Raises:
        InvalidDimensionError: If dimension is out of range for the input array.
    """
    if dim is not None and dim >= x.ndim:
        raise InvalidDimensionError(
            f"Dimension {dim} is out of range for array with {x.ndim} dimensions. "
            f"Please use a dimension less than {x.ndim}."
        )
    
    if is_tensor(x):
        return torch.mean(x, dim=dim)
    return np.mean(x, axis=dim)


def dot(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Compute the dot product of two arrays.

    Args:
        x (ArrayLike): First input array or tensor.
        y (ArrayLike): Second input array or tensor.

    Returns:
        ArrayLike: Dot product of x and y.
        
    Raises:
        IncompatibleTypeError: If input arrays have different types.
    """
    if not isinstance(x, type(y)):
        raise IncompatibleTypeError(
            f"Both arrays must be of the same type, got {type(x).__name__} and {type(y).__name__}. "
            f"Please ensure both inputs are either numpy arrays or torch tensors."
        )
    
    if is_tensor(x):
        return torch.dot(x, y)
    return np.dot(x, y)


def qr(x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute the QR decomposition of a matrix.

    Args:
        x (ArrayLike): Input 2D matrix.

    Returns:
        tuple[ArrayLike, ArrayLike]: Q and R matrices from QR decomposition.
        
    Raises:
        InvalidDimensionError: If input is not a 2D matrix.
        NumericalError: If QR decomposition fails.
    """
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"QR decomposition requires a 2D matrix, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is a 2D matrix."
        )
    
    try:
        if is_tensor(x):
            return torch.linalg.qr(x)
        return np.linalg.qr(x)
    except Exception as e:
        raise NumericalError(f"QR decomposition failed: {e}") from e


def svd(x: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute the Singular Value Decomposition (SVD) of a matrix.

    Args:
        x (ArrayLike): Input 2D matrix.

    Returns:
        tuple[ArrayLike, ArrayLike, ArrayLike]: U, S, V matrices from SVD decomposition.
        
    Raises:
        InvalidDimensionError: If input is not a 2D matrix.
        NumericalError: If SVD decomposition fails.
    """
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"SVD requires a 2D matrix, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is a 2D matrix."
        )
    
    try:
        if is_tensor(x):
            return torch.linalg.svd(x)
        return np.linalg.svd(x)
    except Exception as e:
        raise NumericalError(f"SVD decomposition failed: {e}") from e


def determinant(x: ArrayLike) -> float:
    """
    Compute the determinant of a square matrix.

    Args:
        x (ArrayLike): Input 2D square matrix.

    Returns:
        float: Determinant of the matrix.
        
    Raises:
        InvalidDimensionError: If input is not a 2D matrix.
        InvalidShapeError: If matrix is not square.
        NumericalError: If determinant computation fails.
    """
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"Determinant requires a 2D matrix, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is a 2D matrix."
        )
    
    if x.shape[0] != x.shape[1]:
        raise InvalidShapeError(
            f"Determinant requires a square matrix, got shape {x.shape}. "
            f"Please ensure the matrix has equal width and height."
        )
    
    try:
        if is_tensor(x):
            return torch.det(x).item()
        return np.linalg.det(x)
    except Exception as e:
        raise NumericalError(f"Determinant computation failed: {e}") from e


def inv(x: ArrayLike) -> ArrayLike:
    """
    Compute the inverse of a square matrix.

    Args:
        x (ArrayLike): Input 2D square matrix.

    Returns:
        ArrayLike: Inverse of the matrix.
        
    Raises:
        InvalidDimensionError: If input is not a 2D matrix.
        InvalidShapeError: If matrix is not square.
        NumericalError: If matrix inversion fails (e.g., singular matrix).
    """
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"Matrix inversion requires a 2D matrix, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is a 2D matrix."
        )
    
    if x.shape[0] != x.shape[1]:
        raise InvalidShapeError(
            f"Matrix inversion requires a square matrix, got shape {x.shape}. "
            f"Please ensure the matrix has equal width and height."
        )
    
    try:
        if is_tensor(x):
            return torch.inverse(x)
        return np.linalg.inv(x)
    except Exception as e:
        raise NumericalError(f"Matrix inversion failed: {e}. The matrix may be singular (non-invertible).") from e


# Matrix and Vector Operations
def norm(
    x: ArrayLike,
    order: Optional[Union[int, str]] = None,
    dim: Optional[int] = None,
    keepdim: Optional[bool] = False,
) -> ArrayLike:
    """
    Compute the norm of an array.

    Args:
        x (ArrayLike): Input array or tensor.
        order (Optional[Union[int, str]]): Order of the norm.
        dim (Optional[int]): Dimension along which to compute the norm.
        keepdim (Optional[bool]): Whether to keep the dimensions.

    Returns:
        ArrayLike: Norm of the array.
        
    Raises:
        InvalidDimensionError: If dimension is out of range for the input array.
    """
    if dim is not None and dim >= x.ndim:
        raise InvalidDimensionError(
            f"Dimension {dim} is out of range for array with {x.ndim} dimensions. "
            f"Please use a dimension less than {x.ndim}."
        )
    
    if is_tensor(x):
        return torch.norm(x, p=order, dim=dim, keepdim=keepdim)
    return np.linalg.norm(x, ord=order, axis=dim, keepdims=keepdim)


def normalize(
    x: ArrayLike,
    order: Optional[Union[int, str]] = None,
    dim: Optional[int] = None,
    eps: Optional[float] = EPSILON,
) -> ArrayLike:
    """
    Normalize an array along a specified dimension.

    Args:
        x (ArrayLike): Input array or tensor.
        order (Optional[Union[int, str]]): Order of the norm.
        dim (Optional[int]): Dimension along which to normalize.
        eps (Optional[float]): Small value to avoid division by zero.

    Returns:
        ArrayLike: Normalized array.
    """
    n = norm(x=x, order=order, dim=dim, keepdim=False)
    return x / (n + eps)


def matmul(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Perform matrix multiplication between two arrays.

    Args:
        x (ArrayLike): First input array or tensor.
        y (ArrayLike): Second input array or tensor.

    Returns:
        ArrayLike: Result of matrix multiplication.
        
    Raises:
        InvalidDimensionError: If input arrays are not at least 1D.
        InvalidShapeError: If matrix dimensions are incompatible for multiplication.
    """
    if x.ndim < 1 or y.ndim < 1:
        raise InvalidDimensionError(
            f"Matrix multiplication requires at least 1D arrays, got shapes {x.shape} and {y.shape}. "
            f"Please ensure both inputs are at least 1-dimensional."
        )
    
    if x.shape[-1] != y.shape[0]:
        raise InvalidShapeError(
            f"Matrix multiplication dimension mismatch: {x.shape[-1]} != {y.shape[0]}. "
            f"Please ensure the last dimension of first array matches the first dimension of second array."
        )
    
    if is_tensor(x):
        return torch.matmul(x, y)
    return x @ y


def permute(x: ArrayLike, dims: Tuple[int]) -> ArrayLike:
    """
    Permute the dimensions of an array.

    Args:
        x (ArrayLike): Input array or tensor.
        dims (Tuple[int]): Desired ordering of dimensions.

    Returns:
        ArrayLike: Permuted array.
        
    Raises:
        InvalidShapeError: If permutation dimensions don't match array dimensions.
    """
    if len(x.shape) != len(dims):
        raise InvalidShapeError(
            f"Permutation dimensions {len(dims)} don't match array dimensions {len(x.shape)}. "
            f"Please provide {len(x.shape)} dimension indices for permutation."
        )
    
    if is_tensor(x):
        return x.permute(dims)
    return x.transpose(dims)


def trace(x: ArrayLike) -> ArrayLike:
    """
    Compute the trace of a matrix.

    Args:
        x (ArrayLike): Input 2D matrix.

    Returns:
        ArrayLike: Trace of the matrix.
        
    Raises:
        InvalidDimensionError: If input is not a 2D matrix.
    """
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"Trace requires a 2D matrix, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is a 2D matrix."
        )
    
    if is_tensor(x):
        return torch.trace(x)
    return np.trace(x)


def diag(x: ArrayLike) -> ArrayLike:
    """
    Extract the diagonal of a matrix.

    Args:
        x (ArrayLike): Input 2D matrix.

    Returns:
        ArrayLike: Diagonal elements of the matrix.
        
    Raises:
        InvalidDimensionError: If input is not a 2D matrix.
    """
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"Diagonal extraction requires a 2D matrix, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is a 2D matrix."
        )
    
    if is_tensor(x):
        return torch.diag(x)
    return np.diag(x)


# Transform and Permutation
def rad2deg(x: ArrayLike) -> ArrayLike:
    """
    Convert radians to degrees.

    Args:
        x (ArrayLike): Input array or tensor in radians.

    Returns:
        ArrayLike: Converted array in degrees.
    """
    return x * (180.0 / np.pi)


def deg2rad(x: ArrayLike) -> ArrayLike:
    """
    Convert degrees to radians.

    Args:
        x (ArrayLike): Input array or tensor in degrees.

    Returns:
        ArrayLike: Converted array in radians.
    """
    return x * (np.pi / 180.0)


def exponential_map(mat: ArrayLike) -> ArrayLike:
    """
    Compute the matrix exponential.

    Args:
        mat (ArrayLike): Input square matrix.

    Returns:
        ArrayLike: Matrix exponential.
    """
    if is_tensor(mat):
        return torch.matrix_exp(mat)
    else:
        return expm(mat)


# Trigonometric functions
def sin(x: ArrayLike) -> ArrayLike:
    """
    Compute the sine of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Sine of each element.
    """
    if is_tensor(x):
        return torch.sin(x)
    return np.sin(x)


def cos(x: ArrayLike) -> ArrayLike:
    """
    Compute the cosine of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Cosine of each element.
    """
    if is_tensor(x):
        return torch.cos(x)
    return np.cos(x)


def tan(x: ArrayLike) -> ArrayLike:
    """
    Compute the tangent of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Tangent of each element.
    """
    if is_tensor(x):
        return torch.tan(x)
    return np.tan(x)


def arcsin(x: ArrayLike) -> ArrayLike:
    """
    Compute the arcsine of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Arcsine of each element.
    """
    if is_tensor(x):
        return torch.arcsin(x)
    return np.arcsin(x)


def arccos(x: ArrayLike) -> ArrayLike:
    """
    Compute the arccosine of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Arccosine of each element.
    """
    if is_tensor(x):
        return torch.arccos(x)
    return np.arccos(x)


def arctan(x: ArrayLike) -> ArrayLike:
    """
    Compute the arctangent of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Arctangent of each element.
    """
    if is_tensor(x):
        return torch.arctan(x)
    return np.arctan(x)


def arctan2(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Compute the element-wise arctangent of x/y.

    Args:
        x (ArrayLike): First input array or tensor.
        y (ArrayLike): Second input array or tensor.

    Returns:
        ArrayLike: Element-wise arctangent of x/y.
    """
    if is_tensor(x):
        y = convert_tensor(y, x)
        return torch.arctan2(x, y)
    return np.arctan2(x, y)


# Polynomial functions
def polyval(coeffs: Union[ArrayLike, List[float]], x: ArrayLike) -> ArrayLike:
    """
    Evaluate a polynomial at specific values.

    Args:
        coeffs (Union[ArrayLike,List[float]]): Polynomial coefficients.
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Evaluated polynomial.
    """
    y = zeros_like(x)
    for c in coeffs:
        y = y * x + c
    return y


def polyfit(x: ArrayLike, y: ArrayLike, degree: int) -> ArrayLike:
    """
    Fit a polynomial of a specified degree to data.

    Args:
        x (ArrayLike): Input array or tensor for x-values.
        y (ArrayLike): Input array or tensor for y-values.
        degree (int): Degree of the polynomial.

    Returns:
        ArrayLike: Polynomial coefficients.
    """
    if is_tensor(x):
        x_np = convert_numpy(x)
        y_np = convert_numpy(y)
        coeffs_np = np.polyfit(x_np, y_np, degree)
        return convert_tensor(coeffs_np, x)
    else:
        return np.polyfit(x, y, degree)


# Linear-Algebra Problem
def is_square(x: ArrayLike) -> bool:
    """
    Check if a matrix is square.

    Args:
        x (ArrayLike): Input matrix.

    Returns:
        bool: True if the matrix is square, False otherwise.
    """
    return x.ndim == 2 and x.shape[0] == x.shape[1]


def solve(A: ArrayLike, b: ArrayLike) -> ArrayLike:
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    Args:
        A (ArrayLike): Coefficient matrix.
        b (ArrayLike): Ordinate or dependent variable values.

    Returns:
        ArrayLike: Solution to the system of equations.
        
    Raises:
        IncompatibleTypeError: If input arrays have different types.
        NumericalError: If the linear system cannot be solved.
    """
    if not isinstance(A, type(b)):
        raise IncompatibleTypeError(
            f"Both arrays must be of the same type, got A:{type(A).__name__} and b:{type(b).__name__}. "
            f"Please ensure both inputs are either numpy arrays or torch tensors."
        )
    
    try:
        if is_tensor(A):
            return torch.linalg.solve(A, b)
        return np.linalg.solve(A, b)
    except Exception as e:
        raise NumericalError(f"Linear system solve failed: {e}. The system may be singular or ill-conditioned.") from e


def solve_linear_system(A: ArrayLike, b: Optional[ArrayLike] = None):
    """
    Solve the linear system Ax = b or find the null space if b is None.
    Efficient for small linear systems but may be adapted for larger systems with appropriate libraries.

    Args:
    - A (ArrayLike, ): the coefficient matrix.
    - b (ArrayLike, ): the dependent variable vector. If None, find null space of A.

    Return:
    - Sol (ArrayLike, ) : Solution vector or null space basis vectors.
    """
    if b is not None:
        # Ax = b
        return solve(A, b)
    else:
        # Ax = 0
        # use svd
        _, s, vt = svd(A)
        null_space = transpose2d(vt)[:, s < EPSILON]
        return null_space


# Computer Vision
def vec3_to_skew(x: ArrayLike) -> ArrayLike:
    """
    Convert a 3D vector to a skew-symmetric matrix.

    Args:
        x (ArrayLike): Input 3D vector with shape (3,) or (1, 3).

    Returns:
        ArrayLike: 3x3 skew-symmetric matrix.
        
    Raises:
        InvalidShapeError: If input vector is not 3D with proper shape.
    """
    if x.shape not in [(3,), (1, 3)]:
        raise InvalidShapeError(
            f"Input vector must have shape (3,) or (1, 3), got {x.shape}. "
            f"Please provide a 3D vector with the correct shape."
        )
    
    if x.shape == (1, 3):
        x = reduce_dim(x, 0)
    
    wx = x[0].item()
    wy = x[1].item()
    wz = x[2].item()
    skew_x = np.array([[0.0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    
    if is_tensor(x):
        skew_x = convert_tensor(skew_x, x)
    return skew_x


def homo(x: ArrayLike) -> ArrayLike:
    """
    Convert Euclidean coordinates to Homogeneous coordinates.

    Args:
        x (ArrayLike): Euclidean coordinates with shape (2, N) or (3, N).

    Returns:
        ArrayLike: Homogeneous coordinates with shape (3, N) or (4, N).
        
    Raises:
        InvalidCoordinateError: If input coordinates have invalid shape.

    Details:
    - [x y] -> [x y 1]
    - [x y z] -> [x y z 1]
    """
    if x.ndim != 2 or x.shape[0] not in [2, 3]:
        raise InvalidCoordinateError(
            f"Input coordinates must have shape (2, N) or (3, N), got {x.shape}. "
            f"Please ensure input is a 2D array with 2 or 3 rows."
        )
    
    return concat([x, ones_like(x[0:1, :])], 0)


def dehomo(x: ArrayLike) -> ArrayLike:
    """
    Convert Homogeneous coordinates to Euclidean coordinates.

    Args:
        x (ArrayLike): Homogeneous coordinates with shape (3, N) or (4, N).

    Returns:
        ArrayLike: Euclidean coordinates with shape (2, N) or (3, N).
        
    Raises:
        InvalidCoordinateError: If input coordinates have invalid shape.

    Details:
    - [x y z] -> [x/z y/z]
    - [x y z w] -> [x/w y/w z/w]
    """
    if x.ndim != 2 or x.shape[0] not in [3, 4]:
        raise InvalidCoordinateError(
            f"Input coordinates must have shape (3, N) or (4, N), got {x.shape}. "
            f"Please ensure input is a 2D array with 3 or 4 rows."
        )
    
    euc_coords = x[:-1, :] / x[-1, :]
    return euc_coords
