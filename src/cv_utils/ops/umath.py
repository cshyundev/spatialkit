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
from ..common.logger import LOG_ERROR


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
    """
    if dim is not None:
        assert (
            dim < x.ndim
        ), f"Invalid dimension: expected dimension less than {x.ndim}, but got {dim}."
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
    """
    assert isinstance(
        x, type(y)
    ), f"Invalid type: expected same type for both arrays, but got {type(x)} and {type(y)}"
    if is_tensor(x):
        return torch.dot(x, y)
    return np.dot(x, y)


def qr(x: ArrayLike) -> ArrayLike:
    """
    Compute the QR decomposition of a matrix.

    Args:
        x (ArrayLike): Input 2D matrix.

    Returns:
        Q,R (ArrayLike): QR decomposition of the matrix.
    """
    assert (
        x.ndim == 2
    ), f"Invalid shape for QR: expected a 2D matrix, but got {x.shape}."
    if is_tensor(x):
        return torch.linalg.qr(x)
    return np.linalg.qr(x)


def svd(x: ArrayLike) -> ArrayLike:
    """
    Compute the Singular Value Decomposition (SVD) of a matrix.

    Args:
        x (ArrayLike): Input 2D matrix.

    Returns:
        ArrayLike: SVD of the matrix.
    """
    assert (
        x.ndim == 2
    ), f"Invalid shape for SVD: expected a 2D matrix, but got {x.shape}."
    if is_tensor(x):
        return torch.linalg.svd(x)
    return np.linalg.svd(x)


def determinant(x: ArrayLike) -> float:
    """
    Compute the determinant of a square matrix.

    Args:
        x (ArrayLike): Input 2D square matrix.

    Returns:
        float: Determinant of the matrix.
    """
    assert (
        x.ndim == 2
    ), f"Invalid shape for determinant: expected a 2D square matrix, but got {x.shape}."
    if is_tensor(x):
        return torch.det(x).item()
    return np.linalg.det(x)


def inv(x: ArrayLike) -> ArrayLike:
    """
    Compute the inverse of a square matrix.

    Args:
        x (ArrayLike): Input 2D square matrix.

    Returns:
        ArrayLike: Inverse of the matrix.
    """
    assert (
        x.ndim == 2
    ), f"Invalid shape for inversion: expected a 2D square matrix, but got {x.shape}."
    if is_tensor(x):
        return torch.inverse(x)
    return np.linalg.inv(x)


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
        ord (Optional[Union[int, str]]): Order of the norm.
        dim (Optional[int]): Dimension along which to compute the norm.
        keepdim (Optional[bool]): Whether to keep the dimensions.

    Returns:
        ArrayLike: Norm of the array.
    """
    if dim is not None and dim >= x.ndim:
        assert (
            False
        ), f"Invalid dimension for norm: expected dimension less than {x.ndim}, but got {dim}."
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
    """
    assert (
        x.ndim >= 1 and y.ndim >= 1
    ), f"Invalid shape: expected at least 1D arrays, but got {x.shape} and {y.shape}."
    assert (
        x.shape[-1] == y.shape[0]
    ), f"Invalid shape for matmul: x dimensions {x.shape[-1]}, y dimensions {y.shape[0]}."
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
    """
    assert len(x.shape) == len(
        dims
    ), f"Invalid permutation: expected dimensions {len(x.shape)}, but got {len(dims)}."
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
    """
    assert (
        x.ndim == 2
    ), f"Invalid shape for trace: expected a 2D array, but got {x.shape}."
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
    """
    assert (
        x.ndim == 2
    ), f"Invalid shape for diag: expected a 2D array, but got {x.shape}."
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
    """
    assert isinstance(
        A, type(b)
    ), f"Invalid type: expected same type for both arrays, but got A:{type(A)} and b:{type(b)}"
    if is_tensor(A):
        return torch.linalg.solve(A, b)
    return np.linalg.solve(A, b)


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
        x (ArrayLike): Input 3D vector.

    Returns:
        ArrayLike: Skew-symmetric matrix.
    """
    assert x.shape in [
        (3,),
        (1, 3),
    ], f"Invalid shape. Shape of vector must be (3,) or (1,3), but got {str(x.shape)}"
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

    Arg:
        x (ArrayLike, [2,N] or [3,N]): Euclidean coordinates.

    Return:
        (ArrayLike, [3,N] or [4,N]): Converted homogeneous coordinates.

    Details:
    - [x y] -> [x y 1]
    - [x y z] -> [x y z 1]
    """
    if x.ndim != 2 and x.shape[0] != 2 and x.shape[0] != 3:
        LOG_ERROR(
            f"Invalid Shape. Input Shape must be (2,N) or (3,N), but got {x.shape}"
        )
        raise ValueError
    return concat([x, ones_like(x[0:1, :])], 0)


def dehomo(x: ArrayLike) -> ArrayLike:
    """
    Convert Homogeneous coordinates to Euclidean coordinates.

    Arg:
        x (ArrayLike, [3,N] or [4,N]): Homogeneous coordinates.

    Return:
        (ArrayLike, [2,N] or [3,N]): Converted euclidean coordinates.

    Details:
    - [x y z] -> [x/z y/z]
    - [x y z w] -> [x/w y/w z/w]
    """
    if x.ndim != 2 and x.shape[0] != 3 and x.shape[0] != 4:
        LOG_ERROR(
            f"Invalid Shape. Input Shape must be (3,N) or (4,N), but got {x.shape}"
        )
        raise ValueError
    euc_coords = x[:-1, :] / x[-1, :]
    return euc_coords
