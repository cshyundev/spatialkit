"""
Module Name: umath.py

Description: 
Unified Math (umath) module provides a unified interface for common mathematical operations that can be performed using both Numpy and Torch. 
This module helps to write agnostic code that can handle both Numpy arrays and Torch tensors seamlessly.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

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

from .uops import *
from scipy.linalg import expm
from ..common.constant import EPSILON
from ..common.logger import *


# Basic Mathematical Operations
def abs(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.abs(x)
    return np.abs(x)

def sqrt(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.sqrt(x)
    return np.sqrt(x)

def mean(x: ArrayLike, dim:Optional[int]=None) -> ArrayLike:
    if dim is not None:
        assert dim < x.ndim, f"Invalid dimension: expected dimension less than {x.ndim}, but got {dim}."
    if is_tensor(x): return torch.mean(x, dim=dim)
    return np.mean(x, axis=dim)

def dot(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    assert(type(x) == type(y)), f"Invalid type: expected same type for both arrays, but got {type(x)} and {type(y)}"
    if is_tensor(x): return torch.dot(x, y)
    return np.dot(x, y)

def qr(x :ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Invalid shape for QR: expected a 2D matrix, but got {x.shape}."
    if is_tensor(x): return torch.linalg.qr(x)
    return np.linalg.qr(x)

def svd(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Invalid shape for SVD: expected a 2D matrix, but got {x.shape}."
    if is_tensor(x): return torch.linalg.svd(x)
    return np.linalg.svd(x)

def determinant(x: ArrayLike) -> float:
    assert x.ndim == 2, f"Invalid shape for determinant: expected a 2D square matrix, but got {x.shape}."
    if is_tensor(x):
        return torch.det(x).item()
    return np.linalg.det(x)

def inv(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Invalid shape for inversion: expected a 2D square matrix, but got {x.shape}."
    if is_tensor(x): return torch.inverse(x)
    return np.linalg.inv(x)

# Matrix and Vector Operations
def norm(x: ArrayLike, ord: Optional[Union[int, str]]=None, dim:Optional[int]=None, keepdim:Optional[bool]=False) -> ArrayLike:
    if dim is not None and dim >= x.ndim:
        assert False, f"Invalid dimension for norm: expected dimension less than {x.ndim}, but got {dim}."
    if is_tensor(x): return torch.norm(x, p=ord, dim=dim, keepdim=keepdim)
    return np.linalg.norm(x, ord=ord, axis=dim, keepdims=keepdim)

def normalize(x: ArrayLike, ord:Optional[Union[int, str]]=None, dim:Optional[int]=None, eps:Optional[float]=EPSILON) -> ArrayLike:
    n = norm(x=x, ord=ord, dim=dim, keepdim=False)
    return x / (n + eps)

def matmul(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    assert x.ndim >= 1 and y.ndim >= 1, f"Invalid shape: expected at least 1D arrays, but got {x.shape} and {y.shape}."
    assert x.shape[-1] == y.shape[0], f"Invalid shape for matmul: x dimensions {x.shape[-1]}, y dimensions {y.shape[0]}."
    if is_tensor(x): return torch.matmul(x, y)
    return x @ y

def permute(x: ArrayLike, dims: Tuple[int]) -> ArrayLike:
    assert len(x.shape) == len(dims), f"Invalid permutation: expected dimensions {len(x.shape)}, but got {len(dims)}."
    if is_tensor(x): return x.permute(dims)
    return x.transpose(dims)

def trace(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Invalid shape for trace: expected a 2D array, but got {x.shape}."
    if is_tensor(x): return torch.trace(x)
    return np.trace(x)

def diag(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Invalid shape for diag: expected a 2D array, but got {x.shape}."
    if is_tensor(x): return torch.diag(x)
    return np.diag(x)

# Transform and Permutation
def rad2deg(x: ArrayLike) -> ArrayLike:
    return x * (180. / np.pi)

def deg2rad(x: ArrayLike) -> ArrayLike:
    return x * (np.pi / 180.)

def exponential_map(mat: ArrayLike) -> ArrayLike:
    if is_tensor(mat):
        return torch.matrix_exp(mat)
    else:
        return expm(mat)

# Trigonometric functions
def sin(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.sin(x)
    return np.sin(x)

def cos(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.cos(x)
    return np.cos(x)

def tan(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.tan(x)
    return np.tan(x)

def arcsin(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.arcsin(x)
    return np.arcsin(x)

def arccos(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.arccos(x)
    return np.arccos(x)

def arctan(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.arctan(x)
    return np.arctan(x)

def arctan2(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    if is_tensor(x):
        y = convert_tensor(y,x)
        return torch.arctan2(x, y)
    return np.arctan2(x,y)

# Polynomial functions
def polyval(coeffs: Union[ArrayLike,List[float]], x: ArrayLike) -> ArrayLike:
    y = zeros_like(x)
    for c in coeffs:
        y = y * x + c
    return y

def polyfit(x: ArrayLike, y: ArrayLike, degree: int) -> ArrayLike:
    if is_tensor(x):
        x_np = convert_numpy(x)
        y_np =  convert_numpy(y)
        coeffs_np = np.polyfit(x_np, y_np, degree)
        return convert_tensor(coeffs_np,x)
    else:
        return np.polyfit(x, y, degree)

# Linear-Algebra Problem
def is_square(x: ArrayLike) -> bool:
    return x.ndim == 2 and x.shape[0] == x.shape[1]

def solve(A:ArrayLike, b: ArrayLike) -> ArrayLike:
    assert(type(A) == type(b)), "Two ArrayLike must be same type."
    if is_tensor(A): return torch.linalg.solve(A,b)
    return np.linalg.solve(A,b)

def solve_linear_system(A:ArrayLike, b:Optional[ArrayLike]=None):
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
        return solve(A,b)    
    else:
        # Ax = 0
        # use svd
        _,s,vt = svd(A)
        null_space = transpose2d(vt)[:, s < EPSILON]
        return null_space
    
# Computer Vision
def vec3_to_skew(x: ArrayLike) -> ArrayLike:
    assert (x.shape == (3,) or x.shape == (1,3)), f"Invalid shape. Shape of vector must be (3,) or (1,3), but got {str(x.shape)}"
    if x.shape == (1,3): x = reduce_dim(x,0)
    wx = x[0].item()
    wy = x[1].item()
    wz = x[2].item()
    skew_x = np.array([[0., -wz, wy],
                    [wz,  0, -wx],
                    [-wy, wx,  0]])
    if is_tensor(x): skew_x = convert_tensor(skew_x, x)
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
        LOG_ERROR(f"Invalid Shape. Input Shape must be (2,N) or (3,N), but got {x.shape}")
        raise ValueError
    return concat([x, ones_like(x[0:1,:])], 0)

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
        LOG_ERROR(f"Invalid Shape. Input Shape must be (3,N) or (4,N), but got {x.shape}")
        raise ValueError
    euc_coords = x[:-1,:] / x[-1,:]
    return euc_coords