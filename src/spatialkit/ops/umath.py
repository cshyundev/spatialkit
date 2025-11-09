"""
Module Name: umath.py

Description: 
Unified Math (umath) module provides a unified interface for common mathematical operations that can be performed using both Numpy and Torch. 
This module helps to write agnostic code that can handle both Numpy arrays and Torch tensors seamlessly.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.2

License: MIT LICENSE

Usage:
>>> import numpy as np
>>> import torch
>>> from spatialkit import umath

>>> np_array = np.array([1., 2., 3., 4.])
>>> torch_tensor = torch.tensor([1., 2., 3., 4.])

>>> umath.mean(np_array)
2.5
>>> umath.mean(torch_tensor)
tensor(2.5000)
"""

from typing import Optional, Union, Tuple, List
from scipy.linalg import expm
import numpy as np
import torch

from .uops import *
from ..common.constant import EPSILON
from ..common.exceptions import (
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

def floor(x: ArrayLike) -> ArrayLike:
    """
    Compute the floor of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Floor of each element.
    """
    if is_tensor(x):
        return torch.floor(x)
    return np.floor(x)

def ceil(x: ArrayLike) -> ArrayLike:
    """
    Compute the ceiling of each element in the array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        ArrayLike: Ceiling of each element.
    """
    if is_tensor(x):
        return torch.ceil(x)
    return np.ceil(x)

def min(x: ArrayLike, dim: Optional[int] = None, keepdims: bool = False) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
    """
    Compute the minimum of elements along the specified dimension.

    Args:
        x (ArrayLike): Input array or tensor.
        dim (Optional[int]): Dimension along which to compute the minimum.
        keepdims (bool): Whether to keep the reduced dimension. Default is False.

    Returns:
        ArrayLike: Minimum of elements. For torch with dim specified, returns (values, indices).

    Raises:
        InvalidDimensionError: If dimension is out of range for the input array.
    """
    if dim is not None and dim >= x.ndim:
        raise InvalidDimensionError(
            f"Dimension {dim} is out of range for array with {x.ndim} dimensions. "
            f"Please use a dimension less than {x.ndim}."
        )

    if is_tensor(x):
        if dim is None:
            return torch.min(x)
        result = torch.min(x, dim=dim, keepdim=keepdims)
        return result.values  # Return only values, ignore indices for consistency
    else:
        if dim is None:
            return np.min(x)
        return np.min(x, axis=dim, keepdims=keepdims)


def max(x: ArrayLike, dim: Optional[int] = None, keepdims: bool = False) -> ArrayLike:
    """
    Compute the maximum of elements along the specified dimension.

    Args:
        x (ArrayLike): Input array or tensor.
        dim (Optional[int]): Dimension along which to compute the maximum.
        keepdims (bool): Whether to keep the reduced dimension. Default is False.

    Returns:
        ArrayLike: Maximum of elements.

    Raises:
        InvalidDimensionError: If dimension is out of range for the input array.
    """
    if dim is not None and dim >= x.ndim:
        raise InvalidDimensionError(
            f"Dimension {dim} is out of range for array with {x.ndim} dimensions. "
            f"Please use a dimension less than {x.ndim}."
        )

    if is_tensor(x):
        if dim is None:
            return torch.max(x)
        result = torch.max(x, dim=dim, keepdim=keepdims)
        return result.values  # Return only values, ignore indices for consistency
    else:
        if dim is None:
            return np.max(x)
        return np.max(x, axis=dim, keepdims=keepdims)


def mean(x: ArrayLike, dim: Optional[int] = None, keepdims: bool = False) -> ArrayLike:
    """
    Compute the mean of elements along the specified dimension.

    Args:
        x (ArrayLike): Input array or tensor.
        dim (Optional[int]): Dimension along which to compute the mean.
        keepdims (bool): Whether to keep the reduced dimension. Default is False.

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
        if dim is None:
            return torch.mean(x)
        return torch.mean(x, dim=dim, keepdim=keepdims)
    else:
        if dim is None:
            return np.mean(x)
        return np.mean(x, axis=dim, keepdims=keepdims)


def dot(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Compute the dot product of two arrays.

    Args:
        x (ArrayLike): First input array or tensor.
        y (ArrayLike): Second input array or tensor.
            Supports both float32 and float64. If dtypes differ, they are promoted to higher precision.

    Returns:
        ArrayLike: Dot product of x and y with promoted dtype.

    Raises:
        IncompatibleTypeError: If input arrays have different types (numpy vs tensor).
    """
    if not isinstance(x, type(y)):
        raise IncompatibleTypeError(
            f"Both arrays must be of the same type, got {type(x).__name__} and {type(y).__name__}. "
            f"Please ensure both inputs are either numpy arrays or torch tensors."
        )

    # Promote dtypes if different
    if x.dtype != y.dtype:
        target_dtype = promote_types(x, y)
        if is_tensor(x):
            x = x.type(target_dtype)
            y = y.type(target_dtype)
        else:
            x = x.astype(target_dtype)
            y = y.astype(target_dtype)

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
    Compute the Singular Value Decomposition (SVD) of a matrix or batch of matrices.

    Args:
        x (ArrayLike): Input matrix.
            - 2D array [M, N]: Single matrix
            - 3D array [B, M, N]: Batch of matrices (automatically handled)

    Returns:
        tuple[ArrayLike, ArrayLike, ArrayLike]: U, S, Vt matrices from SVD decomposition.
            - For 2D input: U[M,M], S[min(M,N)], Vt[N,N]
            - For 3D input: U[B,M,M], S[B,min(M,N)], Vt[B,N,N]

    Raises:
        InvalidDimensionError: If input is not 2D or 3D array.
        NumericalError: If SVD decomposition fails.

    Example:
        >>> import numpy as np
        >>> # Single matrix
        >>> A = np.random.rand(4, 3)
        >>> U, S, Vt = svd(A)
        >>> # Batch of matrices
        >>> A_batch = np.random.rand(10, 4, 3)
        >>> U, S, Vt = svd(A_batch)
    """
    if x.ndim not in [2, 3]:
        raise InvalidDimensionError(
            f"SVD requires 2D or 3D array, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is either a single matrix [M,N] or batch of matrices [B,M,N]."
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


def cross(a: ArrayLike, b: ArrayLike, dim: int = -1) -> ArrayLike:
    """
    Compute the cross product of two 3D vectors.

    Args:
        a (ArrayLike): First input vector (must have 3 components along specified dimension).
        b (ArrayLike): Second input vector (must have 3 components along specified dimension).
        dim (int): Dimension along which to compute cross product. Default is -1 (last dimension).

    Returns:
        ArrayLike: Cross product vector with same type as input.

    Raises:
        InvalidShapeError: If vectors don't have exactly 3 components.
        IncompatibleTypeError: If inputs have different types.

    Example:
        >>> a = np.array([1, 0, 0])
        >>> b = np.array([0, 1, 0])
        >>> cross(a, b)
        array([0, 0, 1])
    """


    if not (is_array(a) and is_array(b)):
        raise IncompatibleTypeError(
            f"Both inputs must be arrays, got {type(a)} and {type(b)}."
        )

    if is_tensor(a) != is_tensor(b):
        raise IncompatibleTypeError(
            f"Both inputs must be of same type (both NumPy or both PyTorch), "
            f"got {type(a)} and {type(b)}."
        )

    if is_tensor(a):
        if a.shape[dim] != 3 or b.shape[dim] != 3:
            raise InvalidShapeError(
                f"Cross product requires 3-component vectors, "
                f"got shapes {a.shape} and {b.shape} along dim={dim}."
            )
        return torch.cross(a, b, dim=dim)
    else:
        if a.shape[dim] != 3 or b.shape[dim] != 3:
            raise InvalidShapeError(
                f"Cross product requires 3-component vectors, "
                f"got shapes {a.shape} and {b.shape} along axis={dim}."
            )
        return np.cross(a, b, axis=dim)


def cdist(x1: ArrayLike, x2: ArrayLike, p: float = 2.0) -> ArrayLike:
    """
    Compute pairwise distance between two sets of points.

    Args:
        x1 (ArrayLike, [N, D]): First set of points.
        x2 (ArrayLike, [M, D]): Second set of points.
        p (float): Order of the norm. Default is 2.0 (Euclidean distance).

    Returns:
        ArrayLike: Pairwise distance matrix with shape [N, M].

    Raises:
        InvalidDimensionError: If inputs are not 2D arrays.
        InvalidShapeError: If feature dimensions don't match.
        IncompatibleTypeError: If inputs have different types.

    Example:
        >>> x1 = np.array([[0, 0], [1, 1]])
        >>> x2 = np.array([[0, 0], [2, 2]])
        >>> cdist(x1, x2)
        array([[0.        , 2.82842712],
               [1.41421356, 1.41421356]])
    """
    if x1.ndim != 2 or x2.ndim != 2:
        raise InvalidDimensionError(
            f"cdist requires 2D arrays, got shapes {x1.shape} and {x2.shape}. "
            f"Please ensure both inputs are 2D arrays with shape [N, D] and [M, D]."
        )

    if x1.shape[1] != x2.shape[1]:
        raise InvalidShapeError(
            f"Feature dimensions must match, got {x1.shape[1]} and {x2.shape[1]}. "
            f"Please ensure both arrays have the same number of features (dimension 1)."
        )

    if is_tensor(x1) != is_tensor(x2):
        raise IncompatibleTypeError(
            f"Both inputs must be of same type, got {type(x1).__name__} and {type(x2).__name__}. "
            f"Please ensure both inputs are either numpy arrays or torch tensors."
        )

    if is_tensor(x1):
        return torch.cdist(x1, x2, p=p)
    else:
        # NumPy doesn't have cdist in base, use scipy or manual computation
        from scipy.spatial.distance import cdist as scipy_cdist
        return scipy_cdist(x1, x2, metric='minkowski', p=p)


def topk(x: ArrayLike, k: int, dim: int = -1, largest: bool = True) -> Tuple[ArrayLike, ArrayLike]:
    """
    Find the k largest or smallest elements along a dimension.

    Args:
        x (ArrayLike): Input array or tensor.
        k (int): Number of top elements to return.
        dim (int): Dimension along which to find top-k. Default is -1 (last dimension).
        largest (bool): If True, return k largest elements; if False, return k smallest. Default is True.

    Returns:
        Tuple[ArrayLike, ArrayLike]: (values, indices) of the top-k elements.

    Raises:
        InvalidDimensionError: If dimension is out of range.
        InvalidArgumentError: If k is larger than the dimension size.

    Example:
        >>> x = np.array([[3, 1, 4], [1, 5, 9]])
        >>> values, indices = topk(x, k=2, dim=1, largest=True)
        >>> values
        array([[4, 3],
               [9, 5]])
    """
    if dim >= x.ndim or dim < -x.ndim:
        raise InvalidDimensionError(
            f"Dimension {dim} is out of range for array with {x.ndim} dimensions. "
            f"Please use a dimension in range [{-x.ndim}, {x.ndim})."
        )

    # Normalize negative dimension
    if dim < 0:
        dim = x.ndim + dim

    if k > x.shape[dim]:
        from ..common.exceptions import InvalidArgumentError
        raise InvalidArgumentError(
            f"k={k} is larger than dimension size {x.shape[dim]}. "
            f"Please use k <= {x.shape[dim]}."
        )

    if is_tensor(x):
        return torch.topk(x, k, dim=dim, largest=largest)
    else:
        # NumPy implementation
        if largest:
            indices = np.argpartition(x, -k, axis=dim)
            indices = np.take(indices, np.arange(-k, 0), axis=dim)
            # Sort the top-k to match PyTorch behavior
            values = np.take_along_axis(x, indices, axis=dim)
            sorted_idx = np.argsort(-values, axis=dim)
            indices = np.take_along_axis(indices, sorted_idx, axis=dim)
            values = np.take_along_axis(values, sorted_idx, axis=dim)
        else:
            indices = np.argpartition(x, k, axis=dim)
            indices = np.take(indices, np.arange(k), axis=dim)
            values = np.take_along_axis(x, indices, axis=dim)
            sorted_idx = np.argsort(values, axis=dim)
            indices = np.take_along_axis(indices, sorted_idx, axis=dim)
            values = np.take_along_axis(values, sorted_idx, axis=dim)

        return values, indices


def knn(
    src: ArrayLike,
    tgt: ArrayLike,
    k: int = 1,
    batch_size: Optional[int] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Find k-nearest neighbors using scipy.spatial.cKDTree for optimal performance.

    DEPRECATION WARNING: This function will be removed in a future version.
    It is kept temporarily for backward compatibility during migration.
    Use scipy.spatial.cKDTree or sklearn.neighbors.NearestNeighbors directly instead.

    Args:
        src (ArrayLike, [N, D]): Source points to find neighbors for.
        tgt (ArrayLike, [M, D]): Target points to search from.
        k (int): Number of nearest neighbors. Default is 1.
        batch_size (Optional[int]): DEPRECATED. No longer used with KDTree implementation.

    Returns:
        Tuple[ArrayLike, ArrayLike]: (indices, distances) where:
            - indices: [N, k] indices of nearest neighbors in tgt
            - distances: [N, k] distances to nearest neighbors

    Raises:
        InvalidDimensionError: If inputs are not 2D arrays.
        InvalidShapeError: If feature dimensions don't match.
        IncompatibleTypeError: If inputs have different types.
        InvalidArgumentError: If k is invalid.

    Warning:
        If tensor inputs are provided, they will be converted to numpy arrays,
        breaking differentiability. This function is NOT differentiable.

    Example:
        >>> src = np.array([[0, 0], [1, 1], [2, 2]])
        >>> tgt = np.array([[0, 0], [0.5, 0.5], [2, 2]])
        >>> indices, distances = knn(src, tgt, k=2)
        >>> indices.shape
        (3, 2)

    Note:
        This implementation uses scipy.spatial.cKDTree, which is 60-70x faster
        than the previous cdist-based implementation for large point clouds.
    """
    # Validation
    if src.ndim != 2 or tgt.ndim != 2:
        raise InvalidDimensionError(
            f"knn requires 2D arrays, got shapes {src.shape} and {tgt.shape}. "
            f"Please ensure both inputs are 2D arrays with shape [N, D] and [M, D]."
        )

    if src.shape[1] != tgt.shape[1]:
        raise InvalidShapeError(
            f"Feature dimensions must match, got {src.shape[1]} and {tgt.shape[1]}. "
            f"Please ensure both arrays have the same number of features (dimension 1)."
        )

    if is_tensor(src) != is_tensor(tgt):
        raise IncompatibleTypeError(
            f"Both inputs must be of same type, got {type(src).__name__} and {type(tgt).__name__}. "
            f"Please ensure both inputs are either numpy arrays or torch tensors."
        )

    from ..common.exceptions import InvalidArgumentError
    if k <= 0:
        raise InvalidArgumentError(f"k must be positive, got {k}.")

    if k > tgt.shape[0]:
        raise InvalidArgumentError(
            f"k={k} is larger than target size {tgt.shape[0]}. "
            f"Please use k <= {tgt.shape[0]}."
        )

    # Warn about tensor conversion (breaks differentiability)
    input_is_tensor = is_tensor(src)
    if input_is_tensor:
        from ..common.logger import LOG_WARN
        LOG_WARN(
            "knn() received tensor inputs but will convert to numpy for KDTree computation. "
            "This breaks differentiability! Consider using a differentiable KNN implementation "
            "or perform KNN operations outside the computational graph. "
            "This function is deprecated and will be removed in a future version."
        )

    # Convert to numpy if needed
    from .uops import convert_numpy
    src_np = convert_numpy(src) if input_is_tensor else src
    tgt_np = convert_numpy(tgt) if input_is_tensor else tgt

    # Use scipy.spatial.cKDTree for fast KNN search
    from scipy.spatial import cKDTree

    tree = cKDTree(tgt_np)
    distances, indices = tree.query(src_np, k=k, workers=-1)  # workers=-1 uses all CPUs

    # Ensure consistent output shape
    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Convert back to tensor if input was tensor
    if input_is_tensor:
        from .uops import convert_tensor
        indices = convert_tensor(indices, src)
        distances = convert_tensor(distances, src)

    return indices, distances


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
    n = norm(x=x, order=order, dim=dim, keepdim=True)
    return x / (n + eps)


def matmul(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Perform matrix multiplication between two arrays.

    Args:
        x (ArrayLike): First input array or tensor.
        y (ArrayLike): Second input array or tensor.
            Supports both float32 and float64. If dtypes differ, they are promoted to higher precision.

    Returns:
        ArrayLike: Result of matrix multiplication with promoted dtype.

    Raises:
        InvalidDimensionError: If input arrays are not at least 1D.
        InvalidShapeError: If matrix dimensions are incompatible for multiplication.
        IncompatibleTypeError: If mixing numpy and tensor types.
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

    # Check type compatibility and promote dtypes if different
    if not isinstance(x, type(y)):
        raise IncompatibleTypeError(
            f"Both arrays must be of the same type, got {type(x).__name__} and {type(y).__name__}. "
            f"Please ensure both inputs are either numpy arrays or torch tensors."
        )

    if x.dtype != y.dtype:
        target_dtype = promote_types(x, y)
        if is_tensor(x):
            x = x.type(target_dtype)
            y = y.type(target_dtype)
        else:
            x = x.astype(target_dtype)
            y = y.astype(target_dtype)

    if is_tensor(x):
        return torch.matmul(x, y)
    return x @ y


def bmm3d(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Perform batch matrix multiplication for 3D arrays only.

    This function multiplies batches of matrices: x[i] @ y[i] for all i.
    Both inputs MUST be exactly 3-dimensional with compatible shapes.

    Args:
        x (ArrayLike, [B, N, M]): First batch of matrices (3D only).
        y (ArrayLike, [B, M, K]): Second batch of matrices (3D only).

    Returns:
        ArrayLike, [B, N, K]: Batch matrix multiplication result.

    Raises:
        InvalidDimensionError: If inputs are not exactly 3D.
        InvalidShapeError: If batch dimensions or matrix dimensions are incompatible.
        IncompatibleTypeError: If mixing numpy and tensor types.

    Example:
        >>> import numpy as np
        >>> x = np.random.rand(10, 3, 4)  # batch=10, 3x4 matrices
        >>> y = np.random.rand(10, 4, 5)  # batch=10, 4x5 matrices
        >>> result = bmm3d(x, y)  # shape: [10, 3, 5]
        >>> result.shape
        (10, 3, 5)

    Note:
        This function ONLY supports 3D inputs. For general batched matmul
        with arbitrary dimensions, use matmul() instead.
    """
    if x.ndim != 3 or y.ndim != 3:
        raise InvalidDimensionError(
            f"bmm3d requires exactly 3D arrays, got shapes {x.shape} and {y.shape}. "
            f"Please ensure both inputs are 3D arrays with shape [B, N, M] and [B, M, K]."
        )

    if x.shape[0] != y.shape[0]:
        raise InvalidShapeError(
            f"Batch dimensions must match, got {x.shape[0]} and {y.shape[0]}. "
            f"Please ensure both arrays have the same batch size (dimension 0)."
        )

    if x.shape[2] != y.shape[1]:
        raise InvalidShapeError(
            f"Matrix dimensions are incompatible for multiplication: "
            f"x[{x.shape[0]}, {x.shape[1]}, {x.shape[2]}] @ y[{y.shape[0]}, {y.shape[1]}, {y.shape[2]}]. "
            f"Inner dimensions must match: {x.shape[2]} != {y.shape[1]}."
        )

    if is_tensor(x) != is_tensor(y):
        raise IncompatibleTypeError(
            f"Both inputs must be of same type, got {type(x).__name__} and {type(y).__name__}. "
            f"Please ensure both inputs are either numpy arrays or torch tensors."
        )

    if is_tensor(x):
        return torch.bmm(x, y)
    else:
        # NumPy: use einsum for efficient batch matrix multiplication
        return np.einsum('bij,bjk->bik', x, y)


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
            Supports both float32 and float64. If dtypes differ, they are promoted to higher precision.

    Returns:
        ArrayLike: Element-wise arctangent of x/y with promoted dtype.

    Raises:
        IncompatibleTypeError: If mixing numpy and tensor types.
    """
    # Ensure same type (numpy or tensor)
    if not isinstance(x, type(y)):
        if is_tensor(x):
            y = convert_tensor(y, x)
        else:
            raise IncompatibleTypeError(
                f"Both arrays must be of the same type, got {type(x).__name__} and {type(y).__name__}. "
                f"Please ensure both inputs are either numpy arrays or torch tensors."
            )

    # Promote dtypes if different
    if x.dtype != y.dtype:
        target_dtype = promote_types(x, y)
        if is_tensor(x):
            x = x.type(target_dtype)
            y = y.type(target_dtype)
        else:
            x = x.astype(target_dtype)
            y = y.astype(target_dtype)

    if is_tensor(x):
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
            Supports both float32 and float64. If dtypes differ, they are promoted to higher precision.

    Returns:
        ArrayLike: Solution to the system of equations with promoted dtype.

    Raises:
        IncompatibleTypeError: If input arrays have different types (numpy vs tensor).
        NumericalError: If the linear system cannot be solved.
    """
    if not isinstance(A, type(b)):
        raise IncompatibleTypeError(
            f"Both arrays must be of the same type, got A:{type(A).__name__} and b:{type(b).__name__}. "
            f"Please ensure both inputs are either numpy arrays or torch tensors."
        )

    # Promote dtypes if different
    if A.dtype != b.dtype:
        target_dtype = promote_types(A, b)
        if is_tensor(A):
            A = A.type(target_dtype)
            b = b.type(target_dtype)
        else:
            A = A.astype(target_dtype)
            b = b.astype(target_dtype)

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


__all__ = [
    # Basic math operations
    "abs",
    "sqrt",
    "floor",
    "ceil",
    "mean",
    "dot",
    # Linear algebra decompositions
    "qr",
    "svd",
    # Matrix operations
    "determinant",
    "inv",
    "norm",
    "normalize",
    "matmul",
    "bmm3d",
    "permute",
    "trace",
    "diag",
    # Trigonometric functions
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    # Unit conversion
    "rad2deg",
    "deg2rad",
    # Matrix exponential
    "exponential_map",
    # Polynomial functions
    "polyval",
    "polyfit",
    # Linear systems
    "is_square",
    "solve",
    "solve_linear_system",
    # Computer vision helpers
    "vec3_to_skew",
    "homo",
    "dehomo",
    # Vector operations
    "cross",
    # Distance and nearest neighbors
    "cdist",
    "topk",
    "knn",
]
