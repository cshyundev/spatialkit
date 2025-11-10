"""
Module Name: uops.py

Description: 
Unified Operations (uops) module provides a unified interface for common operations that can be performed using both Numpy and Torch. 
This module helps to write agnostic code that can handle both Numpy arrays and Torch tensors seamlessly.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.2

License: MIT LICENSE

Usage:
>>> import numpy as np
>>> import torch
>>> from spatialkit import uops

>>> np_array = np.array([1, 2, 3])
>>> torch_tensor = torch.tensor([1, 2, 3])

>>> ones_np = uops.ones_like(np_array)
>>> ones_torch = uops.ones_like(torch_tensor)

>>> print(ones_np)
array([1, 1, 1])
>>> print(ones_torch)
tensor([1, 1, 1])
"""

from typing import *
import numpy as np
from numpy import ndarray
from torch import Tensor
import torch

from ..common.exceptions import (
    IncompatibleTypeError,
    InvalidDimensionError,
    InvalidShapeError,
    InvalidArgumentError
)

ArrayLike = Union[ndarray, Tensor]  # Unified ArrayType


def is_tensor(x: ArrayLike) -> bool:
    """
    Checks if the input is a Torch tensor.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        bool: True if the input is a Torch tensor, False otherwise.
    """
    return isinstance(x, Tensor)


def is_numpy(x: ArrayLike) -> bool:
    """
    Checks if the input is a Numpy array.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        bool: True if the input is a Numpy array, False otherwise.
    """
    return isinstance(x, ndarray) or isinstance(x, np.generic)


def is_array(x: Any) -> bool:
    """
    Checks if the input is either a Numpy array or a Torch tensor.

    Args:
        x (Any): The input object.

    Returns:
        bool: True if the input is either a Numpy array or a Torch tensor, False otherwise.
    """
    return is_tensor(x) or is_numpy(x)


def convert_tensor(x: ArrayLike, tensor: Optional[Tensor] = None) -> Tensor:
    """
    Converts an input to a Torch tensor.

    Args:
        x (ArrayLike): The input array-like object.
        tensor (Optional[Tensor]): An optional Torch tensor to specify the desired dtype and device for the conversion.

    Returns:
        Tensor: The converted Torch tensor.
        
    Raises:
        IncompatibleTypeError: If tensor parameter is not a Torch tensor.
    """
    if is_tensor(x):
        return x
    if tensor is not None:
        if not is_tensor(tensor):
            raise IncompatibleTypeError("Expected tensor parameter to be a Torch tensor")
        x_tensor = torch.tensor(x, dtype=tensor.dtype, device=tensor.device)
    else:
        x_tensor = Tensor(x)
    return x_tensor


def convert_numpy(x: ArrayLike) -> ndarray:
    """
    Converts an input to a Numpy array.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        ndarray: The converted Numpy array.
    """
    if is_tensor(x):
        x_numpy = x.detach().cpu().numpy()
    elif is_numpy(x):
        x_numpy = x
    else:
        x_numpy = np.array(x)
    return x_numpy


def convert_array(x: Any, array: ArrayLike) -> ArrayLike:
    """
    Converts an input to either a Numpy array or Torch tensor based on a reference array.

    Args:
        x (Any): The input object to convert.
        array (ArrayLike): The reference array to determine the conversion type.

    Returns:
        ArrayLike: The converted array-like object.
    """
    if is_tensor(array):
        return convert_tensor(x, array)
    return convert_numpy(x)


def numel(x: ArrayLike) -> int:
    """
    Returns the number of elements in the input array.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        int: The number of elements in the array.
        
    Raises:
        IncompatibleTypeError: If input is not a numpy array or Torch tensor.
    """
    if not is_array(x):
        raise IncompatibleTypeError("Invalid type. Input type must be either ndarray or Tensor.")
    if is_tensor(x):
        return x.numel()
    return x.size


def _assert_same_array_type(arrays: Tuple[ArrayLike, ...]):
    """
    Validates that all input arrays are of the same type.

    Args:
        arrays (Tuple[ArrayLike, ...]): A tuple of array-like objects.

    Raises:
        IncompatibleTypeError: If the input arrays are not of the same type.
    """
    if not (all(is_tensor(arr) for arr in arrays) or all(is_numpy(arr) for arr in arrays)):
        raise IncompatibleTypeError("All input arrays must be of the same type")


def convert_dict_tensor(
    dict: Dict[Any, ndarray], tensor: Tensor = None
) -> Dict[Any, Tensor]:
    """
    Converts a dictionary of Numpy arrays to Torch tensors.

    Args:
        dict (Dict[Any, ndarray]): A dictionary with Numpy arrays as values.
        tensor (Tensor, optional): A reference tensor for dtype and device.

    Returns:
        Dict[Any, Tensor]: A dictionary with Torch tensors as values.
    """
    _assert_same_array_type(dict)
    new_dict = {}
    for key in dict.keys():
        new_dict[key] = convert_tensor(dict[key], tensor)
    return new_dict


def expand_dim(x: ArrayLike, dim: int) -> ArrayLike:
    """
    Expands the dimensions of an array.

    Args:
        x (ArrayLike): The input array-like object.
        dim (int): The dimension index to expand.

    Returns:
        ArrayLike: The array with expanded dimensions.
    """
    if is_tensor(x):
        return x.unsqueeze(dim)
    else:
        return np.expand_dims(x, axis=dim)


def reduce_dim(x: ArrayLike, dim: int) -> ArrayLike:
    """
    Reduces the dimensions of an array.

    Args:
        x (ArrayLike): The input array-like object.
        dim (int): The dimension index to reduce.

    Returns:
        ArrayLike: The array with reduced dimensions.
    """
    if is_tensor(x):
        return x.squeeze(dim)
    else:
        return np.squeeze(x, axis=dim)


def concat(x: List[ArrayLike], dim: int) -> ArrayLike:
    """
    Concatenates a list of arrays along a specified dimension.

    Args:
        x (List[ArrayLike]): A list of array-like objects to concatenate.
        dim (int): The dimension along which to concatenate.

    Returns:
        ArrayLike: The concatenated array.
    """
    _assert_same_array_type(x)
    if is_tensor(x[0]):
        return torch.cat(x, dim=dim)
    return np.concatenate(x, axis=dim)


def stack(x: List[ArrayLike], dim: int) -> ArrayLike:
    """
    Stacks a list of arrays along a specified dimension.

    Args:
        x (List[ArrayLike]): A list of array-like objects to stack.
        dim (int): The dimension along which to stack.

    Returns:
        ArrayLike: The stacked array.
    """
    _assert_same_array_type(x)
    if is_tensor(x[0]):
        return torch.stack(x, dim=dim)
    return np.stack(x, axis=dim)


def ones_like(x: ArrayLike) -> ArrayLike:
    """
    Returns an array of ones with the same shape and type as the input.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        ArrayLike: An array of ones with the same shape and type as the input.
        
    Raises:
        IncompatibleTypeError: If input is neither a numpy array nor a Torch tensor.
    """
    if not is_array(x):
        raise IncompatibleTypeError("Invalid Type. It is neither Numpy nor Tensor.")
    if is_tensor(x):
        return torch.ones_like(x)
    return np.ones_like(x)


def ones(shape: Union[int, Tuple[int, ...]], dtype: Optional[Any] = None, like: Optional[ArrayLike] = None) -> ArrayLike:
    """
    Returns a new array of given shape and type, filled with ones.

    Args:
        shape (Union[int, Tuple[int, ...]]): Shape of the new array.
        dtype (Optional[Any]): Desired data type for the array.
        like (Optional[ArrayLike]): Reference array to determine whether to create numpy or torch array.

    Returns:
        ArrayLike: Array of ones with specified shape and type.
    """
    if like is not None and is_tensor(like):
        if dtype is None:
            return torch.ones(shape)
        return torch.ones(shape, dtype=dtype)
    else:
        if dtype is None:
            return np.ones(shape)
        return np.ones(shape, dtype=dtype)


def zeros(shape: Union[int, Tuple[int, ...]], dtype: Optional[Any] = None, like: Optional[ArrayLike] = None) -> ArrayLike:
    """
    Returns a new array of given shape and type, filled with zeros.

    Args:
        shape (int or Tuple[int, ...]): Shape of the new array.
        dtype (Optional[Any]): Data type of the array. Default is None (float64 for NumPy, float32 for PyTorch).
        like (Optional[ArrayLike]): Reference array to determine whether to return NumPy or PyTorch. Default is None (NumPy).

    Returns:
        ArrayLike: Array of zeros with the specified shape and type.

    Example:
        >>> zeros((3, 3))
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
    """
    if like is not None and is_tensor(like):
        if dtype is None:
            return torch.zeros(shape)
        return torch.zeros(shape, dtype=dtype)
    else:
        if dtype is None:
            return np.zeros(shape)
        return np.zeros(shape, dtype=dtype)

def any(x: ArrayLike, dim: Optional[int] = None, keepdims: bool = False) -> ArrayLike:
    """
    Tests whether any array elements along a given dimension evaluate to True.

    Args:
        x (ArrayLike): Input array or tensor.
        dim (Optional[int]): Dimension along which to perform the test. If None, tests all elements.
        keepdims (bool): Whether to keep the reduced dimension. Default is False.

    Returns:
        ArrayLike: Boolean array indicating if any elements are True along the specified dimension.
    """
    if is_tensor(x):
        if dim is None:
            return torch.any(x)
        return torch.any(x, dim=dim, keepdim=keepdims)
    else:
        if dim is None:
            return np.any(x)
        return np.any(x, axis=dim, keepdims=keepdims)


def zeros_like(x: ArrayLike) -> ArrayLike:
    """
    Returns an array of zeros with the same shape and type as the input.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        ArrayLike: An array of zeros with the same shape and type as the input.
    """
    if is_tensor(x):
        return torch.zeros_like(x)
    return np.zeros_like(x)


def empty_like(x: ArrayLike) -> ArrayLike:
    """
    Returns an uninitialized array with the same shape and type as the input.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        ArrayLike: An uninitialized array with the same shape and type as the input.
    """
    if is_tensor(x):
        return torch.empty_like(x)
    return np.empty_like(x)


def full_like(x: ArrayLike, fill_value: Any, dtype: Any = None) -> ArrayLike:
    """
    Returns an array filled with a specified value, with the same shape and type as the input.

    Args:
        x (ArrayLike): The input array-like object.
        fill_value (Any): The value to fill the array with.
        dtype (Any, optional): The desired data type of the output array.

    Returns:
        ArrayLike: An array filled with the specified value.
    """
    if is_tensor(x):
        return torch.full_like(x, fill_value, dtype=dtype)
    return np.full_like(a=x, fill_value=fill_value, dtype=dtype)


def arange(x: ArrayLike, start: Any, stop: Any = None, step: int = 1, dtype=None):
    """
    Returns evenly spaced values within a given interval.

    Args:
        x (ArrayLike): The input array-like object.
        start (Any): The start of the interval.
        stop (Any, optional): The end of the interval. If None, start is used as stop and 0 as start.
        step (int, optional): The spacing between values. Default is 1.
        dtype (Any, optional): The desired data type of the output array.

    Returns:
        ArrayLike: An array of evenly spaced values.
    """
    if stop is None:
        stop = start
        start = 0
    if is_tensor(x):
        return torch.arange(start, stop, step, dtype=dtype)
    return np.arange(start, stop, step, dtype=dtype)


def deep_copy(x: ArrayLike) -> ArrayLike:
    """
    Returns a deep copy of the input array.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        ArrayLike: A deep copy of the input array.
    """
    if is_tensor(x):
        return x.clone()
    return np.copy(x)


def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Returns elements chosen from two arrays based on a condition.

    Args:
        condition (ArrayLike): The condition array.
        x (ArrayLike): The array to choose elements from when the condition is True.
        y (ArrayLike): The array to choose elements from when the condition is False.

    Returns:
        ArrayLike: An array with elements chosen based on the condition.
    """
    if is_tensor(condition):
        return torch.where(condition, x, y)
    return np.where(condition, x, y)


def clip(x: ArrayLike, min: float = None, max: float = None) -> ArrayLike:
    """
    Clips the values of an array within a specified range.

    Args:
        x (ArrayLike): The input array-like object.
        min (float, optional): The minimum value to clip to.
        max (float, optional): The maximum value to clip to.

    Returns:
        ArrayLike: The clipped array.
    """
    if is_tensor(x):
        return torch.clip(x, min, max)
    return np.clip(x, min, max)


def eye(n: int, x: ArrayLike) -> ArrayLike:
    """
    Returns a 2-D identity matrix.

    Args:
        n (int): The number of rows and columns in the identity matrix.
        x (ArrayLike): The input array-like object to determine the type.

    Returns:
        ArrayLike: A 2-D identity matrix.
    """
    if is_tensor(x):
        return torch.eye(n)
    return np.eye(n)


def transpose2d(x: ArrayLike) -> ArrayLike:
    """
    Transposes a 2-D array.

    Args:
        x (ArrayLike): The input 2-D array-like object.

    Returns:
        ArrayLike: The transposed array.

    Raises:
        InvalidShapeError: If the input array is not 2-D.
    """
    if x.ndim != 2:
        raise InvalidShapeError(f"Invalid shape for transpose: expected a 2D array, but got {x.shape}.")
    if is_tensor(x):
        return x.transpose(0, 1)
    return x.T


def swapaxes(x: ArrayLike, axis0: int, axis1: int) -> ArrayLike:
    """
    Swaps two axes of an array.

    Args:
        x (ArrayLike): The input array-like object.
        axis0 (int): The first axis to swap.
        axis1 (int): The second axis to swap.

    Returns:
        ArrayLike: The array with swapped axes.
    """
    if is_tensor(x):
        return torch.swapaxes(x, axis0, axis1)
    return np.swapaxes(x, axis0, axis1)


def as_bool(x: ArrayLike) -> ArrayLike:
    """
    Converts an array to boolean type.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        ArrayLike: The array converted to boolean type.
    """
    if is_tensor(x):
        return x.type(torch.bool)
    return x.astype(bool)


def as_int(x: ArrayLike, n: int = 32) -> ArrayLike:
    """
    Converts an array to integer type with specified bit-width.

    Args:
        x (ArrayLike): The input array-like object.
        n (int, optional): The bit-width of the integer type. Default is 32.

    Returns:
        ArrayLike: The array converted to integer type.
        
    Raises:
        InvalidDimensionError: If the specified bit-width is not supported.
    """
    if is_tensor(x):
        if n == 64:
            return x.type(torch.int64)
        elif n == 32:
            return x.type(torch.int32)
        elif n == 16:
            return x.type(torch.int16)
        else:
            raise InvalidDimensionError(f"Unsupported bit-width {n} for int conversion.")
    elif is_numpy(x):
        if n == 64:
            return x.astype(np.int64)
        elif n == 32:
            return x.astype(np.int32)
        elif n == 16:
            return x.astype(np.int16)
        else:
            raise InvalidDimensionError(f"Unsupported bit-width {n} for int conversion. Supported: 16, 32, 64.")


def as_float(x: ArrayLike, n: int = 32) -> ArrayLike:
    """
    Converts an array to float type with specified bit-width.

    Args:
        x (ArrayLike): The input array-like object.
        n (int, optional): The bit-width of the float type. Default is 32.

    Returns:
        ArrayLike: The array converted to float type.

    Raises:
        InvalidDimensionError: If the specified bit-width is not supported.
    """
    if is_tensor(x):
        if n == 64:
            return x.type(torch.float64)
        elif n == 32:
            return x.type(torch.float32)
        elif n == 16:
            return x.type(torch.float16)
        else:
            raise InvalidDimensionError(f"Unsupported bit-width {n} for float conversion.")
    elif is_numpy(x):
        if n == 64:
            return x.astype(np.float64)
        elif n == 32:
            return x.astype(np.float32)
        elif n == 16:
            return x.astype(np.float16)
        else:
            raise InvalidDimensionError(f"Unsupported bit-width {n} for float conversion. Supported: 16, 32, 64.")


def astype_like(x: ArrayLike, reference: ArrayLike) -> ArrayLike:
    """
    Converts an array to the same dtype as a reference array.

    This function ensures dtype consistency between arrays, which is particularly
    useful for avoiding unwanted dtype promotion in operations (e.g., numpy 2.0+).

    Args:
        x (ArrayLike): The input array to convert.
        reference (ArrayLike): The reference array whose dtype will be used.

    Returns:
        ArrayLike: Array with the same dtype as reference.

    Raises:
        IncompatibleTypeError: If x and reference are not both numpy arrays or both torch tensors.

    Example:
        >>> import numpy as np
        >>> x = np.array([1.0, 2.0], dtype=np.float64)
        >>> ref = np.array([0.0], dtype=np.float32)
        >>> result = astype_like(x, ref)
        >>> result.dtype
        dtype('float32')

        >>> import torch
        >>> x_t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        >>> ref_t = torch.tensor([0.0], dtype=torch.float32)
        >>> result_t = astype_like(x_t, ref_t)
        >>> result_t.dtype
        torch.float32
    """
    x_is_tensor = is_tensor(x)
    ref_is_tensor = is_tensor(reference)

    if x_is_tensor != ref_is_tensor:
        raise IncompatibleTypeError(
            f"x and reference must be the same type (both numpy or both torch). "
            f"Got x: {type(x).__name__}, reference: {type(reference).__name__}"
        )

    if is_tensor(x):
        return x.type(reference.dtype)
    else:
        return x.astype(reference.dtype)


def logical_or(*arrays: ArrayLike) -> ArrayLike:
    """
    Computes the element-wise logical OR of input arrays.

    Args:
        *arrays (ArrayLike): A variable number of array-like objects.

    Returns:
        ArrayLike: The result of the logical OR operation.
        
    Raises:
        InvalidDimensionError: If no input arrays are provided.
        IncompatibleTypeError: If input arrays are not of the same type.
    """
    if len(arrays) == 0:
        raise InvalidDimensionError("At least one input array is required")
    _assert_same_array_type(arrays)

    result = arrays[0]
    for arr in arrays[1:]:
        if is_tensor(result):
            result = torch.logical_or(result, arr)
        else:
            result = np.logical_or(result, arr)

    return result


def logical_and(*arrays: ArrayLike) -> ArrayLike:
    """
    Computes the element-wise logical AND of input arrays.

    Args:
        *arrays (ArrayLike): A variable number of array-like objects.

    Returns:
        ArrayLike: The result of the logical AND operation.
        
    Raises:
        InvalidDimensionError: If fewer than two input arrays are provided.
        IncompatibleTypeError: If input arrays are not of the same type.
    """
    if len(arrays) <= 1:
        raise InvalidDimensionError("At least two input arrays are required")
    _assert_same_array_type(arrays)

    result = arrays[0]
    for arr in arrays[1:]:
        if is_tensor(result):
            result = torch.logical_and(result, arr)
        else:
            result = np.logical_and(result, arr)
    return result


def logical_not(x: ArrayLike) -> ArrayLike:
    """
    Computes the element-wise logical NOT of the input array.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        ArrayLike: The result of the logical NOT operation.
        
    Raises:
        IncompatibleTypeError: If input is not a numpy array or Torch tensor.
    """
    if not is_array(x):
        raise IncompatibleTypeError("Input must be a numpy array or Torch tensor")
    if is_tensor(x):
        return torch.logical_not(x)
    return np.logical_not(x)


def logical_xor(x: ArrayLike) -> ArrayLike:
    """
    Computes the element-wise logical XOR of the input array.

    Args:
        x (ArrayLike): The input array-like object.

    Returns:
        ArrayLike: The result of the logical XOR operation.
        
    Raises:
        IncompatibleTypeError: If input is not a numpy array or Torch tensor.
    """
    if not is_array(x):
        raise IncompatibleTypeError("Input must be a numpy array or Torch tensor")
    if is_tensor(x):
        return torch.logical_xor(x)
    return np.logical_xor(x)


def allclose(
    x: ArrayLike, y: ArrayLike, rtol: float = 0.00001, atol: float = 1e-8
) -> bool:
    """
    Checks if two arrays are element-wise equal within a tolerance.

    Args:
        x (ArrayLike): The first input array-like object.
        y (ArrayLike): The second input array-like object.
        rtol (float, optional): The relative tolerance parameter. Default is 0.00001.
        atol (float, optional): The absolute tolerance parameter. Default is 1e-8.

    Returns:
        bool: True if the arrays are element-wise equal within the tolerance, False otherwise.
        
    Raises:
        IncompatibleTypeError: If the input arrays are not of the same type.
    """
    if not isinstance(x, type(y)):
        raise IncompatibleTypeError(f"Invalid type: expected same type for both arrays, but got {type(x)} and {type(y)}")
    if is_tensor(x):
        return torch.allclose(x, y, rtol=rtol, atol=atol)
    return np.allclose(x, y, rtol=rtol, atol=atol)


def isclose(
    x: ArrayLike, y: Any, rtol: float = 0.00001, atol: float = 1e-8
) -> ArrayLike:
    """
    Checks if elements of an array are close to a given value within a tolerance.

    Args:
        x (ArrayLike): The input array-like object.
        y (Any): The value to compare against.
        rtol (float, optional): The relative tolerance parameter. Default is 0.00001.
        atol (float, optional): The absolute tolerance parameter. Default is 1e-8.

    Returns:
        ArrayLike: An array of booleans indicating where the elements are close to the given value.
    """
    if is_tensor(x):
        return torch.isclose(x, y, rtol=rtol, atol=atol)
    return np.isclose(x, y, rtol=rtol, atol=atol)


def get_dtype(x: ArrayLike) -> Any:
    """
    Get the dtype of an array.

    Args:
        x (ArrayLike): Input array or tensor.

    Returns:
        Any: The dtype of the input. Returns numpy.dtype for numpy arrays,
             torch.dtype for torch tensors.

    Raises:
        IncompatibleTypeError: If input is not a numpy array or Torch tensor.

    Example:
        >>> import numpy as np
        >>> arr = np.array([1.0], dtype=np.float32)
        >>> get_dtype(arr)
        dtype('float32')
        >>> get_dtype(arr) == np.float32
        True
    """
    if not is_array(x):
        raise IncompatibleTypeError(
            f"Input must be a numpy array or Torch tensor, got {type(x)}."
        )
    return x.dtype


def promote_types(*arrays: ArrayLike) -> Any:
    """
    Find the promoted dtype among multiple arrays.
    Returns the highest precision dtype following numpy/torch promotion rules.

    Promotion examples:
        - int32 + float32 → float32
        - float32 + float64 → float64

    Args:
        *arrays (ArrayLike): Variable number of arrays. All must be either numpy arrays
                            or torch tensors (no mixing allowed).

    Returns:
        Any: The promoted dtype (numpy.dtype for numpy arrays, torch.dtype for tensors).

    Raises:
        InvalidDimensionError: If no arrays are provided.
        IncompatibleTypeError: If mixing numpy and tensor types, or if any input is not an array.

    Example:
        >>> import numpy as np
        >>> a = np.array([1], dtype=np.float32)
        >>> b = np.array([2], dtype=np.float64)
        >>> promote_types(a, b)
        dtype('float64')
        >>> a.astype(promote_types(a, b)).dtype
        dtype('float64')
    """
    if len(arrays) == 0:
        raise InvalidDimensionError(
            "At least one array is required for dtype promotion."
        )

    # Validate all inputs are arrays
    for i, arr in enumerate(arrays):
        if not is_array(arr):
            raise IncompatibleTypeError(
                f"All inputs must be numpy arrays or torch tensors. "
                f"Input at index {i} has type {type(arr)}."
            )

    # Check if all numpy or all tensor (no mixing)
    all_numpy = all(is_numpy(arr) for arr in arrays)
    all_tensor = all(is_tensor(arr) for arr in arrays)

    if not (all_numpy or all_tensor):
        raise IncompatibleTypeError(
            "Cannot promote dtype between numpy and tensor types. "
            "All arrays must be of the same type (all numpy or all torch)."
        )

    if all_numpy:
        # Use numpy's promote_types
        result_dtype = arrays[0].dtype
        for arr in arrays[1:]:
            result_dtype = np.promote_types(result_dtype, arr.dtype)
        return result_dtype
    else:
        # Use torch's promote_types
        result_dtype = arrays[0].dtype
        for arr in arrays[1:]:
            result_dtype = torch.promote_types(result_dtype, arr.dtype)
        return result_dtype


def sum(x: ArrayLike, dim: Optional[int] = None, keepdims: bool = False) -> ArrayLike:
    """
    Sum array elements over a given dimension.

    Args:
        x (ArrayLike): Input array or tensor.
        dim (Optional[int]): Dimension along which to sum. If None, sums all elements.
        keepdims (bool): Whether to keep the reduced dimension. Default is False.

    Returns:
        ArrayLike: Sum of array elements.
    """
    if is_tensor(x):
        if dim is None:
            return torch.sum(x)
        return torch.sum(x, dim=dim, keepdim=keepdims)
    else:
        if dim is None:
            return np.sum(x)
        return np.sum(x, axis=dim, keepdims=keepdims)


def maximum(x: ArrayLike, y: Union[ArrayLike, float]) -> ArrayLike:
    """
    Element-wise maximum of array elements.

    Args:
        x (ArrayLike): Input array or tensor.
        y (Union[ArrayLike, float]): Values to compare against.

    Returns:
        ArrayLike: Element-wise maximum.
    """
    if is_tensor(x):
        return torch.maximum(x, y if is_tensor(y) else torch.tensor(y))
    return np.maximum(x, y)

def minimum(x: ArrayLike, y: Union[ArrayLike, float]) -> ArrayLike:
    """
    Element-wise minimum of array elements.

    Args:
        x (ArrayLike): Input array or tensor.
        y (Union[ArrayLike, float]): Values to compare against.

    Returns:
        ArrayLike: Element-wise minimum.
    """
    if is_tensor(x):
        return torch.minimum(x, y if is_tensor(y) else torch.tensor(y))
    return np.minimum(x, y)

def argmin(x: ArrayLike, dim: Optional[int] = None) -> ArrayLike:
    """
    Returns the indices of the minimum values along a dimension.

    Args:
        x (ArrayLike): Input array or tensor.
        dim (Optional[int]): Dimension along which to find minimum. If None, flattens array.

    Returns:
        ArrayLike: Indices of minimum values.
    """
    if is_tensor(x):
        if dim is None:
            return torch.argmin(x)
        return torch.argmin(x, dim=dim)
    else:
        if dim is None:
            return np.argmin(x)
        return np.argmin(x, axis=dim)


def argsort(x: ArrayLike, dim: int = -1) -> ArrayLike:
    """
    Returns the indices that would sort an array.

    Args:
        x (ArrayLike): Input array or tensor.
        dim (int): Dimension along which to sort. Default is -1 (last dimension).

    Returns:
        ArrayLike: Array of indices that sort x.
    """
    if is_tensor(x):
        return torch.argsort(x, dim=dim)
    return np.argsort(x, axis=dim)


def unique(
    x: ArrayLike,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> Union[ArrayLike, Tuple[ArrayLike, ...]]:
    """
    Find the unique elements of an array.

    Args:
        x (ArrayLike): Input array or tensor (1D).
        return_inverse (bool): If True, return indices to reconstruct the input from unique values.
        return_counts (bool): If True, return the count for each unique element.

    Returns:
        unique_values (ArrayLike): The sorted unique values.
        inverse_indices (ArrayLike, optional): Indices to reconstruct input from unique array.
        counts (ArrayLike, optional): The count for each unique value.

    Raises:
        InvalidDimensionError: If input is not 1D.

    Example:
        >>> x = np.array([1, 2, 2, 3, 3, 3])
        >>> unique_vals, inverse, counts = unique(x, return_inverse=True, return_counts=True)
        >>> unique_vals
        array([1, 2, 3])
        >>> inverse
        array([0, 1, 1, 2, 2, 2])
        >>> counts
        array([1, 2, 3])
    """
    if x.ndim != 1:
        raise InvalidDimensionError(
            f"unique() requires 1D array, but got {x.ndim}D array with shape {x.shape}. "
            f"Please flatten the array first if needed."
        )

    if is_tensor(x):
        result = torch.unique(x, return_inverse=return_inverse, return_counts=return_counts)
    else:
        result = np.unique(x, return_inverse=return_inverse, return_counts=return_counts)

    # Return format depends on flags
    if not return_inverse and not return_counts:
        return result
    elif return_inverse and not return_counts:
        return result[0], result[1]
    elif not return_inverse and return_counts:
        return result[0], result[1]
    else:  # Both flags
        return result[0], result[1], result[2]


def scatter_add(
    target: ArrayLike,
    indices: ArrayLike,
    source: ArrayLike,
    dim: int = 0,
) -> ArrayLike:
    """
    Adds all values from source into target at the indices specified in the indices tensor.

    This is an in-place operation that modifies target directly.

    Args:
        target (ArrayLike): Target array to scatter into (modified in-place).
        indices (ArrayLike): Index array (must be integer type).
        source (ArrayLike): Source values to add.
        dim (int): Dimension along which to index. Default is 0.

    Returns:
        ArrayLike: The modified target array (same object as input).

    Example:
        >>> target = np.zeros((5, 3))
        >>> indices = np.array([0, 1, 1, 2])
        >>> source = np.ones((4, 3))
        >>> result = scatter_add(target, indices, source, dim=0)
        >>> result[1]  # 2 additions
        array([2., 2., 2.])
    """
    if is_tensor(target):
        target.index_add_(dim, indices.long(), source)
        return target
    else:
        if dim == 0:
            np.add.at(target, indices, source)
        else:
            raise NotImplementedError(
                f"scatter_add for numpy only supports dim=0, got dim={dim}"
            )
        return target


def as_strided(x: ArrayLike, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> ArrayLike:
    """
    Create a view into an array with a given shape and strides.

    This creates a memory-efficient view without copying data, useful for
    sliding window operations (e.g., extracting patches from images).

    Args:
        x (ArrayLike): Input array or tensor.
        shape (Tuple[int, ...]): Shape of the resulting array view.
        strides (Tuple[int, ...]): Strides of the resulting array view.
            - For NumPy: strides in BYTES (use x.strides)
            - For PyTorch: strides in ELEMENTS (use x.stride())

    Returns:
        ArrayLike: Array view with specified shape and strides.

    Example:
        >>> import numpy as np
        >>> x = np.arange(12).reshape(3, 4)
        >>> # Extract 2x2 sliding windows
        >>> shape = (2, 3, 2, 2)
        >>> strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1])
        >>> windows = as_strided(x, shape, strides)
        >>> windows.shape
        (2, 3, 2, 2)

    Warning:
        Use with caution! Incorrect strides can lead to undefined behavior.
        This function does not perform bounds checking.
    """
    if is_tensor(x):
        return torch.as_strided(x, size=shape, stride=strides)
    else:
        from numpy.lib.stride_tricks import as_strided as np_as_strided
        return np_as_strided(x, shape=shape, strides=strides)


def extract_patches(
    x: ArrayLike,
    patch_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, str] = 0,
) -> ArrayLike:
    """
    Extract sliding window patches from 2D or 3D arrays.

    This function creates overlapping or non-overlapping patches from spatial data
    using memory-efficient views (no copying).

    Args:
        x (ArrayLike): Input array with shape (H,W) or (H,W,C).
            - (H,W): Single-channel 2D data (e.g., depth map)
            - (H,W,C): Multi-channel 2D data (e.g., RGB image, 3D points)
        patch_size (int or (h,w)): Size of extraction window.
            If int, uses square patches.
        stride (int or (h,w)): Step size between patches. Default 1.
            If int, uses same stride for both dimensions.
        padding (int or str): Padding specification:
            - 0 or 'valid': No padding (default)
            - int: Symmetric padding on H and W
            - 'same': Auto-pad to preserve spatial dimensions

    Returns:
        ArrayLike: Extracted patches with shape:
            - (H,W) input → (H', W', Ph*Pw)
            - (H,W,C) input → (H', W', Ph*Pw*C)
        where H' and W' are output spatial dimensions after padding/stride.

    Raises:
        InvalidDimensionError: If input is not 2D or 3D.
        InvalidArgumentError: If patch_size, stride, or padding are invalid.

    Example:
        >>> import numpy as np
        >>> # Depth map patches
        >>> depth = np.random.rand(480, 640)
        >>> patches = extract_patches(depth, patch_size=3, padding='same')
        >>> patches.shape
        (480, 640, 9)

        >>> # RGB image patches
        >>> rgb = np.random.rand(480, 640, 3)
        >>> patches = extract_patches(rgb, patch_size=5, stride=2)
        >>> patches.shape
        (238, 318, 75)  # (480-5)//2+1, (640-5)//2+1, 5*5*3

    Note:
        - Uses `sliding_window_view` for NumPy (view, no copy)
        - Uses `Tensor.unfold()` for PyTorch (view, no copy)
        - Input format must be (H,W,C) with channels last (OpenCV convention)
    """
    # Validate input dimensions
    if x.ndim not in [2, 3]:
        raise InvalidDimensionError(
            f"extract_patches expects 2D (H,W) or 3D (H,W,C) input, got {x.ndim}D with shape {x.shape}."
        )

    # Normalize patch_size and stride to tuples
    import builtins
    if isinstance(patch_size, int):
        if patch_size <= 0:
            raise InvalidArgumentError(f"patch_size must be positive, got {patch_size}.")
        patch_size = (patch_size, patch_size)
    else:
        if builtins.any(p <= 0 for p in patch_size):
            raise InvalidArgumentError(f"patch_size must be positive, got {patch_size}.")

    if isinstance(stride, int):
        if stride <= 0:
            raise InvalidArgumentError(f"stride must be positive, got {stride}.")
        stride = (stride, stride)
    else:
        if builtins.any(s <= 0 for s in stride):
            raise InvalidArgumentError(f"stride must be positive, got {stride}.")

    # Dispatch to implementation
    if is_tensor(x):
        return _extract_patches_torch(x, patch_size, stride, padding)
    else:
        return _extract_patches_numpy(x, patch_size, stride, padding)


def _extract_patches_numpy(
    x: np.ndarray,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Union[int, str],
) -> np.ndarray:
    """NumPy implementation using sliding_window_view."""
    from numpy.lib.stride_tricks import sliding_window_view

    # Handle padding
    if padding == 'same':
        pad_h = (patch_size[0] - 1) // 2
        pad_w = (patch_size[1] - 1) // 2
        if x.ndim == 3:
            x = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
        else:
            x = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    elif isinstance(padding, int) and padding > 0:
        if x.ndim == 3:
            x = np.pad(x, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        else:
            x = np.pad(x, ((padding, padding), (padding, padding)), mode='constant')
    elif padding != 0 and padding != 'valid':
        raise InvalidArgumentError(
            f"Invalid padding: {padding}. Use int, 'same', or 'valid'."
        )

    # Define window shape
    if x.ndim == 3:
        window_shape = (patch_size[0], patch_size[1], x.shape[2])
    else:
        window_shape = patch_size

    # Extract patches using sliding_window_view (creates view, no copy)
    patches = sliding_window_view(x, window_shape)
    # patches shape: (H', W', Ph, Pw, C?) where H' = H - Ph + 1

    # Apply stride via slicing (still a view)
    if stride != (1, 1):
        patches = patches[::stride[0], ::stride[1]]

    # Flatten patch dimensions
    H_out, W_out = patches.shape[:2]
    return patches.reshape(H_out, W_out, -1)


def _extract_patches_torch(
    x: torch.Tensor,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Union[int, str],
) -> torch.Tensor:
    """PyTorch implementation using Tensor.unfold (view-based)."""
    import torch.nn.functional as F

    # Handle padding
    if padding == 'same':
        pad_h = (patch_size[0] - 1) // 2
        pad_w = (patch_size[1] - 1) // 2
        if x.ndim == 3:
            x = F.pad(x, (0, 0, pad_w, pad_w, pad_h, pad_h), mode='replicate')
        else:
            # PyTorch replicate mode requires 3D+ tensors, add temporary dimension
            x = x.unsqueeze(0)  # (1, H, W)
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
            x = x.squeeze(0)  # (H, W)
    elif isinstance(padding, int) and padding > 0:
        if x.ndim == 3:
            x = F.pad(x, (0, 0, padding, padding, padding, padding), mode='constant')
        else:
            x = F.pad(x, (padding, padding, padding, padding), mode='constant')
    elif padding != 0 and padding != 'valid':
        raise InvalidArgumentError(
            f"Invalid padding: {padding}. Use int, 'same', or 'valid'."
        )

    # Unfold along H dimension (dim 0), then W dimension (dim 1)
    # Both unfold operations create views (no memory copy)
    patches = x.unfold(0, patch_size[0], stride[0])
    patches = patches.unfold(1, patch_size[1], stride[1])
    # patches shape: (H', W', C, Ph, Pw) or (H', W', Ph, Pw)

    # Reshape to (H', W', Ph*Pw*C) or (H', W', Ph*Pw)
    H_out, W_out = patches.shape[:2]
    return patches.reshape(H_out, W_out, -1)


def pad(
    x: ArrayLike,
    pad_width: Union[int, tuple, list],
    mode: str = 'constant',
    constant_values: Union[int, float] = 0,
) -> ArrayLike:
    """
    Pad an array.

    Args:
        x (ArrayLike): Input array or tensor.
        pad_width (Union[int, tuple, list]): Number of values padded to the edges.
            - int: Pad all dimensions equally on both sides
            - tuple/list: Padding for each dimension as ((before_1, after_1), (before_2, after_2), ...)
        mode (str): Padding mode. Options:
            - 'constant': Pads with a constant value (default)
            - 'edge': Pads with edge values (replicate)
            - 'reflect': Pads with reflection of values at the edge
        constant_values (Union[int, float]): Value to use for constant padding. Default is 0.

    Returns:
        ArrayLike: Padded array with same type as input.

    Raises:
        InvalidArgumentError: If mode is not supported.

    Example:
        >>> import numpy as np
        >>> x = np.array([[1, 2], [3, 4]])
        >>> pad(x, pad_width=((1, 1), (1, 1)), mode='constant')
        array([[0, 0, 0, 0],
               [0, 1, 2, 0],
               [0, 3, 4, 0],
               [0, 0, 0, 0]])
        >>> pad(x, pad_width=((1, 1), (1, 1)), mode='edge')
        array([[1, 1, 2, 2],
               [1, 1, 2, 2],
               [3, 3, 4, 4],
               [3, 3, 4, 4]])
    """
    if mode not in ['constant', 'edge', 'reflect']:
        raise InvalidArgumentError(
            f"Unsupported padding mode: {mode}. "
            f"Choose from: 'constant', 'edge', 'reflect'"
        )

    if is_tensor(x):
        # PyTorch uses different padding format: (left, right, top, bottom, ...)
        # We need to convert from numpy format to torch format
        import torch.nn.functional as F

        # Normalize pad_width to list of tuples
        if isinstance(pad_width, int):
            pad_width = [(pad_width, pad_width)] * x.ndim
        elif isinstance(pad_width, (tuple, list)):
            if isinstance(pad_width[0], int):
                # Single tuple (before, after) -> apply to all dims
                pad_width = [tuple(pad_width)] * x.ndim

        # Convert to torch format: reverse order and flatten
        torch_pad = []
        for before, after in reversed(pad_width):
            torch_pad.extend([before, after])

        # Map mode names
        if mode == 'edge':
            torch_mode = 'replicate'
        elif mode == 'constant':
            torch_mode = 'constant'
        elif mode == 'reflect':
            torch_mode = 'reflect'

        if torch_mode == 'constant':
            return F.pad(x, torch_pad, mode=torch_mode, value=constant_values)
        else:
            return F.pad(x, torch_pad, mode=torch_mode)
    else:
        # NumPy padding
        if mode == 'constant':
            return np.pad(x, pad_width, mode=mode, constant_values=constant_values)
        else:
            return np.pad(x, pad_width, mode=mode)


__all__ = [
    # Type alias
    "ArrayLike",
    # Type checking
    "is_tensor",
    "is_numpy",
    "is_array",
    # Type conversion
    "convert_tensor",
    "convert_numpy",
    "convert_array",
    "convert_dict_tensor",
    # Array properties
    "numel",
    # Dimension manipulation
    "expand_dim",
    "reduce_dim",
    "pad",
    "as_strided",
    "extract_patches",
    # Array construction
    "concat",
    "stack",
    "ones_like",
    "zeros_like",
    "empty_like",
    "full_like",
    "arange",
    "ones",
    "zeros",
    "any",
    # Array operations
    "sum",
    "maximum",
    "minimum",
    "argmin",
    "argsort",
    "unique",
    "scatter_add",
    "deep_copy",
    "where",
    "clip",
    "eye",
    "transpose2d",
    "swapaxes",
    # Type casting
    "as_bool",
    "as_int",
    "as_float",
    # Dtype utilities
    "get_dtype",
    "promote_types",
    "astype_like",
    # Logical operations
    "logical_or",
    "logical_and",
    "logical_not",
    "logical_xor",
    # Comparison
    "allclose",
    "isclose",
]
