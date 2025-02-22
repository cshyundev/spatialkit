"""
Module Name: uops.py

Description: 
Unified Operations (uops) module provides a unified interface for common operations that can be performed using both Numpy and Torch. 
This module helps to write agnostic code that can handle both Numpy arrays and Torch tensors seamlessly.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1

License: MIT LICENSE

Usage:
>>> import numpy as np
>>> import torch
>>> from cv_utils import uops

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
    """
    if is_tensor(x):
        return x
    if tensor is not None:
        assert is_tensor(tensor)
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
    """
    assert is_array(x), "Invalid type. Input type must be either ndarray or Tensor."
    if is_tensor(x):
        return x.numel()
    return x.size


def _assert_same_array_type(arrays: Tuple[ArrayLike, ...]):
    """
    Asserts that all input arrays are of the same type.

    Args:
        arrays (Tuple[ArrayLike, ...]): A tuple of array-like objects.

    Raises:
        AssertionError: If the input arrays are not of the same type.
    """
    assert all(is_tensor(arr) for arr in arrays) or all(
        is_numpy(arr) for arr in arrays
    ), "All input arrays must be of the same type"


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
    """
    assert is_array(x), "Invalid Type. It is neither Numpy nor Tensor."
    if is_tensor(x):
        return torch.ones_like(x)
    return np.ones_like(x)


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
        AssertionError: If the input array is not 2-D.
    """
    assert (
        x.ndim == 2
    ), f"Invalid shape for transpose: expected a 2D array, but got {x.shape}."
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
    """
    if is_tensor(x):
        if n == 64:
            return x.type(torch.int64)
        elif n == 32:
            return x.type(torch.int32)
        elif n == 16:
            return x.type(torch.int16)
        else:
            raise ValueError(f"Unsupported bit-width {n} for int conversion.")
    elif is_numpy(x):
        if n == 256:
            return x.astype(np.int256)
        elif n == 128:
            return x.astype(np.int128)
        elif n == 64:
            return x.astype(np.int64)
        elif n == 32:
            return x.astype(np.int32)
        elif n == 16:
            return x.astype(np.int16)
        else:
            raise ValueError(f"Unsupported bit-width {n} for int conversion.")


def as_float(x: ArrayLike, n: int = 32) -> ArrayLike:
    """
    Converts an array to float type with specified bit-width.

    Args:
        x (ArrayLike): The input array-like object.
        n (int, optional): The bit-width of the float type. Default is 32.

    Returns:
        ArrayLike: The array converted to float type.
    """
    if is_tensor(x):
        if n == 64:
            return x.type(torch.float64)
        elif n == 32:
            return x.type(torch.float32)
        elif n == 16:
            return x.type(torch.float16)
        else:
            raise ValueError(f"Unsupported bit-width {n} for float conversion.")
    elif is_numpy(x):
        if n == 256:
            return x.astype(np.float256)
        elif n == 128:
            return x.astype(np.float128)
        elif n == 64:
            return x.astype(np.float64)
        elif n == 32:
            return x.astype(np.float32)
        elif n == 16:
            return x.astype(np.float16)
        else:
            raise ValueError(f"Unsupported bit-width {n} for float conversion.")


def logical_or(*arrays: ArrayLike) -> ArrayLike:
    """
    Computes the element-wise logical OR of input arrays.

    Args:
        *arrays (ArrayLike): A variable number of array-like objects.

    Returns:
        ArrayLike: The result of the logical OR operation.
    """
    assert len(arrays) > 0, "At least one input array is required"
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
    """
    assert len(arrays) > 1, "At least one input array is required"
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
    """
    assert is_array(x)
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
    """
    assert is_array(x)
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
    """
    assert isinstance(
        x, type(y)
    ), f"Invalid type: expected same type for both arrays, but got {type(x)} and {type(y)}"
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
