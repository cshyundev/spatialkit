"""
Module Name: uops.py

Description: 
Unified Operations (uops) module provides a unified interface for common operations that can be performed using both Numpy and Torch. 
This module helps to write agnostic code that can handle both Numpy arrays and Torch tensors seamlessly.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

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


import numpy as np
from numpy import ndarray
from torch import Tensor
import torch
from typing import *

ArrayLike = Union[ndarray, Tensor] # Unified ArrayType

def is_tensor(x: ArrayLike) -> bool:
    return type(x) == Tensor

def is_numpy(x: ArrayLike) -> bool:
    return type(x) == ndarray

def is_array(x: Any) -> bool:
    return is_tensor(x) or is_numpy(x)

def convert_tensor(x: ArrayLike, tensor:Optional[Tensor] = None) -> Tensor:
    if is_tensor(x): return x 
    if tensor is not None:    
        assert is_tensor(tensor)
        x_tensor = torch.tensor(x, dtype=tensor.dtype, device=tensor.device)
    else:
        x_tensor = Tensor(x)
    return x_tensor
        
def convert_numpy(x: ArrayLike) -> ndarray:  
    if is_tensor(x): x_numpy = x.detach().cpu().numpy()
    elif is_numpy(x): x_numpy = x
    else: x_numpy = np.array(x)
    return x_numpy

def convert_array(x:Any, array:ArrayLike) -> ArrayLike:
    if is_tensor(array): return convert_tensor(x,array)
    return convert_numpy(x)

def numel(x:ArrayLike) -> ArrayLike:
    assert is_array(x), "Invalid type. Input type must be either ndarray or Tensor."
    if is_tensor(x): return x.numel()
    return x.size

def _assert_same_array_type(arrays: Tuple[ArrayLike, ...]):
    assert all(is_tensor(arr) for arr in arrays) or all(is_numpy(arr) for arr in arrays), "All input arrays must be of the same type"

def convert_dict_tensor(dict: Dict[Any, ndarray], tensor: Tensor=None) -> Dict[Any,Tensor]:
    _assert_same_array_type(dict)
    new_dict = {}
    for key in dict.keys():
        new_dict[key] = convert_tensor(dict[key], tensor)
    return new_dict

def expand_dim(x: ArrayLike, dim: int) -> ArrayLike:
    if is_tensor(x): return x.unsqueeze(dim)
    else: return np.expand_dims(x, axis=dim)

def reduce_dim(x: ArrayLike, dim: int) -> ArrayLike:
    if is_tensor(x): return x.squeeze(dim)
    else: return np.squeeze(x, axis=dim)

def concat(x: List[ArrayLike], dim: int) -> ArrayLike:
    _assert_same_array_type(x)
    if is_tensor(x[0]): return torch.cat(x, dim=dim)
    return np.concatenate(x, axis=dim)

def stack(x: List[ArrayLike], dim:int) -> ArrayLike:
    _assert_same_array_type(x)
    if is_tensor(x[0]): return torch.stack(x, dim=dim)
    return np.stack(x, axis=dim)
  
def ones_like(x: ArrayLike) -> ArrayLike:
    assert is_array(x), ("Invalid Type. It is neither Numpy nor Tensor.")
    if is_tensor(x): return torch.ones_like(x)
    return np.ones_like(x)

def zeros_like(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.zeros_like(x)
    return np.zeros_like(x)

def empty_like(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.empty_like(x)
    return np.empty_like(x)

def full_like(x: ArrayLike, fill_value: Any,dtype:Any=None) -> ArrayLike:
    if is_tensor(x): return torch.full_like(x,fill_value,dtype=dtype)
    return np.full_like(a=x,fill_value=fill_value,dtype=dtype)

def arange(x:ArrayLike, start:Any, stop:Any=None, step:int=1,dtype=None):
    if stop is None:
        stop = start
        start = 0
    if is_tensor(x): return torch.arange(start,stop,step,dtype=dtype)
    return np.arange(start,stop,step,dtype=dtype)

def deep_copy(x:ArrayLike) -> ArrayLike:
    if is_tensor(x): return x.clone()
    return np.copy(x)

def where(condition:ArrayLike, x:ArrayLike, y:ArrayLike) -> ArrayLike:
    if is_tensor(condition): return torch.where(condition,x,y)
    return np.where(condition,x,y)

def clip(x:ArrayLike, min:float=None,max:float=None) -> ArrayLike:
    if is_tensor(x): return torch.clip(x,min,max)
    return np.clip(x,min,max)

def eye(n:int, x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return torch.eye(n)
    return np.eye(n)

def transpose2d(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Invalid shape for transpose: expected a 2D array, but got {x.shape}."
    if is_tensor(x): return x.transpose(0, 1)
    return x.T

def swapaxes(x: ArrayLike, axis0:int,axis1:int) -> ArrayLike:
    if is_tensor(x): return torch.swapaxes(x,axis0,axis1)
    return np.swapaxes(x,axis0,axis1)

def as_bool(x: ArrayLike) -> ArrayLike:
    if is_tensor(x): return x.type(torch.bool)
    return x.astype(bool)

def as_int(x:ArrayLike,n:int=32) -> ArrayLike:
    if is_tensor(x):
        if n == 64: return x.type(torch.int64)
        elif n == 32: return x.type(torch.int32)
        elif n == 16: return x.type(torch.int16)
        else: raise TypeError
    elif is_numpy(x):
        if n == 256: return x.astype(np.int256)
        elif n == 128: return x.astype(np.int128)
        elif n == 64: return x.astype(np.int64)
        elif n == 32: return x.astype(np.int32)
        elif n == 16: return x.astype(np.int16)
        else: raise TypeError

def as_float(x:ArrayLike, n:int=32) -> ArrayLike:
    if is_tensor(x):
        if   n == 64: return x.type(torch.float64)
        elif n == 32: return x.type(torch.float32)
        elif n == 16: return x.type(torch.float16)
        else: raise TypeError
    elif is_numpy(x):
        if   n == 256:return x.astype(np.float256)
        elif n == 128:return x.astype(np.float128)
        elif n == 64: return x.astype(np.float64)
        elif n == 32: return x.astype(np.float32)
        elif n == 16: return x.astype(np.float16)
        else: raise TypeError

def logical_or(*arrays: ArrayLike) -> ArrayLike:
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
    assert len(arrays) > 0, "At least one input array is required"
    _assert_same_array_type(arrays)

    result = arrays[0]
    for arr in arrays[1:]:
        if is_tensor(result):
            result = torch.logical_and(result, arr)
        else:
            result = np.logical_and(result, arr)
    return result

def logical_not(x: ArrayLike) -> ArrayLike:
    assert(is_array(x))
    if is_tensor(x): return torch.logical_not(x)
    return np.logical_not(x)

def logical_xor(x: ArrayLike) -> ArrayLike:
    assert(is_array(x))
    if is_tensor(x): return torch.logical_xor(x)
    return np.logical_xor(x)

def allclose(x:ArrayLike, y:ArrayLike,rtol:float=0.00001,atol:float=1e-8):
    assert type(x) == type(y)
    if is_tensor(x): return torch.allclose(x,y,rtol=rtol,atol=atol)
    return np.allclose(x,y,rtol=rtol,atol=atol)

def isclose(x:ArrayLike, y:Any,rtol:float=0.00001,atol:float=1e-8):
    if is_tensor(x): return torch.isclose(x,y,rtol=rtol,atol=atol)
    return np.isclose(x,y,rtol=rtol,atol=atol)