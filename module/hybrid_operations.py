import numpy as np
from numpy import ndarray
from typing import *

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
    
if TORCH_AVAILABLE:
    Array = Union[ndarray, Tensor]
else:
    Array = ndarray

def is_tensor(x: Array) -> bool:
    if TORCH_AVAILABLE: return type(x) == Tensor
    else: return False    

def is_numpy(x: Array) -> bool:
    return type(x) == ndarray

def is_array(x: Any) -> bool:
    return is_tensor(x) or is_numpy(x)

def convert_tensor(x: Array, tensor: Tensor = None) -> Tensor:
    assert TORCH_AVAILABLE
    if is_tensor(x): return x 
    if tensor is not None:    
        assert is_tensor(tensor)
        x_tensor = torch.tensor(x, dtype=tensor.dtype, device=tensor.device)
    else:
        x_tensor = Tensor(x)
    return x_tensor
        
def convert_numpy(x: Array) -> ndarray:  
    if is_tensor(x): x_numpy = x.detach().cpu().numpy()
    elif is_numpy(x): x_numpy = x
    else: raise TypeError
    return x_numpy

def convert_dict_tensor(dict: Dict[Any, ndarray], tensor: Tensor=None) -> Dict[Any,Tensor]:
    for key in dict.keys():
        if is_numpy(dict[key]): dict[key] = convert_tensor(dict[key], tensor)
    return dict

def expand_dim(x: Array, dim: int) -> Array:
    if is_tensor(x): return x.unsqueeze(dim)
    else: return np.expand_dims(x, axis=dim)

def reduce_dim(x: Array, dim: int) -> Array:
    if is_tensor(x): return x.squeeze(dim)
    else: return np.squeeze(x, axis=dim)

def concat(x: List[Array], dim: int) -> Array:
     if is_tensor(x[0]): return torch.cat(x, dim=dim)
     return np.concatenate(x, axis=dim)

def stack(x: List[Array], dim:int) -> Array:
     if is_tensor(x[0]): return torch.stack(x, dim=dim)
     return np.stack(x, axis=dim)


def ones_like(x: Array) -> Array:
    assert is_array(x), ("Invalid Type. It is neither Numpy nor Tensor.")
    if is_tensor(x): return torch.ones_like(x)
    return np.ones_like(x)

def zeros_like(x: Array) -> Array:
    if is_tensor(x): return torch.zeros_like(x)
    return np.zeros_like(x)

def deep_copy(x:Array) -> Array:
    if is_tensor(x): return x.clone()
    return np.copy(x)

def where(condition:Array, x:Array, y:Array) -> Array:
    if is_tensor(condition): return torch.where(condition,x,y)
    return np.where(condition,x,y)    

if __name__ == '__main__':
    x = np.array(range(30)).reshape(2,3,5)
    
    