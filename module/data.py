import numpy as np
from typing import *

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    

    
if TORCH_AVAILABLE:
    Array = Union[np.ndarray, torch.Tensor]
else:
    Array = np.ndarray

def is_tensor(x: Array) -> bool:
    if TORCH_AVAILABLE: return type(x) == torch.Tensor
    else: return False    

def is_numpy(x: Array) -> bool:
    return type(x) == np.ndarray

def is_array(x: Any) -> bool:
    return is_tensor(x) or is_numpy(x)

def convert_tensor(x: Array, tensor: torch.Tensor = None) -> torch.Tensor:
    assert TORCH_AVAILABLE
    if is_tensor(x): return x 
    if tensor is not None:    
        assert is_tensor(tensor)
        x_tensor = torch.tensor(x, dtype=tensor.dtype, device=tensor.device)
    else:
        x_tensor = torch.Tensor(x)
    return x_tensor
        
def convert_numpy(x: Array) -> np.ndarray:  
    if is_tensor(x): x_numpy = x.detach().cpu().numpy()
    elif is_numpy(x): x_numpy = x
    else: raise TypeError
    return x_numpy

def expand_dim(x: Array, dim: int) -> Array:
    if is_tensor(x): return x.unsqueeze(dim)
    else: return np.expand_dims(x, axis=dim)

def concat(x: List[Array], dim: int) -> Array:
     if is_tensor(x[0]): return torch.cat(x, dim=dim)
     return np.concatenate(x, axis=dim)

def ones_like(x: Array) -> Array:
    assert is_array(x), ("Invalid Type. It is neither Numpy nor Tensor.")
    if is_tensor(x): return torch.ones_like(x)
    return np.ones_like(x)

def zeros_like(x: Array) -> Array:
    if is_tensor(x): return torch.zeros_like(x)
    return np.zeros_like(x)

def norm_l2(x: Array, dim:int, keepdim:bool=True):
    if is_tensor(x): return torch.norm(x,dim=dim,keepdim=keepdim)    
    return np.linalg.norm(x,axis=dim,keepdims=keepdim)

def normalize(x: Array, dim: int, eps:float=1e-6) -> Array:
    norm = norm_l2(x,dim)
    return x / (norm + eps)


if __name__ == '__main__':
    arr = np.ones(4)
    norm_arr = normalize(arr,0)
    norm = norm_l2(arr,0,False)
    
    print(arr)
    print(norm_arr)
    print(norm)
    