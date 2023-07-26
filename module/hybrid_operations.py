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

def convert_dict_tensor(dict: Dict[Any, ndarray], tensor: Tensor) -> Dict[Any,Tensor]:
    for key in dict.keys():
        if is_numpy(dict[key]): dict[key] = convert_tensor(dict[key], tensor)
    return dict

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

def transpose2d(x:Array) -> Array:
    assert len(x.shape) == 2, "Invalid Shape. Array must be 2D."
    if is_tensor(x): return x.transpose(0,1)
    return x.T

def matmul(x:Array, y:Array) -> Array:
    assert(type(x) == type(y)), f"two Array types must be same, but got {str(type(x))} and {str(type(y))}"
    assert(x.shape[-1] == y.shape[0]), f"Invalid Shape. x:[m,n], y:[n,k], but got {str(x.shape)} and {str(y.shape)}."
    
    if is_tensor(x): return torch.matmul(x,y)
    return x@y

def permute(x:Array, dims:Tuple[int]) -> Array:
    assert(len(x.shape) == len(dims)), (f"Shape of Array and Dimensions must be same, but got {str(x.shape)}, {str(dims)}")
    if is_tensor(x): return x.permute(dims)
    return x.transpose(dims)

if __name__ == '__main__':
    x = np.array(range(30)).reshape(2,3,5)
    # print(permute(x,(1,2)).shape)
    x = Tensor(x)
    print(permute(x,(1,2)).shape)
    
    