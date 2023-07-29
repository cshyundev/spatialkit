from module.hybrid_operations import *


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