from hybrid_operations import *
from scipy.linalg import expm, logm

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

def abs(x:Array) -> Array:
    if is_tensor(x): torch.abs(x)
    return np.abs(x)

def sqrt(x: Array) -> Array:
    if is_tensor(x): return torch.sqrt(x)
    return np.sqrt(x)

def exponential_map(mat: Array) -> Array:
    if is_tensor(mat):
        return torch.matrix_exp(mat)
    else:
        return expm(mat)

def sin(x: Array) -> Array:
    if is_tensor(x): torch.sin(x)
    return np.sin(x)

def cos(x: Array) -> Array:
    if is_tensor(x): torch.cos(x)
    return np.cos(x)

def tan(x: Array) -> Array:
    if is_tensor(x): torch.tan(x)
    return np.tan(x)

def arcos(x: Array) -> Array:
    if is_tensor(x): torch.arccos(x)
    return np.arccos(x)

def trace(x: Array) -> Array:
    if is_tensor(x): torch.trace(x)
    return np.trace(x)

def vec3_to_skew(x: Array) -> Array:
    assert (x.shape == (3,) or x.shape == (1,3)), f"Invalid shape. Shape of vector must be (3,) or (1,3), but got {str(x.shape)}"
    if x.shape == (1,3): x = reduce_dim(x,0)
    wx = x[0].item()
    wy = x[1].item()
    wz = x[2].item()
    skew_x = np.array([[0., -wz, wy],
                    [wz,  0, -wx],
                    [-wy, wx,  0]])
    if is_tensor(x): skew_x = convert_tensor(skew_x,x)
    return skew_x

if  __name__ == "__main__":
    vec3  = np.array([1,2,3])
    print(vec3_to_skew(vec3))