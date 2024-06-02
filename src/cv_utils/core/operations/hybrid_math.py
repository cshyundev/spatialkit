from .hybrid_operations import *
from scipy.linalg import expm
from cv_utils.constant import EPSILON

# Basic Mathematical Operations
def abs(x: Array) -> Array:
    if is_tensor(x): return torch.abs(x)
    return np.abs(x)

def sqrt(x: Array) -> Array:
    if is_tensor(x): return torch.sqrt(x)
    return np.sqrt(x)

def mean(x: Array, dim: int = None) -> Array:
    if dim is not None:
        assert dim < x.ndim, f"Invalid dimension: expected dimension less than {x.ndim}, but got {dim}."
    if is_tensor(x): return torch.mean(x, dim=dim)
    return np.mean(x, axis=dim)

def dot(x: Array, y: Array) -> Array:
    assert(type(x) == type(y)), f"Invalid type: expected same type for both arrays, but got {type(x)} and {type(y)}"
    if is_tensor(x): return torch.dot(x, y)
    return np.dot(x, y)

def svd(x: Array) -> Array:
    assert x.ndim == 2, f"Invalid shape for SVD: expected a 2D matrix, but got {x.shape}."
    if is_tensor(x): return torch.svd(x)
    return np.linalg.svd(x)

def determinant(x: Array) -> float:
    assert x.ndim == 2, f"Invalid shape for determinant: expected a 2D square matrix, but got {x.shape}."
    if is_tensor(x):
        return torch.det(x).item()
    return np.linalg.det(x)

def inv(x: Array) -> Array:
    assert x.ndim == 2, f"Invalid shape for inversion: expected a 2D square matrix, but got {x.shape}."
    if is_tensor(x): return torch.inverse(x)
    return np.linalg.inv(x)

# Matrix and Vector Operations
def norm(x: Array, ord: Union[int, str] = None, dim: int = None, keepdim: bool = False) -> Array:
    if dim is not None and dim >= x.ndim:
        assert False, f"Invalid dimension for norm: expected dimension less than {x.ndim}, but got {dim}."
    if is_tensor(x): return torch.norm(x, p=ord, dim=dim, keepdim=keepdim)
    return np.linalg.norm(x, ord=ord, axis=dim, keepdims=keepdim)

def normalize(x: Array, ord: Union[int, str] = None, dim: int = None, eps: float = EPSILON) -> Array:
    n = norm(x=x, ord=ord, dim=dim, keepdim=False)
    return x / (n + eps)

def transpose2d(x: Array) -> Array:
    assert x.ndim == 2, f"Invalid shape for transpose: expected a 2D array, but got {x.shape}."
    if is_tensor(x): return x.transpose(0, 1)
    return x.T

def matmul(x: Array, y: Array) -> Array:
    assert x.ndim >= 1 and y.ndim >= 1, f"Invalid shape: expected at least 1D arrays, but got {x.shape} and {y.shape}."
    assert x.shape[-1] == y.shape[0], f"Invalid shape for matmul: x dimensions {x.shape[-1]}, y dimensions {y.shape[0]}."
    if is_tensor(x): return torch.matmul(x, y)
    return x @ y

def permute(x: Array, dims: Tuple[int]) -> Array:
    assert len(x.shape) == len(dims), f"Invalid permutation: expected dimensions {len(x.shape)}, but got {len(dims)}."
    if is_tensor(x): return x.permute(dims)
    return x.transpose(dims)

def trace(x: Array) -> Array:
    assert x.ndim == 2, f"Invalid shape for trace: expected a 2D array, but got {x.shape}."
    if is_tensor(x): return torch.trace(x)
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
    if is_tensor(x): skew_x = convert_tensor(skew_x, x)
    return skew_x

# Transform and Permutation
def rad2deg(x: Array) -> Array:
    return x * (180. / np.pi)

def deg2rad(x: Array) -> Array:
    return x * (np.pi / 180.)

def exponential_map(mat: Array) -> Array:
    if is_tensor(mat):
        return torch.matrix_exp(mat)
    else:
        return expm(mat)

# Trigonometric functions
def sin(x: Array) -> Array:
    if is_tensor(x): return torch.sin(x)
    return np.sin(x)

def cos(x: Array) -> Array:
    if is_tensor(x): return torch.cos(x)
    return np.cos(x)

def tan(x: Array) -> Array:
    if is_tensor(x): return torch.tan(x)
    return np.tan(x)

def arcsin(x: Array) -> Array:
    if is_tensor(x): return torch.arcsin(x)
    return np.arcsin(x)

def arcos(x: Array) -> Array:
    if is_tensor(x): return torch.arccos(x)
    return np.arccos(x)

def arctan(x: Array) -> Array:
    if is_tensor(x): return torch.arctan(x)
    return np.arctan(x)

def arctan2(x: Array) -> Array:
    if is_tensor(x): return torch.arctan2(x)
    return np.arctan2(x)
