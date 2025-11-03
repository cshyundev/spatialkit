"""
Operations package for unified NumPy/PyTorch operations.

This package provides type-agnostic operations that work seamlessly with both
NumPy arrays and PyTorch tensors.

Modules:
    uops: Basic unified operations (concat, stack, transpose, type checking, etc.)
    umath: Mathematical unified operations (matmul, svd, qr, inv, etc.)
"""

from .uops import *
from .umath import *

__all__ = uops.__all__ + umath.__all__