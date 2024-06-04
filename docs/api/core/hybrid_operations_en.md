# API Documentation: Matrix and Vector Operations

**Last Updated:** June 2, 2024  
**version:** v0.1.0

## Overview

This API provides a set of functions designed to perform matrix and vector operations on data represented using either NumPy arrays or PyTorch tensors. The primary focus is to facilitate seamless integration and manipulation of both data types using unified functions.

## Functions

### `norm`

**Purpose**: Computes the norm (magnitude) of a vector or matrix.

**Parameters**:
- `x` (Array): Input data. Can be either a NumPy array or a PyTorch tensor.
- `ord` (Union[int, str], optional): Order of the norm. Can be an integer (e.g., 1, 2) or a string (e.g., 'fro' for Frobenius norm). Defaults to None, which calculates the 2-norm.
- `dim` (int, optional): Dimension along which to compute the norm. If not specified, the norm is computed over the entire array. Defaults to None.
- `keepdim` (bool, optional): Whether to keep the reduced dimensions in the output. Defaults to False.

**Returns**:
- `Array`: The computed norm. The type of the output matches the type of the input (`x`).

**Raises**:
- `AssertionError`: If the specified dimension (`dim`) is invalid, i.e., greater than or equal to the number of dimensions in `x`.

### `normalize`

**Purpose**: Normalizes the input vector or matrix along the specified dimension.

**Parameters**:
- `x` (Array): Input data. Can be either a NumPy array or a PyTorch tensor.
- `ord` (Union[int, str], optional): Order of the norm used for normalization. Can be an integer (e.g., 1, 2) or a string (e.g., 'fro' for Frobenius norm). Defaults to None, which calculates the 2-norm.
- `dim` (int, optional): Dimension along which to normalize. If not specified, normalization is applied to the entire array. Defaults to None.
- `eps` (float, optional): A small value added to the denominator to avoid division by zero. Defaults to a predefined constant `EPSILON`.

**Returns**:
- `Array`: The normalized array. The type of the output matches the type of the input (`x`).

## Helper Functions

### `is_tensor`

**Purpose**: Checks if the input is a PyTorch tensor.

**Parameters**:
- `x` (Any): Input to be checked.

**Returns**:
- `bool`: True if `x` is a PyTorch tensor, False otherwise.

## Constants

### `EPSILON`

**Purpose**: A small constant used to avoid division by zero in normalization.

**Value**:
```python
EPSILON = 1e-10
