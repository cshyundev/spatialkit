# Code Convention

**Last Updated:** 2025-11-03
**Current Version:** 0.3.2

This document outlines the code conventions and style guidelines for the project. Adhering to these conventions ensures consistency, readability, and maintainability across the codebase.

## 1. Python Style Guide (PEP 8)

All Python code must adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) – the official Python style guide.

### Key PEP 8 Guidelines:

- **Indentation**: 4 spaces per indentation level.
- **Line Length**: Max 79 characters for code, 72 for docstrings/comments.
- **Blank Lines**:
    - Two blank lines between top-level function/class definitions.
    - One blank line between method definitions inside a class.
- **Imports**:
    - Imports should be on separate lines.
    - Grouped in the following order:
        1. Standard library imports
        2. Third-party imports
        3. Local application/library specific imports
    - Each group separated by a blank line.
    - Alphabetical order within each group.
- **Whitespace**: Avoid extraneous whitespace.
- **Naming Conventions**:
    - `lowercase_with_underscores` for functions, methods, variables.
    - `CamelCase` for classes.
    - `UPPERCASE_WITH_UNDERSCORES` for constants.
    - `_single_leading_underscore` for internal use (private).
    - `__double_leading_underscore` for name mangling (avoid if possible).

## 2. Type Hinting (PEP 484)

All new code and modifications should use type hints to improve code clarity, enable static analysis, and reduce bugs.

### Guidelines:

- Use type hints for function arguments, return values, and variables.
- Use `typing` module for complex types (e.g., `List`, `Dict`, `Optional`, `Union`, `Callable`).
- For internal use, type aliases can be defined in `spatialkit.common.types`.
- For NumPy and PyTorch arrays, use `np.ndarray` and `torch.Tensor` respectively.
- For geometry classes, use the class names directly (e.g., `Rotation`, `Pose`).

```python
from typing import List, Optional, Union
import numpy as np
import torch

def process_data(data: Union[np.ndarray, torch.Tensor], threshold: float) -> List[float]:
    """
    Processes input data based on a given threshold.
    """
    if isinstance(data, np.ndarray):
        filtered_data = data[data > threshold]
    else:
        filtered_data = data[data > threshold]
    return filtered_data.tolist()

class MyClass:
    def __init__(self, name: str, value: Optional[int] = None):
        self.name: str = name
        self.value: Optional[int] = value
```

## 3. Docstrings (PEP 257)

All modules, classes, functions, and methods must have docstrings following the [Google Style Guide](https://google.github.io/styleguide/pyguide.html#pyguide-documenting-code-docstrings).

### Guidelines:

- Use triple double quotes `"""Docstring goes here."""`.
- One-liner docstrings for simple cases.
- Multi-line docstrings for complex cases:
    - Summary line.
    - Blank line.
    - Detailed explanation.
    - Sections for `Args`, `Returns`, `Raises`, `Yields`, `Attributes`, `Example`.

```python
def calculate_sum(a: int, b: int) -> int:
    """Calculates the sum of two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of a and b.
    """
    return a + b

class MyProcessor:
    """A class for processing numerical data.

    Attributes:
        data: The data to be processed.
    """
    def __init__(self, data: np.ndarray):
        self.data = data

    def process(self) -> np.ndarray:
        """Processes the stored data.

        Returns:
            The processed data.
        """
        return self.data * 2
```

## 4. Comments

Use comments sparingly to explain *why* a piece of code does something, rather than *what* it does (which should be clear from the code itself).

### Guidelines:

- Use `#` for inline comments.
- Keep comments concise and up-to-date.
- Avoid redundant comments.

```python
# This loop iterates until convergence is reached,
# which is crucial for the stability of the optimization algorithm.
while not converged:
    # ...
```

## 5. Error Handling

Use Python's exception mechanism for error handling. Avoid returning `None` or special values to indicate errors.

### Guidelines:

- Raise specific exceptions (e.g., `ValueError`, `TypeError`, `FileNotFoundError`).
- Define custom exceptions for domain-specific errors (see `spatialkit.common.exceptions`).
- Use `try...except` blocks to handle expected errors gracefully.
- Avoid broad `except Exception:` clauses unless absolutely necessary and re-raising.

```python
from spatialkit.common.exceptions import InvalidArgumentError

def divide(a: float, b: float) -> float:
    if b == 0:
        raise InvalidArgumentError("Cannot divide by zero.")
    return a / b
```

## 6. Logging

Use the `logging` module for reporting events, debugging information, warnings, and errors.

### Guidelines:

- Import `spatialkit.common.logger` for consistent logging.
- Use appropriate logging levels (`debug`, `info`, `warning`, `error`, `critical`).
- Avoid `print()` statements for debugging in production code.

```python
from spatialkit.common.logger import LOG_INFO, LOG_ERROR

def load_config(path: str) -> dict:
    try:
        # ... load config ...
        LOG_INFO(f"Configuration loaded from {path}")
        return config
    except FileNotFoundError:
        LOG_ERROR(f"Configuration file not found at {path}")
        raise
```

## 7. Imports

Follow the import pattern defined in `docs/import_pattern.md`.

### Guidelines:

- Use absolute imports for internal modules.
- Avoid circular dependencies.
- Use `from __future__ import annotations` for postponed evaluation of type annotations.

## 8. Testing

All new features and bug fixes must be accompanied by unit tests.

### Guidelines:

- Use `pytest` framework.
- Test files should mirror the `src` directory structure (e.g., `tests/geom/test_rotation.py`).
- Test functions should start with `test_`.
- Use descriptive test names.
- Aim for high test coverage.

## 9. Version History

- **v0.3.0-alpha** (2025-11-03): Initial draft, basic PEP 8, type hinting, docstring guidelines.
- **v0.2.1-alpha** (2025-01-30): Added exception handling and logging guidelines.
- **v0.2.0-alpha** (2024-12): Added import pattern and testing guidelines.
