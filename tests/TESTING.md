# Testing Guide for cv_utils

This guide explains how to write and run tests for the cv_utils library.

## Quick Start

### Creating a New Test File

Use the test generator script to quickly create a new test file:

```bash
# Basic usage - creates test in tests/ directory
python tests/create_test.py my_module

# Create test in a specific category
python tests/create_test.py camera --category geom
python tests/create_test.py new_solver --category sol

# With author name
python tests/create_test.py my_module --category ops --author "Your Name"
```

Available categories:
- `ops`: Operations and math utilities
- `geom`: Geometry primitives (rotations, poses, cameras)
- `utils`: Utilities (I/O, visualization)
- `sol`: Solutions (markers, MVS)

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run tests in a specific directory
uv run pytest tests/geom/

# Run a specific test file
uv run pytest tests/geom/test_rotation.py

# Run a specific test class
uv run pytest tests/geom/test_rotation.py::TestRotation

# Run a specific test method
uv run pytest tests/geom/test_rotation.py::TestRotation::test_slerp

# Run with verbose output
uv run pytest tests/ -v

# Run and stop on first failure
uv run pytest tests/ -x

# Run with coverage report
uv run pytest tests/ --cov=cv_utils --cov-report=term
uv run pytest tests/ --cov=cv_utils --cov-report=html  # HTML report in htmlcov/
```

## Test Organization

```
tests/
├── test_template.py          # Template for new tests
├── create_test.py            # Script to generate new tests
├── TESTING.md                # This file
├── ops/                      # Tests for operations
│   ├── test_uops.py          # Unified operations tests
│   └── test_umath.py         # Unified math tests
├── geom/                     # Tests for geometry
│   ├── test_rotation.py      # Rotation tests
│   ├── test_pose.py          # Pose tests
│   ├── test_camera.py        # Camera tests
│   └── test_geom_utils.py    # Geometry utilities tests
├── utils/                    # Tests for utilities
│   └── (utility tests)
└── sol/                      # Tests for solutions
    └── marker/               # Marker detection tests
        └── test_detector.py
```

## Import Patterns

The cv_utils library uses a hierarchical import pattern. Follow these guidelines:

### High-Level Classes (Recommended)

Import commonly used classes directly from `cv_utils`:

```python
from cv_utils import Rotation, Pose, Transform, Camera
from cv_utils import PerspectiveCamera, OpenCVFisheyeCamera
from cv_utils import uops, umath
```

### Module-Level Imports

For accessing module contents:

```python
from cv_utils import geom
from cv_utils.geom import rotation, camera, pose

# Then use:
rotation.slerp(r1, r2, t)
camera.PerspectiveCamera(...)
```

### Specific Function Imports

For importing specific functions:

```python
from cv_utils.geom.rotation import slerp, is_SO3
from cv_utils.geom.geom_utils import solve_pnp, triangulate_points
from cv_utils.ops import uops
```

### What to Import

**Commonly Used:**
- `uops`: Array operations (is_array, convert_numpy, concat, etc.)
- `umath`: Math operations (svd, inv, matmul, etc.)
- `Rotation`, `Pose`, `Transform`: Geometry classes
- `Camera` and camera subclasses

**Less Common:**
- Specific utility functions from modules
- Internal helper functions (usually not needed in tests)

## Writing Tests

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_example(self):
    """Test description."""
    # Arrange: Set up test data
    input_data = np.array([1.0, 2.0, 3.0])
    expected_output = np.array([2.0, 4.0, 6.0])

    # Act: Execute the function being tested
    result = function_under_test(input_data)

    # Assert: Verify the result
    np.testing.assert_array_equal(result, expected_output)
```

### Test Naming Conventions

Use descriptive names that explain what is being tested:

```python
def test_rotation_from_so3(self):
    """Test creating rotation from so3 vector."""

def test_invalid_input_raises_exception(self):
    """Test that invalid input raises IncompatibleTypeError."""

def test_slerp_halfway_interpolation(self):
    """Test spherical interpolation at t=0.5."""
```

### Testing Both NumPy and PyTorch

Most functions support both NumPy arrays and PyTorch tensors. Test both:

```python
def test_with_numpy_array(self):
    """Test with NumPy array input."""
    input_np = np.array([1.0, 2.0, 3.0])
    result = function_under_test(input_np)
    self.assertIsInstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)

def test_with_torch_tensor(self):
    """Test with PyTorch tensor input."""
    input_torch = torch.tensor([1.0, 2.0, 3.0])
    result = function_under_test(input_torch)
    self.assertIsInstance(result, torch.Tensor)
    torch.testing.assert_close(result, expected)
```

### Testing Exceptions

Use `assertRaises` context manager:

```python
def test_invalid_shape_raises_error(self):
    """Test that invalid shape raises InvalidShapeError."""
    invalid_input = np.array([1, 2])  # Wrong shape

    with self.assertRaises(InvalidShapeError):
        function_under_test(invalid_input)
```

### Testing Conversions

For functions that convert between representations, test round-trip:

```python
def test_so3_to_SO3_and_back(self):
    """Test conversion from so3 to SO3 and back."""
    original_so3 = np.array([0.1, 0.2, 0.3])

    # Convert to SO3
    SO3 = so3_to_SO3(original_so3)

    # Convert back to so3
    recovered_so3 = SO3_to_so3(SO3)

    # Verify round-trip
    np.testing.assert_almost_equal(original_so3, recovered_so3, decimal=10)
```

## Common Assertions

### NumPy Assertions

```python
np.testing.assert_array_equal(actual, expected)           # Exact equality
np.testing.assert_array_almost_equal(actual, expected)   # Approximate equality
np.testing.assert_allclose(actual, expected, rtol=1e-5)  # Relative tolerance
```

### PyTorch Assertions

```python
torch.testing.assert_close(actual, expected)              # Default tolerances
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
```

### General Assertions

```python
self.assertEqual(a, b)
self.assertAlmostEqual(a, b, places=7)
self.assertTrue(condition)
self.assertFalse(condition)
self.assertIsNone(value)
self.assertIsInstance(obj, type)
self.assertRaises(ExceptionType, callable, *args)
```

## Best Practices

1. **One Test, One Thing**: Each test should verify one specific behavior
2. **Descriptive Names**: Test names should clearly indicate what is being tested
3. **Independent Tests**: Tests should not depend on each other
4. **Use setUp/tearDown**: Initialize common data in setUp(), clean up in tearDown()
5. **Test Edge Cases**: Include tests for boundary conditions, empty inputs, etc.
6. **Test Exceptions**: Verify that functions raise appropriate exceptions
7. **Document Tests**: Add docstrings explaining what each test verifies
8. **Keep Tests Fast**: Avoid slow operations when possible

## Test Coverage

Check which parts of the code are covered by tests:

```bash
# Generate coverage report
uv run pytest tests/ --cov=cv_utils --cov-report=term

# Generate HTML coverage report
uv run pytest tests/ --cov=cv_utils --cov-report=html

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

Current coverage (as of last update):
- **Total: 38%**
- ops/uops.py: 62%
- ops/umath.py: 65%
- geom/rotation.py: 62%
- geom/camera.py: 52%

## Debugging Tests

```bash
# Print output during tests
uv run pytest tests/ -s

# Drop into debugger on failure
uv run pytest tests/ --pdb

# Run only failed tests from last run
uv run pytest tests/ --lf

# Show local variables in tracebacks
uv run pytest tests/ -l
```

## Continuous Integration

Tests are automatically run on every commit. Make sure your tests pass before pushing:

```bash
# Run all tests
uv run pytest tests/

# Check coverage
uv run pytest tests/ --cov=cv_utils --cov-report=term

# Run with multiple Python versions (if configured)
tox
```

## Examples

### Example 1: Testing a Simple Function

```python
import unittest
import numpy as np
from cv_utils import uops

class TestConvertNumpy(unittest.TestCase):
    def test_convert_tensor_to_numpy(self):
        """Test converting PyTorch tensor to NumPy array."""
        import torch
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = uops.convert_numpy(tensor)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
```

### Example 2: Testing a Class

```python
import unittest
import numpy as np
from cv_utils import Rotation

class TestRotation(unittest.TestCase):
    def setUp(self):
        """Initialize test fixtures."""
        self.identity_quat = np.array([0, 0, 0, 1])  # [x, y, z, w]

    def test_identity_rotation(self):
        """Test that identity quaternion produces identity matrix."""
        rot = Rotation.from_quat_xyzw(self.identity_quat)
        np.testing.assert_array_almost_equal(rot.mat(), np.eye(3))
```

### Example 3: Testing Exceptions

```python
import unittest
from cv_utils import Rotation
from cv_utils.exceptions import InvalidShapeError

class TestRotationExceptions(unittest.TestCase):
    def test_invalid_quaternion_shape(self):
        """Test that invalid quaternion shape raises error."""
        invalid_quat = np.array([0, 0, 0])  # Should be 4 elements

        with self.assertRaises(InvalidShapeError):
            Rotation.from_quat_xyzw(invalid_quat)
```

## Troubleshooting

### Import Errors

If you get import errors:
1. Make sure you're using the hierarchical import pattern
2. Check that you're importing from the correct module path
3. Verify the class/function name is correct

### Test Discovery Issues

If pytest doesn't find your tests:
1. Ensure file name starts with `test_`
2. Ensure class name starts with `Test`
3. Ensure method names start with `test_`
4. Check that the file is in a directory with an `__init__.py`

### Slow Tests

If tests are running slowly:
1. Use smaller test data
2. Mock expensive operations
3. Use `pytest -k pattern` to run specific tests
4. Consider using `pytest-xdist` for parallel execution

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [NumPy testing utilities](https://numpy.org/doc/stable/reference/routines.testing.html)
- [PyTorch testing utilities](https://pytorch.org/docs/stable/testing.html)
- [unittest documentation](https://docs.python.org/3/library/unittest.html)
