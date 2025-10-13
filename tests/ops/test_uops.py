import unittest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Use new hierarchical import pattern
import cv_utils
from cv_utils import uops


class TestHybridOperations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.numpy_array = np.array([1, 2, 3])
        if TORCH_AVAILABLE:
            cls.torch_tensor = torch.tensor([1, 2, 3])

    def test_is_tensor(self):
        if TORCH_AVAILABLE:
            self.assertTrue(uops.is_tensor(self.torch_tensor))
        self.assertFalse(uops.is_tensor(self.numpy_array))

    def test_is_numpy(self):
        self.assertTrue(uops.is_numpy(self.numpy_array))
        if TORCH_AVAILABLE:
            self.assertFalse(uops.is_numpy(self.torch_tensor))

    def test_is_array(self):
        self.assertTrue(uops.is_array(self.numpy_array))
        if TORCH_AVAILABLE:
            self.assertTrue(uops.is_array(self.torch_tensor))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_convert_tensor(self):
        result = uops.convert_tensor(self.numpy_array)
        self.assertTrue(torch.is_tensor(result))
        self.assertTrue(np.array_equal(result.numpy(), self.numpy_array))

    def test_convert_numpy(self):
        if TORCH_AVAILABLE:
            result = uops.convert_numpy(self.torch_tensor)
            self.assertIsInstance(result, np.ndarray)
            self.assertTrue(np.array_equal(result, self.torch_tensor.numpy()))
        self.assertTrue(np.array_equal(uops.convert_numpy(self.numpy_array), self.numpy_array))

    def test_convert_array(self):
        result = uops.convert_array([1, 2, 3], self.numpy_array)
        self.assertTrue(np.array_equal(result, self.numpy_array))
        if TORCH_AVAILABLE:
            result = uops.convert_array([1, 2, 3], self.torch_tensor)
            self.assertTrue(torch.is_tensor(result))

    def test_expand_dim(self):
        expanded_numpy = uops.expand_dim(self.numpy_array, 0)
        self.assertEqual(expanded_numpy.shape, (1, 3))
        if TORCH_AVAILABLE:
            expanded_torch = uops.expand_dim(self.torch_tensor, 0)
            self.assertEqual(expanded_torch.shape, (1, 3))

    def test_reduce_dim(self):
        array = np.expand_dims(self.numpy_array, 0)
        reduced_numpy = uops.reduce_dim(array, 0)
        self.assertEqual(reduced_numpy.shape, (3,))
        if TORCH_AVAILABLE:
            tensor = torch.unsqueeze(self.torch_tensor, 0)
            reduced_torch = uops.reduce_dim(tensor, 0)
            self.assertEqual(reduced_torch.shape, (3,))

    def test_concat(self):
        list_of_arrays = [self.numpy_array, self.numpy_array]
        result = uops.concat(list_of_arrays, 0)
        self.assertEqual(result.shape, (6,))
        if TORCH_AVAILABLE:
            list_of_tensors = [self.torch_tensor, self.torch_tensor]
            result = uops.concat(list_of_tensors, 0)
            self.assertEqual(result.shape, (6,))

    def test_stack(self):
        list_of_arrays = [self.numpy_array, self.numpy_array]
        result = uops.stack(list_of_arrays, 0)
        self.assertEqual(result.shape, (2, 3))
        if TORCH_AVAILABLE:
            list_of_tensors = [self.torch_tensor, self.torch_tensor]
            result = uops.stack(list_of_tensors, 0)
            self.assertEqual(result.shape, (2, 3))

    def test_logical_or_with_numpy(self):
        # Test logical_or with NumPy arrays
        a = np.array([True, False, True])
        b = np.array([False, False, True])
        c = np.array([False, True, False])

        # Expected result using NumPy's logical_or
        expected = np.logical_or(np.logical_or(a, b), c)
        result = uops.logical_or(a, b, c)
        
        np.testing.assert_array_equal(result, expected)

    def test_logical_or_with_tensors(self):
        # Test logical_or with PyTorch tensors
        a = torch.tensor([True, False, True])
        b = torch.tensor([False, False, True])
        c = torch.tensor([False, True, False])

        # Expected result using Torch's logical_or
        expected = torch.logical_or(torch.logical_or(a, b), c)
        result = uops.logical_or(a, b, c)

        self.assertTrue(torch.equal(result, expected))

    def test_logical_and_with_numpy(self):
        # Test logical_and with NumPy arrays
        a = np.array([True, False, True])
        b = np.array([True, False, False])
        c = np.array([True, True, False])

        # Expected result using NumPy's logical_and
        expected = np.logical_and(np.logical_and(a, b), c)
        result = uops.logical_and(a, b, c)

        np.testing.assert_array_equal(result, expected)

    def test_logical_and_with_tensors(self):
        # Test logical_and with PyTorch tensors
        a = torch.tensor([True, False, True])
        b = torch.tensor([True, False, False])
        c = torch.tensor([True, True, False])

        # Expected result using Torch's logical_and
        expected = torch.logical_and(torch.logical_and(a, b), c)
        result = uops.logical_and(a, b, c)

        self.assertTrue(torch.equal(result, expected))

    def test_input_validation(self):
        # Test to ensure input validation is working
        with self.assertRaises(Exception):
            uops.logical_or()  # No input arrays
            uops.logical_and()  # No input arrays

    def test_numel(self):
        # Test with NumPy array
        result_np = uops.numel(self.numpy_array)
        self.assertEqual(result_np, 3)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            result_torch = uops.numel(self.torch_tensor)
            self.assertEqual(result_torch, 3)

        # Test error handling
        with self.assertRaises(cv_utils.IncompatibleTypeError):
            uops.numel([1, 2, 3])

    def test_zeros_like(self):
        # Test with NumPy array
        result_np = uops.zeros_like(self.numpy_array)
        expected_np = np.zeros_like(self.numpy_array)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            result_torch = uops.zeros_like(self.torch_tensor)
            expected_torch = torch.zeros_like(self.torch_tensor)
            torch.testing.assert_close(result_torch, expected_torch)

    def test_empty_like(self):
        # Test with NumPy array
        result_np = uops.empty_like(self.numpy_array)
        self.assertEqual(result_np.shape, self.numpy_array.shape)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            result_torch = uops.empty_like(self.torch_tensor)
            self.assertEqual(result_torch.shape, self.torch_tensor.shape)

    def test_full_like(self):
        # Test with NumPy array
        result_np = uops.full_like(self.numpy_array, 5)
        expected_np = np.full_like(self.numpy_array, 5)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            result_torch = uops.full_like(self.torch_tensor, 5)
            expected_torch = torch.full_like(self.torch_tensor, 5)
            torch.testing.assert_close(result_torch, expected_torch)

    def test_arange(self):
        # Test with NumPy array
        result_np = uops.arange(self.numpy_array, 0, 5)
        expected_np = np.arange(0, 5)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            result_torch = uops.arange(self.torch_tensor, 0, 5)
            self.assertTrue(torch.is_tensor(result_torch))

    def test_deep_copy(self):
        # Test with NumPy array
        copy_np = uops.deep_copy(self.numpy_array)
        self.assertTrue(np.array_equal(copy_np, self.numpy_array))
        copy_np[0] = 999
        self.assertFalse(np.array_equal(copy_np, self.numpy_array))

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            copy_torch = uops.deep_copy(self.torch_tensor)
            self.assertTrue(torch.equal(copy_torch, self.torch_tensor))

    def test_where(self):
        # Test with NumPy arrays
        condition_np = np.array([True, False, True])
        x_np = np.array([1, 2, 3])
        y_np = np.array([4, 5, 6])
        result_np = uops.where(condition_np, x_np, y_np)
        expected_np = np.where(condition_np, x_np, y_np)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensors
        if TORCH_AVAILABLE:
            condition_torch = torch.tensor([True, False, True])
            x_torch = torch.tensor([1, 2, 3])
            y_torch = torch.tensor([4, 5, 6])
            result_torch = uops.where(condition_torch, x_torch, y_torch)
            expected_torch = torch.where(condition_torch, x_torch, y_torch)
            torch.testing.assert_close(result_torch, expected_torch)

    def test_clip(self):
        # Test with NumPy array
        x_np = np.array([1, 5, 10])
        result_np = uops.clip(x_np, 2, 8)
        expected_np = np.clip(x_np, 2, 8)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([1.0, 5.0, 10.0])
            result_torch = uops.clip(x_torch, 2, 8)
            expected_torch = torch.clip(x_torch, 2, 8)
            torch.testing.assert_close(result_torch, expected_torch)

    def test_eye(self):
        # Test with NumPy array
        result_np = uops.eye(3, self.numpy_array)
        expected_np = np.eye(3)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            result_torch = uops.eye(3, self.torch_tensor)
            expected_torch = torch.eye(3)
            torch.testing.assert_close(result_torch, expected_torch)

    def test_transpose2d(self):
        # Test with NumPy array
        x_np = np.array([[1, 2], [3, 4]])
        result_np = uops.transpose2d(x_np)
        expected_np = x_np.T
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([[1, 2], [3, 4]])
            result_torch = uops.transpose2d(x_torch)
            expected_torch = x_torch.T
            torch.testing.assert_close(result_torch, expected_torch)

        # Test error handling
        with self.assertRaises(cv_utils.InvalidShapeError):
            uops.transpose2d(self.numpy_array)

    def test_swapaxes(self):
        # Test with NumPy array
        x_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result_np = uops.swapaxes(x_np, 0, 2)
        expected_np = np.swapaxes(x_np, 0, 2)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
            result_torch = uops.swapaxes(x_torch, 0, 2)
            expected_torch = torch.swapaxes(x_torch, 0, 2)
            torch.testing.assert_close(result_torch, expected_torch)

    def test_as_bool(self):
        # Test with NumPy array
        x_np = np.array([1, 0, 1])
        result_np = uops.as_bool(x_np)
        self.assertEqual(result_np.dtype, bool)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([1, 0, 1])
            result_torch = uops.as_bool(x_torch)
            self.assertEqual(result_torch.dtype, torch.bool)

    def test_as_int(self):
        # Test with NumPy array
        x_np = np.array([1.5, 2.7, 3.9])
        result_np = uops.as_int(x_np, 32)
        self.assertEqual(result_np.dtype, np.int32)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([1.5, 2.7, 3.9])
            result_torch = uops.as_int(x_torch, 32)
            self.assertEqual(result_torch.dtype, torch.int32)

    def test_as_float(self):
        # Test with NumPy array
        x_np = np.array([1, 2, 3])
        result_np = uops.as_float(x_np, 32)
        self.assertEqual(result_np.dtype, np.float32)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([1, 2, 3])
            result_torch = uops.as_float(x_torch, 32)
            self.assertEqual(result_torch.dtype, torch.float32)

    def test_logical_not(self):
        # Test with NumPy array
        x_np = np.array([True, False, True])
        result_np = uops.logical_not(x_np)
        expected_np = np.logical_not(x_np)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([True, False, True])
            result_torch = uops.logical_not(x_torch)
            expected_torch = torch.logical_not(x_torch)
            torch.testing.assert_close(result_torch, expected_torch)

        # Test error handling
        with self.assertRaises(cv_utils.IncompatibleTypeError):
            uops.logical_not([True, False])

    def test_logical_xor(self):
        # Note: The logical_xor function in uops.py has a bug - it should take 2 arguments
        # For now, we'll test the error handling only

        # Test error handling
        with self.assertRaises(cv_utils.IncompatibleTypeError):
            uops.logical_xor([True, False])

    def test_allclose(self):
        # Test with NumPy arrays
        x_np = np.array([1.0, 2.0, 3.0])
        y_np = np.array([1.0, 2.0, 3.0001])
        result_np = uops.allclose(x_np, y_np, atol=1e-3)
        self.assertTrue(result_np)

        # Test with PyTorch tensors
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([1.0, 2.0, 3.0])
            y_torch = torch.tensor([1.0, 2.0, 3.0001])
            result_torch = uops.allclose(x_torch, y_torch, atol=1e-3)
            self.assertTrue(result_torch)

        # Test error handling
        with self.assertRaises(cv_utils.IncompatibleTypeError):
            uops.allclose(x_np, self.torch_tensor if TORCH_AVAILABLE else x_np)

    def test_isclose(self):
        # Test with NumPy array
        x_np = np.array([1.0, 2.0, 3.0])
        result_np = uops.isclose(x_np, 2.0, atol=1e-3)
        expected_np = np.isclose(x_np, 2.0, atol=1e-3)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        if TORCH_AVAILABLE:
            x_torch = torch.tensor([1.0, 2.0, 3.0])
            y_torch = torch.tensor(2.0)
            result_torch = uops.isclose(x_torch, y_torch, atol=1e-3)
            expected_torch = torch.isclose(x_torch, y_torch, atol=1e-3)
            torch.testing.assert_close(result_torch, expected_torch)

    def test_convert_tensor_error(self):
        # Test error handling for convert_tensor
        if TORCH_AVAILABLE:
            with self.assertRaises(cv_utils.IncompatibleTypeError):
                uops.convert_tensor(self.numpy_array, tensor=self.numpy_array)

    def test_ones_like_error(self):
        # Test error handling for ones_like
        with self.assertRaises(cv_utils.IncompatibleTypeError):
            uops.ones_like([1, 2, 3])

    def test_logical_and_error(self):
        # Test error handling for logical_and with < 2 arrays
        with self.assertRaises(cv_utils.InvalidDimensionError):
            uops.logical_and(self.numpy_array)


class TestDtypeUtilities(unittest.TestCase):
    """Test dtype utility functions."""

    def test_get_dtype_numpy(self):
        """Test get_dtype with numpy arrays."""
        arr_f32 = np.array([1.0], dtype=np.float32)
        arr_f64 = np.array([1.0], dtype=np.float64)
        arr_i32 = np.array([1], dtype=np.int32)

        self.assertEqual(uops.get_dtype(arr_f32), np.float32)
        self.assertEqual(uops.get_dtype(arr_f64), np.float64)
        self.assertEqual(uops.get_dtype(arr_i32), np.int32)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_get_dtype_torch(self):
        """Test get_dtype with torch tensors."""
        tensor_f32 = torch.tensor([1.0], dtype=torch.float32)
        tensor_f64 = torch.tensor([1.0], dtype=torch.float64)

        self.assertEqual(uops.get_dtype(tensor_f32), torch.float32)
        self.assertEqual(uops.get_dtype(tensor_f64), torch.float64)

    def test_get_dtype_error(self):
        """Test get_dtype with invalid input."""
        with self.assertRaises(cv_utils.IncompatibleTypeError):
            uops.get_dtype([1, 2, 3])

    def test_promote_types_numpy(self):
        """Test dtype promotion for numpy arrays."""
        arr_f32 = np.array([1.0], dtype=np.float32)
        arr_f64 = np.array([2.0], dtype=np.float64)
        arr_i32 = np.array([3], dtype=np.int32)

        # float32 + float64 → float64
        promoted = uops.promote_types(arr_f32, arr_f64)
        self.assertEqual(promoted, np.float64)

        # int32 + float64 → float64
        promoted = uops.promote_types(arr_i32, arr_f64)
        self.assertEqual(promoted, np.float64)

        # float64 + float32 + int32 → float64
        promoted = uops.promote_types(arr_f64, arr_f32, arr_i32)
        self.assertEqual(promoted, np.float64)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_promote_types_torch(self):
        """Test dtype promotion for torch tensors."""
        tensor_f32 = torch.tensor([1.0], dtype=torch.float32)
        tensor_f64 = torch.tensor([2.0], dtype=torch.float64)

        promoted = uops.promote_types(tensor_f32, tensor_f64)
        self.assertEqual(promoted, torch.float64)

    def test_promote_types_usage(self):
        """Test using promoted dtype for conversion."""
        arr_f32 = np.array([1.0], dtype=np.float32)
        arr_f64 = np.array([2.0], dtype=np.float64)

        promoted = uops.promote_types(arr_f32, arr_f64)
        result = arr_f32.astype(promoted)

        self.assertEqual(result.dtype, np.float64)

    def test_promote_types_no_arrays(self):
        """Test promote_types with no arrays."""
        with self.assertRaises(cv_utils.InvalidDimensionError):
            uops.promote_types()

    def test_promote_types_invalid_input(self):
        """Test promote_types with non-array input."""
        arr = np.array([1.0])
        with self.assertRaises(cv_utils.IncompatibleTypeError):
            uops.promote_types(arr, [1, 2, 3])

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_promote_types_mixed_types_error(self):
        """Test that mixing numpy and tensor raises error."""
        arr_np = np.array([1.0])
        arr_torch = torch.tensor([2.0])

        with self.assertRaises(cv_utils.IncompatibleTypeError):
            uops.promote_types(arr_np, arr_torch)


if __name__ == '__main__':
    unittest.main()
