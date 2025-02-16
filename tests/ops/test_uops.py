import unittest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from cv_utils.ops.uops import *


class TestHybridOperations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.numpy_array = np.array([1, 2, 3])
        if TORCH_AVAILABLE:
            cls.torch_tensor = torch.tensor([1, 2, 3])

    def test_is_tensor(self):
        if TORCH_AVAILABLE:
            self.assertTrue(is_tensor(self.torch_tensor))
        self.assertFalse(is_tensor(self.numpy_array))

    def test_is_numpy(self):
        self.assertTrue(is_numpy(self.numpy_array))
        if TORCH_AVAILABLE:
            self.assertFalse(is_numpy(self.torch_tensor))

    def test_is_array(self):
        self.assertTrue(is_array(self.numpy_array))
        if TORCH_AVAILABLE:
            self.assertTrue(is_array(self.torch_tensor))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_convert_tensor(self):
        result = convert_tensor(self.numpy_array)
        self.assertTrue(torch.is_tensor(result))
        self.assertTrue(np.array_equal(result.numpy(), self.numpy_array))

    def test_convert_numpy(self):
        if TORCH_AVAILABLE:
            result = convert_numpy(self.torch_tensor)
            self.assertIsInstance(result, np.ndarray)
            self.assertTrue(np.array_equal(result, self.torch_tensor.numpy()))
        self.assertTrue(np.array_equal(convert_numpy(self.numpy_array), self.numpy_array))

    def test_convert_array(self):
        result = convert_array([1, 2, 3], self.numpy_array)
        self.assertTrue(np.array_equal(result, self.numpy_array))
        if TORCH_AVAILABLE:
            result = convert_array([1, 2, 3], self.torch_tensor)
            self.assertTrue(torch.is_tensor(result))

    def test_expand_dim(self):
        expanded_numpy = expand_dim(self.numpy_array, 0)
        self.assertEqual(expanded_numpy.shape, (1, 3))
        if TORCH_AVAILABLE:
            expanded_torch = expand_dim(self.torch_tensor, 0)
            self.assertEqual(expanded_torch.shape, (1, 3))

    def test_reduce_dim(self):
        array = np.expand_dims(self.numpy_array, 0)
        reduced_numpy = reduce_dim(array, 0)
        self.assertEqual(reduced_numpy.shape, (3,))
        if TORCH_AVAILABLE:
            tensor = torch.unsqueeze(self.torch_tensor, 0)
            reduced_torch = reduce_dim(tensor, 0)
            self.assertEqual(reduced_torch.shape, (3,))

    def test_concat(self):
        list_of_arrays = [self.numpy_array, self.numpy_array]
        result = concat(list_of_arrays, 0)
        self.assertEqual(result.shape, (6,))
        if TORCH_AVAILABLE:
            list_of_tensors = [self.torch_tensor, self.torch_tensor]
            result = concat(list_of_tensors, 0)
            self.assertEqual(result.shape, (6,))

    def test_stack(self):
        list_of_arrays = [self.numpy_array, self.numpy_array]
        result = stack(list_of_arrays, 0)
        self.assertEqual(result.shape, (2, 3))
        if TORCH_AVAILABLE:
            list_of_tensors = [self.torch_tensor, self.torch_tensor]
            result = stack(list_of_tensors, 0)
            self.assertEqual(result.shape, (2, 3))
    
    def test_logical_or_with_numpy(self):
        # Test logical_or with NumPy arrays
        a = np.array([True, False, True])
        b = np.array([False, False, True])
        c = np.array([False, True, False])
        
        # Expected result using NumPy's logical_or
        expected = np.logical_or(np.logical_or(a, b), c)
        result = logical_or(a, b, c)
        
        np.testing.assert_array_equal(result, expected)

    def test_logical_or_with_tensors(self):
        # Test logical_or with PyTorch tensors
        a = torch.tensor([True, False, True])
        b = torch.tensor([False, False, True])
        c = torch.tensor([False, True, False])
        
        # Expected result using Torch's logical_or
        expected = torch.logical_or(torch.logical_or(a, b), c)
        result = logical_or(a, b, c)
        
        self.assertTrue(torch.equal(result, expected))

    def test_logical_and_with_numpy(self):
        # Test logical_and with NumPy arrays
        a = np.array([True, False, True])
        b = np.array([True, False, False])
        c = np.array([True, True, False])
        
        # Expected result using NumPy's logical_and
        expected = np.logical_and(np.logical_and(a, b), c)
        result = logical_and(a, b, c)
        
        np.testing.assert_array_equal(result, expected)

    def test_logical_and_with_tensors(self):
        # Test logical_and with PyTorch tensors
        a = torch.tensor([True, False, True])
        b = torch.tensor([True, False, False])
        c = torch.tensor([True, True, False])
        
        # Expected result using Torch's logical_and
        expected = torch.logical_and(torch.logical_and(a, b), c)
        result = logical_and(a, b, c)
        
        self.assertTrue(torch.equal(result, expected))

    def test_input_validation(self):
        # Test to ensure input validation is working
        with self.assertRaises(AssertionError):
            logical_or()  # No input arrays
            logical_and()  # No input arrays

if __name__ == '__main__':
    unittest.main()
