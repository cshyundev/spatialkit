import numpy as np
import torch
import unittest
from cv_utils.core.operations.hybrid_math import *

class TestHybridMath(unittest.TestCase):
    def test_svd(self):
        # Test with NumPy array
        x_np = np.random.rand(3, 3)
        u, s, vh = svd(x_np)
        np.testing.assert_allclose(x_np, u @ np.diag(s) @ vh, rtol=1e-5, atol=1e-5)
        
        # Test with PyTorch tensor
        x_torch = torch.rand(3, 3)
        u, s, vh = svd(x_torch)
        torch.testing.assert_close(x_torch, u @ torch.diag(s) @ vh, rtol=1e-5, atol=1e-5)

    def test_determinant(self):
        # Test with NumPy array
        x_np = np.array([[1, 2], [3, 4]])
        det_np = determinant(x_np)
        expected_det_np = np.linalg.det(x_np)
        self.assertTrue(np.isclose(det_np, expected_det_np))

        # Test with PyTorch tensor
        x_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        det_torch = determinant(x_torch)
        expected_det_torch = torch.det(x_torch).item()
        self.assertTrue(np.isclose(det_torch, expected_det_torch))

    def test_inverse(self):
        # Test with NumPy array
        x_np = np.array([[4, 7], [2, 6]])
        inv_np = inv(x_np)
        np.testing.assert_allclose(np.dot(x_np, inv_np), np.eye(2), rtol=1e-5, atol=1e-10)
        
        # Test with PyTorch tensor
        x_torch = torch.tensor([[4, 7], [2, 6]], dtype=torch.float32)
        inv_torch = inv(x_torch)
        torch.testing.assert_close(torch.mm(x_torch, inv_torch), torch.eye(2), rtol=1e-5, atol=1e-10)

    def test_matmul(self):
        # Test with NumPy array
        x_np = np.array([[1, 2], [3, 4]])
        y_np = np.array([[2], [1]])
        result_np = matmul(x_np, y_np)
        expected_result_np = np.dot(x_np, y_np)
        np.testing.assert_array_equal(result_np, expected_result_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        y_torch = torch.tensor([[2], [1]], dtype=torch.float32)
        result_torch = matmul(x_torch, y_torch)
        expected_result_torch = torch.mm(x_torch, y_torch)
        torch.testing.assert_close(result_torch, expected_result_torch)

    def test_exponential_map(self):
        # Test with NumPy array
        x_np = np.random.rand(3, 3)
        result_np = exponential_map(x_np)
        expected_result_np = expm(x_np)
        np.testing.assert_allclose(result_np, expected_result_np, rtol=1e-5,atol=1e-5)

        # Test with PyTorch tensor
        x_torch = torch.rand(3, 3)
        result_torch = exponential_map(x_torch)
        expected_result_torch = torch.matrix_exp(x_torch)
        torch.testing.assert_close(result_torch, expected_result_torch, rtol=1e-5,atol=1e-5)

    def test_norm(self):
        # Test with NumPy array
        x_np = np.array([3, 4])
        norm_np = norm(x_np)
        expected_norm_np = np.linalg.norm(x_np)
        self.assertAlmostEqual(norm_np, expected_norm_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([3.0, 4.0])
        norm_torch = norm(x_torch)
        expected_norm_torch = torch.norm(x_torch).item()
        self.assertAlmostEqual(norm_torch, expected_norm_torch)

        # Test with specified order and dimension for tensors
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        norm_torch = norm(x_torch, ord=1, dim=1)
        expected_norm_torch = torch.tensor([3.0, 7.0])
        torch.testing.assert_close(norm_torch, expected_norm_torch)

    def test_permute(self):
        # Test with NumPy array
        x_np = np.array([[1, 2, 3], [4, 5, 6]])
        permuted_np = permute(x_np, (1, 0))
        expected_permuted_np = x_np.transpose(1, 0)
        np.testing.assert_array_equal(permuted_np, expected_permuted_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([[1, 2, 3], [4, 5, 6]])
        permuted_torch = permute(x_torch, (1, 0))
        expected_permuted_torch = x_torch.permute(1, 0)
        torch.testing.assert_close(permuted_torch, expected_permuted_torch)

if __name__ == '__main__':
    unittest.main()
