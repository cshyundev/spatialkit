import numpy as np
import torch
import unittest
from scipy.linalg import expm

# Use new hierarchical import pattern
import cv_utils
from cv_utils import umath

class TestHybridMath(unittest.TestCase):
    
    def test_solve_linear_system_with_solution(self):
        # Define a square, non-singular matrix and a solution vector
        A_square = np.array([[2, 1], [1, 3]])
        b = np.array([1, 2])
        torch_A_square = torch.tensor([[2, 1], [1, 3]], dtype=torch.float32)
        torch_b = torch.tensor([1, 2], dtype=torch.float32)

        # Test solving a system with a known solution
        np_result = umath.solve_linear_system(A_square, b)
        self.assertTrue(np.allclose(np_result, np.linalg.solve(A_square, b)))

        torch_result = umath.solve_linear_system(torch_A_square, torch_b)
        self.assertTrue(torch.allclose(torch_result, torch.linalg.solve(torch_A_square, torch_b)))

    def test_solve_linear_system_without_solution(self):
        # Define a non-square matrix
        A_non_square = np.array([[2, 1], [1, 3], [1, 1]])
        torch_A_non_square = torch.tensor(A_non_square, dtype=torch.float32)

        # Test finding the null space of a non-square matrix
        np_result = umath.solve_linear_system(A_non_square)
        self.assertTrue(isinstance(np_result, np.ndarray))
        # Verify that all vectors in the null space satisfy A*v = 0
        for col in np_result.T:
            self.assertTrue(np.allclose(A_non_square @ col, np.zeros(A_non_square.shape[0])))

        torch_result = umath.solve_linear_system(torch_A_non_square)
        self.assertTrue(isinstance(torch_result, torch.Tensor))
        for col in torch_result.t():
            self.assertTrue(torch.allclose(torch.matmul(torch_A_non_square, col), torch.zeros(A_non_square.shape[0])))

    def test_solve_linear_system_singular_matrix(self):
        # Define a singular matrix (more clearly singular)
        A_singular = np.array([[2.0, 4.0], [1.0, 2.0]], dtype=np.float64)
        torch_A_singular = torch.tensor(A_singular, dtype=torch.float64)

        # Test solving a system with a singular matrix (NumPy)
        np_result = umath.solve_linear_system(A_singular)
        self.assertTrue(isinstance(np_result, np.ndarray))
        # Verify that all vectors in the null space satisfy A*v ≈ 0
        if np_result.size > 0:
            for col in np_result.T:
                self.assertTrue(np.allclose(A_singular @ col, np.zeros(A_singular.shape[0]), atol=1e-10))

        # Test solving a system with a singular matrix (PyTorch)
        torch_result = umath.solve_linear_system(torch_A_singular)
        self.assertTrue(isinstance(torch_result, torch.Tensor))
        # Verify that all vectors in the null space satisfy A*v ≈ 0
        if torch_result.numel() > 0:
            for col in torch_result.t():
                self.assertTrue(torch.allclose(torch.matmul(torch_A_singular, col), torch.zeros(torch_A_singular.shape[0], dtype=torch_A_singular.dtype), atol=1e-10))

    def test_polyval(self):
        # Test 1: Simple polynomial evaluation
        coeffs = np.array([2, 0, -1])  # Corresponds to 2x^2 - 1
        x = np.array([0, 1, 2])
        expected = np.array([-1, 1, 7])  # 2*0^2 - 1 = -1, 2*1^2 - 1 = 1, 2*2^2 - 1 = 7
        result = umath.polyval(coeffs, x)
        np.testing.assert_array_equal(result, expected)

        # Test 2: Polynomial evaluation with different coefficients
        coeffs = np.array([1, -3, 2])  # Corresponds to x^2 - 3x + 2
        x = np.array([0, 1, 2, 3])
        expected = np.array([2, 0, 0, 2])  # x^2 - 3x + 2 evaluated at [0, 1, 2, 3]
        result = umath.polyval(coeffs, x)
        np.testing.assert_array_equal(result, expected)

    def test_polyfit(self):
        # Test 1: Simple linear fit
        x = np.array([0, 1, 2, 3])
        y = np.array([1, 3, 5, 7])  # Corresponds to y = 2x + 1
        degree = 1
        expected = np.array([2, 1])  # Coefficients [2, 1] for 2x + 1
        result = umath.polyfit(x, y, degree)
        np.testing.assert_almost_equal(result, expected, decimal=6)

        # Test 2: Quadratic fit
        x = np.array([-1, 0, 1, 2])
        y = np.array([1, 0, 1, 4])  # Corresponds to y = x^2
        degree = 2
        expected = np.array([1, 0, 0])  # Coefficients [1, 0, 0] for x^2
        result = umath.polyfit(x, y, degree)
        np.testing.assert_almost_equal(result, expected, decimal=6)
    
    def test_svd(self):
        # Test with NumPy array
        x_np = np.random.rand(3, 3)
        u, s, vh = umath.svd(x_np)
        np.testing.assert_allclose(x_np, u @ np.diag(s) @ vh, rtol=1e-5, atol=1e-5)
        
        # Test with PyTorch tensor
        x_torch = torch.rand(3, 3)
        u, s, vh = umath.svd(x_torch)
        torch.testing.assert_close(x_torch, u @ torch.diag(s) @ vh, rtol=1e-5, atol=1e-5)

    def test_determinant(self):
        # Test with NumPy array
        x_np = np.array([[1, 2], [3, 4]])
        det_np = umath.determinant(x_np)
        expected_det_np = np.linalg.det(x_np)
        self.assertTrue(np.isclose(det_np, expected_det_np))

        # Test with PyTorch tensor
        x_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        det_torch = umath.determinant(x_torch)
        expected_det_torch = torch.det(x_torch).item()
        self.assertTrue(np.isclose(det_torch, expected_det_torch))

    def test_inverse(self):
        # Test with NumPy array
        x_np = np.array([[4, 7], [2, 6]])
        inv_np = umath.inv(x_np)
        np.testing.assert_allclose(np.dot(x_np, inv_np), np.eye(2), rtol=1e-5, atol=1e-10)
        
        # Test with PyTorch tensor
        x_torch = torch.tensor([[4, 7], [2, 6]], dtype=torch.float32)
        inv_torch = umath.inv(x_torch)
        torch.testing.assert_close(torch.mm(x_torch, inv_torch), torch.eye(2), rtol=1e-5, atol=1e-10)

    def test_matmul(self):
        # Test with NumPy array
        x_np = np.array([[1, 2], [3, 4]])
        y_np = np.array([[2], [1]])
        result_np = umath.matmul(x_np, y_np)
        expected_result_np = np.dot(x_np, y_np)
        np.testing.assert_array_equal(result_np, expected_result_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        y_torch = torch.tensor([[2], [1]], dtype=torch.float32)
        result_torch = umath.matmul(x_torch, y_torch)
        expected_result_torch = torch.mm(x_torch, y_torch)
        torch.testing.assert_close(result_torch, expected_result_torch)

    def test_exponential_map(self):
        # Test with NumPy array
        x_np = np.random.rand(3, 3)
        result_np = umath.exponential_map(x_np)
        expected_result_np = expm(x_np)
        np.testing.assert_allclose(result_np, expected_result_np, rtol=1e-5,atol=1e-5)

        # Test with PyTorch tensor
        x_torch = torch.rand(3, 3)
        result_torch = umath.exponential_map(x_torch)
        expected_result_torch = torch.matrix_exp(x_torch)
        torch.testing.assert_close(result_torch, expected_result_torch, rtol=1e-5,atol=1e-5)

    def test_norm(self):
        # Test with NumPy array
        x_np = np.array([3, 4])
        norm_np = umath.norm(x_np)
        expected_norm_np = np.linalg.norm(x_np)
        self.assertAlmostEqual(norm_np, expected_norm_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([3.0, 4.0])
        norm_torch = umath.norm(x_torch)
        expected_norm_torch = torch.norm(x_torch).item()
        self.assertAlmostEqual(norm_torch, expected_norm_torch)

        # Test with specified order and dimension for tensors
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        norm_torch = umath.norm(x_torch, order=1, dim=1)
        expected_norm_torch = torch.tensor([3.0, 7.0])
        torch.testing.assert_close(norm_torch, expected_norm_torch)

    def test_permute(self):
        # Test with NumPy array
        x_np = np.array([[1, 2, 3], [4, 5, 6]])
        permuted_np = umath.permute(x_np, (1, 0))
        expected_permuted_np = x_np.transpose(1, 0)
        np.testing.assert_array_equal(permuted_np, expected_permuted_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([[1, 2, 3], [4, 5, 6]])
        permuted_torch = umath.permute(x_torch, (1, 0))
        expected_permuted_torch = x_torch.permute(1, 0)
        torch.testing.assert_close(permuted_torch, expected_permuted_torch)

    def test_abs(self):
        # Test with NumPy array
        x_np = np.array([-3, -4, 5])
        result_np = umath.abs(x_np)
        expected_np = np.abs(x_np)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([-3.0, -4.0, 5.0])
        result_torch = umath.abs(x_torch)
        expected_torch = torch.abs(x_torch)
        torch.testing.assert_close(result_torch, expected_torch)

    def test_sqrt(self):
        # Test with NumPy array
        x_np = np.array([4, 9, 16])
        result_np = umath.sqrt(x_np)
        expected_np = np.sqrt(x_np)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([4.0, 9.0, 16.0])
        result_torch = umath.sqrt(x_torch)
        expected_torch = torch.sqrt(x_torch)
        torch.testing.assert_close(result_torch, expected_torch)

    def test_mean(self):
        # Test with NumPy array
        x_np = np.array([[1, 2, 3], [4, 5, 6]])
        result_np = umath.mean(x_np)
        expected_np = np.mean(x_np)
        self.assertAlmostEqual(result_np, expected_np)

        # Test with dimension
        result_np_dim = umath.mean(x_np, dim=0)
        expected_np_dim = np.mean(x_np, axis=0)
        np.testing.assert_array_equal(result_np_dim, expected_np_dim)

        # Test with PyTorch tensor
        x_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result_torch = umath.mean(x_torch)
        expected_torch = torch.mean(x_torch)
        torch.testing.assert_close(result_torch, expected_torch)

        # Test error handling
        with self.assertRaises(cv_utils.InvalidDimensionError):
            umath.mean(x_np, dim=5)

    def test_dot(self):
        # Test with NumPy arrays
        x_np = np.array([1, 2, 3])
        y_np = np.array([4, 5, 6])
        result_np = umath.dot(x_np, y_np)
        expected_np = np.dot(x_np, y_np)
        self.assertEqual(result_np, expected_np)

        # Test with PyTorch tensors
        x_torch = torch.tensor([1.0, 2.0, 3.0])
        y_torch = torch.tensor([4.0, 5.0, 6.0])
        result_torch = umath.dot(x_torch, y_torch)
        expected_torch = torch.dot(x_torch, y_torch)
        torch.testing.assert_close(result_torch, expected_torch)

        # Test error handling for incompatible types
        with self.assertRaises(cv_utils.IncompatibleTypeError):
            umath.dot(x_np, y_torch)

    def test_qr_error_handling(self):
        # Test error handling for non-2D input
        x_1d = np.array([1, 2, 3])
        with self.assertRaises(cv_utils.InvalidDimensionError):
            umath.qr(x_1d)

    def test_trace(self):
        # Test with NumPy array
        x_np = np.array([[1, 2], [3, 4]])
        result_np = umath.trace(x_np)
        expected_np = np.trace(x_np)
        self.assertEqual(result_np, expected_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result_torch = umath.trace(x_torch)
        expected_torch = torch.trace(x_torch)
        torch.testing.assert_close(result_torch, expected_torch)

        # Test error handling
        with self.assertRaises(cv_utils.InvalidDimensionError):
            umath.trace(np.array([1, 2, 3]))

    def test_diag(self):
        # Test with NumPy array
        x_np = np.array([[1, 2], [3, 4]])
        result_np = umath.diag(x_np)
        expected_np = np.diag(x_np)
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result_torch = umath.diag(x_torch)
        expected_torch = torch.diag(x_torch)
        torch.testing.assert_close(result_torch, expected_torch)

        # Test error handling
        with self.assertRaises(cv_utils.InvalidDimensionError):
            umath.diag(np.array([1, 2, 3]))

    def test_rad2deg_deg2rad(self):
        # Test rad2deg
        x_rad = np.array([0, np.pi/2, np.pi])
        result_deg = umath.rad2deg(x_rad)
        expected_deg = np.array([0, 90, 180])
        np.testing.assert_allclose(result_deg, expected_deg)

        # Test deg2rad
        x_deg = np.array([0, 90, 180])
        result_rad = umath.deg2rad(x_deg)
        expected_rad = np.array([0, np.pi/2, np.pi])
        np.testing.assert_allclose(result_rad, expected_rad)

    def test_trig_functions(self):
        # Test sin
        x_torch = torch.tensor([0.0, np.pi/2])
        result_sin = umath.sin(x_torch)
        expected_sin = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(result_sin, expected_sin, atol=1e-6, rtol=1e-6)

        # Test cos
        result_cos = umath.cos(x_torch)
        expected_cos = torch.tensor([1.0, 0.0])
        torch.testing.assert_close(result_cos, expected_cos, atol=1e-6, rtol=1e-6)

        # Test tan
        x_tan = torch.tensor([0.0, np.pi/4])
        result_tan = umath.tan(x_tan)
        expected_tan = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(result_tan, expected_tan, atol=1e-6, rtol=1e-6)

        # Test arcsin
        x_arcsin = torch.tensor([0.0, 1.0])
        result_arcsin = umath.arcsin(x_arcsin)
        expected_arcsin = torch.tensor([0.0, np.pi/2])
        torch.testing.assert_close(result_arcsin, expected_arcsin, atol=1e-6, rtol=1e-6)

        # Test arccos
        result_arccos = umath.arccos(x_arcsin)
        expected_arccos = torch.tensor([np.pi/2, 0.0])
        torch.testing.assert_close(result_arccos, expected_arccos, atol=1e-6, rtol=1e-6)

        # Test arctan
        x_arctan = torch.tensor([0.0, 1.0])
        result_arctan = umath.arctan(x_arctan)
        expected_arctan = torch.tensor([0.0, np.pi/4])
        torch.testing.assert_close(result_arctan, expected_arctan, atol=1e-6, rtol=1e-6)

    def test_arctan2(self):
        # Test with PyTorch tensors
        x_torch = torch.tensor([1.0, 0.0, -1.0])
        y_torch = torch.tensor([0.0, 1.0, 0.0])
        result_torch = umath.arctan2(x_torch, y_torch)
        expected_torch = torch.arctan2(x_torch, y_torch)
        torch.testing.assert_close(result_torch, expected_torch)

        # Test with NumPy arrays
        x_np = np.array([1.0, 0.0, -1.0])
        y_np = np.array([0.0, 1.0, 0.0])
        result_np = umath.arctan2(x_np, y_np)
        expected_np = np.arctan2(x_np, y_np)
        np.testing.assert_allclose(result_np, expected_np)

    def test_normalize(self):
        # Test with NumPy array
        x_np = np.array([3, 4])
        result_np = umath.normalize(x_np)
        expected_np = x_np / np.linalg.norm(x_np)
        np.testing.assert_allclose(result_np, expected_np)

    def test_is_square(self):
        # Test square matrix
        square_matrix = np.array([[1, 2], [3, 4]])
        self.assertTrue(umath.is_square(square_matrix))

        # Test non-square matrix
        non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(umath.is_square(non_square_matrix))

    def test_vec3_to_skew(self):
        # Test with NumPy array (3,)
        x_np = np.array([1, 2, 3])
        result_np = umath.vec3_to_skew(x_np)
        expected_np = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        np.testing.assert_array_equal(result_np, expected_np)

        # Test with PyTorch tensor (1, 3)
        x_torch = torch.tensor([[1.0, 2.0, 3.0]])
        result_torch = umath.vec3_to_skew(x_torch)
        self.assertTrue(torch.is_tensor(result_torch))

        # Test error handling
        with self.assertRaises(cv_utils.InvalidShapeError):
            umath.vec3_to_skew(np.array([1, 2]))

    def test_homo_dehomo(self):
        # Test homo with 2D points
        pts_2d = np.array([[1, 2, 3], [4, 5, 6]])
        homo_pts = umath.homo(pts_2d)
        expected_homo = np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
        np.testing.assert_array_equal(homo_pts, expected_homo)

        # Test homo error handling
        with self.assertRaises(cv_utils.InvalidCoordinateError):
            umath.homo(np.array([1, 2, 3]))

        # Test dehomo with 3D homogeneous points
        homo_pts_3d = np.array([[2, 4, 6], [3, 6, 9], [2, 2, 2]])
        euc_pts = umath.dehomo(homo_pts_3d)
        expected_euc = np.array([[1, 2, 3], [1.5, 3, 4.5]])
        np.testing.assert_allclose(euc_pts, expected_euc)

        # Test dehomo error handling
        with self.assertRaises(cv_utils.InvalidCoordinateError):
            umath.dehomo(np.array([[1, 2], [3, 4]]))

    def test_polyfit_tensor(self):
        # Test polyfit with PyTorch tensor
        x_torch = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y_torch = torch.tensor([1.0, 3.0, 5.0, 7.0])
        result_torch = umath.polyfit(x_torch, y_torch, 1)
        self.assertTrue(torch.is_tensor(result_torch))

if __name__ == '__main__':
    unittest.main()
