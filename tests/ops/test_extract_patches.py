import unittest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import spatialkit
from spatialkit import uops
from spatialkit.common.exceptions import InvalidDimensionError, InvalidArgumentError


class TestExtractPatches(unittest.TestCase):
    """Test extract_patches function for both NumPy and PyTorch."""

    def test_2d_input_no_padding(self):
        """Test 2D input (H,W) with no padding."""
        x = np.arange(16).reshape(4, 4).astype(np.float32)
        patches = uops.extract_patches(x, patch_size=2, stride=1, padding=0)

        # Expected shape: (3, 3, 4) -> (H-ph+1, W-pw+1, ph*pw)
        self.assertEqual(patches.shape, (3, 3, 4))

        # Check first patch [0,1,4,5]
        expected_first = np.array([0, 1, 4, 5])
        np.testing.assert_array_equal(patches[0, 0], expected_first)

    def test_2d_input_same_padding(self):
        """Test 2D input with 'same' padding."""
        x = np.random.rand(10, 10).astype(np.float32)
        patches = uops.extract_patches(x, patch_size=3, stride=1, padding='same')

        # With 'same' padding, output spatial dims should match input
        self.assertEqual(patches.shape, (10, 10, 9))

    def test_3d_input_rgb(self):
        """Test 3D input (H,W,C) with RGB image."""
        rgb = np.random.rand(8, 8, 3).astype(np.float32)
        patches = uops.extract_patches(rgb, patch_size=3, stride=2, padding=0)

        # Expected: (3, 3, 27) -> ((8-3)//2+1, (8-3)//2+1, 3*3*3)
        self.assertEqual(patches.shape, (3, 3, 27))

    def test_non_square_patch(self):
        """Test non-square patch size."""
        x = np.random.rand(10, 12).astype(np.float32)
        patches = uops.extract_patches(x, patch_size=(3, 5), stride=1, padding=0)

        # Expected: (8, 8, 15) -> (10-3+1, 12-5+1, 3*5)
        self.assertEqual(patches.shape, (8, 8, 15))

    def test_non_uniform_stride(self):
        """Test non-uniform stride."""
        x = np.random.rand(20, 20).astype(np.float32)
        patches = uops.extract_patches(x, patch_size=3, stride=(2, 3), padding=0)

        # Expected: (9, 6, 9) -> ((20-3)//2+1, (20-3)//3+1, 3*3)
        self.assertEqual(patches.shape, (9, 6, 9))

    def test_int_padding(self):
        """Test integer padding."""
        x = np.random.rand(8, 8).astype(np.float32)
        patches = uops.extract_patches(x, patch_size=3, stride=1, padding=1)

        # With padding=1, padded size is (10, 10), output is (8, 8, 9)
        self.assertEqual(patches.shape, (8, 8, 9))

    def test_invalid_dimension(self):
        """Test that 1D or 4D input raises error."""
        x_1d = np.random.rand(10)
        x_4d = np.random.rand(2, 8, 8, 3)

        with self.assertRaises(InvalidDimensionError):
            uops.extract_patches(x_1d, patch_size=3)

        with self.assertRaises(InvalidDimensionError):
            uops.extract_patches(x_4d, patch_size=3)

    def test_invalid_patch_size(self):
        """Test that invalid patch_size raises error."""
        x = np.random.rand(10, 10).astype(np.float32)

        with self.assertRaises(InvalidArgumentError):
            uops.extract_patches(x, patch_size=0)

        with self.assertRaises(InvalidArgumentError):
            uops.extract_patches(x, patch_size=-1)

    def test_invalid_stride(self):
        """Test that invalid stride raises error."""
        x = np.random.rand(10, 10).astype(np.float32)

        with self.assertRaises(InvalidArgumentError):
            uops.extract_patches(x, patch_size=3, stride=0)

    def test_dtype_preservation(self):
        """Test that dtype is preserved."""
        for dtype in [np.float32, np.float64]:
            x = np.random.rand(10, 10).astype(dtype)
            patches = uops.extract_patches(x, patch_size=3, padding='same')
            self.assertEqual(patches.dtype, dtype)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_2d_input(self):
        """Test PyTorch tensor input (2D)."""
        x = torch.randn(10, 10)
        patches = uops.extract_patches(x, patch_size=3, stride=1, padding='same')

        self.assertTrue(torch.is_tensor(patches))
        self.assertEqual(patches.shape, (10, 10, 9))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_3d_input(self):
        """Test PyTorch tensor input (3D)."""
        x = torch.randn(8, 8, 3)
        patches = uops.extract_patches(x, patch_size=3, stride=2, padding=0)

        self.assertTrue(torch.is_tensor(patches))
        self.assertEqual(patches.shape, (3, 3, 27))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_dtype_preservation(self):
        """Test that PyTorch dtype is preserved."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(10, 10, dtype=dtype)
            patches = uops.extract_patches(x, patch_size=3, padding='same')
            self.assertEqual(patches.dtype, dtype)

    def test_memory_view(self):
        """Test that NumPy version uses view when possible (no padding)."""
        x = np.random.rand(10, 10).astype(np.float32)
        original_value = x[1, 1]
        patches = uops.extract_patches(x, patch_size=3, stride=1, padding=0)

        # With sliding_window_view, result is a view
        # Check that the value appears in patches
        self.assertEqual(patches[0, 0, 4], original_value)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_memory_view(self):
        """Test that PyTorch version uses view when possible (no padding)."""
        x = torch.randn(10, 10)
        original_value = x[1, 1].item()
        patches = uops.extract_patches(x, patch_size=3, stride=1, padding=0)

        # With unfold, result is a view
        # Check that the value appears in patches
        self.assertAlmostEqual(patches[0, 0, 4].item(), original_value, places=5)


class TestExtractPatchesEdgeCases(unittest.TestCase):
    """Test edge cases for extract_patches."""

    def test_patch_size_equals_input(self):
        """Test when patch_size equals input size."""
        x = np.random.rand(5, 5).astype(np.float32)
        patches = uops.extract_patches(x, patch_size=5, stride=1, padding=0)

        # Should have single patch
        self.assertEqual(patches.shape, (1, 1, 25))

    def test_large_stride(self):
        """Test stride larger than patch_size."""
        x = np.random.rand(20, 20).astype(np.float32)
        patches = uops.extract_patches(x, patch_size=3, stride=5, padding=0)

        # Expected: (4, 4, 9) -> ((20-3)//5+1, (20-3)//5+1, 9)
        self.assertEqual(patches.shape, (4, 4, 9))

    def test_single_channel_vs_multi_channel(self):
        """Test consistency between (H,W) and (H,W,1) inputs."""
        x_2d = np.random.rand(10, 10).astype(np.float32)
        x_3d = x_2d[:, :, np.newaxis]  # (10, 10, 1)

        patches_2d = uops.extract_patches(x_2d, patch_size=3, padding='same')
        patches_3d = uops.extract_patches(x_3d, patch_size=3, padding='same')

        # 2D should give (10, 10, 9), 3D should give (10, 10, 9)
        self.assertEqual(patches_2d.shape, (10, 10, 9))
        self.assertEqual(patches_3d.shape, (10, 10, 9))

        # Values should match (within 3D version, last dim is flattened)
        np.testing.assert_allclose(patches_2d, patches_3d, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
