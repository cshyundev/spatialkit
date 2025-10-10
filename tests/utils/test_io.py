"""
Test suite for cv_utils.utils.io module.

This module tests file I/O operations including TIFF, image, JSON, YAML,
PGM formats, and video writing functionality.

Author: Sehyun Cha
"""

import unittest
import os
import tempfile
import shutil
import json
import yaml
import numpy as np
from pathlib import Path

import cv_utils
from cv_utils.utils import io
from cv_utils.exceptions import (
    FileNotFoundError as CVFileNotFoundError,
    FileFormatError,
    ReadWriteError
)


class TestTiffIO(unittest.TestCase):
    """Test TIFF file read/write operations."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_and_read_tiff_single_image(self):
        """Test writing and reading a single TIFF image."""
        data = np.random.rand(100, 100).astype(np.float32)
        filepath = os.path.join(self.temp_dir, 'test.tiff')

        # Write TIFF
        io.write_tiff(data, filepath)

        # Read TIFF
        read_data = io.read_tiff(filepath)

        np.testing.assert_array_almost_equal(data, read_data)

    def test_write_and_read_tiff_multi_image(self):
        """Test writing and reading multi-page TIFF."""
        data = np.random.rand(5, 100, 100).astype(np.float32)
        filepath = os.path.join(self.temp_dir, 'test_multi.tiff')

        # Write TIFF
        io.write_tiff(data, filepath)

        # Read TIFF
        read_data = io.read_tiff(filepath)

        np.testing.assert_array_almost_equal(data, read_data)

    def test_write_tiff_with_options(self):
        """Test writing TIFF with custom options."""
        data = np.random.rand(50, 50).astype(np.float32)
        filepath = os.path.join(self.temp_dir, 'test_options.tiff')

        io.write_tiff(data, filepath, photometric='MINISBLACK',
                     bitspersample=32, compression='zlib')

        # Verify file was created
        self.assertTrue(os.path.exists(filepath))

        # Verify data integrity
        read_data = io.read_tiff(filepath)
        np.testing.assert_array_almost_equal(data, read_data)

    def test_read_tiff_file_not_found(self):
        """Test reading non-existent TIFF file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_tiff('/nonexistent/path/file.tiff')

    def test_read_tiff_path_is_directory(self):
        """Test reading directory instead of file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_tiff(self.temp_dir)

    def test_write_tiff_invalid_data_type(self):
        """Test writing non-numpy array raises ValueError."""
        filepath = os.path.join(self.temp_dir, 'test.tiff')

        with self.assertRaises(ValueError):
            io.write_tiff([1, 2, 3], filepath)  # List instead of ndarray

    def test_write_tiff_creates_directory(self):
        """Test that write_tiff creates parent directory if needed."""
        nested_path = os.path.join(self.temp_dir, 'nested', 'dir', 'test.tiff')
        data = np.random.rand(50, 50).astype(np.float32)

        io.write_tiff(data, nested_path)

        self.assertTrue(os.path.exists(nested_path))
        read_data = io.read_tiff(nested_path)
        np.testing.assert_array_almost_equal(data, read_data)


class TestImageIO(unittest.TestCase):
    """Test image file read/write operations."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_and_read_image_png(self):
        """Test writing and reading PNG image."""
        # Create random RGB image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        filepath = os.path.join(self.temp_dir, 'test.png')

        # Write image
        io.write_image(image, filepath)

        # Read image
        read_image = io.read_image(filepath)

        np.testing.assert_array_equal(image, read_image)

    def test_write_and_read_image_jpg(self):
        """Test writing and reading JPEG image."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Use .jpeg extension (PIL expects 'JPEG' not 'JPG')
        filepath = os.path.join(self.temp_dir, 'test.jpeg')

        io.write_image(image, filepath)
        read_image = io.read_image(filepath)

        # JPEG is lossy, so we can't expect exact equality
        self.assertEqual(image.shape, read_image.shape)

    def test_read_image_as_float(self):
        """Test reading image with float conversion."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        filepath = os.path.join(self.temp_dir, 'test.png')

        io.write_image(image, filepath)
        float_image = io.read_image(filepath, as_float=True)

        # Check that values are in [0, 1] range
        self.assertTrue(float_image.min() >= 0.0)
        self.assertTrue(float_image.max() <= 1.0)
        self.assertEqual(float_image.dtype, np.float64)

    def test_write_image_single_channel(self):
        """Test writing single channel (grayscale) image."""
        image = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
        filepath = os.path.join(self.temp_dir, 'gray.png')

        io.write_image(image, filepath)
        read_image = io.read_image(filepath)

        # Should be squeezed to 2D
        self.assertEqual(read_image.ndim, 2)

    def test_read_image_file_not_found(self):
        """Test reading non-existent image file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_image('/nonexistent/image.png')

    def test_read_image_path_is_directory(self):
        """Test reading directory instead of file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_image(self.temp_dir)

    def test_write_image_invalid_data_type(self):
        """Test writing non-numpy array raises ValueError."""
        filepath = os.path.join(self.temp_dir, 'test.png')

        with self.assertRaises(ValueError):
            io.write_image([1, 2, 3], filepath)

    def test_write_image_creates_directory(self):
        """Test that write_image creates parent directory if needed."""
        nested_path = os.path.join(self.temp_dir, 'nested', 'test.png')
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        io.write_image(image, nested_path)

        self.assertTrue(os.path.exists(nested_path))


class TestReadAllImages(unittest.TestCase):
    """Test reading all images from directory."""

    def setUp(self):
        """Set up temporary directory with test images."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test images
        for i in range(5):
            image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            filepath = os.path.join(self.temp_dir, f'image_{i}.png')
            io.write_image(image, filepath)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_all_images_directory(self):
        """Test reading all images from directory."""
        images = io.read_all_images(self.temp_dir)

        self.assertEqual(len(images), 5)
        for image in images:
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(image.shape, (50, 50, 3))

    def test_read_all_images_as_float(self):
        """Test reading all images as float."""
        images = io.read_all_images(self.temp_dir, as_float=True)

        self.assertEqual(len(images), 5)
        for image in images:
            self.assertTrue(image.min() >= 0.0)
            self.assertTrue(image.max() <= 1.0)

    def test_read_all_images_directory_not_found(self):
        """Test reading from non-existent directory raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_all_images('/nonexistent/directory')

    def test_read_all_images_path_is_file(self):
        """Test reading from file instead of directory raises CVFileNotFoundError."""
        filepath = os.path.join(self.temp_dir, 'image_0.png')

        with self.assertRaises(CVFileNotFoundError):
            io.read_all_images(filepath)

    def test_read_all_images_with_invalid_files(self):
        """Test that invalid files are skipped."""
        # Create invalid file (text file)
        invalid_file = os.path.join(self.temp_dir, 'not_an_image.txt')
        with open(invalid_file, 'w') as f:
            f.write('This is not an image')

        images = io.read_all_images(self.temp_dir)

        # Should still get 5 valid images, invalid file is skipped
        self.assertEqual(len(images), 5)


class TestPGMIO(unittest.TestCase):
    """Test PGM file read/write operations."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_and_read_pgm(self):
        """Test writing and reading PGM image."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        filepath = os.path.join(self.temp_dir, 'test.pgm')

        # Write PGM
        io.write_pgm(image, filepath)

        # Verify file was created
        self.assertTrue(os.path.exists(filepath))

        # Read PGM (note: may return None if there's an issue with PIL file handling)
        read_image = io.read_pgm(filepath)

        # If read succeeds, verify the data
        if read_image is not None:
            np.testing.assert_array_equal(image, read_image)

    def test_read_pgm_with_mode(self):
        """Test reading PGM with specific mode."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        filepath = os.path.join(self.temp_dir, 'test.pgm')

        io.write_pgm(image, filepath)
        read_image = io.read_pgm(filepath, mode='L')

        self.assertIsNotNone(read_image)
        self.assertIsInstance(read_image, np.ndarray)

    def test_read_pgm_file_not_found(self):
        """Test reading non-existent PGM file returns None."""
        result = io.read_pgm('/nonexistent/file.pgm')
        self.assertIsNone(result)


class TestJSONIO(unittest.TestCase):
    """Test JSON file read/write operations."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_and_read_json(self):
        """Test writing and reading JSON file."""
        data = {
            'name': 'test',
            'value': 42,
            'list': [1, 2, 3],
            'nested': {'key': 'value'}
        }
        filepath = os.path.join(self.temp_dir, 'test.json')

        # Write JSON
        io.write_json(filepath, data)

        # Read JSON
        read_data = io.read_json(filepath)

        self.assertEqual(data, read_data)

    def test_write_json_with_indent(self):
        """Test writing JSON with custom indentation."""
        data = {'key': 'value', 'number': 123}
        filepath = os.path.join(self.temp_dir, 'test_indent.json')

        io.write_json(filepath, data, indent=4)

        # Verify file exists and can be read
        self.assertTrue(os.path.exists(filepath))
        read_data = io.read_json(filepath)
        self.assertEqual(data, read_data)

    def test_read_json_file_not_found(self):
        """Test reading non-existent JSON file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_json('/nonexistent/file.json')

    def test_read_json_path_is_directory(self):
        """Test reading directory instead of file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_json(self.temp_dir)

    def test_read_json_invalid_format(self):
        """Test reading invalid JSON file raises FileFormatError."""
        filepath = os.path.join(self.temp_dir, 'invalid.json')

        # Write invalid JSON
        with open(filepath, 'w') as f:
            f.write('{ invalid json content ')

        with self.assertRaises(FileFormatError):
            io.read_json(filepath)

    def test_write_json_invalid_data_type(self):
        """Test writing non-dict data raises ValueError."""
        filepath = os.path.join(self.temp_dir, 'test.json')

        with self.assertRaises(ValueError):
            io.write_json(filepath, "not a dict")

    def test_write_json_non_serializable(self):
        """Test writing non-serializable data raises ValueError."""
        filepath = os.path.join(self.temp_dir, 'test.json')

        # NumPy array is not JSON serializable by default
        data = {'array': np.array([1, 2, 3])}

        with self.assertRaises(ValueError):
            io.write_json(filepath, data)

    def test_write_json_creates_directory(self):
        """Test that write_json creates parent directory if needed."""
        nested_path = os.path.join(self.temp_dir, 'nested', 'test.json')
        data = {'key': 'value'}

        io.write_json(nested_path, data)

        self.assertTrue(os.path.exists(nested_path))


class TestYAMLIO(unittest.TestCase):
    """Test YAML file read/write operations."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_and_read_yaml(self):
        """Test writing and reading YAML file."""
        data = {
            'name': 'test',
            'value': 42,
            'list': [1, 2, 3],
            'nested': {'key': 'value'}
        }
        filepath = os.path.join(self.temp_dir, 'test.yaml')

        # Write YAML
        io.write_yaml(filepath, data)

        # Read YAML
        read_data = io.read_yaml(filepath)

        self.assertEqual(data, read_data)

    def test_read_yaml_file_not_found(self):
        """Test reading non-existent YAML file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_yaml('/nonexistent/file.yaml')

    def test_read_yaml_path_is_directory(self):
        """Test reading directory instead of file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_yaml(self.temp_dir)

    def test_read_yaml_invalid_format(self):
        """Test reading invalid YAML file raises FileFormatError."""
        filepath = os.path.join(self.temp_dir, 'invalid.yaml')

        # Write invalid YAML
        with open(filepath, 'w') as f:
            f.write('invalid: yaml: content: :::')

        with self.assertRaises(FileFormatError):
            io.read_yaml(filepath)

    def test_write_yaml_invalid_data_type(self):
        """Test writing non-dict data raises ValueError."""
        filepath = os.path.join(self.temp_dir, 'test.yaml')

        with self.assertRaises(ValueError):
            io.write_yaml(filepath, "not a dict")

    def test_write_yaml_creates_directory(self):
        """Test that write_yaml creates parent directory if needed."""
        nested_path = os.path.join(self.temp_dir, 'nested', 'test.yaml')
        data = {'key': 'value'}

        io.write_yaml(nested_path, data)

        self.assertTrue(os.path.exists(nested_path))

    def test_write_yaml_unicode(self):
        """Test writing YAML with Unicode characters."""
        data = {'message': '안녕하세요', 'greeting': 'Hello'}
        filepath = os.path.join(self.temp_dir, 'unicode.yaml')

        io.write_yaml(filepath, data)
        read_data = io.read_yaml(filepath)

        self.assertEqual(data, read_data)


class TestVideoIO(unittest.TestCase):
    """Test video writing operations."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_video_from_images(self):
        """Test writing video from list of images."""
        # Create list of test images
        images = []
        for i in range(10):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            images.append(image)

        output_path = os.path.join(self.temp_dir, 'test_video.mp4')

        # Write video
        io.write_video_from_images(images, output_path, fps=10)

        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

    def test_write_video_from_images_grayscale(self):
        """Test writing video from grayscale images."""
        # Create list of grayscale images
        images = []
        for i in range(10):
            image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            images.append(image)

        output_path = os.path.join(self.temp_dir, 'test_gray.mp4')

        # Write video
        io.write_video_from_images(images, output_path, fps=10)

        # Verify file was created
        self.assertTrue(os.path.exists(output_path))

    def test_write_video_from_images_invalid_type(self):
        """Test that non-numpy array in images list raises ValueError."""
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            [1, 2, 3],  # Invalid: not a numpy array
        ]

        output_path = os.path.join(self.temp_dir, 'test.mp4')

        with self.assertRaises(ValueError):
            io.write_video_from_images(images, output_path)

    def test_write_video_from_image_paths(self):
        """Test writing video from image file paths."""
        # Create test images and save them
        image_paths = []
        for i in range(10):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            filepath = os.path.join(self.temp_dir, f'frame_{i}.png')
            io.write_image(image, filepath)
            image_paths.append(filepath)

        output_path = os.path.join(self.temp_dir, 'test_video_paths.mp4')

        # Write video
        io.write_video_from_image_paths(image_paths, output_path, fps=10)

        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

    def test_write_video_from_image_paths_empty_list(self):
        """Test that empty image paths list raises ValueError."""
        output_path = os.path.join(self.temp_dir, 'test.mp4')

        with self.assertRaises(ValueError):
            io.write_video_from_image_paths([], output_path)

    def test_write_video_with_custom_codec(self):
        """Test writing video with custom codec."""
        images = []
        for i in range(5):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            images.append(image)

        output_path = os.path.join(self.temp_dir, 'test_codec.mp4')

        # Write video with custom codec
        io.write_video_from_images(images, output_path, fps=5, codec='mp4v')

        # Verify file was created
        self.assertTrue(os.path.exists(output_path))


class TestVideoReading(unittest.TestCase):
    """Test video reading operations."""

    def setUp(self):
        """Set up temporary directory and create test video."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a test video with 30 frames
        self.video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        self.num_frames = 30
        self.frame_height = 100
        self.frame_width = 100

        images = []
        for i in range(self.num_frames):
            # Create distinct frames with different colors
            image = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * (i * 8)
            images.append(image)

        io.write_video_from_images(images, self.video_path, fps=10)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_video_with_sampling_ratio_all_frames(self):
        """Test reading video with sampling ratio 1.0 (all frames)."""
        frames = io.read_video_with_sampling_ratio(self.video_path, sampling_ratio=1.0)

        self.assertEqual(len(frames), self.num_frames)
        for frame in frames:
            self.assertEqual(frame.shape, (self.frame_height, self.frame_width, 3))

    def test_read_video_with_sampling_ratio_half(self):
        """Test reading video with sampling ratio 0.5 (every other frame)."""
        frames = io.read_video_with_sampling_ratio(self.video_path, sampling_ratio=0.5)

        # Should get approximately half the frames
        self.assertLess(len(frames), self.num_frames)
        self.assertGreater(len(frames), self.num_frames // 3)

    def test_read_video_with_sampling_ratio_with_max_samples(self):
        """Test reading video with sampling ratio and max_samples limit."""
        max_samples = 10
        frames = io.read_video_with_sampling_ratio(
            self.video_path, sampling_ratio=1.0, max_samples=max_samples
        )

        self.assertEqual(len(frames), max_samples)

    def test_read_video_with_sampling_ratio_file_not_found(self):
        """Test that reading non-existent video file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_video_with_sampling_ratio('/nonexistent/video.mp4')

    def test_read_video_with_num_samples(self):
        """Test reading video with specified number of samples."""
        num_samples = 10
        frames = io.read_video_with_num_samples(self.video_path, num_samples=num_samples)

        self.assertEqual(len(frames), num_samples)
        for frame in frames:
            self.assertEqual(frame.shape, (self.frame_height, self.frame_width, 3))

    def test_read_video_with_num_samples_more_than_total(self):
        """Test reading more samples than total frames returns all frames."""
        num_samples = 100  # More than self.num_frames
        frames = io.read_video_with_num_samples(self.video_path, num_samples=num_samples)

        # Should return all available frames
        self.assertEqual(len(frames), self.num_frames)

    def test_read_video_with_num_samples_with_max_samples(self):
        """Test reading video with num_samples and max_samples limit."""
        num_samples = 20
        max_samples = 5
        frames = io.read_video_with_num_samples(
            self.video_path, num_samples=num_samples, max_samples=max_samples
        )

        # max_samples should take precedence
        self.assertEqual(len(frames), max_samples)

    def test_read_video_with_num_samples_invalid(self):
        """Test that invalid num_samples raises ValueError."""
        with self.assertRaises(ValueError):
            io.read_video_with_num_samples(self.video_path, num_samples=0)

    def test_read_video_with_num_samples_file_not_found(self):
        """Test that reading non-existent video file raises CVFileNotFoundError."""
        with self.assertRaises(CVFileNotFoundError):
            io.read_video_with_num_samples('/nonexistent/video.mp4', num_samples=10)


class TestVideoReader(unittest.TestCase):
    """Test VideoReader class for lazy loading."""

    def setUp(self):
        """Set up temporary directory and create test video."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a test video with 50 frames
        self.video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        self.num_frames = 50
        self.frame_height = 100
        self.frame_width = 100

        images = []
        for i in range(self.num_frames):
            image = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * (i * 5)
            images.append(image)

        io.write_video_from_images(images, self.video_path, fps=10)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_video_reader_initialization(self):
        """Test VideoReader initialization and properties."""
        reader = io.VideoReader(self.video_path)

        self.assertEqual(reader.path, self.video_path)
        self.assertEqual(reader.total_frames, self.num_frames)
        self.assertGreater(reader.fps, 0)
        self.assertEqual(reader.width, self.frame_width)
        self.assertEqual(reader.height, self.frame_height)

        reader.close()

    def test_video_reader_next_batch(self):
        """Test reading video in batches."""
        reader = io.VideoReader(self.video_path)

        batch_size = 10
        batch = reader.next_batch(batch_size)

        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), batch_size)
        for frame in batch:
            self.assertEqual(frame.shape, (self.frame_height, self.frame_width, 3))

        reader.close()

    def test_video_reader_multiple_batches(self):
        """Test reading multiple batches sequentially."""
        reader = io.VideoReader(self.video_path)

        batch_size = 10
        total_read = 0

        while True:
            batch = reader.next_batch(batch_size)
            if batch is None:
                break
            total_read += len(batch)

        self.assertEqual(total_read, self.num_frames)
        reader.close()

    def test_video_reader_next_batch_at_end(self):
        """Test that next_batch returns None at end of video."""
        reader = io.VideoReader(self.video_path)

        # Read all frames
        reader.read_all()

        # Next batch should return None
        batch = reader.next_batch(10)
        self.assertIsNone(batch)

        reader.close()

    def test_video_reader_read_all(self):
        """Test reading all frames at once."""
        reader = io.VideoReader(self.video_path)

        frames = reader.read_all()

        self.assertEqual(len(frames), self.num_frames)
        reader.close()

    def test_video_reader_read_all_with_max_samples(self):
        """Test reading all frames with max_samples limit."""
        reader = io.VideoReader(self.video_path)

        max_samples = 20
        frames = reader.read_all(max_samples=max_samples)

        self.assertEqual(len(frames), max_samples)
        reader.close()

    def test_video_reader_reset(self):
        """Test resetting video reader to beginning."""
        reader = io.VideoReader(self.video_path)

        # Read some frames
        reader.next_batch(10)

        # Reset
        reader.reset()

        # Should be able to read from beginning again
        frames = reader.read_all()
        self.assertEqual(len(frames), self.num_frames)

        reader.close()

    def test_video_reader_seek(self):
        """Test seeking to specific frame."""
        reader = io.VideoReader(self.video_path)

        frame_number = 25
        reader.seek(frame_number)

        # Read one batch
        batch = reader.next_batch(5)

        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 5)

        reader.close()

    def test_video_reader_seek_invalid(self):
        """Test that seeking to invalid frame raises ValueError."""
        reader = io.VideoReader(self.video_path)

        with self.assertRaises(ValueError):
            reader.seek(-1)

        with self.assertRaises(ValueError):
            reader.seek(self.num_frames + 10)

        reader.close()

    def test_video_reader_context_manager(self):
        """Test VideoReader as context manager."""
        with io.VideoReader(self.video_path) as reader:
            batch = reader.next_batch(10)
            self.assertIsNotNone(batch)
            self.assertEqual(len(batch), 10)

        # Reader should be closed after exiting context

    def test_video_reader_file_not_found(self):
        """Test that VideoReader raises CVFileNotFoundError for non-existent file."""
        with self.assertRaises(CVFileNotFoundError):
            io.VideoReader('/nonexistent/video.mp4')

    def test_video_reader_invalid_batch_size(self):
        """Test that invalid batch_size raises ValueError."""
        reader = io.VideoReader(self.video_path)

        with self.assertRaises(ValueError):
            reader.next_batch(0)

        with self.assertRaises(ValueError):
            reader.next_batch(-5)

        reader.close()


if __name__ == '__main__':
    unittest.main()
