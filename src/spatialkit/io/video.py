"""
Video I/O operations with support for sampling and lazy loading.

This module provides functions for reading and writing video files with
various sampling strategies and a lazy-loading VideoReader class.

Functions:
    - write_video_from_image_paths: Create video from image file paths
    - write_video_from_images: Create video from image arrays
    - read_video_with_sampling_ratio: Read video with frame sampling
    - read_video_with_num_samples: Read evenly distributed frames

Classes:
    - VideoReader: Lazy-loading video reader for batch processing

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT License
"""

import os.path as osp
from typing import Optional, List

import numpy as np
import cv2 as cv

from ..common.exceptions import (
    FileNotFoundError as CVFileNotFoundError,
    ReadWriteError
)

# Import read_image from image module (for video_from_image_paths)
from .image import read_image
def write_video_from_image_paths(
    image_paths: List[str], output_path: str, fps: int = 30, codec: str = "mp4v"
) -> None:
    """
    Write a video from a list of image file paths.

    Args:
        image_paths (List[str]): List of image file paths.
        output_path (str): Output path for the video file.
        fps (int): Frames per second for the video. Default is 30.
        codec (str): FourCC code for the video codec. Default is 'mp4v'.
    """
    if not image_paths:
        raise ValueError("The image paths list is empty")

    first_image = read_image(image_paths[0])
    if first_image is None:
        return

    height, width = first_image.shape[0:2]

    # Initialize the video writer
    fourcc = cv.VideoWriter_fourcc(*codec)
    video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_path in image_paths:
        image = read_image(image_path)
        if image is None:
            return
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        video_writer.write(image)
    video_writer.release()


def write_video_from_images(
    images: List[np.ndarray], output_path: str, fps: int = 30, codec: str = "mp4v"
) -> None:
    """
    Write a video from a list of images.

    Args:
        images (List[np.ndarray]): List of images.
        output_path (str): Output path for the video file.
        fps (int): Frames per second for the video. Default is 30.
        codec (str): FourCC code for the video codec. Default is 'mp4v'.
    """

    # Get the size from the first image
    first_image = images[0]
    height, width = first_image.shape[0:2]

    # Initialize the video writer
    fourcc = cv.VideoWriter_fourcc(*codec)
    video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"All items in the images list must be numpy arrays, got {type(image)}. "
                f"Please ensure all images are valid numpy arrays."
            )
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        video_writer.write(image)
    video_writer.release()


def read_video_with_sampling_ratio(
    path: str, sampling_ratio: float = 1.0, max_samples: Optional[int] = None
) -> List[np.ndarray]:
    """
    Read video frames with a specified sampling ratio.

    Args:
        path (str): Path to the video file.
        sampling_ratio (float): Ratio for sampling frames (e.g., 0.5 reads every other frame,
                                2.0 reads every second frame). Default is 1.0 (all frames).
        max_samples (Optional[int]): Maximum number of frames to read. If None, reads all sampled frames.

    Returns:
        List[np.ndarray]: List of sampled frames as RGB images.

    Raises:
        CVFileNotFoundError: If video file is not found.
        ReadWriteError: If video reading fails.

    Example:
        frames = read_video_with_sampling_ratio('video.mp4', sampling_ratio=0.5, max_samples=100)
    """
    if not osp.exists(path):
        raise CVFileNotFoundError(
            f"Video file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )

    if not osp.isfile(path):
        raise CVFileNotFoundError(
            f"Path is not a file: {path}. "
            f"Please provide a valid video file path."
        )

    try:
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            raise ReadWriteError(
                f"Failed to open video file: {path}. "
                f"Please ensure the file is a valid video format."
            )

        frames = []
        frame_idx = 0
        next_frame_to_read = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if current frame should be sampled
            if frame_idx >= int(next_frame_to_read):
                # Convert BGR to RGB
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                # Update next frame to read
                next_frame_to_read += 1.0 / sampling_ratio

                # Check max_samples limit
                if max_samples is not None and len(frames) >= max_samples:
                    break

            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            raise ReadWriteError(
                f"No frames could be read from video: {path}. "
                f"Please check if the video file is valid."
            )

        return frames

    except (CVFileNotFoundError, ReadWriteError):
        raise
    except Exception as e:
        raise ReadWriteError(
            f"Failed to read video file {path}: {e}. "
            f"Please check file format and permissions."
        ) from e


def read_video_with_num_samples(
    path: str, num_samples: int, max_samples: Optional[int] = None
) -> List[np.ndarray]:
    """
    Read a specified number of evenly distributed frames from a video.

    Args:
        path (str): Path to the video file.
        num_samples (int): Number of frames to sample evenly from the video.
        max_samples (Optional[int]): Maximum number of frames to read. If specified and less than
                                     num_samples, max_samples takes precedence.

    Returns:
        List[np.ndarray]: List of sampled frames as RGB images.

    Raises:
        CVFileNotFoundError: If video file is not found.
        ValueError: If num_samples is less than 1.
        ReadWriteError: If video reading fails.

    Example:
        frames = read_video_with_num_samples('video.mp4', num_samples=50, max_samples=100)
    """
    if not osp.exists(path):
        raise CVFileNotFoundError(
            f"Video file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )

    if not osp.isfile(path):
        raise CVFileNotFoundError(
            f"Path is not a file: {path}. "
            f"Please provide a valid video file path."
        )

    if num_samples < 1:
        raise ValueError(
            f"num_samples must be at least 1, got {num_samples}. "
            f"Please provide a valid number of samples."
        )

    # Apply max_samples limit
    effective_num_samples = num_samples
    if max_samples is not None:
        effective_num_samples = min(num_samples, max_samples)

    try:
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            raise ReadWriteError(
                f"Failed to open video file: {path}. "
                f"Please ensure the file is a valid video format."
            )

        # Get total number of frames
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            raise ReadWriteError(
                f"Video contains no frames: {path}. "
                f"Please check if the video file is valid."
            )

        # Calculate frame indices to sample
        if effective_num_samples >= total_frames:
            # If requesting more samples than available, return all frames
            frame_indices = list(range(total_frames))
        else:
            # Evenly distribute samples
            step = total_frames / effective_num_samples
            frame_indices = [int(i * step) for i in range(effective_num_samples)]

        frames = []
        for idx in frame_indices:
            cap.set(cv.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()

        if len(frames) == 0:
            raise ReadWriteError(
                f"No frames could be read from video: {path}. "
                f"Please check if the video file is valid."
            )

        return frames

    except (CVFileNotFoundError, ValueError, ReadWriteError):
        raise
    except Exception as e:
        raise ReadWriteError(
            f"Failed to read video file {path}: {e}. "
            f"Please check file format and permissions."
        ) from e


class VideoReader:
    """
    Lazy loading video reader for efficiently reading video frames in batches.

    This class allows reading video frames on-demand without loading the entire
    video into memory at once.

    Attributes:
        path (str): Path to the video file.
        total_frames (int): Total number of frames in the video.
        fps (float): Frames per second of the video.
        width (int): Width of video frames.
        height (int): Height of video frames.

    Example:
        reader = VideoReader('video.mp4')
        while True:
            batch = reader.next_batch(batch_size=10)
            if batch is None:
                break
            # Process batch
        reader.close()
    """

    def __init__(self, path: str):
        """
        Initialize VideoReader.

        Args:
            path (str): Path to the video file.

        Raises:
            CVFileNotFoundError: If video file is not found.
            ReadWriteError: If video cannot be opened.
        """
        if not osp.exists(path):
            raise CVFileNotFoundError(
                f"Video file not found: {path}. "
                f"Please check the file path and ensure the file exists."
            )

        if not osp.isfile(path):
            raise CVFileNotFoundError(
                f"Path is not a file: {path}. "
                f"Please provide a valid video file path."
            )

        self.path = path
        self._cap = cv.VideoCapture(path)

        if not self._cap.isOpened():
            raise ReadWriteError(
                f"Failed to open video file: {path}. "
                f"Please ensure the file is a valid video format."
            )

        # Get video properties
        self.total_frames = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = self._cap.get(cv.CAP_PROP_FPS)
        self.width = int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self._current_frame = 0

    def next_batch(self, batch_size: int) -> Optional[List[np.ndarray]]:
        """
        Read the next batch of frames.

        Args:
            batch_size (int): Number of frames to read in this batch.

        Returns:
            Optional[List[np.ndarray]]: List of frames as RGB images, or None if end of video.

        Raises:
            ValueError: If batch_size is less than 1.
        """
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be at least 1, got {batch_size}. "
                f"Please provide a valid batch size."
            )

        if self._current_frame >= self.total_frames:
            return None

        frames = []
        for _ in range(batch_size):
            ret, frame = self._cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            self._current_frame += 1

        return frames if len(frames) > 0 else None

    def read_all(self, max_samples: Optional[int] = None) -> List[np.ndarray]:
        """
        Read all remaining frames from current position.

        Args:
            max_samples (Optional[int]): Maximum number of frames to read. If None, reads all.

        Returns:
            List[np.ndarray]: List of frames as RGB images.
        """
        frames = []
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            self._current_frame += 1

            if max_samples is not None and len(frames) >= max_samples:
                break

        return frames

    def reset(self):
        """Reset the video reader to the beginning."""
        self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        self._current_frame = 0

    def seek(self, frame_number: int):
        """
        Seek to a specific frame number.

        Args:
            frame_number (int): Frame number to seek to (0-indexed).

        Raises:
            ValueError: If frame_number is out of range.
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            raise ValueError(
                f"frame_number must be between 0 and {self.total_frames - 1}, got {frame_number}. "
                f"Please provide a valid frame number."
            )

        self._cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        self._current_frame = frame_number

    def close(self):
        """Release video capture resources."""
        if hasattr(self, '_cap') and self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()
