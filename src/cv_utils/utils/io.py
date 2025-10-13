"""
Module Name: io.py

Description:
This module provides utility functions for reading and writing various file formats, including TIFF, image files, JSON, YAML, and video files.
It is designed to facilitate data handling in image processing and computer vision tasks.

Main Functions:
- read_tiff: Reads a TIFF file and returns the data as a NumPy array.
- write_tiff: Writes data to a TIFF file with specified options.
- read_all_images: Reads all images from a directory and returns them as a list of NumPy arrays.
- read_image: Reads an image file and optionally converts it to a float representation.
- write_image: Writes image data to a file.
- read_pgm: Reads a PGM image file and returns the data as a NumPy array.
- write_pgm: Writes NumPy array image data to a PGM file.
- read_json: Reads a JSON file and returns the data as a dictionary.
- write_json: Writes a dictionary to a JSON file with specified indentation.
- read_yaml: Reads a YAML file and returns the data as a dictionary.
- write_yaml: Writes a dictionary to a YAML file.
- write_video_from_image_paths: Writes a video from a list of image file paths.
- write_video_from_images: Writes a video from a list of images.
- read_video_with_sampling_ratio: Reads video frames with a specified sampling ratio.
- read_video_with_num_samples: Reads a specified number of evenly distributed frames from a video.
- VideoReader: Lazy loading video reader class for reading frames in batches.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1

License: MIT License

"""

import os
import os.path as osp

import json

from typing import *
import tifffile
import numpy as np
import skimage.io as skio
import skimage
import yaml
from PIL import Image
import cv2 as cv

from ..common.logger import LOG_ERROR
from ..exceptions import (
    FileNotFoundError as CVFileNotFoundError,
    FileFormatError,
    ReadWriteError
)


def read_tiff(path: str) -> np.ndarray:
    """
    Reads a TIFF file.

    Args:
        path (str): Path to the TIFF file.

    Returns:
        np.ndarray: Data read from the TIFF file.
        
    Raises:
        CVFileNotFoundError: If TIFF file is not found.
        FileFormatError: If TIFF file format is invalid or empty.
        ReadWriteError: If reading TIFF file fails.

    Example:
        data = read_tiff('example.tiff')
    """
    if not osp.exists(path):
        raise CVFileNotFoundError(
            f"TIFF file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )
    
    if not osp.isfile(path):
        raise CVFileNotFoundError(
            f"Path is not a file: {path}. "
            f"Please provide a valid TIFF file path."
        )
    
    try:
        multi_datas = tifffile.TiffFile(path)
        num_datas = len(multi_datas.pages)
        
        if num_datas == 0:
            raise FileFormatError(
                f"TIFF file contains no images: {path}. "
                f"Please provide a valid TIFF file with image data."
            )
        
        if num_datas == 1:
            data = multi_datas.pages[0].asarray().squeeze()
        else:
            data = np.concatenate(
                [np.expand_dims(x.asarray(), 0) for x in multi_datas.pages], 0
            )
        return data
    except FileFormatError:
        raise  # Re-raise our own exceptions
    except Exception as e:
        raise ReadWriteError(
            f"Failed to read TIFF file {path}: {e}. "
            f"Please check file format and permissions."
        ) from e


def write_tiff(
    data: np.ndarray,
    path: str,
    photometric: str = "MINISBLACK",
    bitspersample: int = 32,
    compression: str = "zlib",
):
    """
    Writes data to a TIFF file.

    Args:
        data (np.ndarray): The main data to write.
        path (str): Path to save the TIFF file.
        photometric (str): Photometric interpretation of the data.
        bitspersample (int): Number of bits per sample. Default is 32.
        compression (str): Compression type to use. Default is 'zlib'.
        
    Raises:
        ValueError: If data is not a valid numpy array.
        ReadWriteError: If writing TIFF file fails.

    Example:
        data = np.random.rand(100, 100).astype(np.float32)
        write_tiff(data, 'output_with_thumbnail.tiff')
    """
    if not isinstance(data, np.ndarray):
        raise ValueError(
            f"Data must be a numpy array, got {type(data)}. "
            f"Please provide valid image data as numpy array."
        )
    
    # Create directory if it doesn't exist
    dir_path = osp.dirname(path)
    if dir_path and not osp.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            raise ReadWriteError(
                f"Failed to create directory {dir_path}: {e}. "
                f"Please check directory permissions."
            ) from e
    
    try:
        with tifffile.TiffWriter(path) as tiff:
            options = dict(
                photometric=photometric,
                bitspersample=bitspersample,
                compression=compression,
            )
            tiff.write(data, subifds=0, **options)
    except Exception as e:
        raise ReadWriteError(
            f"Failed to write TIFF file {path}: {e}. "
            f"Please check file path, permissions, and data format."
        ) from e


def read_all_images(
    image_dir: str, as_float: Optional[bool] = False
) -> List[np.ndarray]:
    """
    Reads all images in a directory.

    Args:
        image_dir (str): Path to the directory containing images.
        as_float (Optional[bool]): Flag to convert images to float representation.

    Returns:
        List[np.ndarray]: List of images read from the directory.
        
    Raises:
        CVFileNotFoundError: If image directory is not found.
        ReadWriteError: If reading images fails.

    Example:
        images = read_all_images('path/to/images', as_float=True)
    """
    if not osp.exists(image_dir):
        raise CVFileNotFoundError(
            f"Image directory not found: {image_dir}. "
            f"Please check the directory path and ensure it exists."
        )
    
    if not osp.isdir(image_dir):
        raise CVFileNotFoundError(
            f"Path is not a directory: {image_dir}. "
            f"Please provide a valid directory path."
        )
    
    try:
        image_list = os.listdir(image_dir)
        images = []
        failed_files = []
        
        for image_name in image_list:
            image_path = osp.join(image_dir, image_name)
            try:
                image = read_image(image_path, as_float)
                if image is not None:
                    images.append(image)
            except Exception:
                failed_files.append(image_name)
                continue  # Skip invalid files
        
        if failed_files:
            LOG_ERROR(f"Failed to read {len(failed_files)} files: {failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")
        
        return images
    except Exception as e:
        raise ReadWriteError(
            f"Failed to read images from directory {image_dir}: {e}. "
            f"Please check directory permissions and file formats."
        ) from e


def read_image(path: str, as_float: Optional[bool] = False) -> np.ndarray:
    """
    Reads an image file.

    Args:
        path (str): Path to the image file.
        as_float (Optional[bool]): Flag to convert the image to float representation.

    Returns:
        np.ndarray: Image data read from the file.
        
    Raises:
        CVFileNotFoundError: If image file is not found.
        FileFormatError: If image file format is invalid.
        ReadWriteError: If reading image file fails.

    Example:
        image = read_image('example.png', as_float=True)
    """
    if not osp.exists(path):
        raise CVFileNotFoundError(
            f"Image file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )
    
    if not osp.isfile(path):
        raise CVFileNotFoundError(
            f"Path is not a file: {path}. "
            f"Please provide a valid image file path."
        )
    
    try:
        image = skio.imread(path)
        if as_float:
            image = skimage.img_as_float(image)  # normalize [0,255] -> [0.,1.]
        return image
    except Exception as e:
        raise ReadWriteError(
            f"Failed to read image file {path}: {e}. "
            f"Please ensure the file is a valid image format."
        ) from e


def write_image(image: np.ndarray, path: str):
    """
    Writes image data to a file.

    Args:
        image (np.ndarray): Image data to write.
        path (str): Path to save the image file.
        
    Raises:
        ValueError: If image data is not a valid numpy array.
        ReadWriteError: If writing image file fails.

    Example:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        write_image(image, 'output.png')
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"Image must be a numpy array, got {type(image)}. "
            f"Please provide valid image data as numpy array."
        )
    
    # Create directory if it doesn't exist
    dir_path = osp.dirname(path)
    if dir_path and not osp.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            raise ReadWriteError(
                f"Failed to create directory {dir_path}: {e}. "
                f"Please check directory permissions."
            ) from e
    
    try:
        # Handle single channel images
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image = np.squeeze(image)
        
        pil_image = Image.fromarray(image)
        extension = path.split(sep=".")[-1].upper()
        pil_image.save(path, extension)
    except Exception as e:
        raise ReadWriteError(
            f"Failed to write image file {path}: {e}. "
            f"Please check file path, permissions, and image format."
        ) from e


def read_pgm(path: str, mode: Optional[str] = None) -> np.ndarray:
    """
    Reads a PGM image file.

    Args:
        path (str): Path to the PGM image file.
        mode (Optional[str]): Mode to convert the image using PIL.

    Return:
        np.ndarray: Image data read from the file.

    Details:
    - 'L' - (8-bit pixels, black and white)
    - 'RGB' - (3x8-bit pixels, true color)
    - 'RGBA' - (4x8-bit pixels, true color with transparency mask)
    - 'CMYK' - (4x8-bit pixels, color separation)
    - 'YCbCr' - (3x8-bit pixels, color video format)
    - 'I' - (32-bit signed integer pixels)
    - 'F' - (32-bit floating point pixels)

    Example:
        image = read_pgm('example.pgm', mode='L')
    """

    try:
        with open(path, "rb") as f:
            image = Image.open(f)
            if mode is not None:
                image = image.convert(mode)
        return np.array(image)

    except (FileNotFoundError, IsADirectoryError) as e:
        LOG_ERROR(f"File not found or is a directory: {path}. Error: {e}")
    except ValueError as e:
        LOG_ERROR(f"Value error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to read Image file: {path}. Error: {e}.")
    return None


def write_pgm(image: np.ndarray, path: str):
    """
    Writes numpy array image data to a PGM file.

    Args:
        image (np.ndarray): Image data to write.
        path (str): Path to save the PGM image file.

    Example:
        image = np.random.random((100, 100)).astype(np.uint8)
        write_pgm(image, 'output.pgm')
    """
    img = Image.fromarray(image)
    img.save(path)


def read_json(path: str) -> Dict[str, Any]:
    """
    Reads a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Data read from the JSON file.
        
    Raises:
        CVFileNotFoundError: If JSON file is not found.
        FileFormatError: If JSON file format is invalid.
        ReadWriteError: If reading JSON file fails.

    Example:
        data = read_json('example.json')
    """
    if not osp.exists(path):
        raise CVFileNotFoundError(
            f"JSON file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )
    
    if not osp.isfile(path):
        raise CVFileNotFoundError(
            f"Path is not a file: {path}. "
            f"Please provide a valid JSON file path."
        )
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        return json_dict
    except json.JSONDecodeError as e:
        raise FileFormatError(
            f"Invalid JSON format in file {path}: {e}. "
            f"Please check JSON syntax and format."
        ) from e
    except Exception as e:
        raise ReadWriteError(
            f"Failed to read JSON file {path}: {e}. "
            f"Please check file permissions and format."
        ) from e


def write_json(path: str, json_dict: Dict[str, Any], indent: int = 1):
    """
    Writes a dictionary to a JSON file.

    Args:
        path (str): Path to save the JSON file.
        json_dict (Dict[str, Any]): Dictionary to write to the JSON file.
        indent (int): Indentation level for pretty-printing. Default is 1.
        
    Raises:
        ValueError: If json_dict is not serializable.
        ReadWriteError: If writing JSON file fails.

    Example:
        data = {'key': 'value'}
        write_json('output.json', data, indent=2)
    """
    if not isinstance(json_dict, dict):
        raise ValueError(
            f"json_dict must be a dictionary, got {type(json_dict)}. "
            f"Please provide a valid dictionary to write as JSON."
        )
    
    # Create directory if it doesn't exist
    dir_path = osp.dirname(path)
    if dir_path and not osp.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            raise ReadWriteError(
                f"Failed to create directory {dir_path}: {e}. "
                f"Please check directory permissions."
            ) from e
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=indent)
    except TypeError as e:
        raise ValueError(
            f"Failed to serialize data to JSON: {e}. "
            f"Please ensure all data types are JSON serializable."
        ) from e
    except Exception as e:
        raise ReadWriteError(
            f"Failed to write JSON file {path}: {e}. "
            f"Please check file path and permissions."
        ) from e


def read_yaml(path: str) -> Dict[str, Any]:
    """
    Reads a YAML file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Data read from the YAML file.
        
    Raises:
        CVFileNotFoundError: If YAML file is not found.
        FileFormatError: If YAML file format is invalid.
        ReadWriteError: If reading YAML file fails.

    Example:
        data = read_yaml('example.yaml')
    """
    if not osp.exists(path):
        raise CVFileNotFoundError(
            f"YAML file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )
    
    if not osp.isfile(path):
        raise CVFileNotFoundError(
            f"Path is not a file: {path}. "
            f"Please provide a valid YAML file path."
        )
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)
        return data_dict
    except yaml.YAMLError as e:
        raise FileFormatError(
            f"Invalid YAML format in file {path}: {e}. "
            f"Please check YAML syntax and format."
        ) from e
    except Exception as e:
        raise ReadWriteError(
            f"Failed to read YAML file {path}: {e}. "
            f"Please check file permissions and format."
        ) from e


def write_yaml(path: str, yaml_dict: Dict[str, Any]):
    """
    Writes a dictionary to a YAML file.

    Args:
        path (str): Path to save the YAML file.
        yaml_dict (Dict[str, Any]): Dictionary to write to the YAML file.
        
    Raises:
        ValueError: If yaml_dict is not a valid dictionary.
        ReadWriteError: If writing YAML file fails.

    Example:
        data = {'key': 'value'}
        write_yaml('output.yaml', data)
    """
    if not isinstance(yaml_dict, dict):
        raise ValueError(
            f"yaml_dict must be a dictionary, got {type(yaml_dict)}. "
            f"Please provide a valid dictionary to write as YAML."
        )
    
    # Create directory if it doesn't exist
    dir_path = osp.dirname(path)
    if dir_path and not osp.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            raise ReadWriteError(
                f"Failed to create directory {dir_path}: {e}. "
                f"Please check directory permissions."
            ) from e
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        raise ReadWriteError(
            f"Failed to write YAML file {path}: {e}. "
            f"Please check file path and permissions."
        ) from e


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
