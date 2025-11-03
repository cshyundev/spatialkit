"""
Image I/O operations including TIFF, PNG, JPG, and PGM formats.

This module provides functions for reading and writing various image formats
with support for float conversion and batch operations.

Functions:
    - read_tiff: Read TIFF files
    - write_tiff: Write TIFF files with compression
    - read_image: Read images in various formats
    - read_all_images: Read all images from a directory
    - write_image: Write images
    - read_pgm: Read PGM format images
    - write_pgm: Write PGM format images

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT License
"""

import os
import os.path as osp
from typing import Optional, List

import tifffile
import numpy as np
import skimage.io as skio
import skimage
from PIL import Image

from ..common.logger import LOG_ERROR
from ..common.exceptions import (
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


