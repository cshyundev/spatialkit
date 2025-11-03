"""
Display utilities for visualizing images and correspondences.

This module provides functions for displaying images using matplotlib.
"""

import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..common.exceptions import DisplayError, InvalidDimensionError
from .convert import concat_images, float_to_image


def show_image(image: np.ndarray, title: str = "image"):
    """
    Displays an image with a title.

    Args:
        image (np.ndarray): Image to display.
        title (str): Title of the image. Default is "image".

    Raises:
        InvalidDimensionError: If image is not a valid array.
        DisplayError: If image display fails.

    Example:
        show_image(image, title="Example Image")
    """
    if not isinstance(image, np.ndarray):
        raise InvalidDimensionError(
            f"Image must be numpy array, got {type(image)}. "
            f"Please provide a valid image array."
        )

    try:
        plt.imshow(image)
        plt.title(title)
        plt.show()
    except Exception as e:
        raise DisplayError(
            f"Failed to display image '{title}': {e}. "
            f"Please check image format and display environment."
        ) from e


def show_two_images(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Left image",
    title2: str = "Right image",
):
    """
    Displays two images side by side.

    Args:
        image1 (np.ndarray): First image to display.
        image2 (np.ndarray): Second image to display.
        title1 (str): Title for the first image. Default is "Left image".
        title2 (str): Title for the second image. Default is "Right image".

    Raises:
        InvalidDimensionError: If images are not valid arrays.
        DisplayError: If image display fails.

    Details:
    - Each images can have different size.

    Example:
        show_two_images(image1, image2, title1="Image 1", title2="Image 2")
    """
    if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
        raise InvalidDimensionError(
            f"Both images must be numpy arrays, got {type(image1)} and {type(image2)}. "
            f"Please provide valid image arrays."
        )

    try:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image1)
        axes[0].set_title(title1)
        axes[0].axis("off")

        axes[1].imshow(image2)
        axes[1].set_title(title2)
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        raise DisplayError(
            f"Failed to display two images '{title1}' and '{title2}': {e}. "
            f"Please check image formats and display environment."
        ) from e


def show_correspondences(
    image1: np.ndarray,
    image2: np.ndarray,
    pts1: List[Tuple[float, float]],
    pts2: List[Tuple[float, float]],
    margin_width: int = 20,
):
    """
    Plots corresponding points between two images with an optional white margin between them.

    Args:
        image1 (np.ndarray): First input image.
        image2 (np.ndarray): Second input image.
        pts1 (List[Tuple[float, float]]): Points in the first image.
        pts2 (List[Tuple[float, float]]): Points in the second image.
        margin_width (int): Width of the white margin between the images. Default is 20.

    Raises:
        InvalidDimensionError: If images are not valid arrays.
        ValueError: If point lists have different lengths.
        DisplayError: If correspondence display fails.

    Example:
        show_correspondences(image1, image2, pts1, pts2, margin_width=30)
    """
    if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
        raise InvalidDimensionError(
            f"Both images must be numpy arrays, got {type(image1)} and {type(image2)}. "
            f"Please provide valid image arrays."
        )

    if len(pts1) != len(pts2):
        raise ValueError(
            f"Point lists must have same length, got {len(pts1)} and {len(pts2)} points. "
            f"Please provide equal number of corresponding points."
        )

    if margin_width < 0:
        raise ValueError(
            f"Margin width must be non-negative, got {margin_width}. "
            f"Please provide a valid margin width."
        )

    try:
        height = image1.shape[0]
        white_margin = float_to_image(
            np.ones((height, margin_width)), 0.0, 1.0, color_map="gray"
        )

        combined_image = concat_images([image1, white_margin, image2], vertical=False)

        _, ax = plt.subplots()
        ax.imshow(combined_image, cmap="gray")
        ax.set_axis_off()

        offset = image1.shape[1] + margin_width

        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            color = "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
            ax.plot([x1, x2 + offset], [y1, y2], linestyle="-", color=color)
            ax.plot(x1, y1, "o", mfc="none", mec=color, mew=2)
            ax.plot(x2 + offset, y2, "o", mfc="none", mec=color, mew=2)

        plt.show()
    except Exception as e:
        raise DisplayError(
            f"Failed to display correspondences between {len(pts1)} point pairs: {e}. "
            f"Please check image formats and point coordinates."
        ) from e
