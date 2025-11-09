"""
Drawing utilities for 2D images.

This module provides functions for drawing shapes and annotations on images.

Functions:
    - draw_circle: Draw circles on images
    - draw_line_by_points: Draw lines between two points
    - draw_line_by_line: Draw lines specified by line equation
    - draw_polygon: Draw polygons
    - draw_lines: Draw multiple lines
    - draw_bbox: Draw bounding boxes with labels and confidence scores
    - draw_segmentation_mask: Draw segmentation masks with transparency

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT License
"""

from typing import Optional, List, Tuple, Union
import numpy as np
import cv2 as cv

from ..common.exceptions import (
    InvalidDimensionError,
    RenderingError
)
def draw_circle(
    image: np.ndarray,
    pt2d: Tuple[int, int],
    radius: int = 1,
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
):
    """
    Draws a circle on the image.

    Args:
        image (np.ndarray, [H,W,3] or [H,W]): Input image array.
        pt2d (Tuple[int, int]): Center of the circle.
        radius (int): Radius of the circle. Default is 1.
        rgb (Optional[Tuple[int, int, int]]): Color of the circle in RGB. Default is random.
        thickness (int): Thickness of the circle outline. Default is 2.

    Returns:
        np.ndarray: Image with the drawn circle.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        RenderingError: If circle drawing fails.

    Example:
        image_with_circle = draw_circle(image, (50, 50), radius=5)
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    if radius <= 0:
        raise ValueError(
            f"Radius must be positive, got {radius}. "
            f"Please provide a positive radius value."
        )
    
    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())
        pt2d = (int(pt2d[0]), int(pt2d[1]))
        return cv.circle(image, pt2d, radius, rgb, thickness)
    except Exception as e:
        raise RenderingError(
            f"Failed to draw circle at {pt2d} with radius {radius}: {e}. "
            f"Please check point coordinates are within image bounds."
        ) from e


def draw_line_by_points(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[float, float],
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draws a line between two points on the image.

    Args:
        image (np.ndarray): Input image array.
        pt1 (Tuple[int, int]): Starting point of the line.
        pt2 (Tuple[float, float]): Ending point of the line.
        rgb (Optional[Tuple[int, int, int]]): Color of the line in RGB. Default is random.
        thickness (int): Thickness of the line. Default is 2.

    Returns:
        np.ndarray: Image with the drawn line.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        RenderingError: If line drawing fails.

    Example:
        image_with_line = draw_line_by_points(image, (10, 10), (100, 100))
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())
        pt1_int = (int(pt1[0]), int(pt1[1]))
        pt2_int = (int(pt2[0]), int(pt2[1]))
        return cv.line(image, pt1_int, pt2_int, rgb, thickness)
    except Exception as e:
        raise RenderingError(
            f"Failed to draw line from {pt1} to {pt2}: {e}. "
            f"Please check point coordinates are valid."
        ) from e


def draw_line_by_line(
    image: np.ndarray,
    line: Tuple[float, float, float],
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw Line by (a,b,c), which (a,b,c) means a line: ax + by + c = 0

    Args:
        image (np.ndarray, [H,W,3] or [H,W]): Input image array.
        line (Tuple[float], [3,]): line parameter (a,b,c)
        rgb (Tuple[int], [3,] ): RGB color
        thickness (int): thickness of the line

    Returns:
        np.ndarray: Image with the drawn line.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        ValueError: If line parameters are invalid.
        RenderingError: If line drawing fails.
        
    Example:
        image_with_line = draw_line_by_line(image, (1.0, -1.0, 50.0))
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    if len(line) != 3:
        raise ValueError(
            f"Line parameters must be tuple of 3 values (a, b, c), got {len(line)} values. "
            f"Please provide line equation coefficients ax + by + c = 0."
        )
    
    a, b, c = line
    if a == 0 and b == 0:
        raise ValueError(
            f"Invalid line parameters: both a and b cannot be zero. "
            f"Please provide valid line equation coefficients."
        )
    
    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())
        h, w = image.shape[:2]
        
        if b == 0.0:
            # Vertical line: x = -c/a
            x0 = x1 = int(-c / a)
            y0 = 0
            y1 = h
        else:
            # General line
            x0, y0 = map(int, [0, -c / b])
            x1, y1 = map(int, [w, -(c + a * w) / b])
        
        return cv.line(image, (x0, y0), (x1, y1), rgb, thickness)
    except Exception as e:
        raise RenderingError(
            f"Failed to draw line with parameters {line}: {e}. "
            f"Please check line parameters and image dimensions."
        ) from e


def draw_polygon(
    image: np.ndarray,
    pts: Union[List[Tuple[int, int]], np.ndarray],
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 3,
) -> np.ndarray:
    """
    Draws a polygon on the image based on provided points using OpenCV.

    Args:
        image (np.ndarray): The input image on which the polygon will be drawn.
        pts (Union[List[Tuple[int, int]], np.ndarray], [N,2]): List of points (x, y) that define the vertices of the polygon.
        rgb (Optional[Tuple[int, int, int]]): Color of the polygon in RGB. Default is green.
        thickness (int): Thickness of the polygon lines. Default is 3.

    Returns:
        np.ndarray: The image with the drawn polygon.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        ValueError: If insufficient points or invalid point format.
        RenderingError: If polygon drawing fails.

    Example:
        image_with_polygon = draw_polygon(image, [(10, 10), (20, 20), (30, 10)])
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    if len(pts) < 3:
        raise ValueError(
            f"Polygon requires at least 3 points, got {len(pts)} points. "
            f"Please provide at least 3 vertices to form a polygon."
        )
    
    if isinstance(pts, list):
        if not all(isinstance(pt, tuple) and len(pt) == 2 for pt in pts):
            raise ValueError(
                f"Each point must be a tuple of two numbers (x, y). "
                f"Please ensure all points are in format (x, y)."
            )
    
    try:
        points_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        
        if rgb is None:
            rgb = (0, 255, 0)  # default color = green
        cv.polylines(image, [points_array], isClosed=True, color=rgb, thickness=thickness)
        
        return image
    except Exception as e:
        raise RenderingError(
            f"Failed to draw polygon with {len(pts)} points: {e}. "
            f"Please check point coordinates and image dimensions."
        ) from e


def draw_lines(
    image,
    pts: List[Tuple[int, int]],
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
):
    """
    Draws lines connecting a series of points on an image in the order.

    Args:
        image (np.ndarray): The image on which to draw the lines.
        pts (List[Tuple[int, int]]): List of (x, y) tuples representing the points.
        rgb (Optional[Tuple[int, int, int]]): Color of the line in RGB format. Default is random.
        thickness (int): Thickness of the lines. Default is 2.

    Returns:
        np.ndarray: The image with lines drawn on it.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        ValueError: If insufficient points provided.
        RenderingError: If line drawing fails.

    Example:
        image_with_lines = draw_lines(image, [(10, 10), (20, 20), (30, 10)])
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    if len(pts) < 2:
        raise ValueError(
            f"Need at least 2 points to draw lines, got {len(pts)} points. "
            f"Please provide at least 2 points to connect with lines."
        )
    
    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())
        for i in range(len(pts) - 1):
            pt1 = (int(pts[i][0]), int(pts[i][1]))
            pt2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
            cv.line(image, pt1, pt2, rgb, thickness)
        return image
    except Exception as e:
        raise RenderingError(
            f"Failed to draw lines connecting {len(pts)} points: {e}. "
            f"Please check point coordinates are valid."
        ) from e


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: Optional[str] = None,
    confidence: Optional[float] = None,
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    Draws a bounding box with optional label and confidence score on the image.

    Args:
        image (np.ndarray, [H,W,3] or [H,W]): Input image array.
        bbox (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2) where
            (x1, y1) is top-left corner and (x2, y2) is bottom-right corner.
        label (Optional[str]): Class label to display above the bounding box.
        confidence (Optional[float]): Confidence score (0.0-1.0) to display with the label.
        rgb (Optional[Tuple[int, int, int]]): Color of the bounding box in RGB. Default is random.
        thickness (int): Thickness of the bounding box lines. Default is 2.
        font_scale (float): Font scale for label text. Default is 0.5.
        font_thickness (int): Thickness of the label text. Default is 1.

    Returns:
        np.ndarray: Image with the drawn bounding box.

    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        ValueError: If bbox coordinates are invalid or confidence is out of range.
        RenderingError: If drawing fails.

    Example:
        >>> image_with_bbox = draw_bbox(image, (10, 10, 100, 100), label="person", confidence=0.95)
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )

    if len(bbox) != 4:
        raise ValueError(
            f"Bounding box must have 4 coordinates (x1, y1, x2, y2), got {len(bbox)} values. "
            f"Please provide valid bounding box coordinates."
        )

    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"Invalid bounding box coordinates: ({x1}, {y1}, {x2}, {y2}). "
            f"Ensure x1 < x2 and y1 < y2."
        )

    if confidence is not None and not (0.0 <= confidence <= 1.0):
        raise ValueError(
            f"Confidence score must be between 0.0 and 1.0, got {confidence}. "
            f"Please provide a valid confidence score."
        )

    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())

        # Draw bounding box rectangle
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv.rectangle(image, pt1, pt2, rgb, thickness)

        # Draw label and confidence if provided
        if label is not None or confidence is not None:
            text_parts = []
            if label is not None:
                text_parts.append(label)
            if confidence is not None:
                text_parts.append(f"{confidence:.2f}")
            text = " ".join(text_parts)

            # Calculate text size for background
            (text_width, text_height), baseline = cv.getTextSize(
                text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Draw filled rectangle as background for text
            text_bg_pt1 = (int(x1), int(y1) - text_height - baseline - 5)
            text_bg_pt2 = (int(x1) + text_width + 5, int(y1))
            cv.rectangle(image, text_bg_pt1, text_bg_pt2, rgb, -1)

            # Draw text
            text_org = (int(x1) + 2, int(y1) - baseline - 2)
            cv.putText(
                image,
                text,
                text_org,
                cv.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv.LINE_AA,
            )

        return image
    except Exception as e:
        raise RenderingError(
            f"Failed to draw bounding box at {bbox}: {e}. "
            f"Please check coordinates are within image bounds."
        ) from e


def draw_segmentation_mask(
    image: np.ndarray,
    mask: np.ndarray,
    rgb: Optional[Tuple[int, int, int]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Draws a segmentation mask overlay on the image with transparency.

    Args:
        image (np.ndarray, [H,W,3]): Input image array (must be 3-channel RGB/BGR).
        mask (np.ndarray, [H,W]): Binary segmentation mask where non-zero pixels are masked.
            Can be boolean or integer array.
        rgb (Optional[Tuple[int, int, int]]): Color of the mask overlay in RGB. Default is random.
        alpha (float): Transparency of the mask overlay (0.0-1.0).
            0.0 is fully transparent, 1.0 is fully opaque. Default is 0.5.

    Returns:
        np.ndarray: Image with the segmentation mask overlay.

    Raises:
        InvalidDimensionError: If image is not 3D array or mask is not 2D array.
        ValueError: If image and mask shapes don't match or alpha is out of range.
        RenderingError: If mask drawing fails.

    Example:
        >>> mask = np.zeros((480, 640), dtype=bool)
        >>> mask[100:200, 100:200] = True
        >>> image_with_mask = draw_segmentation_mask(image, mask, rgb=(255, 0, 0), alpha=0.6)
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise InvalidDimensionError(
            f"Image must be 3D numpy array (H,W,3), got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid RGB/BGR image array."
        )

    if not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
        raise InvalidDimensionError(
            f"Mask must be 2D numpy array (H,W), got {type(mask)} with shape {getattr(mask, 'shape', 'unknown')}. "
            f"Please provide a valid binary mask array."
        )

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image and mask dimensions must match. "
            f"Got image shape {image.shape[:2]} and mask shape {mask.shape}. "
            f"Please ensure both have the same height and width."
        )

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(
            f"Alpha must be between 0.0 and 1.0, got {alpha}. "
            f"Please provide a valid transparency value."
        )

    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())

        # Create colored mask overlay
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = rgb

        # Blend the mask with the original image
        result = image.copy()
        mask_area = mask > 0
        result[mask_area] = (
            alpha * colored_mask[mask_area] + (1 - alpha) * image[mask_area]
        ).astype(np.uint8)

        return result
    except Exception as e:
        raise RenderingError(
            f"Failed to draw segmentation mask: {e}. "
            f"Please check image and mask are compatible."
        ) from e


