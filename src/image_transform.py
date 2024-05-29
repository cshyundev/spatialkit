import numpy as np
from typing import *
from src.hybrid_operations import *
from src.hybrid_math import *
from scipy.ndimage import map_coordinates


def make_pixel_grid(width:int, height:int) -> np.ndarray:
    u,v = np.meshgrid(range(width), range(height))
    uv = concat([u.reshape((1,-1)), v.reshape((1,-1))], 0)
    return uv

def translation(tx: int = 0, ty: int = 0) -> np.ndarray:
    """
    Translation transformation matrix

    Args:
        tx (int, optional): Translation in x-direction. Defaults to 0.
        ty (int, optional): Translation in y-direction. Defaults to 0.
    Returns:
         Translation transformation matrix
    """
    mat33 = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]], np.float32)
    return mat33

def rotation(angle: float) -> np.ndarray:
    """
    Rotation transformation matrix

    Args:
        angle (float): Rotation angle in degrees.
    Returns:
        Rotation transformation matrix
    """
    rad = np.deg2rad(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    mat33 = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]], np.float32)
    return mat33

def shear(shx: float = 0, shy: float = 0) -> np.ndarray:
    """
    Shear transformation matrix

    Args:
        shx (float, optional): Shear in x-direction. Defaults to 0.
        shy (float, optional): Shear in y-direction. Defaults to 0.
    Returns:
        Shear transformation matrix
    """
    mat33 = np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]], np.float32)
    return mat33

def scaling(sx: float = 1, sy: float = 1) -> np.ndarray:
    """
    Scaling transformation matrix

    Args:
        sx (float, optional): Scaling factor in x-direction. Defaults to 1.
        sy (float, optional): Scaling factor in y-direction. Defaults to 1.
    Returns:
        Scaling transformation matrix
    """
    mat33 = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]], np.float32)
    return mat33

def similarity(angle: float, tx: int = 0, ty: int = 0, scale: float = 1.0) -> np.ndarray:
    """
    Similarity transformation matrix (combining rotation, translation, and scaling)

    Args:
        angle (float): Rotation angle in degrees.
        tx (int, optional): Translation in x-direction. Defaults to 0.
        ty (int, optional): Translation in y-direction. Defaults to 0.
        scale (float, optional): Scaling factor. Defaults to 1.0.
    Returns:
        Similarity transformation matrix
    """
    rad = np.deg2rad(angle)
    s_cos_a = np.cos(rad) * scale
    s_sin_a = np.sin(rad) * scale
    mat33 = np.array([
        [s_cos_a, -s_sin_a, tx],
        [s_sin_a, s_cos_a, ty],
        [0, 0, 1]], np.float32)
    return mat33

class ImageWarper:
    def __init__(self, matrix, input_shape, output_shape, mode='inverse'):
        self.matrix = matrix
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.mode = mode
        self.warp_table = self._precompute_warp_table()

    def _precompute_inverse_warp_table(self):
        height, width = self.output_shape
        inv_matrix = np.linalg.inv(self.matrix)
        warp_table = np.zeros((height, width, 2), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                src_coords = np.dot(inv_matrix, [x, y, 1])
                src_coords /= src_coords[2]
                warp_table[y, x] = src_coords[:2]

        return warp_table

    def _precompute_forward_warp_table(self):
        height, width = self.input_shape
        warp_table = np.zeros((height, width, 2), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                dst_coords = np.dot(self.matrix, [x, y, 1])
                dst_coords /= dst_coords[2]
                if 0 <= dst_coords[0] < self.output_shape[1] and 0 <= dst_coords[1] < self.output_shape[0]:
                    warp_table[y, x] = dst_coords[:2]

        return warp_table

    def _precompute_warp_table(self):
        if self.mode == 'inverse':
            return self._precompute_inverse_warp_table()
        elif self.mode == 'forward':
            return self._precompute_forward_warp_table()
        else:
            raise ValueError("Mode should be either 'inverse' or 'forward'")

    def _apply_warp_table(self, image):
        height, width = self.output_shape
        if(len(image.shape) == 3):
            warped_image = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
        else:
            warped_image = np.zeros((height, width), dtype=image.dtype)
            

        for y in range(height):
            for x in range(width):
                src_x, src_y = self.warp_table[y, x]
                if 0 <= src_x < image.shape[1] and 0 <= src_y < image.shape[0]:
                    warped_image[y, x] = image[int(src_y), int(src_x)]

        return warped_image

    def __call__(self, image):
        return self._apply_warp_table(image)