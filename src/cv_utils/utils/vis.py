import matplotlib.pyplot as plt
import numpy as np
from typing import *
import cv2 as cv
import random
from ..common.logger import LOG_ERROR

def _is_image(image:np.ndarray) -> bool:
    """
        Checks if the given array is an image with 3 channels.

        Args:
            image (np.ndarray): Array to check.

        Return:
            bool: True if the array is an image with 3 channels, False otherwise.

        Example:
            result = is_image(np.random.rand(100, 100, 3))
    """
    if len(image.shape) != 3: return False
    if image.shape[0] == 3 or image.shape[-1] == 3: return True
    return False

def convert_to_uint8_image(image: np.ndarray) -> np.ndarray:
    """
        Normalizes data to the range [0, 255] and converts it to uint8 type.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Converted image array.

        Details:
            - Checks if the input image has a valid shape: (H,W,3) or (3,H,W).
            - Transposes the image if its shape is (3,H,W) to (H,W,3).
            - If the image data type is not uint8, normalizes the data to the range [0, 255] and converts it to uint8.

        Example:
            converted_image = convert_to_uint8_image(np.random.rand(3, 100, 100))
    """
    assert _is_image(image), "Invalid Shape. Image's shape must be (H,W,3) or (3,H,W)."
    if image.shape[0] == 3:  # (3,H,W)
        image = np.transpose(image, (1, 2, 0))  # (H,W,3)
    if image.dtype != np.uint8:
        image = image * 255
        image = image.astype(np.uint8)
    return image

# convert float to color image such as gray or depth
def float_to_image(v:np.ndarray,min_v:Optional[float]=None,max_v:Optional[float]=None, color_map:str='magma') -> np.ndarray:
    """
        Converts a float array to a color image using a color map.

        Args:
            v (np.ndarray): Input float array.
            min_v (Optional[float]): Minimum value for normalization. Default is None.
            max_v (Optional[float]): Maximum value for normalization. Default is None.
            color_map (str): Color map to use. Default is 'magma'.

        Return:
            np.ndarray: Color mapped image.
    """
    if len(v.shape) == 3: v = np.squeeze(v)
    if min_v is None: min_v = v.min()
    if max_v is None: max_v = v.max()
    v = np.clip(v,min_v,max_v)
    normalized_v = (v - min_v) / (max_v -min_v)
    color_mapped = plt.cm.get_cmap(color_map)(normalized_v)
    # remove alpha channel
    color_mapped = (color_mapped[:, :, :3] * 255).astype(np.uint8)
    return color_mapped

# openGL coordinate mapping
def normal_to_image(normal: np.ndarray) -> np.ndarray:
    """
        Converts normal map to an image.

        Args:
            normal (np.ndarray): Input normal map array.

        Return:
            np.ndarray: Converted image.
        
        Details:
            - Converts the normal map coordinates to OpenGL coordinates.
            - The normal map is expected to have 3 channels.
            - The red channel is mapped from (x + 1) / 2.
            - The green channel is mapped from (-y + 1) / 2.
            - The blue channel is mapped from (-z + 1) / 2.
    """
    assert len(normal.shape) == 3 or normal.shape[-1] == 3
    r = (normal[:,:,0] + 1) / 2.0 # (H,W,3)
    g = (-normal[:,:,1] + 1) / 2.0
    b = (-normal[:,:,2] + 1) / 2.0
    color_mapped = convert_to_uint8_image(np.stack((r, g, b), -1)) 
    return color_mapped

def concat_images(images:List[np.ndarray], vertical:bool=False):
        """
        Concatenates a list of images either vertically or horizontally.

        Args:
            images (List[np.ndarray]): List of images to concatenate.
            vertical (bool): Flag to concatenate vertically. Default is False.

        Returns:
            np.ndarray: Concatenated image.

        Example:
            combined_image = concat_images([image1, image2], vertical=True)
    """
        # Assume all images have same size
        concated_image = images[0] 
        for image in images[1:]:
            if vertical: concated_image = np.concatenate([concated_image, image],0)
            else: concated_image = np.concatenate([concated_image, image],1)
        return concated_image

def draw_circle(image:np.ndarray, pt2d:Tuple[int,int], radius:int=1,
                rgb:Optional[Tuple[int,int,int]]=None, thickness:int=2):
    """
        Draws a circle on the image.

        Args:
            image (np.ndarray): Input image array.
            pt2d (Tuple[int, int]): Center of the circle.
            radius (int): Radius of the circle. Default is 1.
            rgb (Optional[Tuple[int, int, int]]): Color of the circle in RGB. Default is random.
            thickness (int): Thickness of the circle outline. Default is 2.

        Return:
            np.ndarray: Image with the drawn circle.

        Example:
            image_with_circle = draw_circle(image, (50, 50), radius=5)
    """
    if rgb is None: rgb = tuple(np.random.randint(0,255,3).tolist())

    return cv.circle(image, pt2d, radius, rgb, thickness)

def draw_line_by_points(image:np.ndarray, pt1: Tuple[int,int], pt2: Tuple[float,float],
              rgb:Optional[Tuple[int,int,int]]=None, thickness:int=2) -> np.ndarray:
    """
        Draws a line between two points on the image.

        Args:
            image (np.ndarray): Input image array.
            pt1 (Tuple[int, int]): Starting point of the line.
            pt2 (Tuple[float, float]): Ending point of the line.
            rgb (Optional[Tuple[int, int, int]]): Color of the line in RGB. Default is random.
            thickness (int): Thickness of the line. Default is 2.

        Return:
            np.ndarray: Image with the drawn line.

        Example:
            image_with_line = draw_line_by_points(image, (10, 10), (100, 100))
    """
    if rgb is None: rgb = tuple(np.random.randint(0,255,3).tolist())
    return cv.line(image, pt1, pt2, rgb, thickness)

def draw_line_by_line(image:np.ndarray, line: Tuple[float,float,float],
                      rgb:Optional[Tuple[int,int,int]]=None, thickness:int=2) ->np.ndarray:
    """
        Draw Line by (a,b,c). (a,b,c) means a line: ax + by + c = 0

        Args:
            image: (H,W,3) or (H,W), float
            line: (3,), float, line parameter (a,b,c)
            rgb: (3,), int, RGB color
            thickness: scalar, int, thickness of the line

        Return:
            np.ndarray
    """
    if rgb is None: rgb = tuple(np.random.randint(0,255,3).tolist())
    h,w = image.shape[:2]
    if line[1] == 0.:
        # ax +0y + c = 0
        x0 = x1 = int(-line[2] / line[0])
        y0 = 0
        y1 = h
    else:
        x0,y0 = map(int, [0, -line[2]/line[1] ])
        x1,y1 = map(int, [w, -(line[2]+line[0]*w)/line[1]])
    return cv.line(image,(x0,y0), (x1,y1), rgb, thickness)

def draw_polygon(image: np.ndarray, pts: Union[List[Tuple[int, int]],np.ndarray],\
                 rgb:Optional[Tuple[int,int,int]]=None, thickness:int=3) -> np.ndarray:
    """
        Draws a polygon on the image based on provided points using OpenCV.

        Args:
            image (np.ndarray): The input image on which the polygon will be drawn.
            pts (Union[List[Tuple[int, int]], np.ndarray], [N,2]): List of points (x, y) that define the vertices of the polygon.
            rgb (Optional[Tuple[int, int, int]]): Color of the polygon in RGB. Default is green.
            thickness (int): Thickness of the polygon lines. Default is 3.

        Returns:
            np.ndarray: The image with the drawn polygon.

        Example:
            image_with_polygon = draw_polygon(image, [(10, 10), (20, 20), (30, 10)])
    """
    assert len(pts) >= 3, "There must be at least three points to form a polygon."
    if isinstance(pts,list): 
        assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in pts), "Each point must be a tuple of two integers."

    points_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    
    if rgb is None: rgb = (0,255,0) # default color = green
    cv.polylines(image, [points_array], isClosed=True, color=rgb, thickness=thickness)
    
    return image

def draw_lines(image, pts:List[Tuple[int,int]], rgb:Optional[Tuple[int,int,int]]=None, thickness:int=2):
    """
        Draws lines connecting a series of points on an image in the order.

        Args:
            image (np.ndarray): The image on which to draw the lines.
            pts (List[Tuple[int, int]]): List of (x, y) tuples representing the points.
            rgb (Optional[Tuple[int, int, int]]): Color of the line in RGB format. Default is random.
            thickness (int): Thickness of the lines. Default is 2.

        Returns:
            np.ndarray: The image with lines drawn on it.

        Example:
            image_with_lines = draw_lines(image, [(10, 10), (20, 20), (30, 10)])
    """
    if rgb is None: rgb = tuple(np.random.randint(0,255,3).tolist())
    for i in range(len(pts) - 1):
       cv.line(image, pts[i], pts[i + 1], rgb, thickness)
    return image

def show_image(image:np.ndarray, title:str="image"):
    """
        Displays an image with a title.

        Args:
            image (np.ndarray): Image to display.
            title (str): Title of the image. Default is "image".

        Example:
            show_image(image, title="Example Image")
    """
    plt.imshow(image)
    plt.title(title)
    plt.show()

def show_two_images(image1:np.ndarray, image2:np.ndarray, title1:str="Left image", title2:str="Right image"):
    """
        Displays two images side by side.

        Args:
            image1 (np.ndarray): First image to display.
            image2 (np.ndarray): Second image to display.
            title1 (str): Title for the first image. Default is "Left image".
            title2 (str): Title for the second image. Default is "Right image".

        Details:
        - Each images can have different size. 

        Example:
            show_two_images(image1, image2, title1="Image 1", title2="Image 2")
    """
    # Create a figure to hold the subplots
    _, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    
    # Display the first image
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis('off')  # Turn off axis numbers and ticks

    # Display the second image
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis('off')  # Turn off axis numbers and ticks

    plt.tight_layout()  # Adjust subplots to give some padding between them
    plt.show()

def show_correspondences(image1: np.ndarray, image2: np.ndarray,
                         pts1: List[Tuple[float,float]], pts2: List[Tuple[float,float]],
                         margin_width: int=20):
    """
        Plots corresponding points between two images with an optional white margin between them.

        Args:
            image1 (np.ndarray): First input image.
            image2 (np.ndarray): Second input image.
            pts1 (List[Tuple[float, float]]): Points in the first image.
            pts2 (List[Tuple[float, float]]): Points in the second image.
            margin_width (int): Width of the white margin between the images. Default is 20.

        Example:
            show_correspondences(image1, image2, pts1, pts2, margin_width=30)
    """
    # Create white margin
    height = image1.shape[0]
    white_margin = float_to_image(np.ones((height, margin_width)),0.,1.,color_map='gray') 

    combined_image = concat_images([image1, white_margin, image2], vertical=False)

    _, ax = plt.subplots()
    ax.imshow(combined_image, cmap='gray')
    ax.set_axis_off()

    offset = image1.shape[1] + margin_width

    # Draw points and lines connecting them
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        ax.plot([x1, x2 + offset], [y1, y2], linestyle='-', color=color)
        ax.plot(x1, y1, 'o', mfc='none', mec=color, mew=2)
        ax.plot(x2 + offset, y2, 'o', mfc='none', mec=color, mew=2)

    plt.show()