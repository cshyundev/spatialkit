from hybrid_operations import *
from hybrid_math import *
import matplotlib.pyplot as plt
import cv2 as cv

def is_image(image:Array) -> bool:
    if len(image.shape) != 3: return False
    if image.shape[0] == 3 or image.shape[-1] == 3: return True
    return False 

def convert_image(image: Array) -> Array:
    assert (is_image(image)), (f"Invalid Shape. Image's shape must be (H,W,3) or (3,H,W).")
    image = convert_numpy(image)
    if image.shape[0] == 3: # (3,H,W)
        image = permute(image, (2,0,1)) # (H,W,3)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image
        
def concat_images(images:List[Array], vertical:bool=False):
        concated_images = convert_image(images[0]) 
        for image in images[1:]:
            if vertical: concated_images = concat([concated_images, convert_image(image)],0)
            else: concated_images = concat([concated_images, convert_image(image)],1)
        return concated_images

def draw_circle(image:np.ndarray, pt2d:Tuple[int,int], radius:int=1,
                rgb:Tuple[int,int,int]=None, thickness:int=2):
    if rgb is None: rgb = tuple(np.random.randint(0,255,3).tolist())
    return cv.circle(image, pt2d, radius, rgb, thickness)

def draw_line_by_points(image:np.ndarray, pt1: Tuple[int,int], pt2: Tuple[float,float],
              rgb:Tuple[int,int,int]=None, thickness:int=2) -> np.ndarray:
    if rgb is None: rgb = tuple(np.random.randint(0,255,3).tolist())
    return cv.line(image, pt1, pt2, rgb, thickness)

def draw_line_by_line(image:np.ndarray, line: Tuple[float,float,float],
                      rgb:Tuple[int,int,int]=None, thickness:int=2) ->np.ndarray:
    """
    Draw Line by (a,b,c). (a,b,c) means a line: ax + by + c = 0
    Args:
        image: (H,W,3) or (H,W), float
        line: (3,), float, line parameter (a,b,c)
        rgb: (3,), int, RGB color
        thickness: scalar, int, thickness of the line
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
    
def show_image(image:np.ndarray):
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    from file_utils import read_image    
    image = read_image("/home/sehyun/workspace/computer_vision_python/replica/scan1/000000_rgb.png")
    pt1 = (246,110)
    pt2 = (134,222)
    line = (1.,-2.,0.)
    # image = draw_line_by_points(image,pt1,pt2)
    image = draw_line_by_line(image, line)
    show_image(image)