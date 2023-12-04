from module.hybrid_operations import *
from module.hybrid_math import *
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
        image = image * 255
        image = image.astype(np.uint8)
    return image

# convert float to color image such as gray or depth
def float_to_image(v:Array,min_v:float=None,max_v:float=None, color_map:str='magma') -> np.ndarray:
    if len(v.shape) == 3: v = reduce_dim(v,dim=2)
    v = convert_numpy(v)
    if min_v is None: min_v = v.min()
    if max_v is None: max_v = v.max()
    v = np.clip(v,min_v,max_v)
    normalized_v = (v - min_v) / (max_v -min_v)
    color_mapped = plt.cm.get_cmap(color_map)(normalized_v)
    # remove alpha channel
    color_mapped = (color_mapped[:, :, :3] * 255).astype(np.uint8)
    return color_mapped

# openGL coordinate mapping
def normal_to_image(normal: Array) -> np.ndarray:
    assert len(normal.shape) == 3 or normal.shape[-1] == 3

    normal = convert_numpy(normal)
    r = (normal[:,:,0] + 1) / 2.0 # (H,W,3)
    g = (-normal[:,:,1] + 1) / 2.0
    b = (-normal[:,:,2] + 1) / 2.0
    color_mapped = convert_image(np.stack((r, g, b), -1)) 
    return color_mapped

def concat_images(images:List[Array], vertical:bool=False):
        concated_image = convert_image(images[0]) 
        for image in images[1:]:
            if vertical: concated_image = concat([concated_image, convert_image(image)],0)
            else: concated_image = concat([concated_image, convert_image(image)],1)
        return concated_image

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
    from file_utils import write_image
    normal = np.random.rand(480,480,1)*100
    color_normal = float_to_image(normal)
    write_image(color_normal, "/home/sehyun/normal.png")