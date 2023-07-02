from module.data import *
import matplotlib.pyplot as plt
import os
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
        concated_imgs = convert_image(images[0]) 
        for image in images[1:]:
            if vertical: concated_imgs = concat([concated_imgs, convert_image(image)],0)
            else: concated_imgs = concat([concated_imgs, convert_image(image)],1)
        return concated_imgs

def draw_circles(image:np.ndarray, pts2d:List[Tuple[int,int]], radius:int=1,
                rgb:Tuple[int,int,int]=(255,0,0), thickness:int=2):
    for pt2d in pts2d:
        image = cv.circle(image, tuple(reversed(pt2d)), radius, rgb, thickness)
    return image

def show_image(image:np.ndarray):
    cv.imshow("Image", image)

if __name__ == "__main__":
    
    image = np.ones(640*480) * 255
    image = image.reshape(640,480)
    
    pt2d = [(380,290)]
    image = draw_circles(image, pt2d, 2)