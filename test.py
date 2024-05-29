from src.rotation import *
from src.hybrid_operations import *
from src.hybrid_math import *
from src.file_utils import *
from src.camera import PinholeCamera, Camera
from src.geometry import *
from src.pose import interpolate_pose, Pose
import numpy as np
import absl.app as app
from cv2 import undistortPoints
from numpy.testing import assert_almost_equal
from src.camera import PinholeCamera
from src.image_transform import *

def main(unused_args):

    img = read_image("./image.png")
    
    def show_images(left_image: np.ndarray, right_image: np.ndarray, left_title: str, right_title: str):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(left_image)
        axes[0].set_title(left_title)
        axes[0].axis('off')

        axes[1].imshow(right_image)
        axes[1].set_title(right_title)
        axes[1].axis('off')

        plt.show()
    
    camera = PinholeCamera.from_fov(img.shape[0:2], 60.)
    inv_K = camera.inv_K
    camera = PinholeCamera.from_fov((150,150), 60.)
    new_K = camera.K
    
    
    # fwd_wrapper = ImageWarper(similarity(90.,135.,.93,1.), img.shape[0:2], (271,186), mode='forward')
    # inv_wrapper = ImageWarper( similarity(90.,186,.93,1.), img.shape[0:2], (271,186), mode='inverse')
    inv_wrapper = ImageWarper(similarity(90.,150,0,1.) @ inv_K @ new_K, img.shape[0:2], (150,150), mode='inverse')
    
    # fwd_img = fwd_wrapper(img)
    inv_img = inv_wrapper(img)
    show_image(inv_img)
    # show_image(np.rot90(img,-1))
    # show_images(left_image=fwd_img,right_image=inv_img,left_title="Forward",right_title="Inverse")

    return None

if __name__ == "__main__":
    app.run(main)