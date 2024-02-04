from module.rotation import *
from module.hybrid_operations import *
from module.hybrid_math import *
from module.file_utils import *
from module.camera import PinholeCamera, Camera
from module.geometry import *
from module.pose import interpolate_pose, Pose
import numpy as np
import absl.app as app

from module.geometry import *


def main(unused_args):
    
    frame_dict = {}
    root = "./garage_pinhole"
    frame_dict["root"] = root
    frames = []


    cam = {
    "cam_type": "PINHOLE",
    "image_size": [640, 480],
    "focal_length": [554.256, 415.692],
    "principal_point": [320, 240],
    "skew": 0.,
    "radial": [0.0, 0.0, 0.0],
    "tangential": [0.0, 0.0]
   }

    for i in range(1):
        frame = {}
        # camera
        frame["cam"] = cam
        # pose
        pose = np.loadtxt(osp.join(root,f"pose/0_train_{i:04d}.txt"))
        # frame["camtoworld"] = pose
        frame["camtoworld"] = [[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]]
        frame["image"] = f"rgb/0_train_{i:04d}.png"
        frame["depth"] = f"depth/0_train_{i:04d}.tiff"
        frames.append(frame)

    frame_dict["frames"] = frames 

    multiview =  MultiView.from_dict(frame_dict)
    multiview.save_point_cloud("./pcd.ply")

    return None

if __name__ == "__main__":
    app.run(main)