from module.rotation import *
from module.hybrid_operations import *
from module.hybrid_math import *
from module.file_utils import *
from module.camera import PinholeCamera, Camera
from module.geometry import *
from module.pose import interpolate_pose, Pose
import numpy as np
import absl.app as app
from module.parser import parse_monosdf_dataset

from module.geometry import *


def main(unused_args):
    
    frame_dict = parse_monosdf_dataset("./replica/scan1","center_crop_for_replica","PINHOLE",1.)

    multiview =  MultiView.from_dict(frame_dict)
    multiview.save_point_cloud("./pcd_replica.ply")

    return None

if __name__ == "__main__":
    app.run(main)