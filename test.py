from module.rotation import *
from module.hybrid_operations import *
from module.hybrid_math import *
from module.file_utils import *
from module.camera import PinholeCamera, Camera
from module.geometry import *
from module.pose import interpolate_pose, Pose
import numpy as np
import absl.app as app


def main(unused_args):
    
    # data_dict = read_json("/home/sehyun/nerf-project/dataset/monosdf_replica.json")
    # multivew = MultiView.from_dict(data_dict)
    # depth_dir = "/home/sehyun/nerf_result/vanilla_nerf_wo_coarse/render_output_100000"
    # depth_file_fmt = "depth_%06d.npy" 
    # depth_path = [ osp.join(depth_dir,depth_file_fmt%i) for i in range(100)]
    # multivew.depth_path = depth_path
    # multivew.save_point_cloud("/home/sehyun/nerf_result/vanilla_nerf_wo_coarse/pcd_vn.ply")
    os.makedirs("/home/sehyun/acc_images")
    acc_path = "/home/sehyun/nerf_result/vanilla_nerf_wo_coarse/tsdf_output_res_1024"
    acc_file_fmt = "acc_%05d.npy"
    for i in range(99):
        acc = read_float(osp.join(acc_path, acc_file_fmt%i))
        acc_image = (acc * 255.).astype(np.uint8)
        write_image(reduce_dim(acc_image,dim=-1),f"/home/sehyun/acc_images/acc_{i:05d}.png")
    
    return None

if __name__ == "__main__":
    app.run(main)