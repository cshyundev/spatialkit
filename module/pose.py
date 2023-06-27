import numpy as np
import torch
from typing import Tuple

import os
import sys
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rotation import Rotation
from data import *

    
class Pose:
    """
        Rigid Transfrom Class
        Basically Save Camera Coord. to World Coord. Transform
    
        Notation
        - T: normally means tranformation (tr & rot)
        - t: normally means translation
        - rot: normally means rotation
        - c: normally means scale factor
    """
    def __init__(self, t:Array = None, rot:Rotation = None):
        """
        Initialization Pose Instance
        Args:
            t: [n,3], float, translation
            Rotation: Rotation Instance
        """
        if t is None: t = np.array([[0.,0.,0.]])
        if rot is None: rot = Rotation.from_mat3(expand_dim(np.eye(3),0))

        assert (is_array(t)), ('translation must be array type(Tensor or Numpy)')
        assert(type(t) == rot.type), ('Rotation and Tranlsation must be same array type(Tensor or Numpy)')
        
        if len(t.shape) == 1: t = expand_dim(t,0) 
        self.t:Array = t
        self.rot:Rotation = rot
        
        self.n_poses = t.shape[0]
        
    def get_origin_direction(self, rays: Array):
        """
        get camera origin and direction vectors from rays(camera coordinates)
        Args:
            rays: (n,3), float, ray from camera origin
        Return:
            origin: (n,3), float, camera origin in world coord. origin = t
            direction: (n,3), float, direction vector in world coord. direction = R.T*rays
        """
        assert len(rays.shape) <= 2, "Invalid Shape. Ray's shape must be (n,3) or (3)"
        assert self.n_poses == 1, "Only 1 pose can be transformed by Origin and Direction"
        if len(rays.shape) == 2:
            n_rays = rays.shape[0]
        else:
            rays = expand_dim(rays, 0)
            n_rays = 1
        origin = np.tile(self.t, (n_rays,1))
        if is_tensor(rays): origin = convert_tensor(origin, rays)
        direction = self.rot.apply_rot(rays)
        return origin, direction        
    
    @staticmethod
    def from_se3(se3: Array):
        # assert(is_array(se3)), 'se3 must be Array type(Tensor or Numpy)'
        # assert(se3.shape[-1] == 6), 'Invalid Shape. se3 must be (n,6) or (6)'
        
        # if len(se3.shape) == 1: # (6)
        #     se3 = expand_dim(se3,0) # (1,6)
        # return Pose(tr, rot)
        raise NotImplementedError

    @staticmethod
    def from_mat4(mat4: Array):
        assert is_array(mat4), ('mat4 must be array type(Tensor or Numpy)')
        assert((mat4.shape[-1] == 4) and (mat4.shape[-2] == 4)), ('Invalid Shape. The shape of mat4 must be (n,4,4) or (4,4)')
        
        if len(mat4.shape) == 2: # (4,4) => (1,4,4)
            mat4 = expand_dim(mat4,0)        
        return Pose(mat4[:,0:3,3], Rotation.from_mat3(mat4[:,:3,:3]))
    
    @staticmethod
    def from_dict(dict: Dict[str, Any]):
        assert hasattr(dict, 'camtoworld'), ("No Value Error. There is no Cam to World Transform in Dict.")
        mat = np.array(dict['camtoworld']) 
    
        if mat.shape == (3,4):
            mat = concat([mat, np.array([0.,0.,0.,1.])], 1)
        return Pose.from_mat4(mat)        
    
    def rot_mat(self,idx:int=0):
        return self.rot.mat(idx)
    
    def translation(self,idx:int=0):
        if idx < 0: return self.t
        return self.t[idx,:]
    
    def skew_t(self,idx:int=0):
        t = self.t[idx,:]
        wx = t[0]
        wy = t[1]
        wz = t[2]
        skew_t = np.array([[0., -wz, wy],
                           [wz,  0, -wx],
                           [-wy, wx,  0]])
        if is_tensor(t): skew_t = convert_tensor(skew_t,t)
        return skew_t
    
    def get_t_rot_mat(self, idx:int=0):
        return self.translation(idx), self.rot_mat(idx)
    
    def inverse_pose(self, idx:int=0):
        t = self.translation(idx)
        R_inv = Rotation.from_mat3(self.rot.inverse_rot_mat(idx)) 
        t_inv = -R_inv.apply_rot(t)
        return Pose(t_inv, R_inv)
    
def merge_transform(pose1:Pose, pose2:Pose) -> Pose:
    t1, rot_mat1 = pose1.get_t_rot_mat()
    t2, rot_mat2 = pose2.get_t_rot_mat()
    
    rot_mat = matmul(rot_mat2, rot_mat1)
    t = matmul(rot_mat2,t1) + t2
    return Pose(t, Rotation.from_mat3(rot_mat)) 
    
if __name__ == '__main__':
    # data = np.random.rand(16).reshape(4,4)
    data = torch.rand(16).reshape(4,4)
    w = Pose.from_mat4(data)
    print(data)
    # print(w.rot_mat())
    # print(w.translation())
    inv_w = w.inverse_pose() 
    I = merge_transform(w,inv_w)
    print(I.get_t_rot_mat())
    
    
    