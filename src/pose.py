import numpy as np
from .rotation import Rotation, SO3_to_so3, slerp
from .hybrid_operations import *
from .hybrid_math import *



def SE3_to_se3(SE3: Array) -> Array:
    # covert SE3(matrix) to se3(so3,R^3)
    R = SE3[:3,:3]
    t = SE3[3,:]
    so3 = SO3_to_so3(R)    
    se3 = stack([so3, t], dim=0)
    return se3

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
        if t is None: t = np.array([0.,0.,0.])
        if rot is None: rot = Rotation.from_mat3(np.eye(3))

        assert (is_array(t)), ('translation must be array type(Tensor or Numpy).')
        assert(type(t) == rot.type), ('Rotation and Tranlsation must be same array type(Tensor or Numpy).')
        assert(t.size == 3), ('size of translation must be 3.')

        if t.shape == (3,):
            t = t.reshape(1,3)
        
        self.t:Array = t
        self.rot:Rotation = rot
        
    def get_origin_direction(self, rays: Array):
        """
        get camera origin and direction vectors from rays(camera coordinates)
        Args:
            rays: (3,n), float, ray from camera origin
        Return:
            origin: (n,3), float, camera origin in world coord. origin = t
            direction: (n,3), float, direction vector in world coord. direction = R.T*rays
        """
        assert len(rays.shape) <= 2 and rays.shape[0] == 3, "Invalid Shape. Ray's shape must be (n,3) or (3)"
        if len(rays.shape) == 2:
            n_rays = rays.shape[1]
        else:
            rays = expand_dim(rays, 1)
            n_rays = 1
        origin = np.tile(self.t, (n_rays,1))
        if is_tensor(rays): origin = convert_tensor(origin, rays)
        direction = transpose2d(self.rot.apply_pts3d(rays)).reshape((-1,3)) 
        return origin, direction        
    
    @staticmethod
    def from_rot_vec_t(rot_vec: Array, t:Array) -> 'Pose':
        assert(is_array(rot_vec)), 'rotation vector must be Array type(Tensor or Numpy)'
        assert(is_array(t)), 'translation vector must be Array type(Tensor or Numpy)'
        assert(rot_vec.shape[-1] == 3), 'Invalid Shape. rotation vector must be (3,)'
        assert(rot_vec.shape[-1] == 3), 'Invalid Shape. translation vector must be (3,)'
        rot = Rotation.from_so3(rot_vec)
        return Pose(t,rot)

    @staticmethod
    def from_mat(mat4: Array) -> 'Pose':
        assert is_array(mat4), ('mat4 must be array type(Tensor or Numpy)')
        assert((mat4.shape[-1] == 4) and (mat4.shape[-2] == 3 or mat4.shape[-2] == 4)), ('Invalid Shape. The shape of mat4 must be (3,4) or (4,4)')
        return Pose(mat4[0:3,3], Rotation.from_mat3(mat4[:3,:3]))
    
    @staticmethod
    def from_dict(dict: Dict[str, Any]) -> 'Pose':
        assert('camtoworld' in dict), ("No Value Error. There is no Cam to World Transform in Dict.")
        mat = np.array(dict['camtoworld']) 
        return Pose.from_mat(mat)        
    
    def rot_mat(self):
        return self.rot.mat()
    
    def mat34(self):
        return concat([self.rot_mat(), transpose2d(self.t)], 1)
    
    def mat44(self):
        mat34 = self.mat34()
        last = np.array([0., 0., 0., 1.]).reshape(1,4)
        if is_tensor(mat34): last = convert_tensor(last,mat34)        
        return concat([mat34, last], 0)
    
    def rot_vec_t(self):
        return self.rot.so3(), self.t
        
    def skew_t(self):
        return vec3_to_skew(self.t)
    
    def get_t_rot_mat(self):
        return self.t, self.rot_mat()
    
    def inverse(self):
        R_inv = self.rot.inverse()
        t_inv = -R_inv.apply_pts3d(transpose2d(self.t))
        return Pose(transpose2d(t_inv), R_inv)

    def merge(self, pose:'Pose') -> 'Pose':
        """
        |R t||R' t'| = |RR' Rt' + t|
        |0 1||0  1 |   |0       1  |
        """   
        mat4 = self.mat44()@pose.mat44() 
        t = mat4[:3,3]
        rot = Rotation.from_mat3(mat4[:3,:3])
        return Pose(t, rot)
    
    def apply_pts3d(self, pts3d: Array) -> Array:
        # P' = R@P + t
        p = self.rot.apply_pts3d(pts3d)
        if is_tensor(pts3d): return p + convert_tensor(self.t, pts3d)
        return p + convert_numpy(self.t) 

def interpolate_pose(p1:Pose,p2:Pose,t:float) -> Pose:
    """
    Interpolate Pose using Lerp and SLerp.
    translation: linear interpolation(Lerp)
    rotation: spherical linear interpolation(Slerp)
    Args:
        p1: Pose, start Pose
        p2: Pose, end Pose
        t: interpolation parameter
    Return:
        Interpolated Pose
    """
    r=slerp(p1.rot,p2.rot,t)
    trans1, trans2 = p1.t, p2.t
    trans = trans1 * (1.- t) + trans2 * t
    return Pose(t=trans,rot=r)
    
if __name__ == '__main__':
    rot = [  0.9958109, -0.0487703,  0.0773446,
           0.0526372,  0.9974220, -0.0487703,
          -0.0747667,  0.0526372,  0.9958109 ]
    rot = Rotation.from_mat3(np.array(rot).reshape(3,3))
    t1 = np.array([1.,2.,3.])
    t2 = np.array([2.,3.,4.])
    
    # cam1tow = Pose(t1,rot) # cam1 to world
    # cam2tow = Pose(t2,rot) # cam2 to world
    # rel = cam2tow.inverse().merge(cam1tow) # cam1 to cam2
    # print(rel.mat34())
    # cam1tow_mat = cam1tow.mat44()
    # wtocam2_mat = cam2tow.inverse().mat44()
    # rel_mat = wtocam2_mat@cam1tow_mat 
    # print(rel_mat)
    p = Pose(t1,rot)
    skew_t = p.skew_t()
    print(vec3_to_skew(t1))
    # print(skew_t)
    



    
    
    