import numpy as np
from module.rotation import Rotation
from module.hybrid_operations import *

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

        assert (is_array(t)), ('translation must be array type(Tensor or Numpy)')
        assert(type(t) == rot.type), ('Rotation and Tranlsation must be same array type(Tensor or Numpy)')

        if t.shape == (3,):
            t = t.reshape(1,3)
        
        self.t:Array = t
        self.rot:Rotation = rot
        
    def get_origin_direction(self, rays: Array):
        """
        get camera origin and direction vectors from rays(camera coordinates)
        Args:
            rays: (n,3), float, ray from camera origin
        Return:
            origin: (n,3), float, camera origin in world coord. origin = t
            direction: (n,3), float, direction vector in world coord. direction = R.T*rays
        """
        assert len(rays.shape) <= 2 and rays.shape[-1] == 3, "Invalid Shape. Ray's shape must be (n,3) or (3)"
        if len(rays.shape) == 2:
            n_rays = rays.shape[0]
        else:
            rays = expand_dim(rays, 0)
            n_rays = 1
        origin = np.tile(self.t, (n_rays,1))
        if is_tensor(rays): origin = convert_tensor(origin, rays)
        direction = self.rot.apply_pts3d(rays)
        return origin, direction        
    
    @staticmethod
    def from_se3(se3: Array) -> 'Pose':
        # assert(is_array(se3)), 'se3 must be Array type(Tensor or Numpy)'
        # assert(se3.shape[-1] == 6), 'Invalid Shape. se3 must be (n,6) or (6)'
        
        # if len(se3.shape) == 1: # (6)
        #     se3 = expand_dim(se3,0) # (1,6)
        # return Pose(tr, rot)
        raise NotImplementedError

    @staticmethod
    def from_mat(mat4: Array) -> 'Pose':
        assert is_array(mat4), ('mat4 must be array type(Tensor or Numpy)')
        assert((mat4.shape[-1] == 4) and (mat4.shape[-2] == 3 or mat4.shape[-2] == 4)), ('Invalid Shape. The shape of mat4 must be (3,4) or (4,4)')
        return Pose(mat4[0:3,3], Rotation.from_mat3(mat4[:3,:3]))
    
    @staticmethod
    def from_dict(dict: Dict[str, Any]) -> 'Pose':
        assert hasattr(dict, 'camtoworld'), ("No Value Error. There is no Cam to World Transform in Dict.")
        mat = np.array(dict['camtoworld']) 
    
        if mat.shape == (3,4):
            mat = concat([mat, np.array([0.,0.,0.,1.])], 1)
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
    
    def se3(self):
        

    def skew_t(self):
        t = self.t
        wx = t[0,0]
        wy = t[0,1]
        wz = t[0,2]
        skew_t = np.array([[0., -wz, wy],
                           [wz,  0, -wx],
                           [-wy, wx,  0]])
        if is_tensor(t): skew_t = convert_tensor(skew_t,t)
        return skew_t
    
    def get_t_rot_mat(self):
        return self.t, self.rot_mat()
    
    def inverse(self):
        R_inv = Rotation.from_mat3(self.rot.inverse_mat()) 
        t_inv = -R_inv.apply_pts3d(self.t)
        return Pose(t_inv, R_inv)

    def merge(self, pose:'Pose') -> 'Pose':    
        rot = pose.rot.dot(self.rot)
        t = pose.rot.apply_pts3d(self.t) + pose.t
        return Pose(t, rot)
    
    
if __name__ == '__main__':
    rot = [  0.9958109, -0.0487703,  0.0773446,
           0.0526372,  0.9974220, -0.0487703,
          -0.0747667,  0.0526372,  0.9958109 ]
    rot = Rotation.from_mat3(np.array(rot).reshape(3,3))     
    t = np.array([1.,2.,3.])
    w = Pose(t,rot)
    print(w.mat34())
    inv_w = w.inverse() 
    print(inv_w.mat34())
    I = w.merge(inv_w)
    print(I.mat34())
    
    
    