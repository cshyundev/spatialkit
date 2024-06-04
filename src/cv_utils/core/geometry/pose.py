import numpy as np
from rotation import Rotation, SO3_to_so3, slerp, so3_to_SO3
from ..operations.hybrid_operations import *
from ..operations.hybrid_math import *



def SE3_to_se3(SE3: Array) -> Array:
    """
    Convert SE3 matrix to se3 vector.
    Args:
        SE3: (4,4), float, SE3 transformation matrix
    Returns:
        se3: (6,), float, se3 vector
    """
    R = SE3[:3,:3]
    t = SE3[:3,3]
    so3 = SO3_to_so3(R)    
    se3 = concat([so3, t], dim=0)
    return se3

def se3_to_SE3(se3: Array) -> Array:
    """
    Convert se3 vector to SE3 matrix.
    Args:
        se3: (6,), float, se3 vector (3 for so3 and 3 for translation)
    Returns:
        SE3: (4,4), float, SE3 transformation matrix
    """
    assert se3.shape == (6,), 'se3 vector must be of shape (6,)'

    so3 = se3[:3]
    t = se3[3:]

    R = so3_to_SO3(so3)
    SE3 = eye(4,se3)
    SE3[:3, :3] = R
    SE3[:3, 3] = t

    return SE3

class Pose:
    """
    Pose Class representing a 3D pose with position(x,y,z) and orientation.
    This class encapsulates both the translation and rotation components of a pose.

    Attributes:
        t (np.ndarray): Translation vector of shape (1, 3)
        rot (Rotation): Rotation instance represented by a Rotation object
    """
    def __init__(self, t:Array = None, rot:Rotation = None):
        """
        Initialization Pose Instance
        Args:
            t: (1,3), float, translation
            Rotation: Rotation Instance
        """
        if t is None: t = np.array([0.,0.,0.])
        if rot is None: rot = Rotation.from_mat3(np.eye(3))

        assert (is_array(t)), ('translation must be array type(Tensor or Numpy).')
        assert(type(t) == rot.type), ('Rotation and Tranlsation must be same array type(Tensor or Numpy).')
        assert(t.size == 3), ('size of translation must be 3.')

        if t.shape == (3,): t = t.reshape(1,3)
        
        self.t:Array = t
        self.rot:Rotation = rot      
    
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
    r = slerp(p1.rot,p2.rot,t)
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
    



    
    
    