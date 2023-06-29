import numpy as np
from data import *
if TORCH_AVAILABLE:
    from pytorch3d import transforms as tr
from scipy.spatial.transform import Rotation as R
import os
import sys
# print(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# The lists of available rotation parameterization
ROT_TYPES = [
    'SO3',  
    'so3',  # so(3) axis angle
    'quat_xyzw', # Quaternion with 'xyzw' ordering
    'quat_wxyz', # Quaternion with 'wxyz' ordering
    # 'rpy'   # Roll-Pitch-Yaw, Note that Ordering is matter # TODO
]

def is_SO3(x: Array) -> bool:
    """
    Check given rotation array's type is either SO3 or not.
    If the type is SO3, the shape of array is (n,3,3) or (3,3)
    """
    shape = x.shape
    if len(shape) <= 1 or len(shape) > 3: return False # invalid shape

    # [3,3] or [n,3,3]
    return shape[-2] == 3 and shape[-1] == 3 

def is_so3(x: Array) -> bool:
    """
    Check given rotation array's type is either so3 or not.
    If the type is SO3, the shape of array is (n,3) or (3)
    """
    shape = x.shape
    if len(shape) > 2: return False # invalid shape

    # [3,3] or [n,3,3]
    return shape[-1] == 3 

def is_quat(x: Array) -> bool:
    """
    Check given rotation array's type is either quaternion or not.
    If the type is Quaternion, the shape of array is (n,4) or (4)
    """
    shape = x.shape
    if len(shape) > 2: return False # invalid shape

    # [4] or [n,4]
    return shape[-1] == 4

def so3_to_SO3(so3: Array) -> Array:
    """
        Transform so3 to Rotation Matrix(SO3)
        Args:
            so3:(n,3), float, so3
        return:
            Mat:(n,3,3), float, Rotation Matrix
    """
    if is_tensor(so3): return tr.so3_exponential_map(so3)
    else: return R.from_rotvec(so3).as_matrix()
    
def quat_to_SO3(quat: Array, is_xyzw: bool) -> Array:
    """
        Transform Quaternion to Rotation Matrix(SO3)
        Args:
            quat:(n,4), float, quaternion
            is_xyzw:scalar, bool, quat is real part first. Otherwise, real part is last channel
        return:
            Mat:(n,3,3), float, Rotation Matrix
    """
    if is_xyzw: # real part last
        real = quat[:,3:4]
        img = quat[:,:3]
        quat = concat([real, img], 1)
    
    if is_tensor(quat): return tr.quaternion_to_matrix(quat)
    else: return R.from_quat(quat).as_matrix() 

class Rotation:
    """
    Rotation Class. This can be recieved one of types [SO3, so3, quat].
    - if SO3(Rotation matrix), shape must be [3, 3]
    - if so3(axis angle), shape must be [3]
    - if quat(Quaternion), shape must be [4]
    - default type is SO3 and default shape is [3, 3]
    """
    # class ROTATION_TYPE:
    #     SO3       = 0x0001 # SO(3) Rotation Matrix
    #     so3       = 0x0002 # so(3) axis angle
    #     QUAT_XYZW = 0x0003 # Quaternion with 'xyzw' ordering
    #     QUAT_WXYZ = 0x0004 # Quaternion with 'wxyz' ordering    
    def __init__(self, data: Array, type_str: str) -> None:
        
        assert is_array(data)
        assert len(data.shape) < 3 # invalid data shape        
        assert type_str in ROT_TYPES # invalid type string
        
        if type_str == 'SO3':
            if is_SO3(data) is False: 
                raise Exception(f'Invalid Shape Error. SO3 must be (n,3,3) or (3,3), but got {data.shape}')
            else: self.data = data
        elif type_str == 'so3':
            if is_so3(data) is False:
                raise Exception(f'Invalid Shape Error. so3 must be (n,3) or (3), but got {data.shape}')
            self.data = so3_to_SO3(data) 
        elif type_str == 'quat_xyzw' or type_str == 'quat_wxyz':
            if is_quat(data) is False:
                raise Exception(f'Invalid Shape Error. Quaternion must be (n,4) or (4), but got {data.shape}')
            self.data = quat_to_SO3(data, type_str=='quat_xyzw')        
    
    # constructor
    @staticmethod
    def from_mat3(mat3: Array):
        return Rotation(mat3, 'SO3')
    @staticmethod
    def from_so3(so3: Array):
        return Rotation(so3, 'so3')
    @staticmethod
    def from_quat_xyzw(quat: Array):
        return Rotation(quat, 'quat_xyzw')
    @staticmethod
    def from_quat_wxyz(quat: Array):
        return Rotation(quat, 'quat_wxyz')
    
    def to_tensor(self, device: Any = 'cpu') -> None:
        if is_numpy(self.data): self.data = torch.tensor(self.data,dtype=torch.float32, device=device)        
        if is_tensor(self.data): self.data = self.data.to(device)

    @property
    def type(self):
        return type(self.data)
    
    def mat(self):
        return self.data
    
    def apply_pts3d(self, pts3d: Array):
        ## R*pt3d: [3,3] * [n,3]
        mat = self.mat()
        if is_tensor(pts3d): mat = convert_tensor(mat,pts3d)
        
        pts3d = transpose2d(pts3d) # [3,n]
        pts3d = matmul(mat, pts3d) # [3,3] * [3,n] = [3,n]
        pts3d = transpose2d(pts3d) # [n,3]
        return pts3d
    
    def inverse_mat(self)->Array:
        return transpose2d(self.data)
    
    def inverse(self) -> 'Rotation':
        return Rotation.from_mat3(self.inverse_mat())
    
    def dot(self, rot:'Rotation') -> 'Rotation':
        rot1_mat = self.mat()
        rot2_mat = rot.mat()
        if is_tensor(rot1_mat): rot2_mat = convert_tensor(rot2_mat,rot1_mat)    
        rot_mat = matmul(rot1_mat,rot2_mat)
        return Rotation.from_mat3(rot_mat)
    
if __name__ == '__main__':
    rot = [  0.9958109, -0.0487703,  0.0773446,
           0.0526372,  0.9974220, -0.0487703,
          -0.0747667,  0.0526372,  0.9958109 ]

    mat3 = np.array(rot).reshape(3,3)
    rot = Rotation.from_mat3(mat3)    
    inv_rot = rot.inverse()
    print(rot.dot(inv_rot).mat())
