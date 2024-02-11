import numpy as np
from .hybrid_operations import *
from .hybrid_math import *
from scipy.spatial.transform import Rotation as R
from enum import Enum

# The lists of available rotation parameterization
ROT_TYPES = [
    'SO3',  # SO(3): Special orthogonal group 
    'so3',  # so(3): axis angle
    'quat_xyzw', # Quaternion with 'xyzw' ordering
    'quat_wxyz', # Quaternion with 'wxyz' ordering
    # 'rpy'   # Roll-Pitch-Yaw, Note that Ordering is matter # TODO
]

class RotType(Enum):
    SO3 = ("SO3", "SO(3): Special orthogonal group")
    so3 = ("so3", "so(3): Lie Algebra of SO(3)")
    QUAT_XYZW = ("QUAT_XYZW", "Quaternion with 'xyzw' ordering")
    QUAT_WXYZ = ("QUAT_WXYZ", "Quaternion with 'wxyz' ordering")
    NONE = ("NONE", "NoneType")

    @staticmethod
    def from_string(type_str: str)-> 'RotType':
        if type_str == 'SO3':
            return RotType.SO3
        elif type_str == 'so3':
            return RotType.so3
        elif type_str == 'QUAT_XYZW':
            return RotType.QUAT_XYZW
        elif type_str == 'QUAT_WXYZ':
            return RotType.QUAT_XYZW
        else:
            return RotType.NONE



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
            so3:(3,), float, so3
        return:
            Mat:(3,3), float, Rotation Matrix
    """
    assert is_so3(so3) and len(so3.shape) == 1, (f"Invaild Shape. so3 must be (3), but got {str(so3.shape)}")
    theta = sqrt(so3[0]**2 + so3[1]**2 + so3[2]**2)
    vec = so3 / (theta + 1e-15) 
    skew_vec = vec3_to_skew(vec)
    return exponential_map(skew_vec*theta)

def quat_to_SO3(quat: Array, is_xyzw: bool) -> Array:
    """
        Transform Quaternion to Rotation Matrix(SO3)
        Args:
            quat:(4), float, quaternion
            is_xyzw:scalar, bool, quat is real part first. Otherwise, real part is last channel
        return:
            Mat:(n,3,3), float, Rotation Matrix
    """
    if is_xyzw: # real part last
        x, y, z, w = quat[0],quat[1],quat[2],quat[3] 
    else:
        w, x, y, z = quat[0],quat[1],quat[2],quat[3] 

    # Compute the elements of the rotation matrix
    m00 = 1 - 2 * (y**2 + z**2)
    m01 = 2 * (x * y + z * w)
    m02 = 2 * (x * z - y * w)
    
    m10 = 2 * (x * y - z * w)
    m11 = 1 - 2 * (x**2 + z**2)
    m12 = 2 * (y * z + x * w)

    m20 = 2 * (x * z + y * w)
    m21 = 2 * (y * z - x * w)
    m22 = 1 - 2 * (x**2 + y**2)
    
    # Create the rotation matrices for each quaternion
    rotation_matrix = stack([stack([m00, m01, m02], dim=0),
                              stack([m10, m11, m12], dim=0),
                              stack([m20, m21, m22], dim=0)], dim=1)

    return rotation_matrix

def SO3_to_so3(SO3: Array) -> Array:
    assert(is_SO3(SO3)), (f"Invaild Shape. SO3 must be (3,3), but got {str(so3.shape)}")
    
    theta = arcos((trace(SO3) - 1.)*0.5)
    vec = 0.5 / sin(theta) * stack([SO3[2,1]-SO3[1,2],SO3[0,2] - SO3[2,0],SO3[1,0] - SO3[0,1]], 0)   
    return theta * vec

def SO3_to_quat(SO3: Array) -> Array:
     # Extract the elements of the rotation matrix
    r11, r12, r13 = SO3[0, 0], SO3[0, 1], SO3[0, 2]
    r21, r22, r23 = SO3[1, 0], SO3[1, 1], SO3[1, 2]
    r31, r32, r33 = SO3[2, 0], SO3[2, 1], SO3[2, 2]

    # Calculate the quaternion components
    trace = r11 + r22 + r33
    if trace > 0:
        S = sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (r32 - r23) / S
        y = (r13 - r31) / S
        z = (r21 - r12) / S
    elif (r11 > r22) and (r11 > r33):
        S = sqrt(1.0 + r11 - r22 - r33) * 2
        w = (r32 - r23) / S
        x = 0.25 * S
        y = (r12 + r21) / S
        z = (r13 + r31) / S
    elif r22 > r33:
        S = sqrt(1.0 + r22 - r11 - r33) * 2
        w = (r13 - r31) / S
        x = (r12 + r21) / S
        y = 0.25 * S
        z = (r23 + r32) / S
    else:
        S = sqrt(1.0 + r33 - r11 - r22) * 2
        w = (r21 - r12) / S
        x = (r13 + r31) / S
        y = (r23 + r32) / S
        z = 0.25 * S

    return np.array([w, x, y, z])

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
    
    def so3(self):
        return SO3_to_so3(self.data)  
    
    def quat(self):
        return SO3_to_quat(self.data)
    
    def apply_pts3d(self, pts3d: Array):
        ## R*pt3d: [3,3] * [3,n]
        assert pts3d.shape[0] == 3, f"Invalid Shape. pts3d's shape should be (3,n), but got {str(pts3d.shape)}."
        mat = self.mat()
        if is_tensor(pts3d): mat = convert_tensor(mat,pts3d)        
        pts3d = matmul(mat,pts3d) # [3,3] * [3,n] = [3,n]
        # pts3d = transpose2d(pts3d).reshape(-1,3) # [n,3]
        return pts3d # [3,n]
    
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


def slerp(r1:Rotation,r2:Rotation, t:float):
    """
    Spherical Linear Interpolation between two Rotion
    1. transfrom Rotations to unit quaternions q1,q2
    2. compute angle "w" between two quaternions: w = cos^-1(q1*q2)
    3. compute slerp(q1,q2,t) =  sin((1-t)*w)/sin(w)*q1 + sin(tw)/sin(w)*q2

    Args:
        r1: Rotation, rotation instance
        r2: Rotation, rotation instance
        t: float, interploation parameters, 0<=t<=1
    Return:
        slerp(q1,q2,t): Rotation
    """
    assert (t <=1. and t >= 0.), ("Interpolation parameters must be between 0 and 1.")
    
    if t == 1.: return r2
    if t == 0.: return r1

    q1,q2 = r1.quat(),r2.quat() # (w,x,y,z)
    cos_omega = matmul(q1,q2)
    if cos_omega < 0.: # If negative dot product, negate one of the quaternions to take shorter arc
        q1 = -q1
        cos_omega = -cos_omega
    if cos_omega > 0.9999: # If the quaternions are very close, use linear interpolation
        q = normalize(q1*(1-t) + q2*t,dim=0)
    else:
        omega = arcos(cos_omega)
        sin_omega = sin(omega)
        scale1 = sin((1-t)*omega) / sin_omega
        scale2 = sin(t*omega) / sin_omega
        q = normalize(scale1*q1 + scale2*q2,dim=0)
    return Rotation.from_quat_wxyz(q)  