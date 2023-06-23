import numpy as np
import torch
from typing import Tuple

import os
import sys
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Rotation import Rotation
from data import *

class Pose:
    """
        Notation
        - T: normally means tranformation (tr & rot)
        - t: normally means translation
        - rot: normally means rotation
        - c: normally means scale factor
    """
    def __init__(self, t: Array, rot: Rotation) -> None:
        """
        Initialization Pose Instance
        Args:
            t: [n,3], float, translation
            Rotation: Rotation Instance
        """
        assert (is_array(t)), ('translation must be array type(Tensor or Numpy)')
        assert(type(t) == rot.type), ('Rotation and Tranlsation must be same array type(Tensor or Numpy)')
        
        self.t:Array = t
        self.rot:Rotation = rot


    @property
    def length(self):
        assert (self.t_data.length == self.R_data.length), 'translation and rotation data should have same length!'
        return self.t_data.length

    def numpy(self, out_Rtype:str) -> Tuple[np.ndarray, np.ndarray, str]:
        if out_Rtype == 'SO3':
            self.R_data = self.R_data.SO3()
        elif out_Rtype == 'so3':
            self.R_data = self.R_data.so3()
        if out_Rtype == 'quat':
            self.R_data = self.R_data.quat()
        else:
            raise AssertionError('Unvalid rotation type!')

        t = self.t_data.numpy()
        R, R_type = self.R_data.numpy()
        return t, R, R_type

    def SE3(self):
        raise NotImplementedError

    def se3(self):
        raise NotImplementedError

    @staticmethod
    def create_from_se3(se3: Array):
        # assert(is_array(se3)), 'se3 must be Array type(Tensor or Numpy)'
        # assert(se3.shape[-1] == 6), 'Invalid Shape. se3 must be (n,6) or (6)'
        
        # if len(se3.shape) == 1: # (6)
        #     se3 = expand_dim(se3,0) # (1,6)
        # return Pose(tr, rot)
        raise NotImplementedError

    @staticmethod
    def create_from_mat4(mat4: Array):
        assert is_array(mat4), 'mat4 must be array type(Tensor or Numpy)'
        assert((mat4.shape[-1] == 4) and (mat4.shape[-2] == 4)), 'Invalid Shape. The shape of mat4 must be (n,4,4) or (4,4)'
        
        if len(mat4.shape) == 2: # (4,4) => (1,4,4)
            mat4 = expand_dim(mat4,0)        
        return Pose(mat4[:,:,:3], Rotation.create_from_mat3(mat4[:,:3,:3]))
    
    
if __name__ == '__main__':
    data = torch.randn((4, 4))
    w = Pose.create_from_mat4(data)
    print(w.t)
    print(w.rot.data)
    