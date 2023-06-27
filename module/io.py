import numpy as np
from typing import *
from data import *
import os
import os.path as osp
import tifffile
import skimage.io as skio

def read_float(path: str) -> np.ndarray:
    extension = path.split(sep=".")[-1]
    
    try:
        if extension == "npy":
            data = np.load(path)
    except Exception as e:
        print("reading data failed.")
        data = None
    return data    

def read_image(path: str) -> np.ndarray:
    
    try:
        return skio.imread(path)
    except Exception as e:
        print("reading image failed.")
        return None

if __name__ == '__main__':
    path = "./hello/world.npy"
    
    print(read_float(path))