import numpy as np
from typing import *
from module.data import *
import os
import os.path as osp
# import tifffile
# import skimage.io as skio
from PIL import Image

def read_float(path: str) -> np.ndarray:
    extension = path.split(sep=".")[-1]
    try:
        if extension == "npy":
            data = np.load(path)
        elif extension == 'tiff':
            multi_datas = tifffile.TiffFile(path)
            num_datas = len(multi_datas.pages())
            if num_datas == 0: raise Exception("No Images.")
            elif num_datas == 1: data = multi_datas.pages[0].asarray().squeeze()
            else: data = concat([expand_dim(x.asarray(),0) for x in multi_datas.pages], 0)
        else:
            raise Exception("No Support Extension.")
    except Exception as e:
        print("reading data failed.")
        data = None
    return data    

def write_float(image:Array, path:str):
    x = convert_numpy(x)
    extension = path.split(sep=".")[-1]
    try:
        if extension == "npy":
            np.save(data, path)
        elif extension == "tiff":
            with tifffile.TiffWriter(path) as tiff:
                if image.dtype is not np.float32:
                    image = image.astype(np.float32)
                tiff.write(image, photometric='MINISBLACK',
                    bitspersample=32, compression='zlib')
        else: raise Exception("No Support Extension.")
    except Exception as e:
        print("reading data failed.")
        data = None
    
def read_image(path: str) -> np.ndarray:
    try:
        return skio.imread(path)
    except Exception as e:
        print("reading image failed.")
        return None

def write_image(image:Array, path: str):
    pil_image = Image.fromarray(image)
    extension = path.split(sep=".")[-1]
    pil_image.save(path, 'extension')

if __name__ == '__main__':
    path = "./hello/world.npy"
    print(read_float(path))