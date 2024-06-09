import numpy as np
from typing import *
import tifffile
import skimage.io as skio
from skimage import img_as_float
import json
import yaml
from PIL import Image

def read_float(path: str) -> np.ndarray:
    extension = path.split(sep=".")[-1]
    try:
        if extension == "npy":
            data = np.load(path)
        elif extension == 'tiff':
            multi_datas = tifffile.TiffFile(path)
            num_datas = len(multi_datas.pages)
            if num_datas == 0: raise Exception("No Images.")
            elif num_datas == 1: data = multi_datas.pages[0].asarray().squeeze()
            else: data = np.concatenate([np.expand_dim(x.asarray(),0) for x in multi_datas.pages], 0)
        else:
            raise Exception("No Support Extension.")
    except Exception as e:
        print("reading data failed.")
        print(e)
        data = None
    return data    

def write_float(data:np.ndarray, path:str):
    extension = path.split(sep=".")[-1]
    try:
        if extension == "npy":
            np.save(path,data)
        elif extension == "tiff":
            with tifffile.TiffWriter(path) as tiff:
                if data.dtype is not np.float32:
                    data = data.astype(np.float32)
                tiff.write(data, photometric='MINISBLACK',
                    bitspersample=32, compression='zlib')
        else: raise Exception("No Support Extension.")
    except Exception as e:
        print("writing data failed.")
    
def read_image(path: str, as_float:bool=False) -> np.ndarray:
    try:
        image = skio.imread(path)
        if as_float:
             image = img_as_float(image) # normalize [0,255] -> [0.,1.]
        return image 
    except Exception as e:
        print("reading image failed.")
        return None

def write_image(image:np.ndarray, path: str):
    if len(image.shape) == 3 and image.shape[-1]== 1:
        image = np.squeeze(image)
    pil_image = Image.fromarray(image)
    extension = path.split(sep=".")[-1]
    pil_image.save(path, extension)

def read_json(path: str) -> Dict[str,Any]:
    try:
        with open(path, "r") as f:
            json_dict = json.load(f)
        return json_dict 
    except:
        raise Exception("Reading json file Failed.")

def write_json(path:str, dict:Dict[str,Any], indent:int=1):
    try:
        with open(path,"w") as f:
            json.dump(dict,f, indent=indent)
    except:
        raise Exception("Writing json file Failed.")

def read_yaml(path:str) -> Dict[str, Any]:
    try:
        with open(path,"r") as f:
            dict = yaml.load(f,Loader=yaml.FullLoader)
        return dict
    except:
        raise Exception("Reading yaml file failed.")

def write_yaml(path:str, dict:Dict[str,Any]):
    with open(path,"w") as f:
        yaml.dump(dict,f)
