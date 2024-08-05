import numpy as np
from typing import *
import tifffile
import skimage.io as skio
import skimage
import json
import yaml
from PIL import Image
import os
import os.path as osp
from ..common.logger import  LOG_ERROR
import cv2 as cv

def read_tiff(path: str) -> np.ndarray:
    """
        Reads a TIFF file.

        Args:
            path (str): Path to the TIFF file.

        Return:
            data (np.ndarray): Data read from the TIFF file.

        Example:
            data = read_tiff('example.tiff')
    """
    try:
        multi_datas = tifffile.TiffFile(path)
        num_datas = len(multi_datas.pages)
        if num_datas == 0:
            raise Exception("No Images.")
        elif num_datas == 1:
            data = multi_datas.pages[0].asarray().squeeze()
        else:
            data = np.concatenate([np.expand_dims(x.asarray(), 0) for x in multi_datas.pages], 0)
        return data
    except (FileNotFoundError, IsADirectoryError) as e:
        LOG_ERROR(f"File not found or is a directory: {path}. Error: {e}")
    except ValueError as e:
        LOG_ERROR(f"Value error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to read TIFF file: {path}. Error: {e}.")
    
    return None

def write_tiff(data: np.ndarray, path: str, photometric: str='MINISBLACK',
                bitspersample: int=32,
                compression: str='zlib'):
    """
        Writes data to a TIFF file.

        Args:
            data (np.ndarray): The main data to write.
            path (str): Path to save the TIFF file.
            photometric (str): Photometric interpretation of the data.
            bitspersample (int): Number of bits per sample. Default is 32.
            compression (str): Compression type to use. Default is 'zlib'.

        Example:
            data = np.random.rand(100, 100).astype(np.float32)
            write_tiff(data, 'output_with_thumbnail.tiff')
    """
    try:
        with tifffile.TiffWriter(path) as tiff:
            options = dict(photometric=photometric, bitspersample=bitspersample, compression=compression)
            tiff.write(data, subifds=0, **options)
    except (PermissionError, IsADirectoryError) as e:
        LOG_ERROR(f"Permission error or path is a directory: {path}. Error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to write TIFF file: {path}. Error: {e}")

def read_all_images(image_dir: str, as_float: Optional[bool]=False) -> List[np.ndarray]:
    """
        Reads all images in a directory.

        Args:
            image_dir (str): Path to the directory containing images.
            as_float (Optional[bool]): Flag to convert images to float representation.

        Return:
            images (List[np.ndarray]): List of images read from the directory.
    """
    image_list = os.listdir(image_dir)
    images = []
    for image_name in image_list:
        image_path = osp.join(image_dir, image_name)
        image = read_image(image_path, as_float)
        if image is not None: images.append(image)
    return images 

def read_image(path: str, as_float:Optional[bool]=False) -> np.ndarray:
    """
        Reads an image file.

        Args:
            path (str): Path to the image file.
            as_float (Optional[bool]): Flag to convert the image to float representation.

        Return:
            image (np.ndarray): Image data read from the file.
        
        Example:
            image = read_image('example.png', as_float=True)
    """
    try:
        image = skio.imread(path)
        if as_float:
             image = skimage.img_as_float(image) # normalize [0,255] -> [0.,1.]
        return image 
    except (FileNotFoundError, IsADirectoryError) as e:
        LOG_ERROR(f"File not found or is a directory: {path}. Error: {e}")
    except ValueError as e:
        LOG_ERROR(f"Value error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to read Image file: {path}. Error: {e}.")
    return None

def write_image(image:np.ndarray, path: str):
    """
        Writes image data to a file.

        Args:
            image (np.ndarray): Image data to write.
            path (str): Path to save the image file.

        Example:
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            write_image(image, 'output.png')
    """
    if len(image.shape) == 3 and image.shape[-1]== 1:
        image = np.squeeze(image)
    try:
        pil_image = Image.fromarray(image)
        extension = path.split(sep=".")[-1]
        pil_image.save(path, extension)
    except (PermissionError, IsADirectoryError) as e:
        LOG_ERROR(f"Permission error or path is a directory: {path}. Error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to write Image file: {path}. Error: {e}")

def read_pgm(path: str, mode:Optional[str]=None) -> np.ndarray:
    """
        Reads a PGM image file.

        Args:
            path (str): Path to the PGM image file.
            mode (Optional[str]): Mode to convert the image using PIL.

        Return:
            np.ndarray: Image data read from the file.

        Details:
        - 'L' - (8-bit pixels, black and white)
        - 'RGB' - (3x8-bit pixels, true color)
        - 'RGBA' - (4x8-bit pixels, true color with transparency mask)
        - 'CMYK' - (4x8-bit pixels, color separation)
        - 'YCbCr' - (3x8-bit pixels, color video format)
        - 'I' - (32-bit signed integer pixels)
        - 'F' - (32-bit floating point pixels)
            
        Example:
            image = read_pgm('example.pgm', mode='L')
    """

    try:
        with open(path, 'rb') as f:
            image = Image.open(f)
            if mode is not None:
                image = image.convert(mode)
        return np.array(image)
    
    except (FileNotFoundError, IsADirectoryError) as e:
        LOG_ERROR(f"File not found or is a directory: {path}. Error: {e}")
    except ValueError as e:
        LOG_ERROR(f"Value error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to read Image file: {path}. Error: {e}.")
    return None

def write_pgm(image: np.ndarray, path: str):
    """
        Writes numpy array image data to a PGM file.

        Args:
            image (np.ndarray): Image data to write.
            path (str): Path to save the PGM image file.
    
        Example:
            image = np.random.random((100, 100)).astype(np.uint8)
            write_pgm(image, 'output.pgm')
    """
    img = Image.fromarray(image)
    img.save(path)

def read_json(path: str) -> Dict[str,Any]:
    """
        Reads a JSON file.

        Args:
            path (str): Path to the JSON file.

        Return:
            json_dict (Dict[str, Any]): Data read from the JSON file.

        Example:
            data = read_json('example.json')
    """
    try:
        with open(path, "r") as f:
            json_dict = json.load(f)
        return json_dict
    except (FileNotFoundError, IsADirectoryError) as e:
        LOG_ERROR(f"File not found or is a directory: {path}. Error: {e}")
    except ValueError as e:
        LOG_ERROR(f"Value error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to read Json File: {path}. Error: {e}.")
    return None

def write_json(path:str, dict:Dict[str,Any], indent:int=1):
    """
        Writes a dictionary to a JSON file.

        Args:
            path (str): Path to save the JSON file.
            dict (Dict[str, Any]): Dictionary to write to the JSON file.
            indent (int): Indentation level for pretty-printing. Default is 1.

        Example:
            data = {'key': 'value'}
            write_json('output.json', data, indent=2)
    """
    try:
        with open(path,"w") as f:
            json.dump(dict,f, indent=indent)
    except (PermissionError, IsADirectoryError) as e:
        LOG_ERROR(f"Permission error or path is a directory: {path}. Error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to write JSON file: {path}. Error: {e}")

def read_yaml(path:str) -> Dict[str, Any]:
    """
        Reads a YAML file.

        Args:
            path (str): Path to the YAML file.

        Return:
            data_dict (Dict[str, Any]): Data read from the YAML file.

        Example:
            data = read_yaml('example.yaml')
    """
    try:
        with open(path,"r") as f:
            dict = yaml.load(f,Loader=yaml.FullLoader)
        return dict
    except (FileNotFoundError, IsADirectoryError) as e:
        LOG_ERROR(f"File not found or is a directory: {path}. Error: {e}")
    except ValueError as e:
        LOG_ERROR(f"Value error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to read YAML file: {path}. Error: {e}.")
    return None
    
def write_yaml(path:str, dict:Dict[str,Any]):
    """
        Writes a dictionary to a YAML file.

        Args:
            path (str): Path to save the YAML file.
            dict (Dict[str, Any]): Dictionary to write to the YAML file.

        Example:
            data = {'key': 'value'}
            write_yaml('output.yaml', data)
    """
    try:
        with open(path,"w") as f:
            yaml.dump(dict,f)
    except (PermissionError, IsADirectoryError) as e:
        LOG_ERROR(f"Permission error or path is a directory: {path}. Error: {e}")
    except Exception as e:
        LOG_ERROR(f"Failed to write YAML file: {path}. Error: {e}")

def write_video_from_image_paths(image_paths: List[str], output_path:str, fps:int = 30, codec:str = 'mp4v') -> None:
    """
        Write a video from a list of image file paths.

        Args:
            image_paths (List[str]): List of image file paths.
            output_path (str): Output path for the video file.
            fps (int): Frames per second for the video. Default is 30.
            codec (str): FourCC code for the video codec. Default is 'mp4v'.
    """
    if not image_paths:
        raise ValueError("The image paths list is empty")

    first_image = read_image(image_paths[0])
    if first_image is None:
        return
    
    height, width = first_image.shape[0:2]

    # Initialize the video writer
    fourcc = cv.VideoWriter_fourcc(*codec)
    video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_path in image_paths:
        image = read_image(image_path)
        if image is None:
            return
        if image.ndim == 3: image = cv.cvtColor(image,cv.COLOR_RGB2BGR)        
        video_writer.write(image)
    video_writer.release()

def write_video_from_images(images:List[np.ndarray], output_path:str, fps:int = 30, codec:str = 'mp4v') -> None:
    """
        Write a video from a list of images.

        Args:
            images (List[np.ndarray]): List of images.
            output_path (str): Output path for the video file.
            fps (int): Frames per second for the video. Default is 30.
            codec (str): FourCC code for the video codec. Default is 'mp4v'.
    """

    # Get the size from the first image
    first_image = images[0]
    height, width = first_image.shape[0:2]

    # Initialize the video writer
    fourcc = cv.VideoWriter_fourcc(*codec)
    video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        if not isinstance(image, np.ndarray):
            LOG_ERROR("All items in the images list must be numpy arrays.")
            return
        if image.ndim == 3: image = cv.cvtColor(image,cv.COLOR_RGB2BGR)
        video_writer.write(image)
    video_writer.release()
