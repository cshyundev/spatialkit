import os.path as osp
import numpy as np
from src.camera import *
from src.plot import show_image, draw_polygon, show_two_images
from src.file_utils import read_image
from src.image_transform import *
from absl import app
import matplotlib.pyplot as plt
import argparse

def main(argv):
    """
    Homography Test

    To conduct the test, you need:
        1. images.
        2. output image size
    
    Test Process
        1. Set image path and the necessary information.
        2. Select 4 points.
        3. Show the result.
    """   

    ###################################################
    # Fill the value below!!!
    
    # Set Parameters
    parser = argparse.ArgumentParser(description='Homography Test')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('output_width', type=int, help='Width of the output image')
    parser.add_argument('output_height', type=int, help='Height of the output image')
    args = parser.parse_args(argv[1:])

    image_path = args.image_path
    output_width = args.output_width
    output_height = args.output_height

    ###################################################
    image = read_image(image_path)    

    # 2. choose 4 points each 
    plt.imshow(image)
    plt.title("Take 4 dots counterclockwise from the top left corner")
    pts = plt.ginput(4)
    pts = np.int32(pts)
    plt.close()

    output_pts = [(0,0),(0, output_height),(output_width,output_height),(output_width,0)]

    H = compute_homography(pts,output_pts)

    print("Homography Matrix")
    print(H)

    output_image = apply_transform(image, H, (output_width,output_height))
    original_image = draw_polygon(image,pts)

    show_two_images(original_image,output_image,"Original Image","Warped Image")
    
    return

 

if __name__ == "__main__":
    app.run(main)