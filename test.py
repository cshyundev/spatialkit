import numpy as np
from module.plot import *
from module.io import write_image


def main():
    image = np.zeros(640*480*3, dtype=np.uint8)
    image = image.reshape(640,480,3) # H,W,3
    
    pt2d = [(380,200), (250,380),(580,20),(30,150)]
    image = draw_circles(image, pt2d, 1)
    write_image(image, "./test.png")
    

if __name__ == "__main__":
    main() 