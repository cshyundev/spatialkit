import cv2
import numpy as np
from module.geometry import compute_depth_to_normal, compute_pca



data = np.arange(5*5*3*9).reshape(5,5,3,9)
eig_vec, eig_val = compute_pca(data)

print(eig_vec.shape)
