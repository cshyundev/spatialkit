from module.rotation import Rotation
from module.hybrid_math import transpose2d
from module.camera import PinholeCamera, Camera
from module.geometry import compute_depth_to_normal, compute_pca
import numpy as np

cam = Camera.create_cam(
    cam_dict={
        "cam_type": "PINHOLE",
        "image_size": [384,384],
        "focal_length": [338.82354736328125,338.8235168457031],
        "principal_point": [191.7176513671875,191.71763610839844],
        "skew": 0.,
        "radial": [0.,0.,0.],
        "tangential": [0.,0.],
    }
)

camtoworld = np.array([
    [
     -0.35449257493019104,
     0.35950005054473877,
     -0.8631886839866638,
     0.12671881914138794
    ],
    [
     0.9350588321685791,
     0.1362909972667694,
     -0.32724571228027344,
     -0.11779534071683884
    ],
    [
     -1.8574656257541733e-09,
     -0.9231383800506592,
     -0.38446786999702454,
     0.14816303551197052
    ],
    [
     0.0,
     0.0,
     0.0,
     1.0
    ]
   ], dtype=np.float32)

r = Rotation.from_mat3(camtoworld[:3,:3])
rays = cam.get_rays() 
# print(rays[0,100])
direction = r.apply_pts3d(rays)
print(direction[100,:])