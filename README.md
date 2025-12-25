# spatialkit

**Current Version:** 0.3.3 \
**Last Updated:** December 25, 2025

## License

`spatialkit` is freely available under the MIT License. For details, please refer to the [LICENSE](LICENSE) file.

## Introduction

`spatialkit` is a personal library designed to support research and development in computer vision and robotics. This library provides various features and functions necessary for developing and testing computer vision algorithms, including 3D vision. `spatialkit` includes tools and features that help users effectively process and analyze complex data.

### Key Features

- **Prototyping and Research Test Code**: Provides frequently needed test code during the development and testing stages of computer vision algorithms.
- **PyTorch Support**: Offers functions and classes for handling PyTorch tensors to process and analyze 3D data.
- **Integration of Major Libraries**: Provides simplified usage by integrating core features of popular libraries such as NumPy, OpenCV, SciPy, and PyTorch.

### Recommended For

- **Computer Vision and Robotics Beginners**: The code is simpler and easier to understand compared to other libraries, especially facilitating code-level understanding in 3D tasks.
- **3D Vision Researchers**: Provides PyTorch-based code and test code, which can shorten the programming process in various 3D vision research, including deep learning.

## Caution

- **Performance and Efficiency Issues**: Some features may be slower than existing OpenCV or other libraries, so caution should be exercised when used in research and development where optimization and speed are important.
- **Limited PyTorch Support in Some Functions**: Certain functions can only receive NumPy input. This is due to the complexity of implementing them efficiently in PyTorch and the limited practical need for PyTorch support in those specific cases.

## Getting Started (Development Mode)

### Requirements

- **Python Version**: Python >= 3.10
- **Package Manager**: uv (recommended) or pip
- **Dependencies**: All required dependencies will be automatically installed during installation

### Installation Using uv (Recommended)

1. Install uv (if needed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository
   ```bash
   git clone https://github.com/cshyundev/spatialkit.git
   cd spatialkit
   ```

3. Create virtual environment and install dependencies
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   uv pip install -e .
   ```

### Installation Using pip

1. Clone the repository
   ```bash
   git clone https://github.com/cshyundev/spatialkit.git
   ```
2. Install in development mode
     ```bash
    cd spatialkit
    pip install -e .
    ```

## Simple Examples

### Unified NumPy/PyTorch Interface

`spatialkit` provides a unified interface for both NumPy and PyTorch, with geometry classes that preserve input types:

```python
import spatialkit as sp
import numpy as np
import torch

# Create 3D points (3, N) - works with both NumPy and PyTorch
pts_np = np.random.rand(3, 100)
pts_torch = torch.rand(3, 100)

# Create rotation from RPY (Roll-Pitch-Yaw)
rot = sp.Rotation.from_rpy(np.array([0, np.pi/4, 0]))  # 45° pitch

# Apply rotation using multiplication operator - type is preserved
rotated_np = rot * pts_np        # NumPy in → NumPy out
rotated_torch = rot * pts_torch  # Torch in → Torch out

print(type(rotated_np))    # <class 'numpy.ndarray'>
print(type(rotated_torch)) # <class 'torch.Tensor'>

# Create transform (rotation + translation)
tf = sp.Transform(t=np.array([1., 0., 0.]), rot=rot)

# Apply transform using multiplication operator
pts_transformed = tf * pts_np

# Chain transformations
tf_combined = tf * tf.inverse()  # Returns identity transform

print(tf_combined.mat44())
```

### Multiple Camera Models with Image Warping

`spatialkit` supports various camera models and enables image warping between different camera models:

| 360° Cubemap | Perspective | Fisheye | Double Sphere (180° FOV) |
|:------------:|:-----------:|:-------:|:------------------------:|
| ![360](assets/cubemap_360.png) | ![Perspective](assets/camera_perspective.png) | ![Fisheye](assets/camera_fisheye.png) | ![DoubleSphere](assets/camera_doublesphere.png) |

```python
import spatialkit as sp
from spatialkit.imgproc.synthesis import transition_camera_view
from spatialkit.vis2d import show_image

equirect_360 = sp.io.read_image("assets/cubemap_360.png")
img_size = [640, 480]

# 1. Equirectangular camera (source 360 image)
equirect_cam = sp.EquirectangularCamera.from_image_size([1024, 512])

# 2. Perspective (Pinhole) camera
K_perspective = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
perspective_cam = sp.PerspectiveCamera.from_K(K_perspective, img_size)

# 3. Fisheye camera (OpenCV model)
K_fisheye = [[300, 0, 320], [0, 300, 240], [0, 0, 1]]
D_fisheye = [-0.042595202508066574, 0.031307765215775184, -0.04104704724832258, 0.015343014605793324]
fisheye_cam = sp.OpenCVFisheyeCamera.from_K_D(K_fisheye, img_size, D_fisheye)

# 4. Double Sphere camera (180° FOV)
double_sphere_cam = sp.DoubleSphereCamera(
    {
        'image_size': [640, 480],
        'cam_type': 'DOUBLESPHERE',
        'principal_point': [318.86121757059797, 235.7432966284313],
        'focal_length': [122.5533262583915, 121.79271712838818],
        'xi': -0.02235598738719681,
        'alpha': 0.562863934931952,
        'fov_deg': 180.0
    }
)

# Warp image between camera models
perspective_warped = transition_camera_view(equirect_360, equirect_cam, perspective_cam)
fisheye_warped = transition_camera_view(equirect_360, equirect_cam, fisheye_cam)
double_sphere_warped = transition_camera_view(equirect_360, equirect_cam, double_sphere_cam)

# Display results
show_image(equirect_360, title="Equirectangular 360 Image")
show_image(perspective_warped, title="Perspective Camera View")
show_image(fisheye_warped, title="Fisheye Camera View")
show_image(double_sphere_warped, title="Double Sphere Camera View")
```

