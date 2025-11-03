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