"""
Module Name: common.py

Description:
This module provides shorthand notations for commonly used Open3D geometry classes. 
These notations simplify the usage of Open3D's geometry functionalities in the code, 
making it easier to work with 3D data structures such as geometries, triangle meshes, 
line sets, and point clouds.

Constants:
- O3dGeometry: Alias for o3d.geometry.Geometry.
- O3dTriMesh: Alias for o3d.geometry.TriangleMesh.
- O3dLineSet: Alias for o3d.geometry.LineSet.
- O3dPCD: Alias for o3d.geometry.PointCloud.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1

License: MIT LICENSE
"""

import open3d as o3d

# Notation
O3dGeometry = o3d.geometry.Geometry
O3dTriMesh = o3d.geometry.TriangleMesh
O3dLineSet = o3d.geometry.LineSet
O3dPCD = o3d.geometry.PointCloud
