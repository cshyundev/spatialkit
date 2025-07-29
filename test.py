import cv2
import numpy as np
import cv_utils as cvu

# depth map 파일 읽기 (.npy 포맷)
depth_map = np.load("depth_map.npy")  # 실제 파일 경로로 수정 필요

# 360도 카메라 모델 생성 (equirectangular)
height, width = depth_map.shape
cam = cvu.camera.EquirectangularCamera.from_image_size((width, height))

# depth map을 point cloud로 변환 (MSI 타입 - spherical distance)
points_3d = cvu.geom_utils.convert_depth_to_point_cloud(depth_map, cam, map_type="MSI")

# point cloud 시각화
pcd = cvu.vis3d.o3dutils.create_point_cloud(points_3d)
cvu.vis3d.o3dutils.visualize_geometries([pcd], "360 Camera Point Cloud")
