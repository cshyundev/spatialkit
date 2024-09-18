import cv_utils as cvu

image = cvu.io.read_image("./Frame_00009_FinalColor.png")[:,:,:3]

spherical_camera = cvu.o3dcomp.create_spherical_camera_indicator_frame(image=image)

cvu.o3dutils.visualize_geometries(spherical_camera)
