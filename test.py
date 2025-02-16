import cv2
import numpy as np
import cv_utils as cvu


equi_cam = cvu.camera.EquirectangularCamera.from_image_size([5952,2976])
pinhole_cam = cvu.camera.PerspectiveCamera.from_fov(image_size=[640,480],fov=90.)
ds_cam = cvu.camera.DoubleSphereCamera(cam_dict = {
                     "image_size": [640,480],
                     "fov_deg": 180.,
                     "principal_point": (318.86121757059797,235.7432966284313),
                     "focal_length": (122.5533262583915,121.79271712838818),
                     "xi": -0.02235598738719681,
                     "alpha": 0.562863934931952
                    })
marker_size = 0.05

equi_aruco_detector = cvu.marker.OpenCVMarkerDetector(equi_cam,marker_size,cvu.FiducialMarkerType.ARUCO_6X6_250)
pinhole_aruco_detector = cvu.marker.OpenCVMarkerDetector(pinhole_cam,marker_size,cvu.FiducialMarkerType.ARUCO_6X6_250)
ds_aruco_detector = cvu.marker.OpenCVMarkerDetector(ds_cam,marker_size,cvu.FiducialMarkerType.ARUCO_6X6_250)

equi_image = cvu.io.read_image("./images/markers/marker_test_image.jpg")
pinhole_image = cvu.geom.transition_camera_view(equi_image,equi_cam,pinhole_cam)
ds_image = cvu.geom.transition_camera_view(equi_image,equi_cam,ds_cam)

# Draw Test, check only aruco

# Equi image, Color
# markers = equi_aruco_detector.detect_marker(equi_image)
# out_image = equi_aruco_detector.draw_markers(equi_image,markers,True,3,0.04)

# Pinhole image, Color
# markers = pinhole_aruco_detector.detect_marker(pinhole_image)
# out_image = pinhole_aruco_detector.draw_markers(pinhole_image,markers,True,3,0.04)
# cvu.vis.show_image(out_image,"PINHOLE")

# DS image, color
markers = ds_aruco_detector.detect_marker(ds_image)
out_image = ds_aruco_detector.draw_markers(ds_image,markers,True,3,0.04)
cvu.vis.show_image(out_image,"DS")

# aruco_markers = aruco_detector.detect_marker(equi_image,True)
# detected_aruco_image = aruco_detector.draw_markers(equi_image,aruco_markers,True)
# cvu.vis.show_image(detected_aruco_image,"ARUCO")
# cvu.vis.show_image(pinhole_image,"PINHOLE")
