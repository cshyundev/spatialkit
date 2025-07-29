import unittest
import numpy as np

import cv_utils as cvu
from cv_utils.sol import marker
from cv_utils import FiducialMarkerType
from cv_utils.exceptions import InvalidMarkerTypeError, IncompatibleTypeError
import cv2 as cv

class TestInitDetector(unittest.TestCase):

    def setUp(self):
        # CAMERA
        self.pinhole_cam = cvu.camera.PerspectiveCamera.from_fov(image_size=[640,480],fov=90.)
        self.ds_cam = cvu.camera.DoubleSphereCamera(cam_dict =
                                                       {
                                                        "image_size": [640,480],
                                                        "fov_deg": 180.,
                                                        "principal_point": (318.86121757059797,235.7432966284313),
                                                        "focal_length": (122.5533262583915,121.79271712838818),
                                                        "xi": -0.02235598738719681,
                                                        "alpha": 0.562863934931952
                                                       })
        self.equi_cam = cvu.camera.EquirectangularCamera.from_image_size([5952,2976])

        self.marker_size = 0.05 # 0.05m
        self.image_path = "../../../images/markers/marker_test_image.jpg"

        self.equi_image = cvu.io.read_image(self.image_path)

        self.ds_image = cvu.geom_utils.transition_camera_view(self.equi_image,
                                                              self.equi_cam,
                                                              self.ds_cam)
        self.pinhole_image = cvu.geom_utils.transition_camera_view(self.equi_image,
                                                                    self.equi_cam,
                                                                    self.pinhole_cam)

        self.aruco_type = FiducialMarkerType.ARUCO_6X6_250
        self.apriltag_type = FiducialMarkerType.APRILTAG_36H11
        self.stag_type = FiducialMarkerType.STAG_HD21

        self.aruco_ids = [0,1]
        self.apriltag_ids = [0,1]
        self.stag_ids = [0,1]

    def test_opencv_detector_initialization(self):
        # OpenCV MarkerDetector Initialization Test
        VALID_ARUCO_MARKER_TYPE = \
        [   
            # ARUCO
            FiducialMarkerType.ARUCO_4X4_50,
            FiducialMarkerType.ARUCO_4X4_100,
            FiducialMarkerType.ARUCO_4X4_250,
            FiducialMarkerType.ARUCO_4X4_1000,
            FiducialMarkerType.ARUCO_5X5_50,
            FiducialMarkerType.ARUCO_5X5_100,
            FiducialMarkerType.ARUCO_5X5_250,
            FiducialMarkerType.ARUCO_5X5_1000,
            FiducialMarkerType.ARUCO_6X6_50,
            FiducialMarkerType.ARUCO_6X6_100,
            FiducialMarkerType.ARUCO_6X6_250,
            FiducialMarkerType.ARUCO_6X6_1000,
            FiducialMarkerType.ARUCO_7X7_50,
            FiducialMarkerType.ARUCO_7X7_100,
            FiducialMarkerType.ARUCO_7X7_250,
            FiducialMarkerType.ARUCO_7X7_1000,
            FiducialMarkerType.ARUCO_ORIGINAL,
            FiducialMarkerType.ARUCO_MIP_36H12,

            # APRIL
            FiducialMarkerType.APRILTAG_16H5,
            FiducialMarkerType.APRILTAG_25H9,
            FiducialMarkerType.APRILTAG_36H10,
            FiducialMarkerType.APRILTAG_36H11,
        ]

        for marker_type in FiducialMarkerType:
            # initialization Test
            if marker_type in VALID_ARUCO_MARKER_TYPE: 
                try:
                    marker.OpenCVMarkerDetector(self.pinhole_cam,
                            self.marker_size,
                            marker_type)
                except:
                    # Initialization Failed
                    self.fail("OPENCV Detector Initialization Failed.")
            else:
                with self.assertRaises(InvalidMarkerTypeError):
                    marker.OpenCVMarkerDetector(self.pinhole_cam,
                            self.marker_size,
                            marker_type)

    def test_apriltag_detector_initialization(self):
        # AprilTag MarkerDetector Initialization Test
        
        VALID_APRIL_MARKER_TYPE = \
        [   
            # APRIL
            FiducialMarkerType.APRILTAG_16H5,
            FiducialMarkerType.APRILTAG_25H9,
            FiducialMarkerType.APRILTAG_36H10,
            FiducialMarkerType.APRILTAG_36H11,
            FiducialMarkerType.APRILTAG_CUSTOM48H12,
            FiducialMarkerType.APRILTAG_STANDARD41H12,
            FiducialMarkerType.APRILTAG_STANDARD52H13,
        ]

        for marker_type in FiducialMarkerType:
            # initialization Test
            if marker_type in VALID_APRIL_MARKER_TYPE: 
                try:
                    marker.AprilTagMarkerDetector(self.pinhole_cam,
                            self.marker_size,
                            marker_type)
                except:
                    # Initialization Failed
                    self.fail("AprilTag Detector Initialization Failed.")
            else:
                with self.assertRaises(InvalidMarkerTypeError):
                    marker.AprilTagMarkerDetector(self.pinhole_cam,
                            self.marker_size,
                            marker_type)

    def test_stag_detector_initialization(self):

        VALID_STAG_MARKER_TYPE = \
        [   
            # STAG
            FiducialMarkerType.STAG_HD11,
            FiducialMarkerType.STAG_HD13,
            FiducialMarkerType.STAG_HD15,
            FiducialMarkerType.STAG_HD17,
            FiducialMarkerType.STAG_HD19,
            FiducialMarkerType.STAG_HD21,
            FiducialMarkerType.STAG_HD23,
        ]

        for marker_type in FiducialMarkerType:
            # initialization Test
            if marker_type in VALID_STAG_MARKER_TYPE: 
                try:
                    marker.STagMarkerDetector(self.pinhole_cam,
                            self.marker_size,
                            marker_type)
                except:
                    # Initialization Failed
                    self.fail("Stag Detector Initialization Failed.")
            else:
                with self.assertRaises(InvalidMarkerTypeError):
                    marker.STagMarkerDetector(self.pinhole_cam,
                            self.marker_size,
                            marker_type)

    def test_initialization_arguments_error(self):

        with self.assertRaises(IncompatibleTypeError):
            # wrong cam type
            marker.OpenCVMarkerDetector(None, self.marker_size, self.aruco_type)
        
        with self.assertRaises(InvalidMarkerTypeError):
            # wrong marker_size type
            marker.OpenCVMarkerDetector(self.pinhole_cam, None, self.aruco_type)

        with self.assertRaises(InvalidMarkerTypeError):
            # wrong marker_type
            marker.OpenCVMarkerDetector(self.pinhole_cam, self.marker_size, None)
        
        with self.assertRaises(InvalidMarkerTypeError):
            # negative marker size
            marker.OpenCVMarkerDetector(self.pinhole_cam,-1., self.aruco_type)

class TestDetectMarker(unittest.TestCase):
    
    def setUp(self):
        # CAMERA
        self.pinhole_cam = cvu.camera.PerspectiveCamera.from_fov(image_size=[640,480],fov=90.)
        self.ds_cam = cvu.camera.DoubleSphereCamera(cam_dict =
                                                       {
                                                        "image_size": [640,480],
                                                        "fov_deg": 180.,
                                                        "principal_point": (318.86121757059797,235.7432966284313),
                                                        "focal_length": (122.5533262583915,121.79271712838818),
                                                        "xi": -0.02235598738719681,
                                                        "alpha": 0.562863934931952
                                                       })
        self.equi_cam = cvu.camera.EquirectangularCamera.from_image_size([5952,2976])

        self.marker_size = 0.05 # 0.05m
        self.image_path = "../../../images/markers/marker_test_image.jpg"

        self.equi_image = cvu.io.read_image(self.image_path)

        self.ds_image = cvu.geom_utils.transition_camera_view(self.equi_image,
                                                              self.equi_cam,
                                                              self.ds_cam)
        self.pinhole_image = cvu.geom_utils.transition_camera_view(self.equi_image,
                                                                    self.equi_cam,
                                                                    self.pinhole_cam)

        self.aruco_type = FiducialMarkerType.ARUCO_6X6_250
        self.apriltag_type = FiducialMarkerType.APRILTAG_36H11
        self.stag_type = FiducialMarkerType.STAG_HD21

        self.aruco_ids = [0,1]
        self.apriltag_ids = [0,1]
        self.stag_ids = [0,1]     

    def test_detect_marker_argument(self):
        opencv_detector = marker.OpenCVMarkerDetector(
            self.pinhole_cam,
            self.marker_size,
            self.aruco_type
        )

        apriltag_detector = marker.AprilTagMarkerDetector(
            self.pinhole_cam,
            self.marker_size,
            self.apriltag_type
        )

        stag_detector = marker.STagMarkerDetector(
            self.pinhole_cam,
            self.marker_size,
            self.stag_type
        )

        # invalid image type
        with self.assertRaises(IncompatibleTypeError):
            opencv_detector.detect_marker(None)
        with self.assertRaises(IncompatibleTypeError):
            apriltag_detector.detect_marker(None)
        with self.assertRaises(IncompatibleTypeError):
            stag_detector.detect_marker(None)

    def test_equi_cam_opencv_detect_marker(self):
        opencv_detector = marker.OpenCVMarkerDetector(
            self.equi_cam,
            self.marker_size,
            self.aruco_type
            )
        detected_aruco_markers = opencv_detector.detect_marker(self.equi_image,False)        
        
        self.assertEqual(len(self.aruco_ids), len(detected_aruco_markers))

        for m in detected_aruco_markers:
            self.assertEqual(True, m.id in self.aruco_ids)

        gray_image = cv.cvtColor(self.equi_image,cv.COLOR_RGB2GRAY)

        detected_aruco_markers = opencv_detector.detect_marker(gray_image,False)        
        for m in detected_aruco_markers:
            self.assertEqual(True, m.id in self.aruco_ids)  
        self.assertEqual(len(self.aruco_ids) , len(detected_aruco_markers))

    def test_equi_cam_apriltag_detect_marker(self):
        apriltag_detector = marker.AprilTagMarkerDetector(
            self.equi_cam,
            self.marker_size,
            self.apriltag_type
            )
        
        detected_april_markers = apriltag_detector.detect_marker(self.equi_image,False)        

        self.assertEqual(len(self.apriltag_ids) , len(detected_april_markers))

        for m in detected_april_markers:
            self.assertEqual(True, m.id in self.apriltag_ids)  

        gray_image = cv.cvtColor(self.equi_image,cv.COLOR_RGB2GRAY)
        detected_april_markers = apriltag_detector.detect_marker(gray_image,False)        

        for m in detected_april_markers:
            self.assertEqual(True, m.id in self.apriltag_ids)     
        self.assertEqual(len(self.apriltag_ids) , len(detected_april_markers))

    def test_equi_cam_stag_detect_marker(self):

        stag_detector = marker.STagMarkerDetector(
            self.equi_cam,
            self.marker_size,
            self.stag_type
            )

        detected_stag_markers = stag_detector.detect_marker(self.equi_image,False)
        self.assertEqual(len(self.stag_ids), len(detected_stag_markers))

        for m in detected_stag_markers:
            self.assertEqual(True, m.id in self.stag_ids)
        
        # GrayScale
        gray_image = cv.cvtColor(self.equi_image,cv.COLOR_RGB2GRAY)
        detected_stag_markers = stag_detector.detect_marker(gray_image,False)
   
        for m in detected_stag_markers:
            self.assertEqual(True, m.id in self.stag_ids)

        self.assertEqual(len(self.stag_ids), len(detected_stag_markers))

    def test_pinhole_cam_opencv_detect_marker(self):
        opencv_detector = marker.OpenCVMarkerDetector(
            self.pinhole_cam,
            self.marker_size,
            self.aruco_type
            )
        detected_aruco_markers = opencv_detector.detect_marker(self.pinhole_image,False)        

        self.assertEqual(len(self.aruco_ids) , len(detected_aruco_markers))
        for m in detected_aruco_markers:
            self.assertEqual(True, m.id in self.aruco_ids)    

        gray_image = cv.cvtColor(self.pinhole_image,cv.COLOR_RGB2GRAY)
        detected_aruco_markers = opencv_detector.detect_marker(gray_image,False)        

        for m in detected_aruco_markers:
            self.assertEqual(True, m.id in self.aruco_ids)  
        self.assertEqual(len(self.aruco_ids) , len(detected_aruco_markers))

    def test_pinhole_cam_april_detect_marker(self):
        apriltag_detector = marker.AprilTagMarkerDetector(
            self.pinhole_cam,
            self.marker_size,
            self.apriltag_type
            )
        detected_april_markers = apriltag_detector.detect_marker(self.pinhole_image,False)        
        
        self.assertEqual(len(self.apriltag_ids) , len(detected_april_markers))

        for m in detected_april_markers:
            self.assertEqual(True, m.id in self.apriltag_ids)  
        
        gray_image = cv.cvtColor(self.pinhole_image,cv.COLOR_RGB2GRAY)
        
        detected_april_markers = apriltag_detector.detect_marker(gray_image,False)        
        for m in detected_april_markers:
            self.assertEqual(True, m.id in self.apriltag_ids)   
    
        self.assertEqual(len(self.apriltag_ids) , len(detected_april_markers))
    
    def test_pinhole_cam_stag_detect_marker(self):

        stag_detector = marker.STagMarkerDetector(
            self.pinhole_cam,
            self.marker_size,
            self.stag_type
            )

        detected_stag_markers = stag_detector.detect_marker(self.pinhole_image,False)
        self.assertEqual(len(self.stag_ids), len(detected_stag_markers))
      
        for m in detected_stag_markers:
            self.assertEqual(True, m.id in self.stag_ids)
        
        # GrayScale
        gray_image = cv.cvtColor(self.pinhole_image,cv.COLOR_RGB2GRAY)
        detected_stag_markers = stag_detector.detect_marker(gray_image,False)
      
        for m in detected_stag_markers:
            self.assertEqual(True, m.id in self.stag_ids)
        self.assertEqual(len(self.stag_ids), len(detected_stag_markers))

    def test_ds_cam_opencv_detect_marker(self):
        opencv_detector = marker.OpenCVMarkerDetector(
            self.ds_cam,
            self.marker_size,
            self.aruco_type
            )
        detected_aruco_markers = opencv_detector.detect_marker(self.ds_image,False)        
        self.assertEqual(len(self.aruco_ids) , len(detected_aruco_markers))
        for m in detected_aruco_markers:
            self.assertEqual(True, m.id in self.aruco_ids)  

        gray_image = cv.cvtColor(self.equi_image,cv.COLOR_RGB2GRAY)
        detected_aruco_markers = opencv_detector.detect_marker(gray_image,False)        
        for m in detected_aruco_markers:
            self.assertEqual(True, m.id in self.aruco_ids)  
        self.assertEqual(len(self.aruco_ids) , len(detected_aruco_markers))

    def test_ds_cam_apriltag_detect_marker(self):
        apriltag_detector = marker.AprilTagMarkerDetector(
            self.ds_cam,
            self.marker_size,
            self.apriltag_type
            )
        detected_april_markers = apriltag_detector.detect_marker(self.ds_image,False)        
        self.assertEqual(len(self.apriltag_ids) , len(detected_april_markers))
        for m in detected_april_markers:
            self.assertEqual(True, m.id in self.apriltag_ids)  

        gray_image = cv.cvtColor(self.ds_image,cv.COLOR_RGB2GRAY)
        detected_april_markers = apriltag_detector.detect_marker(gray_image,False)        
        for m in detected_april_markers:
            self.assertEqual(True, m.id in self.apriltag_ids) 

        self.assertEqual(len(self.apriltag_ids) , len(detected_april_markers))

    def test_ds_cam_stag_detect_marker(self):
        stag_detector = marker.STagMarkerDetector(
            self.ds_cam,
            self.marker_size,
            self.stag_type
            )
        detected_stag_markers = stag_detector.detect_marker(self.ds_image,False)
        self.assertEqual(len(self.stag_ids), len(detected_stag_markers))
      
        for m in detected_stag_markers:
            self.assertEqual(True, m.id in self.stag_ids)
        
        # GrayScale
        gray_image = cv.cvtColor(self.ds_image,cv.COLOR_RGB2GRAY)
        detected_stag_markers = stag_detector.detect_marker(gray_image,False)
        for m in detected_stag_markers:
            self.assertEqual(True, m.id in self.stag_ids)
        self.assertEqual(len(self.stag_ids), len(detected_stag_markers)) 
    
class TestDetectorPoseEstimation(unittest.TestCase):
    
    def setUp(self):
        # CAMERA
        self.pinhole_cam = cvu.camera.PerspectiveCamera.from_fov(image_size=[640,480],fov=90.)
        self.ds_cam = cvu.camera.DoubleSphereCamera(cam_dict =
                                                       {
                                                        "image_size": [640,480],
                                                        "fov_deg": 180.,
                                                        "principal_point": (318.86121757059797,235.7432966284313),
                                                        "focal_length": (122.5533262583915,121.79271712838818),
                                                        "xi": -0.02235598738719681,
                                                        "alpha": 0.562863934931952
                                                       })
        self.equi_cam = cvu.camera.EquirectangularCamera.from_image_size([5952,2976])

        self.marker_size = 0.05 # 0.05m
        self.image_path = "../../../images/markers/marker_test_image.jpg"

        self.equi_image = cvu.io.read_image(self.image_path)

        self.ds_image = cvu.geom_utils.transition_camera_view(self.equi_image,
                                                              self.equi_cam,
                                                              self.ds_cam)
        self.pinhole_image = cvu.geom_utils.transition_camera_view(self.equi_image,
                                                                    self.equi_cam,
                                                                    self.pinhole_cam)

        self.aruco_type = FiducialMarkerType.ARUCO_6X6_250
        self.apriltag_type = FiducialMarkerType.APRILTAG_36H11
        self.stag_type = FiducialMarkerType.STAG_HD21

        self.aruco_ids = [0,1]
        self.apriltag_ids = [0,1]
        self.stag_ids = [0,1]
    
    def test_equi_cam_opencv_pose_estimation(self):
        opencv_detector = marker.OpenCVMarkerDetector(
            self.equi_cam,
            self.marker_size,
            self.aruco_type
            )
        detected_markers = opencv_detector.detect_marker(self.equi_image,True)        

        m2m_color = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        """
        Difference
         - translation(m): 0.06
         - angle(deg): 90 
        """
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.rot.so3()),np.pi/2.,delta=np.pi/16)

        gray_image = cv.cvtColor(self.equi_image,cv.COLOR_RGB2GRAY)
        detected_markers = opencv_detector.detect_marker(gray_image,True)
        
        m2m_gray = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.rot.so3()),np.pi/2.,delta=np.pi/16)

    def test_equi_cam_apriltag_pose_estimation(self):
        april_detector = marker.AprilTagMarkerDetector(
            self.equi_cam,
            self.marker_size,
            self.apriltag_type
            )
        detected_markers = april_detector.detect_marker(self.equi_image,True)        

        m2m_color = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        """
        Difference
         - translation(m): 0.06
         - angle(deg): 90 
        """
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.rot.so3()),np.pi/2.,delta=np.pi/16)

        gray_image = cv.cvtColor(self.equi_image,cv.COLOR_RGB2GRAY)
        detected_markers = april_detector.detect_marker(gray_image,True)
        
        m2m_gray = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.rot.so3()),np.pi/2.,delta=np.pi/16)

    def test_equi_cam_stag_pose_estimation(self):
        stag_detector = marker.STagMarkerDetector(
            self.equi_cam,
            self.marker_size,
            self.stag_type
            )
        detected_markers = stag_detector.detect_marker(self.equi_image,True)        

        m2m_color = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        """
        Difference
         - translation(m): 0.06
         - angle(deg): 90 
        """
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.rot.so3()),np.pi/2.,delta=np.pi/16)


        gray_image = cv.cvtColor(self.equi_image,cv.COLOR_RGB2GRAY)
        detected_markers = stag_detector.detect_marker(gray_image,True)
        
        m2m_gray = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.rot.so3()),np.pi/2.,delta=np.pi/16)

    def test_pinhole_cam_opencv_pose_estimation(self):
        opencv_detector = marker.OpenCVMarkerDetector(
            self.pinhole_cam,
            self.marker_size,
            self.aruco_type
            )
        detected_markers = opencv_detector.detect_marker(self.pinhole_image,True)        

        m2m_color = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        """
        Difference
         - translation(m): 0.06
         - angle(deg): 90 
        """
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.rot.so3()),np.pi/2.,delta=np.pi/16)

        gray_image = cv.cvtColor(self.pinhole_image,cv.COLOR_RGB2GRAY)
        detected_markers = opencv_detector.detect_marker(gray_image,True)
        
        m2m_gray = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.rot.so3()),np.pi/2.,delta=np.pi/16)

    def test_pinhole_cam_april_pose_estimation(self):
        april_detector = marker.AprilTagMarkerDetector(
            self.pinhole_cam,
            self.marker_size,
            self.apriltag_type
            )
        detected_markers = april_detector.detect_marker(self.pinhole_image,True)        

        m2m_color = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        """
        Difference
         - translation(m): 0.06
         - angle(deg): 90 
        """
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.rot.so3()),np.pi/2.,delta=np.pi/16)


        gray_image = cv.cvtColor(self.pinhole_image,cv.COLOR_RGB2GRAY)
        detected_markers = april_detector.detect_marker(gray_image,True)
        
        m2m_gray = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.rot.so3()),np.pi/2.,delta=np.pi/16)
    
    def test_pinhole_cam_stag_pose_estimation(self):
        april_detector = marker.STagMarkerDetector(
            self.pinhole_cam,
            self.marker_size,
            self.stag_type
            )
        detected_markers = april_detector.detect_marker(self.pinhole_image,True)        

        m2m_color = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        """
        Difference
         - translation(m): 0.06
         - angle(deg): 90 
        """
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.rot.so3()),np.pi/2.,delta=np.pi/16)


        gray_image = cv.cvtColor(self.pinhole_image,cv.COLOR_RGB2GRAY)
        detected_markers = april_detector.detect_marker(gray_image,True)
        
        m2m_gray = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.rot.so3()),np.pi/2.,delta=np.pi/16)

    def test_ds_cam_opencv_pose_estimation(self):
        opencv_detector = marker.OpenCVMarkerDetector(
            self.ds_cam,
            self.marker_size,
            self.aruco_type
            )
        detected_markers = opencv_detector.detect_marker(self.ds_image,True)        

        m2m_color = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        """
        Difference
         - translation(m): 0.06
         - angle(deg): 90 
        """
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.rot.so3()),np.pi/2.,delta=np.pi/16)

        gray_image = cv.cvtColor(self.ds_image,cv.COLOR_RGB2GRAY)
        detected_markers = opencv_detector.detect_marker(gray_image,True)
        
        m2m_gray = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.rot.so3()),np.pi/2.,delta=np.pi/16)

    def test_ds_cam_apriltag_pose_estimation(self):
        apriltag_detector = marker.AprilTagMarkerDetector(
            self.ds_cam,
            self.marker_size,
            self.apriltag_type
            )
        detected_markers = apriltag_detector.detect_marker(self.ds_image,True)        

        m2m_color = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        """
        Difference
         - translation(m): 0.06
         - angle(deg): 90 
        """
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.rot.so3()),np.pi/2.,delta=np.pi/16)


        gray_image = cv.cvtColor(self.ds_image,cv.COLOR_RGB2GRAY)
        detected_markers = apriltag_detector.detect_marker(gray_image,True)
        
        m2m_gray = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.rot.so3()),np.pi/2.,delta=np.pi/16)

    def test_ds_cam_stag_pose_estimation(self):
        stag_detector = marker.STagMarkerDetector(
            self.ds_cam,
            self.marker_size,
            self.stag_type
            )
        detected_markers = stag_detector.detect_marker(self.ds_image,True)        

        m2m_color = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        """
        Difference
         - translation(m): 0.06
         - angle(deg): 90 
        """
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_color.rot.so3()),np.pi/2.,delta=np.pi/16)


        gray_image = cv.cvtColor(self.ds_image,cv.COLOR_RGB2GRAY)
        detected_markers = stag_detector.detect_marker(gray_image,True)
        
        m2m_gray = detected_markers[0].marker2cam.inverse() * \
            detected_markers[1].marker2cam # marker -> cam * cam -> marker = marker -> marker
        
        # Translation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.t),0.06,delta=0.01)
        # Rotation
        self.assertAlmostEqual(cvu.ops.norm(m2m_gray.rot.so3()),np.pi/2.,delta=np.pi/16)

if __name__ == '__main__':
    unittest.main()