import numpy as np
from cv_utils.utils.io import read_yaml, read_image
from cv_utils.geom import Transform, Rotation
from cv_utils.geom import compute_fundamental_matrix, compute_essential_matrix
from cv_utils.geom.rotation import RotType
from cv_utils.geom.camera import PerpectiveCamera, OpenCVFisheyeCamera, RadialCamera
from cv_utils.common.logger import *
from cv_utils.utils.point_selector import DoubleImagesPointSelector
import absl.app as app


def parse_yaml(file_path):
    
    config = read_yaml(file_path)
    
    images_info = []

    for image_info in config.get('images'):
        try:
            path = image_info['image']['path']
            camera_model = image_info['image']['camera_model']
            intrinsic = image_info['image']['intrinsic']
            extrinsic = image_info['image']['extrinsic']
            distortion = image_info['image']['distortion']
        except KeyError as e:
            LOG_ERROR(f"Missing key in YAML structure: {e}")
            return None
        # Parse intrinsic parameters
        try:
            K = [[intrinsic['fx'],intrinsic['skew'],intrinsic['cx']],
                 [0., intrinsic['fy'], intrinsic['cy']],
                 [0., 0., 1.]]
            image_size = (intrinsic['width'],intrinsic['height'])
        except KeyError as e:
            LOG_ERROR(f"Missing intrinsic parameter: {e}")
            return None
        # Parse distortion parameters
        try:
            if camera_model == 'pinhole':
                distortion_params = [distortion['k1'],distortion['k2'],distortion['p1'],distortion['p2'],distortion['k3']]
            elif camera_model == 'opencv_fisheye':
                distortion_params = [distortion['k1'],distortion['k2'],distortion['k3'],distortion['k4']]
            else:
                LOG_ERROR(f"Unknown camera_model: {camera_model}")
                return None
        except KeyError as e:
            LOG_ERROR(f"Missing distortion parameter: {e}")
            return None
        if camera_model == 'pinhole':
            cam = PerpectiveCamera.from_K(K,image_size,distortion_params)
        elif cam == 'opencv_fisheye':
            cam = OpenCVFisheyeCamera.from_K_D(K,image_size,distortion_params)
        # Parse extrinsic parameters
        rotation_type = extrinsic.get('rotation_type')
        try:
            if rotation_type == 'rotation_matrix':
                rotation = np.array(extrinsic['rotation_matrix'])
                rot_type = RotType.SO3
            elif rotation_type == 'quaternion_xyzw':
                rotation = np.array(extrinsic['quaternion'])
                rot_type = RotType.QUAT_XYZW
            elif rotation_type == 'quaternion_wxyz':
                rotation = np.array(extrinsic['quaternion'])
                rot_type = RotType.QUAT_WXYZ
            elif rotation_type == 'rpy':
                rotation = np.array(extrinsic['roll_pitch_yaw'])
                rot_type = RotType.RPY
            elif rotation_type == 'so3':
                rotation = np.array(extrinsic['so3'])
                rot_type = RotType.so3
            else:
                LOG_ERROR(f"Unknown rotation_type: {rotation_type}")
                return None
            translation = np.array(extrinsic['translation_vector'])
            rot = Rotation(rotation, rot_type)
            tr = Transform(translation,rot)
        except KeyError as e:
            LOG_ERROR(f"Missing extrinsic parameter: {e}")
            return None
        except ValueError as e:
            LOG_ERROR(f"Invalid rotation_type: {e}")
            return None

        image_data = {
            'path': path,
            'cam': cam,
            'camtoworld': tr
        }
        images_info.append(image_data)
    
    try:
        num_points = config["num_points"]
    except KeyError as e:
        LOG_ERROR(f"Missing Number of Points in YAML structure: {e}")
        return None

    return images_info, num_points

def main(unused_args):
    """
    Epipolar Line Test to Validate That a Pose and is Correct

    Currently available camera models
        1. Pinhole Camera (no distortion)

    To conduct the test, you need:
        1. A pair of images.
        2. Intrinsic Matrices
        3. Poses for each image (Camera to World).

    Fill above information into ./images_info.yaml
        
    Test Process
        1. Set image path and the necessary information.
        2. Select several points(default number is 4).
        3. Show the result.
    """

    result = parse_yaml(file_path="./test_info.yaml")

    if result is None: return
    image_info, num_points = result

    image1 = read_image(image_info[0]["path"])
    image2 = read_image(image_info[1]["path"])

    cam1:RadialCamera = image_info[0]["cam"]
    cam2:RadialCamera = image_info[1]["cam"]

    tf1:Transform = image_info[0]["camtoworld"]
    tf2:Transform = image_info[1]["camtoworld"]
    tf = tf2.inverse() * tf1  # compute relative transform: cam1 to cam2

    # undistort image
    image1 = cam1.undistort_image(image1)
    image2 = cam2.undistort_image(image2)

    point_selector = DoubleImagesPointSelector(image1, image2, 3)
    point_selector.connect()
    pts1, pts2 = point_selector.get_points()

    ## draw dots and epipolar lines

    for pt1,pt2 in zip(pts1, pts2):
        rgb = tuple(np.random.randint(0,255,3).tolist())
        E = compute_essential_matrix(rel_p=tf)
        F = compute_fundamental_matrix(K1=cam1.K,K2=cam2.K,E=E)
        pt_homo = np.array((pt1[0],pt1[1],1.))
        image1 = draw_circle(image1,pt1,1,rgb,2)
        image2 = draw_line_by_line(image2,tuple((F@pt_homo).tolist()),rgb,1)
        image2 = draw_circle(image2,pt2,1,rgb,2)

    show_two_images(image1,image2)


if __name__ == "__main__":
    app.run(main)