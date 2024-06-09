import os.path as osp
import numpy as np
from cv_utils.utils.file_utils import read_yaml, read_image
from cv_utils.core.geometry import Transform, Rotation
from cv_utils import RotType 
from cv_utils.core.geometry.camera import PinholeCamera, EquidistantCamera,Camera
from cv_utils.utils.logger import *
from cv_utils.vis.point_selector import DoubleImagesPointSelector
from cv_utils.vis.image_utils import *
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
            cam = PinholeCamera.from_K(K,image_size,distortion_params)
        elif cam == 'opencv_fisheye':
            cam = EquidistantCamera.from_K_D(K,image_size,distortion_params)
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

    cam1:Camera = image_info[0]["cam"]
    cam2:Camera = image_info[1]["cam"]

    tf1:Transform = image_info[0]["camtoworld"]
    tf2:Transform = image_info[1]["camtoworld"]
    tf = tf1.inverse() * tf2 # compute relative transform
    tf = tf.inverse()

    # point_selector = DoubleImagesPointSelector(image1, image2, 3)
    # point_selector.connect()
    # pts1, pts2 = point_selector.get_points()

    pts1=[(285.85930735930737, 239.35714285714283), (287.04978354978357, 187.92857142857142), (369.3831168831169, 238.64285714285714)]
    pts2=[(182.85497835497836, 248.1190476190476), (180.75974025974028, 193.59523809523807), (273.42640692640697, 247.73809523809524)]

    zs = np.linspace(1.,15., 100)

    ## draw dots and epipolar lines
    for pt1,pt2 in zip(pts1, pts2):
        rays = cam1.get_rays(np.array(pt1).reshape(2,-1))
        back_projected_pt1 = zs*rays
        transformed_rays = tf.apply_pts3d(back_projected_pt1)
        reprojected_pt1 = cam2.project_rays(transformed_rays).reshape(-1,2).tolist()
        rgb = tuple(np.random.randint(0,255,3).tolist())
        pt1 = (int(pt1[0]),int(pt1[1]))
        pt2 = (int(pt2[0]),int(pt2[1]))
        image1 = draw_circle(image1,pt1,rgb=rgb)
        image2 = draw_lines(image2,reprojected_pt1,rgb=rgb)
        image2 = draw_circle(image2,pt2,rgb=rgb)

    show_two_images(image1,image2)

    
   


if __name__ == "__main__":
    app.run(main)