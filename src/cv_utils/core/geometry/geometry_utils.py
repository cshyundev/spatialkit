import numpy as np
from ..operations.hybrid_operations import *
from .pose import Pose
from .rotation import Rotation 
from .tf import Transform
from .camera import *
from typing import *
import cv2 as cv
from scipy.ndimage import map_coordinates

def compute_essential_matrix(K1: Optional[np.ndarray]=None, K2: Optional[np.ndarray]=None,F: Optional[np.ndarray]=None, rel_p:Optional[Union[Pose,Transform]]=None) -> np.ndarray:
    """
    Compute the essential matrix from either the fundamental matrix or relative camera pose.
    
    Parameters:
    - K1 (np.ndarray): Intrinsic camera matrix for the first camera (3x3).
    - K2 (np.ndarray): Intrinsic camera matrix for the second camera (3x3).
    - F (np.ndarray, optional): Fundamental matrix (3x3), if available.
    - rel_p (Pose, optional): Relative pose object containing rotation and translation between cameras.

    Returns:
    - np.ndarray: The computed essential matrix (3x3).
    
    Raises:
    - AssertionError: If neither the fundamental matrix nor the relative pose is provided.
    """
    assert (F is not None or rel_p is not None), "For computing essential matrix, Fundamental Matrix or Relative Pose is required."

    if F is not None:
        return K2.T @ F @ K1
    else:
        # E = [t]×R  
        skew_t = rel_p.skew_t()
        r_mat = rel_p.rot_mat()
        return convert_numpy(skew_t @r_mat)

def eight_point_algorithm(pts1:Union[np.ndarray, List[Tuple[int]]], pts2:Union[np.ndarray, List[Tuple[int]]]):
    """
    Compute the fundamental matrix using the 8-point algorithm from point correspondences.

    Parameters:
    - - pts1, pts2 (Union[np.ndarray, List[Tuple[int]]], optional): Corresponding points in each image.
    Returns:
    - np.ndarray: The computed fundamental matrix (3x3) after enforcing a rank-2 constraint.
    """
    # Ensure pts1 and pts2 are numpy arrays
    if isinstance(pts1, list):
        pts1 = np.array(pts1)
    if isinstance(pts2, list):
        pts2 = np.array(pts2)
    assert(pts1.shape == pts2.shape)

    n = pts1.shape[0]

    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

    U, S, Vt = svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce the rank-2 constraint
    U, S, Vt = svd(F)
    S[-1] = 0
    F = U @ diag(S) @ Vt
    return F

def compute_fundamental_matrix(pts1: Optional[Union[np.ndarray, List[Tuple[int]]]] = None,
                               pts2: Optional[Union[np.ndarray, List[Tuple[int]]]] = None,
                               K1: Optional[np.ndarray] = None, 
                               K2: Optional[np.ndarray] = None, 
                               E: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the fundamental matrix given point correspondences or camera intrinsics and an essential matrix.

    Parameters:
    - pts1, pts2 (Union[np.ndarray, List[Tuple[int]]], optional): Corresponding points in each image.
    - K1, K2 (np.ndarray, optional): Intrinsic matrices of the two cameras.
    - E (np.ndarray, optional): Essential matrix.

    Returns:
    - np.ndarray: The computed fundamental matrix (3x3).

    Raises:
    - ValueError: If insufficient or incorrect data is provided.
    """
    if pts1 is not None and pts2 is not None:
        # Ensure pts1 and pts2 are numpy arrays
        if isinstance(pts1, list):
            pts1 = np.array(pts1)
        if isinstance(pts2, list):
            pts2 = np.array(pts2)

        assert pts1.shape[1] == 2 and pts2.shape[1] == 2, "Point arrays must have two coordinates."
        assert pts1.shape == pts2.shape, "Point arrays must have the same shape."
        assert pts1.shape[0] >= 8, f"The number of corresponding points must be larger than 8, but got {pts1.shape[0]}"


        # Normalize points
        pts1 = np.vstack((pts1, np.ones((1, pts1.shape[1]))))
        pts2 = np.vstack((pts2, np.ones((1, pts2.shape[1]))))

        # Construct matrix A for linear equation system
        A = np.zeros((pts1.shape[1], 9))
        for i in range(pts1.shape[1]):
            x1, y1, _ = pts1[:, i]
            x2, y2, _ = pts2[:, i]
            A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

        # Solve using SVD
        _, _, Vt = svd(A)
        F = Vt[-1].reshape(3, 3)

        # Enforce the rank-2 constraint
        U, S, Vt = svd(F)
        S[-1] = 0
        F = U @ diag(S) @ Vt

        return F

    elif K1 is not None and K2 is not None and E is not None:
        assert K1.shape == (3, 3) and K2.shape == (3, 3) and E.shape == (3, 3), "Intrinsic and Essential matrices must be 3x3."
        # Compute fundamental matrix from essential matrix
        F = inv(K2).T @ E @ inv(K1)
        return F
    else:
        raise ValueError("Insufficient or incorrect data provided. Please provide either point correspondences or intrinsics with an essential matrix.")

def compute_fundamental_matrix_using_ransac(pts1: Optional[Union[np.ndarray, List[Tuple[int]]]],
                                            pts2: Optional[Union[np.ndarray, List[Tuple[int]]]],
                                            threshold:float=1e-3,
                                            max_iterations:int=1000):
    
    # Ensure pts1 and pts2 are numpy arrays
    if isinstance(pts1, list):
        pts1 = np.array(pts1)
    if isinstance(pts2, list):
        pts2 = np.array(pts2)
    assert pts1.shape[1] == 2 and pts2.shape[1] == 2, "Point arrays must have two coordinates."
    assert pts1.shape == pts2.shape, "Point arrays must have the same shape."
    assert pts1.shape[0] >= 8, f"The number of corresponding points must be larger than 8, but got {pts1.shape[0]}"

    def _normalize_points(pts):
        """ Normalize the points to improve numerical stability for SVD. """
        mean = np.mean(pts, axis=0)
        std = np.std(pts)
        T = np.array([
            [1/std, 0, -mean[0]/std],
            [0, 1/std, -mean[1]/std],
            [0, 0, 1]
        ])
        pts_normalized = np.dot(T, np.vstack((pts.T, np.ones(pts.shape[0]))))
        return pts_normalized[:2].T, T

    best_F = None
    best_inliers = 0

    for _ in range(max_iterations):
        # Randomly select 8 points for the minimal sample set
        indices = np.random.choice(pts1.shape[0], 8, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        # Normalize points
        sample_pts1_norm, T1 = _normalize_points(sample_pts1)
        sample_pts2_norm, T2 = _normalize_points(sample_pts2)

        # Compute the fundamental matrix
        F_norm = eight_point_algorithm(sample_pts1_norm, sample_pts2_norm)

        # Denormalize the fundamental matrix
        F = T2.T @ F_norm @ T1

        # Calculate the number of inliers
        inliers = 0
        for i in range(pts1.shape[0]):
            pt1 = np.append(pts1[i], 1)
            pt2 = np.append(pts2[i], 1)
            error = abs(pt2.T @ F @ pt1)
            if error < threshold:
                inliers += 1

        # Update the best model if current model has more inliers
        if inliers > best_inliers:
            best_F = F
            best_inliers = inliers

    return best_F, best_inliers

def recover_pose(E: np.ndarray) -> Pose:
    """
    Decompose the essential matrix into possible rotations and translations.
    """
    U, _, Vt = svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    
    if determinant(R1) < 0:
        R1 = -R1
    if determinant(R2) < 0:
        R2 = -R2

    return Pose(t, Rotation.from_mat3(R1))

def solve_pnp(pts2d: np.ndarray, pts3d: np.ndarray, cam:RadialCamera, cv_flags:Any=None) -> Transform:
    """
       Computes the camera pose using the Perspective-n-Point (PnP) problem solution.
       Parameters:
            pts2d (np.ndarray): 2D image points (N x 2).
            pts3d (np.ndarray): Corresponding 3D scene points (N x 3).
            cam: Camera instance, which can be one of Radial models.
            cv_flags (Any, optional): Additional options for solving PnP.

        Returns:
            Transform: A Transform Instance from object coordinates to camera coordinates.

        Raises:
            AssertionError: If the camera type is not supported.
    """
    cam_type = cam.cam_type
    available_cam_types = [ CamType.PRESPECTIVE , CamType.OPENCV, CamType.THINPRISM]
    assert(cam_type in available_cam_types), f"Unavailable Camera Type: {cam_type.name}"

    pts2d_undist =  cam.undistort_pixel(pts2d) # 2 * N
    
    _, rvec, tvec = cv.solvePnP(pts3d,pts2d_undist,cam.K,np.zeros([0., 0., 0., 0., 0.]),flags=cv_flags)

    tr = Transform.from_rot_vec_t(rvec,tvec)
    return tr

def transition_camera_view(src_image: np.ndarray, src_cam: Camera, dst_cam: Camera, transform: Transform = None) -> np.ndarray:
    """
    Transition the view from one camera to another with a specified transformation.

    Parameters:
    image (np.ndarray): The input image array from the source camera.
    src_cam (Camera): Source camera object with get_rays method.
    dst_cam (Camera): Destination camera object with project_pixel method.
    transform (Transform): A Transform object applied in normalized coordinates.

    Returns:
    np.ndarray: The output image transformed and projected onto the destination camera's resolution.
    """
    
    out_height,out_width  = dst_cam.hw
    # Prepare the output image array
    if src_image.ndim == 3:
        output_image = np.zeros((out_height, out_width, src_image.shape[2]), dtype=src_image.dtype)
    else:
        output_image = np.zeros((out_height, out_width), dtype=src_image.dtype)

    output_rays, dst_valid_mask = dst_cam.get_rays()  # 3 * N
    # Apply inverse transform
    if transform:
        inverse_transform = transform.inverse()
        output_rays = inverse_transform.apply_pts3d(output_rays)  # 3 * N

    # Project ray onto source camera
    input_coords, src_valid_mask  = src_cam.project_rays(output_rays, out_subpixel=True)  # 2 * N
    # input_coords = input_coords[:,src_valid_mask]
    # Split coordinates
    input_x, input_y = input_coords[0, :], input_coords[1, :]
    if src_image.ndim == 3:
        # For multi-channel images, handle each channel separately
        for c in range(src_image.shape[2]):
            output_image[..., c] = map_coordinates(src_image[..., c], [input_y, input_x], order=1, mode='constant').reshape((out_height, out_width))
    else:
        # For single-channel images
        output_image = map_coordinates(src_image, [input_y, input_x], order=1, mode='constant').reshape((out_height, out_width))
    mask = logical_and(src_valid_mask,dst_valid_mask).reshape((out_height,out_width))
    output_image[~mask] = 0.    

    return output_image

def find_corresponding_points(image1: np.ndarray, image2: np.ndarray, 
                              feature_type: str = 'SIFT', max_matches: int = 50) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Extracts features from two images and finds matching points between them.

    Parameters:
    - image1 (np.ndarray): First input image.
    - image2 (np.ndarray): Second input image.
    - feature_type (str): Type of feature extractor to use ('SIFT', 'ORB', etc.).
    - max_matches (int): Maximum number of matching points to return.

    Returns:
    - Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]: Lists of matching points in the first and second images.
    """
    # Initialize the feature detector and descriptor based on user input
    if feature_type == 'SIFT':
        detector = cv.SIFT_create()
    elif feature_type == 'ORB':
        detector = cv.ORB_create(nfeatures=max_matches)  # Limit the number of features
    else:
        raise ValueError("Unsupported feature type. Choose 'SIFT' or 'ORB'.")

    # Find keypoints and descriptors with chosen detector
    keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

    # Match descriptors between images
    if feature_type == 'SIFT':
        # Use FLANN based matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    elif feature_type == 'ORB':
        # Use BFMatcher (Brute Force Matcher) for ORB
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Perform matching
    matches = matcher.match(descriptors1, descriptors2)
    # Sort matches by their distance (quality)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    pts1 = []
    pts2 = []

    for m in matches[:max_matches]:
        pts1.append(keypoints1[m.queryIdx].pt)
        pts2.append(keypoints2[m.trainIdx].pt)

    return pts1, pts2


# def triangulate_points(pts1: Union[np.ndarray, List[Tuple[int]]],
#                        pts2: Union[np.ndarray, List[Tuple[int]]],
#                        cam1:Camera, cam2:Camera, rel_pose12:Union[Pose,Transform]) -> np.ndarray:
#     """
#     Triangulate points using the linear method.

#     Parameters:
#     - pts1, pts2 (Union[np.ndarray, List[Tuple[int]]]): Corresponding points in each image.
#     - cam1, cam2 (Camera): Camera objects containing intrinsic parameters.
#     - rel_pose12 (Union[Pose, Transform]): Relative pose or transformation from the first to the second camera.

#     Returns:
#     - np.ndarray: Array containing the 3D coordinates of the triangulated points.
#     """
#     # Convert points to numpy array if they are given as lists
#     if isinstance(pts1, list):
#         pts1 = np.array(pts1)
#     if isinstance(pts2, list):
#         pts2 = np.array(pts2)    


# class MultiView:
#     """
#     Multiview Geometry
    
#     Available Contents
#     Epipolar Line: epipolar_line_test.py
#     make point clouds: TODO
#     """
#     def __init__(self,
#                 image_path: List[str],
#                 cameras: List[Camera],
#                 poses: List[Pose],
#                 depth_path: Optional[List[str]]=None,
#                 normal_path: Optional[List[str]]=None
#                 ) -> None:
#         assert(len(image_path) == len(cameras) and len(image_path) == len(poses)), "Number of images, camera, and poses must be same."
#         if depth_path is not None:
#             assert(len(image_path) == len(depth_path)), "Number of images and these depth must be same."
#         if normal_path is not None:
#             assert(len(image_path) == len(normal_path)), "Number of images and these normal must be same."
            

#         self.image_path = image_path
#         self.cameras = cameras
#         self.poses = poses
#         self.depth_path = depth_path
#         self.normal_path = normal_path
#         self.n_views = len(cameras)

#         self.has_depth = True if len(self.depth_path) != 0 else False
                
#     @staticmethod
#     def from_meta_data(
#         root_path: str,
#         meta_data:Dict[str,Any]) -> 'MultiView':
#         image_path = []
#         cameras = []
#         poses = []
#         depth_path = []
#         normal_path = []
        
#         height = meta_data["height"]
#         width = meta_data["width"]
        
#         frames = meta_data["frames"]
#         for frame in frames:
#             image_path.append(osp.join(root_path,frame["rgb_path"]))
#             depth_path.append(osp.join(root_path,frame["mono_depth_path"]))
#             normal_path.append(osp.join(root_path,frame["mono_normal_path"])) 
#             poses.append(Pose.from_mat(np.array(frame["camtoworld"]))) 
#             # create camera
#             K = frame["intrinsics"]
#             cam_dict = {}
#             cam_dict["fx"] = K[0][0]
#             cam_dict["fy"] = K[1][1]
#             cam_dict["cx"] = K[0][2]
#             cam_dict["cy"] = K[1][2]
#             cam_dict["skew"] = K[0][1]
#             cam_dict["height"] = height
#             cam_dict["width"] = width
#             cam_dict["cam_type"] = meta_data["camera_model"]
#             cameras.append(Camera.create_cam(cam_dict))
            
#         return MultiView(image_path,cameras,poses,
#                          depth_path,normal_path)   
    
#     @staticmethod
#     def from_dict(frame_dict:Dict[str,Any]) -> 'MultiView':
#         root = frame_dict["root"]
#         frames = frame_dict["frames"]
#         image_path = []
#         depth_path = []
#         normal_path = []
#         cameras = []
#         poses = []
#         for frame in frames:
#             cameras.append(Camera.create_cam(frame["cam"]))
#             poses.append(Pose.from_mat(np.array(frame["camtoworld"])))
#             image_path.append(osp.join(root,frame["image"]))
#             if "depth" in frame:
#                 depth_path.append(osp.join(root,frame["depth"]))
        
#         return MultiView(image_path,cameras,poses,depth_path,normal_path)

#     def fundamental_matrix(self, idx1:int, idx2:int):
#         ## F = K_1^-1*(t×R)*K_2^-1
#         ## [R,t]: relative pose from frame1 to frame2
        
#         inv_K1 = self.cameras[idx1].inv_K
#         K2 = self.cameras[idx2].K
#         E = self.essential_matrix(idx1,idx2)
#         F = matmul(np.linalg.inv(K2.T) ,E)
#         F = matmul(F,inv_K1)
#         return F
    
#     def essential_matrix(self, idx1:int, idx2:int):
#         ## E = [t]×R,  
#         c1_to_c2 = self.relative_pose(idx1,idx2)
#         skew_t = c1_to_c2.skew_t()
#         r_mat = c1_to_c2.rot_mat()
#         return matmul(skew_t,r_mat)        
    
#     def relative_pose(self, idx1:int, idx2:int) -> Pose:
#         ## [R,t]: relative pose from frame1 to frame2
#         ## Relative Pose = Pose2 - Pose1
#         c1tow = self.poses[idx1]
#         wtoc2 = self.poses[idx2].inverse()
#         return wtoc2.merge(c1tow)
    
#     def choice_points(self, idx1:int, idx2:int, n_points:int) -> List[Tuple[int,int]]:
#         left_image, right_image = read_image(self.image_path[idx1]),read_image(self.image_path[idx2])  
#         fig, axs = plt.subplots(1, 2)
#         axs[0].imshow(left_image)
#         axs[1].imshow(right_image)
#         axs[0].set_title(f"Choose {n_points} points on left image")
#         pts = plt.ginput(n_points)
#         pts = np.int32(pts)
#         plt.close(fig)
#         return pts.tolist()
    
#     def draw_epipolar_line(self, idx1:int, idx2:int,
#                            left_pt2d:List[Tuple[int]]=None):
#         cam1 = self.cameras[idx1]
#         cam2 = self.cameras[idx2]
#         if (cam1.cam_type == cam2.cam_type) and cam1.cam_type == CamType.PINHOLE:
#             return self._draw_epipolar_line_between_pinholes(idx1,idx2, left_pt2d)
    
#     def _draw_epipolar_line_between_pinholes(self, idx1:int, idx2:int,
#                                               left_pt2d:List[Tuple[int,int]]) -> np.ndarray:
#         """
#         Draw Epipolar Line between Pinhole Cameras
#         Args:
#             idx1,idx2: left and right camera inices respectively
#             left_pt2d: (n,2), 2D points in left image
#             save_path: path to save the result
#         Return:
#             Image: (H,2W,3) or (H,2W), float, 
#         """
#         assert(len(left_pt2d) > 0), ("To draw epipolar line, 2D points must be needed.")
#         left, right = read_image(self.image_path[idx1]),read_image(self.image_path[idx2])
        
#         F = self.fundamental_matrix(idx1,idx2)
#         F = convert_numpy(F)
#         for pt in left_pt2d:
#             pt_homo = np.array([pt[0], pt[1], 1.])
#             color = tuple(np.random.randint(0,255,3).tolist())
#             left = draw_circle(left, pt, 1,color,2)
#             right = draw_line_by_line(right,tuple((F@pt_homo).tolist()), color, 2)            
                     
#         image = concat_images([left,right], vertical=False)
#         return image        
    
#     def save_point_cloud(self, save_path: str):
        
#         pts3d = []
#         colors = []
        
#         for i in range(self.n_views):
#             rays = self.cameras[i].get_rays(norm=False) # [n,3]
#             origin, direction = self.poses[i].get_origin_direction(rays) # [n,3], [n,3]
#             depth = read_float(self.depth_path[i]).reshape(-1,1)
#             if depth is not None:
#                 pts3d_w = origin + depth * direction            
#                 pts3d.append(pts3d_w)
#                 color = read_image(self.image_path[i],as_float=True).reshape(-1,3) # as_float:0~255 -> 0~1.          
#                 colors.append(color)
#         pts3d = concat(pts3d, 0)
#         colors = concat(colors, 0)            
        
#         make_point_cloud(pts3d, colors, save_path)
    
#     def get_image(self, idx:int) -> np.ndarray:
#         return read_image(self.image_path[idx])
    
#     def get_camera(self, idx:int) -> Camera:
#         return self.cameras[idx]

#     def __normalize_points(self, pts: Array) -> Tuple[np.ndarray,np.ndarray]:
#         """
#         Normalize the points so that the mean is 0 and the average distance is sqrt(2).
#         """
            
#         pts = convert_numpy(pts)
#         centroid = np.mean(pts, axis=0)
#         centered_pts = pts - centroid
#         scale = np.sqrt(2) / np.mean(np.linalg.norm(centered_pts, axis=1))
#         transform = np.array([[scale, 0, -scale * centroid[0]],
#                               [0, scale, -scale * centroid[1]],
#                               [0, 0, 1]])
#         normalized_points = np.dot(transform, np.concatenate((pts.T, np.ones((1, pts.shape[0])))))
#         return normalized_points.T, transform
    
#     def recover_pose(self, left_pts: List[Tuple[int,int]],right_pts: List[Tuple[int,int]], K:Array=None) -> Pose:
#         assert(len(left_pts) >= 3), f"To recover the relative pose, correspondence pairs must be larger than 3, but got {len(left_pts)}"
#         assert(len(right_pts) >= 3), f"To recover the relative pose, correspondence pairs must be larger than 3, but got {len(right_pts)}"
#         assert(len(left_pts) == len(right_pts)), "Correspondence pairs must be same."
        
#         left_pts = np.array(left_pts)
#         right_pts = np.array(right_pts)
        
#         left_pts_norm, T1 = self.__normalize_points(left_pts)
#         right_pts_norm, T2 = self.__normalize_points(right_pts)
        
#         A = []
#         for l,r in zip(left_pts_norm,right_pts_norm):
#             x1, y1 = l 
#             x2, y2 = r
#             A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
#             A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

#         A = np.array(A)
#         # Compute Homography using SVD decomposition 
#         _, _, Vt = np.linalg.svd(A)
#         H = Vt[-1].reshape((3, 3))

#         # denormalize
#         H = np.dot(np.linalg.inv(T2), np.dot(H, T1))

#         # Make the last element 1
#         H /= H[-1, -1]
        
#         return H


    
# # if __name__ == '__main__':
    
    
    
    