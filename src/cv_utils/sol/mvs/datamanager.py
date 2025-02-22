"""
Module Name: datamanager.py

Description: 
This module provides tools for managing and utilizing multi-view stereo (MVS) datasets. 
It includes methods for loading, preprocessing, and 
manipulating the dataset for 3D reconstruction and scene understanding tasks.

Class:
    - ScannetV1: Handles loading, preprocessing, and accessing the Scannet dataset. 

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

License: MIT LICENSE
"""

from .parser import *
from ...geom.camera import *
from ...geom.tf import *
from ...geom.pose import *
from ...geom import geom_utils as geou
from ...ops.umath import *
from ...ops.uops import *
from ...utils import io

from ...externals._o3d.common import O3dPCD, O3dTriMesh
from ...externals._o3d import o3dutils


class BaseMVSManger:
    """
    Base class for managing Multi-View Stereo (MVS) datasets.

    Attributes:
        has_depth (bool): Flag indicating if depth data is available.
        has_normal (bool): Flag indicating if normal data is available.
        use_buffer (bool): Flag indicating if all data should be loaded into memory.
        is_single_cam (bool): Flag indicating if there is only one camera.
        image_paths (List[str]): List of paths to image files.
        depth_paths (List[str]): List of paths to depth map files.
        normal_paths (List[str]): List of paths to normal map files.
        image_buffer (np.ndarray): Array containing loaded images with shape [B, H, W, 3].
        depth_buffer (np.ndarray): Array containing loaded depth maps with shape [B, H, W].
        normal_buffer (np.ndarray): Array containing loaded normal maps with shape [B, H, W, 3].
        cam (Camera): A single camera instance.
        cams (List[Camera]): List of multiple camera instances.
        tfs (List[Pose]): List of camera poses (cam to world).
        indices (Dict[Any, int]): Dictionary mapping IDs to indices.
        ids (List[Any]): List of IDs.
        num_frames (int): number of frames

    Abstract Methods:
        get_cam: Retrieve the camera instace for a given ID.
        get_image: Retrieve the image for a given ID.
        get_depth: Retrieve the depth map for a given ID.
        compute_point_cloud: Compute the point cloud from the MVS data.
        compute_mesh: Compute the mesh from the point cloud.
    """

    def __init__(self) -> None:
        """
        Initialize the BaseMVSManger instance.
        """
        self.has_depth: bool = False
        self.has_normal: bool = False
        self.use_buffer: bool = False
        self.is_single_cam: bool = True

        self.image_paths: List[str] = []
        self.depth_paths: List[str] = []
        self.normal_paths: List[str] = []

        self.image_buffer: np.ndarray = None  # [B,H,W,3]
        self.depth_buffer: np.ndarray = None  # [B,H,W]
        self.normal_buffer: np.ndarray = None  # [B,H,W,3]

        self.cam: Camera = None
        self.cams: List[Camera] = []  # multiple cameras
        self.tfs: List[Transform] = []  # poses
        self.indices: Dict[Any, int] = {}  # { id: index(int)}
        self.ids: List[Any] = []

        self.num_frames = 0

    def get_cam(self, id: int) -> Camera:
        """
        Retrieve the camera matrix for a given ID.

        Args:
            id (int): The ID of the camera.

        Returns:
            camera (Camera): The camera instance corresponding to the given ID.
        """
        if self.is_single_cam:
            return self.cam
        return self.cams[self.indices[id]]

    def get_image(self, id: int) -> np.ndarray:
        """
        Retrieve the image for a given ID.

        Args:
            id (int): The ID of the image.

        Returns:
            image (np.ndarry, [H,W,3]): The image corresponding to the given ID.
        """
        index = self.indices[id]
        if self.use_buffer:
            return self.image_buffer[index]
        return io.read_image(self.image_paths[index])

    def get_depth(self, id: int) -> np.ndarray:
        """
        Retrieve the depth map for a given ID.

        Args:
            id (int): The ID of the depth map.

        Returns:
            depth map (np.ndarry, [H,W]): The depth map corresponding to the given ID.
        """
        raise NotImplementedError

    def get_pose(self, id: int) -> Transform:
        """
        Retrieve the transform for a given ID.

        Args:
            id (int): The ID of the pose.

        Returns:
            tf (Transform): The transform instance corresponding to the given ID.
        """
        index = self.indices[id]
        return self.tfs[index]

    def compute_point_cloud(self, ids: Optional[List[Any]] = None) -> O3dPCD:
        """
        Compute the point cloud from the MVS data.

        Args:
            ids (List[int]): List of IDs.

        Returns:
            pcd (O3DPCD): The computed point cloud.
        """
        if self.has_depth is False:
            LOG_ERROR("The Dataset does not have depth map.")
            return None

        if ids is None:
            ids = self.ids

        pcds, colors = [], []
        for id in ids:
            depth, image, tf, cam = (
                self.get_depth(id),
                self.get_image(id),
                self.get_pose(id),
                self.get_cam(id),
            )
            pcd, color = geou.convert_depth_to_point_cloud(depth, cam, image, pose=tf)
            pcds.append(pcd)
            colors.append(color)

        pcds = concat(pcds, 0)
        colors = concat(colors, 0)
        return o3dutils.create_point_cloud(pcds, colors)

    def compute_mesh_using_pcd(self, ids: Optional[List[Any]] = None) -> O3dTriMesh:
        """
        Compute the mesh from the point cloud.

        Args:
            ids (List[int]): List of IDs.

        Returns:
            mesh (O3DTriMesh) The computed mesh.
        """
        if self.has_depth is False:
            LOG_ERROR("The Dataset does not have depth map.")
            return None
        pcd = self.compute_point_cloud(ids)
        return o3dutils.create_mesh_from_pcd(pcd, normal_radius=0.04)


class ScannetV1Manager(BaseMVSManger):
    """
    A data manager for the ScanNetV1 dataset.

    This class loads, preprocesses, and provides access to the ScanNetV1 dataset. It parses
    the dataset directory to retrieve image and depth map paths, camera calibration parameters,
    and pose information. Additionally, it performs depth alignment to the color image coordinate
    frame using provided calibration matrices and transforms. Optionally, the loaded images and
    depth maps can be buffered in memory for faster access.

    Attributes:
        cam (PerspectiveCamera): The camera instance for color image calibration.
        depth_cam (PerspectiveCamera): The camera instance for depth image calibration.
        depth_shift (float): The scaling factor applied to raw depth values.
        depth2cam (Transform): The transformation from the depth camera frame to the color camera frame.
        image_paths (List[str]): List of file paths for the color images.
        depth_paths (List[str]): List of file paths for the depth maps.
        tfs (List[Transform]): List of camera poses (as transforms) corresponding to each frame.
        indices (Dict[Any, int]): Mapping from frame IDs to their corresponding index in the dataset.
        ids (List[Any]): List of frame IDs.
        num_frames (int): Total number of frames in the dataset.
        use_buffer (bool): Flag indicating whether to load all data into memory.
        image_buffer (np.ndarray): Buffer for loaded color images (if use_buffer is True).
        depth_buffer (np.ndarray): Buffer for loaded depth maps (if use_buffer is True).

    Args:
        dataset_path (str): The file path to the ScanNetV1 dataset directory.
        use_buffer (bool, optional): If True, load and buffer images and depth maps into memory for faster access.
                                     Defaults to False.
    """

    def __init__(self, dataset_path: str, use_buffer: bool = False) -> None:
        super().__init__()
        dataset_dict = parse_scannetv1_dataset(dataset_path)
        if dataset_dict is None:
            LOG_ERROR("Failed to load scannet dataset files.")
            return

        self.cam = PerspectiveCamera.from_K(
            dataset_dict["image_calib"], dataset_dict["image_size"]
        )
        self.is_single_cam = True
        self.has_depth = True

        self.depth_cam = PerspectiveCamera.from_K(
            dataset_dict["depth_calib"], dataset_dict["depth_size"]
        )
        self.depth_shift = dataset_dict["depth_shift"]

        world2cam = Transform.from_mat(
            dataset_dict["m_calibrationColorExtrinsic"]
        ).inverse()
        depth2world = Transform.from_mat(dataset_dict["m_calibrationDepthExtrinsic"])
        self.depth2cam = world2cam * depth2world

        for idx, frame in enumerate(dataset_dict["frames"]):
            self.image_paths.append(frame["color"])
            self.depth_paths.append(frame["depth"])
            self.tfs.append(Transform.from_mat(np.loadtxt(frame["pose"])))
            self.indices[frame["id"]] = idx
            self.ids.append(frame["id"])

        self.num_frames = len(self.image_paths)

        width, height = dataset_dict["image_size"]
        if use_buffer:
            self.image_buffer = np.empty((self.num_frames, height, width, 3))
            self.depth_buffer = np.empty((self.num_frames, height, width))
            for id in self.ids:
                self.image_buffer[id] = self.get_image(id)
                self.depth_buffer[id] = self.get_depth(id)
        self.use_buffer = use_buffer

    def get_depth(self, id: int):
        index = self.indices[id]
        if self.use_buffer:
            return self.depth_buffer[index]

        depth_map = io.read_pgm(self.depth_paths[index], mode="F") / self.depth_shift
        # align depth map into color image
        pcd = geou.convert_depth_to_point_cloud(
            depth_map, self.depth_cam, map_type="MPI", pose=self.depth2cam
        )
        aligned_depth = geou.convert_point_cloud_to_depth(
            pcd, self.get_cam(id), map_type="MPI"
        )
        return aligned_depth
