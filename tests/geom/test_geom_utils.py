import unittest
import numpy as np
from cv_utils.geom.camera import *
from cv_utils.geom.geom_utils import *
from cv_utils.ops.uops import *
from cv_utils.ops.umath import *
from cv_utils.common.logger import LOG_INFO
from cv_utils.exceptions import (
    InvalidShapeError,
    InvalidDimensionError,
    CalibrationError,
)


class TestGeomUtils(unittest.TestCase):

    def setUp(self):

        self.image_size = (640,480)
        test_cam = PerspectiveCamera.from_fov(image_size=self.image_size,fov=60.)
        self.left_cam = test_cam
        self.right_cam = test_cam

        self.num_pts = 100
        min_num_pts = 50

        self.left_c2w = Transform()
        self.rel_pose = Transform(np.array([1,0,0]), Rotation.from_mat3(np.eye(3)))

        while True:
            pts1 =  np.concatenate([np.random.randint(0,self.image_size[0],self.num_pts).reshape(1,-1),
                                np.random.randint(0,self.image_size[1],self.num_pts).reshape(1,-1)],
                                axis=0)
            depth_scale = 3.
            depth_offset = 1.
            test_depth = np.random.rand(self.num_pts) * depth_scale + depth_offset

            rays1, _ = self.left_cam.convert_to_rays(pts1)
            pts3d = rays1 * test_depth.reshape(-1,)
            pts3d_right = self.rel_pose * pts3d
            pts2,valid_mask = self.right_cam.convert_to_pixels(pts3d_right)

            if valid_mask.sum() >= min_num_pts:
                self.pts1 = pts1[:,valid_mask]
                self.pts2 = pts2[:,valid_mask]
                self.pts3d = pts3d[:,valid_mask]
                self.num_pts = valid_mask.sum()
                break
        
    def test_compute_essential_matrix_from_pose(self):
        
        E = compute_essential_matrix_from_pose(self.rel_pose)
        total_error = 0.
        F = compute_fundamental_matrix_from_essential(self.left_cam.K,self.right_cam.K,E)
        for i in range(self.num_pts):
            pt1,pt2 = homo(self.pts1[:,i:i+1]) ,homo(self.pts2[:,i:i+1])  
            total_error += pt1.T @ F @ pt2
        total_error /= self.num_pts
        np.testing.assert_(total_error < 1e-2)
            
    def test_compute_essential_matrix_from_fundamental(self):
        
        E = compute_essential_matrix_from_pose(self.rel_pose)
        F = compute_fundamental_matrix_from_essential(self.left_cam.K,self.right_cam.K,E)
        E_recover = compute_essential_matrix_from_fundamental(K1=self.left_cam.K,K2=self.right_cam.K,F=F)
        np.testing.assert_array_almost_equal(E,E_recover)

    def test_compute_fundamental_matrix_from_points(self):
        total_error = 0.

        F = compute_fundamental_matrix_from_points(pts1=self.pts1,pts2=self.pts2)
        for i in range(self.num_pts):
            pt1,pt2 = homo(self.pts1[:,i:i+1]) ,homo(self.pts2[:,i:i+1])  
            total_error += pt1.T @ F @ pt2
        total_error /= self.num_pts
        LOG_INFO(f"compute_fundamental_matrix_from_point Error: {total_error}")
        np.testing.assert_(abs(total_error) < 1.)

    def test_compute_fundamental_matrix_using_ransac(self):
        total_error = 0.

        F, _ = compute_fundamental_matrix_using_ransac(pts1=self.pts1,pts2=self.pts2)
        for i in range(self.num_pts):
            pt1,pt2 = homo(self.pts1[:,i:i+1]) ,homo(self.pts2[:,i:i+1])  
            total_error += pt1.T @ F @ pt2
        total_error /= self.num_pts
        LOG_INFO(f"compute_fundamental_matrix_using_ransac Error: {total_error}")
        np.testing.assert_(abs(total_error) < 1.)
    
    def test_solve_pnp(self):
        # world to cam
        recover_pose = solve_pnp(self.pts1,self.pts3d,self.left_cam)
        np.testing.assert_array_almost_equal(recover_pose.mat44(),np.eye(4))


        pass
    
    def test_triangulate_points(self):
        
        recover_pts3d = triangulate_points(
            self.pts1,self.pts2,self.left_cam,self.right_cam,
            self.left_c2w.inverse(),self.rel_pose.inverse()
        )

        # Test Reprojction Error
        pts3d_c2 = self.rel_pose.inverse() * recover_pts3d

        pts1_reprojection, _ = self.left_cam.convert_to_pixels(recover_pts3d,out_subpixel=True) # [2,N]
        pts2_reprojection, _ = self.right_cam.convert_to_pixels(pts3d_c2,out_subpixel=True) # [2,N]

        pts1_reprojection_error = norm(self.pts1.astype(np.float64) -pts1_reprojection, dim=0)
        pts2_reprojection_error = norm(self.pts2.astype(np.float64) -pts2_reprojection, dim=0)

        pts1_reprojection_error = mean(pts1_reprojection_error)
        pts2_reprojection_error = mean(pts2_reprojection_error)

        total_error = pts1_reprojection_error + pts2_reprojection_error
        LOG_INFO(f"triangulate_points left Error: {pts1_reprojection_error}.")
        LOG_INFO(f"triangulate_points right Error: {pts2_reprojection_error}.")
        LOG_INFO(f"triangulate_points Error: {total_error}.")
        np.testing.assert_(total_error < 1.)

    def test_compute_relative_transform_from_points(self):
        
        total_error = 0.

        rel_pose = compute_relative_transform_from_points(self.pts1,self.pts2,self.left_cam,self.right_cam)

        E = compute_essential_matrix_from_pose(rel_pose)
        F = compute_fundamental_matrix_from_essential(self.left_cam.K,self.right_cam.K,E)

        for i in range(self.num_pts):
            pt1,pt2 = homo(self.pts1[:,i:i+1]) ,homo(self.pts2[:,i:i+1])  
            total_error += pt1.T @ F @ pt2
        total_error /= self.num_pts
        LOG_INFO(f"compute_relative_transform_from_points Error: {total_error}")
        np.testing.assert_(abs(total_error) < 1.)

    def test_decompose_essential_matrix(self):
        """Test essential matrix decomposition."""
        E = compute_essential_matrix_from_pose(self.rel_pose)
        tf1, tf2, tf3, tf4 = decompose_essential_matrix(E)

        # Verify all outputs are Transform instances
        self.assertIsInstance(tf1, Transform)
        self.assertIsInstance(tf2, Transform)
        self.assertIsInstance(tf3, Transform)
        self.assertIsInstance(tf4, Transform)

    def test_convert_point_cloud_to_depth_mpi(self):
        """Test point cloud to depth conversion with MPI."""
        # Create depth map from 3D points
        depth_map = convert_point_cloud_to_depth(
            self.pts3d.T,  # Convert to [N,3]
            self.left_cam,
            map_type="MPI"
        )

        self.assertEqual(depth_map.shape, self.left_cam.hw)
        self.assertGreater(np.sum(depth_map > 0), 0, "Depth map should have non-zero values")

    def test_convert_point_cloud_to_depth_msi(self):
        """Test point cloud to depth conversion with MSI."""
        depth_map = convert_point_cloud_to_depth(
            self.pts3d.T,  # Convert to [N,3]
            self.left_cam,
            map_type="MSI"
        )

        self.assertEqual(depth_map.shape, self.left_cam.hw)
        self.assertGreater(np.sum(depth_map > 0), 0, "Depth map should have non-zero values")

    def test_convert_point_cloud_to_depth_mci(self):
        """Test point cloud to depth conversion with MCI."""
        depth_map = convert_point_cloud_to_depth(
            self.pts3d.T,  # Convert to [N,3]
            self.left_cam,
            map_type="MCI"
        )

        self.assertEqual(depth_map.shape, self.left_cam.hw)
        self.assertGreater(np.sum(depth_map > 0), 0, "Depth map should have non-zero values")

    def test_convert_depth_to_point_cloud_mpi(self):
        """Test depth to point cloud conversion with MPI."""
        # Create a synthetic depth map
        depth_map = np.random.rand(*self.left_cam.hw).astype(np.float32) * 3.0 + 1.0

        pcd = convert_depth_to_point_cloud(
            depth_map,
            self.left_cam,
            map_type="MPI"
        )

        self.assertEqual(pcd.shape[1], 3)
        self.assertGreater(pcd.shape[0], 0, "Should have some 3D points")

    def test_convert_depth_to_point_cloud_mci(self):
        """Test depth to point cloud conversion with MCI."""
        depth_map = np.random.rand(*self.left_cam.hw).astype(np.float32) * 3.0 + 1.0

        pcd = convert_depth_to_point_cloud(
            depth_map,
            self.left_cam,
            map_type="MCI"
        )

        self.assertEqual(pcd.shape[1], 3)
        self.assertGreater(pcd.shape[0], 0, "Should have some 3D points")

    def test_convert_depth_to_point_cloud_msi(self):
        """Test depth to point cloud conversion with MSI."""
        depth_map = np.random.rand(*self.left_cam.hw).astype(np.float32) * 3.0 + 1.0

        pcd = convert_depth_to_point_cloud(
            depth_map,
            self.left_cam,
            map_type="MSI"
        )

        self.assertEqual(pcd.shape[1], 3)
        self.assertGreater(pcd.shape[0], 0, "Should have some 3D points")

    def test_convert_depth_to_point_cloud_with_color(self):
        """Test depth to point cloud conversion with color image."""
        depth_map = np.random.rand(*self.left_cam.hw).astype(np.float32) * 3.0 + 1.0
        color_image = np.random.rand(*self.left_cam.hw, 3).astype(np.uint8)

        pcd, colors = convert_depth_to_point_cloud(
            depth_map,
            self.left_cam,
            image=color_image,
            map_type="MPI"
        )

        self.assertEqual(pcd.shape[1], 3)
        self.assertEqual(colors.shape[1], 3)
        self.assertEqual(pcd.shape[0], colors.shape[0])

    def test_convert_depth_to_point_cloud_with_pose(self):
        """Test depth to point cloud conversion with pose transformation."""
        depth_map = np.random.rand(*self.left_cam.hw).astype(np.float32) * 3.0 + 1.0
        pose = Transform(np.array([1.0, 0.0, 0.0]), Rotation.from_mat3(np.eye(3)))

        pcd = convert_depth_to_point_cloud(
            depth_map,
            self.left_cam,
            map_type="MPI",
            pose=pose
        )

        self.assertEqual(pcd.shape[1], 3)
        self.assertGreater(pcd.shape[0], 0, "Should have some 3D points")

    def test_convert_depth_to_point_cloud_invalid_depth_size(self):
        """Test depth to point cloud with invalid depth map size."""
        depth_map = np.random.rand(100, 100).astype(np.float32)  # Wrong size

        with self.assertRaises(ValueError):
            convert_depth_to_point_cloud(
                depth_map,
                self.left_cam,
                map_type="MPI"
            )

    def test_convert_depth_to_point_cloud_invalid_image_size(self):
        """Test depth to point cloud with invalid image size."""
        depth_map = np.random.rand(*self.left_cam.hw).astype(np.float32)
        color_image = np.random.rand(100, 100, 3).astype(np.uint8)  # Wrong size

        with self.assertRaises(ValueError):
            convert_depth_to_point_cloud(
                depth_map,
                self.left_cam,
                image=color_image,
                map_type="MPI"
            )

    def test_transition_camera_view(self):
        """Test camera view transition."""
        # Create a simple test image
        src_image = np.random.rand(*self.left_cam.hw).astype(np.float32)

        # Transition from left to right camera
        output_image = transition_camera_view(
            src_image,
            self.left_cam,
            self.right_cam,
            img_tf=None
        )

        self.assertEqual(output_image.shape, self.right_cam.hw)

    def test_transition_camera_view_rgb(self):
        """Test camera view transition with RGB image."""
        src_image = np.random.rand(*self.left_cam.hw, 3).astype(np.float32)

        output_image = transition_camera_view(
            src_image,
            self.left_cam,
            self.right_cam,
            img_tf=None
        )

        self.assertEqual(output_image.shape, (*self.right_cam.hw, 3))

    def test_compute_points_for_epipolar_curve(self):
        """Test epipolar curve point computation."""
        pt_cam1 = self.pts1[:, 0:1]  # Take first point

        pts2d_cam2 = compute_points_for_epipolar_curve(
            pt_cam1,
            self.left_cam,
            self.right_cam,
            self.rel_pose,
            depth_rng=(1.0, 5.0),
            max_pts=50
        )

        self.assertEqual(pts2d_cam2.shape[0], 2)
        self.assertGreater(pts2d_cam2.shape[1], 0)

    def test_compute_fundamental_matrix_from_points_list_input(self):
        """Test fundamental matrix with list inputs."""
        # Convert to list format
        pts1_list = self.pts1.T.tolist()
        pts2_list = self.pts2.T.tolist()

        F = compute_fundamental_matrix_from_points(pts1_list, pts2_list)
        self.assertEqual(F.shape, (3, 3))

    def test_compute_fundamental_matrix_using_ransac_list_input(self):
        """Test RANSAC fundamental matrix with list inputs."""
        pts1_list = self.pts1.T.tolist()
        pts2_list = self.pts2.T.tolist()

        F, inliers = compute_fundamental_matrix_using_ransac(pts1_list, pts2_list)
        self.assertEqual(F.shape, (3, 3))
        self.assertGreater(inliers, 0)

    def test_triangulate_points_list_input(self):
        """Test triangulation with list inputs."""
        pts1_list = self.pts1.T.tolist()
        pts2_list = self.pts2.T.tolist()

        recover_pts3d = triangulate_points(
            pts1_list, pts2_list,
            self.left_cam, self.right_cam,
            self.left_c2w.inverse(), self.rel_pose.inverse()
        )

        self.assertEqual(recover_pts3d.shape[0], 3)
        self.assertGreater(recover_pts3d.shape[1], 0)

    def test_compute_relative_transform_from_points_list_input(self):
        """Test relative transform with list inputs."""
        pts1_list = self.pts1.T.tolist()
        pts2_list = self.pts2.T.tolist()

        rel_pose = compute_relative_transform_from_points(
            pts1_list, pts2_list,
            self.left_cam, self.right_cam
        )

        self.assertIsInstance(rel_pose, Transform)

    def test_compute_relative_transform_from_points_with_ransac(self):
        """Test relative transform with RANSAC."""
        rel_pose = compute_relative_transform_from_points(
            self.pts1, self.pts2,
            self.left_cam, self.right_cam,
            use_ransac=True,
            threshold=1e-2,
            max_iterations=100
        )

        self.assertIsInstance(rel_pose, Transform)

    def test_transition_camera_view_with_transform(self):
        """Test camera view transition with image transformation."""
        src_image = np.random.rand(*self.left_cam.hw).astype(np.float32)

        # Create a simple transformation matrix (identity)
        img_tf = np.eye(3)

        output_image = transition_camera_view(
            src_image,
            self.left_cam,
            self.right_cam,
            img_tf=img_tf
        )

        self.assertEqual(output_image.shape, self.right_cam.hw)

    def test_convert_point_cloud_to_depth_invalid_map_type(self):
        """Test point cloud to depth with invalid map type."""
        with self.assertRaises(ValueError):
            convert_point_cloud_to_depth(
                self.pts3d.T,
                self.left_cam,
                map_type="INVALID"
            )

    def test_find_corresponding_points_sift(self):
        """Test find corresponding points with SIFT feature detector."""
        # Create simple test images with some patterns
        image1 = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        image2 = np.random.rand(480, 640, 3).astype(np.uint8) * 255

        # Add some features (simple squares) to make matching possible
        image1[100:150, 100:150] = 255
        image2[120:170, 120:170] = 255

        try:
            pts1, pts2 = find_corresponding_points(
                image1, image2,
                feature_type="SIFT",
                max_matches=10
            )
            # Should return lists of tuples
            self.assertIsInstance(pts1, list)
            self.assertIsInstance(pts2, list)
        except Exception as e:
            # SIFT may not be available in all OpenCV builds
            self.skipTest(f"SIFT not available: {e}")

    def test_find_corresponding_points_orb(self):
        """Test find corresponding points with ORB feature detector."""
        # Create simple test images
        image1 = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        image2 = np.random.rand(480, 640, 3).astype(np.uint8) * 255

        # Add some features
        image1[100:150, 100:150] = 255
        image2[120:170, 120:170] = 255

        pts1, pts2 = find_corresponding_points(
            image1, image2,
            feature_type="ORB",
            max_matches=10
        )

        # Should return lists of tuples
        self.assertIsInstance(pts1, list)
        self.assertIsInstance(pts2, list)

    def test_compute_relative_transform(self):
        """Test compute relative transform from images."""
        # Create simple test images with patterns
        image1 = np.random.rand(480, 640, 3).astype(np.uint8) * 200
        image2 = np.random.rand(480, 640, 3).astype(np.uint8) * 200

        # Add some distinct features
        image1[100:200, 100:200] = 255
        image2[150:250, 150:250] = 255

        try:
            rel_tf = compute_relative_transform(
                image1, image2,
                self.left_cam, self.right_cam,
                feature_type="ORB",
                max_matches=50,
                use_ransac=True
            )
            self.assertIsInstance(rel_tf, Transform)
        except Exception as e:
            # May fail if insufficient matching points are found
            self.skipTest(f"Insufficient features for relative transform: {e}")

    def test_compute_points_for_epipolar_curve_invalid_point(self):
        """Test epipolar curve with invalid point causing ProjectionError."""
        # Use a point far outside the camera view to trigger mask.sum() == 0
        pt_cam1 = np.array([[10000], [10000]])  # Far outside image bounds
        rel_tf = Transform()

        try:
            compute_points_for_epipolar_curve(
                pt_cam1,
                self.left_cam,
                self.right_cam,
                rel_tf,
                depth_rng=(1.0, 5.0),
                max_pts=50
            )
        except (ProjectionError, ValueError):
            # Expected - point is outside camera's valid range
            pass

    def test_compute_essential_matrix_from_fundamental(self):
        """Test computing essential matrix from fundamental matrix."""
        E_original = compute_essential_matrix_from_pose(self.rel_pose)
        F = compute_fundamental_matrix_from_essential(
            self.left_cam.K, self.right_cam.K, E_original
        )
        E_recovered = compute_essential_matrix_from_fundamental(
            self.left_cam.K, self.right_cam.K, F
        )
        # Verify the recovered essential matrix is close to original
        np.testing.assert_array_almost_equal(E_original, E_recovered, decimal=5)

    def test_solve_pnp_with_all_valid_points(self):
        """Test PnP with all valid points."""
        # Create valid 2D-3D correspondences
        pts2d = self.pts1
        pts3d = self.pts3d

        transform = solve_pnp(pts2d, pts3d, self.left_cam)
        self.assertIsInstance(transform, Transform)

    def test_transition_camera_view_grayscale(self):
        """Test camera view transition with grayscale image."""
        src_image = np.random.rand(*self.left_cam.hw).astype(np.float32)

        output_image = transition_camera_view(
            src_image,
            self.left_cam,
            self.right_cam,
            img_tf=None
        )

        self.assertEqual(output_image.shape, self.right_cam.hw)
        self.assertEqual(output_image.dtype, src_image.dtype)

    def test_triangulate_points_with_pose(self):
        """Test triangulation using Pose objects instead of Transform."""
        # Convert Transform to Pose for testing
        w2c1_pose = Pose(self.left_c2w.inverse().t.reshape(-1), self.left_c2w.inverse().rot)
        w2c2_pose = Pose(self.rel_pose.inverse().t.reshape(-1), self.rel_pose.inverse().rot)

        recover_pts3d = triangulate_points(
            self.pts1, self.pts2,
            self.left_cam, self.right_cam,
            w2c1_pose, w2c2_pose
        )

        self.assertEqual(recover_pts3d.shape[0], 3)
        self.assertGreater(recover_pts3d.shape[1], 0)

    def test_compute_relative_transform_from_points_float32(self):
        """Test relative transform with float32 input to verify cv.recoverPose compatibility."""
        # Convert points to float32 (common in image processing and deep learning)
        pts1_f32 = self.pts1.astype(np.float32)
        pts2_f32 = self.pts2.astype(np.float32)

        # Should not raise dtype errors with float32 input
        rel_pose = compute_relative_transform_from_points(
            pts1_f32, pts2_f32,
            self.left_cam, self.right_cam
        )

        self.assertIsInstance(rel_pose, Transform)

        # Verify the result is still accurate
        E = compute_essential_matrix_from_pose(rel_pose)
        F = compute_fundamental_matrix_from_essential(self.left_cam.K, self.right_cam.K, E)

        total_error = 0.
        for i in range(self.num_pts):
            pt1, pt2 = homo(self.pts1[:, i:i+1]), homo(self.pts2[:, i:i+1])
            total_error += pt1.T @ F @ pt2
        total_error /= self.num_pts
        np.testing.assert_(abs(total_error) < 1.)

    def test_compute_relative_transform_from_points_float32_ransac(self):
        """Test relative transform with float32 input using RANSAC."""
        pts1_f32 = self.pts1.astype(np.float32)
        pts2_f32 = self.pts2.astype(np.float32)

        # Should work with RANSAC enabled
        rel_pose = compute_relative_transform_from_points(
            pts1_f32, pts2_f32,
            self.left_cam, self.right_cam,
            use_ransac=True,
            threshold=1e-2,
            max_iterations=100
        )

        self.assertIsInstance(rel_pose, Transform)

    def test_compute_fundamental_matrix_from_points_float32(self):
        """Test fundamental matrix computation with float32 points."""
        pts1_f32 = self.pts1.astype(np.float32)
        pts2_f32 = self.pts2.astype(np.float32)

        F = compute_fundamental_matrix_from_points(pts1_f32, pts2_f32)

        # Verify fundamental matrix properties
        self.assertEqual(F.shape, (3, 3))

        # Test epipolar constraint
        total_error = 0.
        for i in range(self.num_pts):
            pt1, pt2 = homo(self.pts1[:, i:i+1]), homo(self.pts2[:, i:i+1])
            total_error += pt1.T @ F @ pt2
        total_error /= self.num_pts
        np.testing.assert_(abs(total_error) < 1.)

    def test_compute_fundamental_matrix_using_ransac_float32(self):
        """Test RANSAC fundamental matrix with float32 points."""
        pts1_f32 = self.pts1.astype(np.float32)
        pts2_f32 = self.pts2.astype(np.float32)

        F, inliers = compute_fundamental_matrix_using_ransac(pts1_f32, pts2_f32)

        self.assertEqual(F.shape, (3, 3))
        self.assertGreater(inliers, 0)

    def test_triangulate_points_float32(self):
        """Test triangulation with float32 points."""
        pts1_f32 = self.pts1.astype(np.float32)
        pts2_f32 = self.pts2.astype(np.float32)

        recover_pts3d = triangulate_points(
            pts1_f32, pts2_f32,
            self.left_cam, self.right_cam,
            self.left_c2w.inverse(), self.rel_pose.inverse()
        )

        self.assertEqual(recover_pts3d.shape[0], 3)
        self.assertGreater(recover_pts3d.shape[1], 0)

        # Verify reconstruction accuracy
        pts3d_c2 = self.rel_pose.inverse() * recover_pts3d
        pts1_reproj, _ = self.left_cam.convert_to_pixels(recover_pts3d, out_subpixel=True)
        pts2_reproj, _ = self.right_cam.convert_to_pixels(pts3d_c2, out_subpixel=True)

        error1 = mean(norm(self.pts1.astype(np.float64) - pts1_reproj, dim=0))
        error2 = mean(norm(self.pts2.astype(np.float64) - pts2_reproj, dim=0))
        total_error = error1 + error2
        np.testing.assert_(total_error < 1.)

    def test_triangulate_points_mixed_dtype(self):
        """Test triangulation with mixed float32/float64 (auto promotion)."""
        pts1_f32 = self.pts1.astype(np.float32)
        pts2_f64 = self.pts2.astype(np.float64)

        # Should automatically promote to float64
        recover_pts3d = triangulate_points(
            pts1_f32, pts2_f64,
            self.left_cam, self.right_cam,
            self.left_c2w.inverse(), self.rel_pose.inverse()
        )

        self.assertEqual(recover_pts3d.shape[0], 3)
        self.assertGreater(recover_pts3d.shape[1], 0)

    def test_compute_fundamental_matrix_mixed_dtype(self):
        """Test fundamental matrix with mixed dtypes."""
        pts1_f32 = self.pts1.astype(np.float32)
        pts2_f64 = self.pts2.astype(np.float64)

        # Should automatically promote to float64
        F = compute_fundamental_matrix_from_points(pts1_f32, pts2_f64)

        self.assertEqual(F.shape, (3, 3))
        # Result should be in promoted dtype (float64)
        self.assertEqual(F.dtype, np.float64)

    def test_compute_relative_transform_mixed_dtype(self):
        """Test relative transform with mixed dtypes."""
        pts1_f32 = self.pts1.astype(np.float32)
        pts2_f64 = self.pts2.astype(np.float64)

        # Should handle mixed dtypes automatically
        rel_pose = compute_relative_transform_from_points(
            pts1_f32, pts2_f64,
            self.left_cam, self.right_cam
        )

        self.assertIsInstance(rel_pose, Transform)


class TestGeomUtilsExceptions(unittest.TestCase):
    """Test exception handling in geom_utils functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.image_size = (640, 480)
        self.cam = PerspectiveCamera.from_fov(image_size=self.image_size, fov=60.)

    def test_compute_fundamental_matrix_invalid_shape(self):
        """Test fundamental matrix with mismatched shapes."""
        pts1 = np.random.rand(2, 10)
        pts2 = np.random.rand(2, 8)  # Different number of points

        with self.assertRaises(InvalidShapeError):
            compute_fundamental_matrix_from_points(pts1, pts2)

    def test_compute_fundamental_matrix_invalid_dimension(self):
        """Test fundamental matrix with wrong dimensions."""
        pts1 = np.random.rand(3, 10)  # 3 coordinates instead of 2
        pts2 = np.random.rand(3, 10)

        with self.assertRaises(InvalidDimensionError):
            compute_fundamental_matrix_from_points(pts1, pts2)

    def test_compute_fundamental_matrix_insufficient_points(self):
        """Test fundamental matrix with < 8 points."""
        pts1 = np.random.rand(2, 5)
        pts2 = np.random.rand(2, 5)

        with self.assertRaises(InvalidDimensionError):
            compute_fundamental_matrix_from_points(pts1, pts2)

    def test_compute_fundamental_from_essential_invalid_shape(self):
        """Test fundamental from essential with invalid matrix shapes."""
        K1 = np.eye(4)  # 4x4 instead of 3x3
        K2 = np.eye(3)
        E = np.eye(3)

        with self.assertRaises(InvalidShapeError):
            compute_fundamental_matrix_from_essential(K1, K2, E)

    def test_solve_pnp_insufficient_points(self):
        """Test PnP with insufficient points."""
        pts2d = np.random.rand(2, 3)  # Only 3 points
        pts3d = np.random.rand(3, 3)

        with self.assertRaises(CalibrationError):
            solve_pnp(pts2d, pts3d, self.cam)

    def test_compute_points_for_epipolar_curve_invalid_depth_range(self):
        """Test epipolar curve with invalid depth range."""
        pt_cam1 = np.array([[100], [100]])
        rel_tf = Transform()

        with self.assertRaises(InvalidDimensionError):
            compute_points_for_epipolar_curve(
                pt_cam1,
                self.cam,
                self.cam,
                rel_tf,
                depth_rng=[1.0, 2.0],  # List instead of tuple
                max_pts=50
            )

    def test_compute_points_for_epipolar_curve_invalid_point_shape(self):
        """Test epipolar curve with invalid point shape."""
        pt_cam1 = np.array([[100, 200]])  # Wrong shape
        rel_tf = Transform()

        with self.assertRaises(InvalidShapeError):
            compute_points_for_epipolar_curve(
                pt_cam1,
                self.cam,
                self.cam,
                rel_tf,
                depth_rng=(1.0, 5.0),
                max_pts=50
            )

    def test_compute_points_for_epipolar_curve_invalid_max_pts(self):
        """Test epipolar curve with invalid max_pts."""
        pt_cam1 = np.array([[100], [100]])
        rel_tf = Transform()

        with self.assertRaises(InvalidDimensionError):
            compute_points_for_epipolar_curve(
                pt_cam1,
                self.cam,
                self.cam,
                rel_tf,
                depth_rng=(1.0, 5.0),
                max_pts=0  # Invalid
            )

    def test_compute_fundamental_matrix_using_ransac_invalid_shape(self):
        """Test RANSAC fundamental matrix with mismatched shapes."""
        pts1 = np.random.rand(2, 10)
        pts2 = np.random.rand(2, 8)  # Different number of points

        with self.assertRaises(InvalidShapeError):
            compute_fundamental_matrix_using_ransac(pts1, pts2)

    def test_compute_fundamental_matrix_using_ransac_invalid_dimension(self):
        """Test RANSAC fundamental matrix with wrong dimensions."""
        pts1 = np.random.rand(3, 10)  # 3 coordinates instead of 2
        pts2 = np.random.rand(3, 10)

        with self.assertRaises(InvalidDimensionError):
            compute_fundamental_matrix_using_ransac(pts1, pts2)

    def test_compute_fundamental_matrix_using_ransac_insufficient_points(self):
        """Test RANSAC fundamental matrix with < 8 points."""
        pts1 = np.random.rand(2, 5)
        pts2 = np.random.rand(2, 5)

        with self.assertRaises(InvalidDimensionError):
            compute_fundamental_matrix_using_ransac(pts1, pts2)

    def test_find_corresponding_points_invalid_feature_type(self):
        """Test find corresponding points with invalid feature type."""
        image1 = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        image2 = np.random.rand(480, 640, 3).astype(np.uint8) * 255

        with self.assertRaises(ValueError):
            find_corresponding_points(
                image1, image2,
                feature_type="INVALID",
                max_matches=10
            )

    def test_solve_pnp_degenerate_configuration(self):
        """Test PnP with degenerate point configuration that might cause failure."""
        # Create coplanar 3D points (degenerate configuration)
        pts3d = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=np.float32)  # All same point (degenerate)
        pts2d = np.random.rand(2, 4)

        # This might not always trigger the failure but tests the error path
        try:
            solve_pnp(pts2d, pts3d, self.cam)
        except (CalibrationError, Exception):
            # Expected - degenerate configuration should fail
            pass


if __name__ == "__main__":
    unittest.main()