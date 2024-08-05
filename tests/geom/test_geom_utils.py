import unittest
import numpy as np
from cv_utils.geom.camera import *
from cv_utils.geom.geom_utils import *
from cv_utils.ops.uops import *
from cv_utils.ops.umath import *


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

    def test_find_corresponding_points(self):
        pass

    def test_convert_point_cloud_to_depth(self):
        pass
    def test_convert_depth_to_point_cloud(self):
        pass



if __name__ == "__main__":
    unittest.main()