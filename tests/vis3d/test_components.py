"""Tests for vis3d plane creation functions."""

import numpy as np
import pytest

from spatialkit.vis3d import (
    create_axis_aligned_plane,
    create_plane_from_corners,
    create_plane_from_normal,
)
from spatialkit.common.exceptions import InvalidShapeError, GeometryError


class TestCreateAxisAlignedPlane:
    """Tests for create_axis_aligned_plane function."""

    def test_xy_plane_basic(self):
        """Test creating a basic XY plane."""
        plane = create_axis_aligned_plane("xy", width=2.0, height=2.0)

        vertices = np.asarray(plane.vertices)
        triangles = np.asarray(plane.triangles)

        assert len(vertices) == 4
        assert len(triangles) == 2
        # All Z coordinates should be 0
        np.testing.assert_array_almost_equal(vertices[:, 2], [0, 0, 0, 0])

    def test_xz_plane_basic(self):
        """Test creating a basic XZ plane."""
        plane = create_axis_aligned_plane("xz", width=2.0, height=2.0)

        vertices = np.asarray(plane.vertices)
        # All Y coordinates should be 0
        np.testing.assert_array_almost_equal(vertices[:, 1], [0, 0, 0, 0])

    def test_yz_plane_basic(self):
        """Test creating a basic YZ plane."""
        plane = create_axis_aligned_plane("yz", width=2.0, height=2.0)

        vertices = np.asarray(plane.vertices)
        # All X coordinates should be 0
        np.testing.assert_array_almost_equal(vertices[:, 0], [0, 0, 0, 0])

    def test_with_offset(self):
        """Test plane with offset."""
        plane = create_axis_aligned_plane("xy", width=1.0, height=1.0, offset=5.0)

        vertices = np.asarray(plane.vertices)
        # All Z coordinates should be 5.0
        np.testing.assert_array_almost_equal(vertices[:, 2], [5, 5, 5, 5])

    def test_with_pose(self):
        """Test plane with pose transformation."""
        pose = np.eye(4)
        pose[:3, 3] = [1.0, 2.0, 3.0]  # Translation

        plane = create_axis_aligned_plane("xy", width=1.0, height=1.0, pose=pose)

        vertices = np.asarray(plane.vertices)
        # Center should be at (1, 2, 3)
        center = vertices.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [1.0, 2.0, 3.0])

    def test_custom_color(self):
        """Test plane with custom color."""
        plane = create_axis_aligned_plane("xy", width=1.0, height=1.0, color=(255, 0, 0))

        colors = np.asarray(plane.vertex_colors)
        expected = np.array([[1.0, 0.0, 0.0]] * 4)
        np.testing.assert_array_almost_equal(colors, expected)

    def test_invalid_plane_string(self):
        """Test that invalid plane string raises GeometryError."""
        with pytest.raises(GeometryError):
            create_axis_aligned_plane("abc", width=1.0, height=1.0)

        with pytest.raises(GeometryError):
            create_axis_aligned_plane("XY", width=1.0, height=1.0)  # Case sensitive

    def test_invalid_pose_shape(self):
        """Test that invalid pose shape raises InvalidShapeError."""
        with pytest.raises(InvalidShapeError):
            create_axis_aligned_plane("xy", width=1.0, height=1.0, pose=np.eye(3))


class TestCreatePlaneFromCorners:
    """Tests for create_plane_from_corners function."""

    def test_basic_square(self):
        """Test creating a plane from 4 corners."""
        corners = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ])
        plane = create_plane_from_corners(corners)

        assert len(plane.vertices) == 4
        assert len(plane.triangles) == 2

    def test_tilted_plane(self):
        """Test creating a tilted plane."""
        corners = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.0],
        ])
        plane = create_plane_from_corners(corners)
        assert len(plane.vertices) == 4

    def test_with_pose(self):
        """Test plane with pose transformation."""
        corners = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ])
        pose = np.eye(4)
        pose[:3, 3] = [10.0, 0.0, 0.0]

        plane = create_plane_from_corners(corners, pose=pose)

        vertices = np.asarray(plane.vertices)
        center = vertices.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [10.0, 0.0, 0.0])

    def test_colinear_corners(self):
        """Test that colinear corners raise GeometryError."""
        corners = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        with pytest.raises(GeometryError):
            create_plane_from_corners(corners)

    def test_invalid_corners_shape_2d(self):
        """Test that 2D corners raise InvalidShapeError."""
        with pytest.raises(InvalidShapeError):
            create_plane_from_corners(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))

    def test_invalid_corners_shape_3_points(self):
        """Test that 3 corners raise InvalidShapeError."""
        with pytest.raises(InvalidShapeError):
            create_plane_from_corners(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))

    def test_invalid_pose_shape(self):
        """Test that invalid pose shape raises InvalidShapeError."""
        corners = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ])
        with pytest.raises(InvalidShapeError):
            create_plane_from_corners(corners, pose=np.eye(3))


class TestCreatePlaneFromNormal:
    """Tests for create_plane_from_normal function."""

    def test_z_up_normal(self):
        """Test creating a plane with Z-up normal."""
        normal = np.array([0.0, 0.0, 1.0])
        plane = create_plane_from_normal(normal, distance=0.0, width=2.0, height=2.0)

        vertices = np.asarray(plane.vertices)
        assert len(vertices) == 4
        # All Z should be 0 (distance=0)
        np.testing.assert_array_almost_equal(vertices[:, 2], [0, 0, 0, 0])

    def test_z_up_normal_with_distance(self):
        """Test creating a plane with Z-up normal and distance."""
        normal = np.array([0.0, 0.0, 1.0])
        plane = create_plane_from_normal(normal, distance=5.0, width=2.0, height=2.0)

        vertices = np.asarray(plane.vertices)
        # All Z should be 5
        np.testing.assert_array_almost_equal(vertices[:, 2], [5, 5, 5, 5])

    def test_x_forward_normal(self):
        """Test creating a plane with X-forward normal."""
        normal = np.array([1.0, 0.0, 0.0])
        plane = create_plane_from_normal(normal, distance=3.0, width=1.0, height=1.0)

        vertices = np.asarray(plane.vertices)
        # All X should be 3
        np.testing.assert_array_almost_equal(vertices[:, 0], [3, 3, 3, 3])

    def test_arbitrary_normal(self):
        """Test creating a plane with arbitrary normal."""
        normal = np.array([1.0, 1.0, 1.0])
        plane = create_plane_from_normal(normal, distance=1.0, width=1.0, height=1.0)

        assert len(plane.vertices) == 4

    def test_non_unit_normal(self):
        """Test that non-unit normals are normalized."""
        normal = np.array([0.0, 0.0, 10.0])  # Non-unit
        plane = create_plane_from_normal(normal, distance=2.0, width=1.0, height=1.0)

        vertices = np.asarray(plane.vertices)
        # Should be normalized, so Z = 2
        np.testing.assert_array_almost_equal(vertices[:, 2], [2, 2, 2, 2])

    def test_with_pose(self):
        """Test plane with additional pose transformation."""
        normal = np.array([0.0, 0.0, 1.0])
        pose = np.eye(4)
        pose[:3, 3] = [1.0, 2.0, 3.0]

        plane = create_plane_from_normal(
            normal, distance=0.0, width=1.0, height=1.0, pose=pose
        )

        vertices = np.asarray(plane.vertices)
        center = vertices.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [1.0, 2.0, 3.0])

    def test_invalid_normal_shape(self):
        """Test that invalid normal shape raises InvalidShapeError."""
        with pytest.raises(InvalidShapeError):
            create_plane_from_normal(
                np.array([1.0, 0.0]), distance=0.0, width=1.0, height=1.0
            )

    def test_zero_normal(self):
        """Test that zero normal raises GeometryError."""
        with pytest.raises(GeometryError):
            create_plane_from_normal(
                np.array([0.0, 0.0, 0.0]), distance=0.0, width=1.0, height=1.0
            )

    def test_invalid_pose_shape(self):
        """Test that invalid pose shape raises InvalidShapeError."""
        with pytest.raises(InvalidShapeError):
            create_plane_from_normal(
                np.array([0.0, 0.0, 1.0]),
                distance=0.0,
                width=1.0,
                height=1.0,
                pose=np.eye(3),
            )


class TestPlaneGeometry:
    """Tests for plane geometry correctness."""

    def test_plane_dimensions_xy(self):
        """Test that XY plane has correct dimensions."""
        width, height = 4.0, 2.0
        plane = create_axis_aligned_plane("xy", width=width, height=height)

        vertices = np.asarray(plane.vertices)
        actual_width = vertices[:, 0].max() - vertices[:, 0].min()
        actual_height = vertices[:, 1].max() - vertices[:, 1].min()

        np.testing.assert_almost_equal(actual_width, width)
        np.testing.assert_almost_equal(actual_height, height)

    def test_plane_centered_at_origin(self):
        """Test that planes are centered at origin by default."""
        plane = create_axis_aligned_plane("xy", width=2.0, height=2.0)

        vertices = np.asarray(plane.vertices)
        center = vertices.mean(axis=0)

        np.testing.assert_array_almost_equal(center, [0, 0, 0])

    def test_corners_plane_preserves_vertices(self):
        """Test that corners are preserved in the mesh."""
        corners = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ])
        plane = create_plane_from_corners(corners)

        vertices = np.asarray(plane.vertices)
        np.testing.assert_array_almost_equal(vertices, corners)
