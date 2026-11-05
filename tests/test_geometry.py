"""Unit tests for camera geometry, pinhole back-projection, and depth sampling."""
import pytest
import numpy as np
from backend.spatial.geometry import (
    backproject_pixel_to_camera_point,
    sample_robust_depth_from_box,
    quaternion_to_rotation_matrix,
    camera_to_world_transform
)

def test_pinhole_backprojection_center():
    """Pixel at principal point (cx, cy) back-projects directly along camera forward optical axis."""
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    depth = 2.0

    x_c, y_c, z_c = backproject_pixel_to_camera_point(cx, cy, depth, fx, fy, cx, cy)
    assert x_c == 0.0
    assert y_c == 0.0
    assert z_c == 2.0

def test_pinhole_backprojection_offset():
    """Pixel shifted horizontally by fx pixels gives X_cam = depth."""
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    depth = 3.0

    # u = cx + fx -> (u - cx) * depth / fx = depth
    x_c, y_c, z_c = backproject_pixel_to_camera_point(cx + fx, cy, depth, fx, fy, cx, cy)
    assert np.isclose(x_c, 3.0)
    assert np.isclose(y_c, 0.0)
    assert np.isclose(z_c, 3.0)

def test_sample_robust_depth_median():
    """Verify robust depth sampling filters noisy borders and returns median of inner patch."""
    depth_map = np.ones((100, 100), dtype=np.float32) * 5.0
    # Add noise on outer borders
    depth_map[10:90, 10:90] = 3.2
    # Inset patch should capture 3.2
    med = sample_robust_depth_from_box(depth_map, xmin=10, ymin=10, xmax=90, ymax=90)
    assert med is not None
    assert np.isclose(med, 3.2)

def test_quaternion_identity_rotation():
    """Identity quaternion [0, 0, 0, 1] gives 3x3 identity matrix."""
    q_id = [0.0, 0.0, 0.0, 1.0]
    R = quaternion_to_rotation_matrix(q_id)
    assert np.allclose(R, np.eye(3))

def test_camera_to_world_transform_identity():
    """Camera at [0, 1.5, 0] with zero rotation transforms local [1, 0, 2] to world [1, 1.5, 2]."""
    cam_pos = [0.0, 1.5, 0.0]
    cam_rot = [0.0, 0.0, 0.0, 1.0]
    p_cam = (1.0, 0.0, 2.0)  # X=1 (right), Y=0, Z=2 (forward)

    p_world = camera_to_world_transform(p_cam, cam_pos, cam_rot)
    assert np.isclose(p_world[0], 1.0)
    assert np.isclose(p_world[1], 1.5)
    assert np.isclose(p_world[2], 2.0)
