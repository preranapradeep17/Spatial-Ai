"""Pinhole camera geometry, depth sampling, and 3D coordinate transformations."""
from typing import Tuple, List, Optional
import numpy as np

def backproject_pixel_to_camera_point(
    u: float,
    v: float,
    depth: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> Tuple[float, float, float]:
    """Back-projects image pixel (u, v) with known depth into 3D camera-space coordinates.

    Math:
        X_cam = (u - cx) * depth / fx
        Y_cam = (v - cy) * depth / fy
        Z_cam = depth

    Note: In computer vision image coordinates, v increases downward.
    For standard optical camera frames:
    X points right, Y points down, Z points forward.
    """
    if depth <= 0.0 or fx == 0.0 or fy == 0.0:
        return (0.0, 0.0, 0.0)

    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = float(depth)
    return (float(x_cam), float(y_cam), float(z_cam))

def sample_robust_depth_from_box(
    depth_map: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    sampling_ratio: float = 0.4
) -> Optional[float]:
    """Extracts a robust depth estimate by computing the median of the inner bounding box patch.

    This avoids background depth bleeding near object contours.
    """
    h, w = depth_map.shape[:2]
    # Clamp bounding box coordinates
    x1 = max(0, min(w - 1, int(xmin)))
    y1 = max(0, min(h - 1, int(ymin)))
    x2 = max(0, min(w, int(xmax)))
    y2 = max(0, min(h, int(ymax)))

    if x2 <= x1 or y2 <= y1:
        return None

    # Calculate inner patch
    box_w = x2 - x1
    box_h = y2 - y1
    inset_x = int(box_w * (1.0 - sampling_ratio) / 2.0)
    inset_y = int(box_h * (1.0 - sampling_ratio) / 2.0)

    patch_x1 = x1 + inset_x
    patch_x2 = max(patch_x1 + 1, x2 - inset_x)
    patch_y1 = y1 + inset_y
    patch_y2 = max(patch_y1 + 1, y2 - inset_y)

    patch = depth_map[patch_y1:patch_y2, patch_x1:patch_x2]
    # Filter valid depth: non-zero and non-inf/nan
    valid = patch[(patch > 0.1) & np.isfinite(patch)]
    if len(valid) == 0:
        return None

    # Return median depth in meters
    return float(np.median(valid))

def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """Converts a unit quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = q
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    else:
        return np.eye(3)

    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float32)

def camera_to_world_transform(
    camera_point: Tuple[float, float, float],
    camera_pos: List[float],
    camera_rot_quat: List[float]
) -> List[float]:
    """Transforms a 3D camera-space point to Unity 3D world space.

    In Unity:
    X points Right, Y points Up, Z points Forward.
    Camera optical frame: X right, Y down, Z forward.
    We convert camera optical Y (down) to Unity Y (up) by negating Y before applying camera rotation.
    """
    # Optical frame -> Unity camera local frame
    x_c, y_c, z_c = camera_point
    p_local = np.array([x_c, -y_c, z_c], dtype=np.float32)

    R = quaternion_to_rotation_matrix(camera_rot_quat)
    T = np.array(camera_pos, dtype=np.float32)

    p_world = R @ p_local + T
    return [float(p_world[0]), float(p_world[1]), float(p_world[2])]


def is_object_in_camera_fov(
    object_pos: List[float],
    camera_pos: List[float],
    camera_yaw_deg: float = 0.0,
    fov_deg: float = 60.0,
    max_distance: float = 30.0
) -> bool:
    """Calculates whether a 3D object position is within the camera's horizontal Field of View (FOV).
    
    Academic ground truth: CS4231 Camera Frustum & Field of View.
    - Bearing angle: theta = atan2(X_obj - X_cam, Z_obj - Z_cam)
    - Angular delta: |(theta - yaw + pi) % (2*pi) - pi|
    - Frustum test: delta <= radians(fov / 2) and Euclidean distance <= max_distance
    """
    import math

    dx = object_pos[0] - camera_pos[0]
    dz = object_pos[2] - camera_pos[2] if len(object_pos) > 2 else object_pos[1] - camera_pos[1]
    dist = math.sqrt(dx * dx + dz * dz)

    if dist < 0.01:
        return True
    if dist > max_distance:
        return False

    bearing = math.atan2(dx, dz)
    yaw_rad = math.radians(camera_yaw_deg)
    diff = (bearing - yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff) <= math.radians(fov_deg / 2.0)

