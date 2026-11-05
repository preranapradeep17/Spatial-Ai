"""3D Spatial Localization and Evaluation against Physical Ground Truth.

Computes 3D camera-space coordinates from 2D bounding boxes and physical depth maps
via calibrated pinhole back-projection, evaluating 3D Euclidean localization error
against ground-truth physical 3D centroids supplied by the SUN RGB-D benchmark.
"""
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from backend.spatial.geometry import sample_robust_depth_from_box, backproject_pixel_to_camera_point

def estimate_3d_position_from_bbox(
    bbox: Dict[str, float],
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    r_tilt: Optional[np.ndarray] = None
) -> Tuple[Optional[List[float]], Optional[float]]:
    """Estimates metric 3D position [X, Y, Z] from 2D bounding box and depth map.

    If r_tilt is provided, transforms from optical frame (X right, Y down, Z forward)
    into the tilt-aligned canonical frame (X right, Y forward, Z up) matching SUN RGB-D 3D ground truth.
    """
    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    # 1. Robust depth sampling from inner 40% patch
    depth_est = sample_robust_depth_from_box(depth_map, xmin, ymin, xmax, ymax, sampling_ratio=0.4)
    if depth_est is None or depth_est <= 0.1:
        return None, None

    # 2. Box center in pixel coordinates
    u_center = (xmin + xmax) / 2.0
    v_center = (ymin + ymax) / 2.0

    # 3. Pinhole back-projection
    x_c, y_c, z_c = backproject_pixel_to_camera_point(
        u=u_center,
        v=v_center,
        depth=depth_est,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    if r_tilt is not None:
        # Optical -> Canonical SUN frame: X right, Y forward (depth), Z up (-y)
        p_canonical = np.array([x_c, z_c, -y_c], dtype=np.float32)
        p_aligned = r_tilt @ p_canonical
        return [round(float(p_aligned[0]), 3), round(float(p_aligned[1]), 3), round(float(p_aligned[2]), 3)], round(depth_est, 3)

    return [round(x_c, 3), round(y_c, 3), round(z_c, 3)], round(depth_est, 3)

def evaluate_3d_localization_error(
    estimated_pos: List[float],
    ground_truth_pos: List[float]
) -> float:
    """Computes 3D Euclidean distance error between estimated and physical ground-truth centroid (in meters)."""
    dx = estimated_pos[0] - ground_truth_pos[0]
    dy = estimated_pos[1] - ground_truth_pos[1]
    dz = estimated_pos[2] - ground_truth_pos[2]
    return float(math.sqrt(dx**2 + dy**2 + dz**2))

def benchmark_localization_on_dataset(
    dataset_dir: str = "data/sunrgbd"
) -> Dict[str, Any]:
    """Runs calibrated 3D localization across all annotated test samples and calculates error statistics."""
    import glob
    import json
    import os
    import scipy.io

    annot_files = sorted(glob.glob(os.path.join(dataset_dir, "annotations", "*.json")))
    mat3d_path = os.path.join(dataset_dir, "metadata", "SUNRGBDMeta3DBB_v2.mat")
    mat3d = None
    if os.path.exists(mat3d_path):
        mat3d = scipy.io.loadmat(mat3d_path, squeeze_me=True, struct_as_record=False)["SUNRGBDMeta"]

    errors = []
    per_class_errors: Dict[str, List[float]] = {}

    for af in annot_files:
        with open(af, "r", encoding="utf-8") as f:
            meta = json.load(f)

        depth_path = os.path.join(dataset_dir, meta["depth_filename"])
        if not os.path.exists(depth_path):
            continue
        depth_map = np.load(depth_path)

        cam = meta["camera"]
        fx, fy = cam["fx"], cam["fy"]
        cx, cy = cam["cx"], cam["cy"]

        # Find Rtilt if available
        r_tilt = None
        if mat3d is not None:
            seq = meta.get("original_sequence", "")
            for m in mat3d:
                if seq in getattr(m, "sequenceName", ""):
                    r_tilt = getattr(m, "Rtilt", None)
                    break

        for obj in meta.get("objects", []):
            gt_3d = obj.get("position_camera_3d")
            if gt_3d is None:
                continue

            est_3d, _ = estimate_3d_position_from_bbox(
                bbox=obj["bbox_2d"],
                depth_map=depth_map,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                r_tilt=r_tilt
            )

            if est_3d is not None:
                err = evaluate_3d_localization_error(est_3d, gt_3d)
                errors.append(err)
                cname = obj["class_name"]
                if cname not in per_class_errors:
                    per_class_errors[cname] = []
                per_class_errors[cname].append(err)

    if not errors:
        return {"num_evaluated": 0, "mean_error_meters": 0.0}

    return {
        "num_evaluated": len(errors),
        "mean_error_meters": round(float(np.mean(errors)), 3),
        "median_error_meters": round(float(np.median(errors)), 3),
        "std_error_meters": round(float(np.std(errors)), 3),
        "per_class_mean_error_meters": {
            c: round(float(np.mean(errs)), 3) for c, errs in per_class_errors.items()
        }
    }
