"""Dataset Validation and Verification Script for SpatialAI.

Validates annotations, image integrity, depth arrays, bounding box boundaries,
coordinate geometry, and class distributions.
Generates visual verification overlays with drawn bounding boxes for inspection.
"""
import os
import glob
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = ["table", "chair", "laptop", "monitor", "door", "fire_extinguisher"]

def validate_dataset(data_dir: str, save_visual_debug: bool = True) -> dict:
    """Validates all samples in data_dir and returns diagnostic summary."""
    annotations_dir = os.path.join(data_dir, "annotations")
    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    annot_files = sorted(glob.glob(os.path.join(annotations_dir, "*.json")))
    if not annot_files:
        raise ValueError(f"No annotation JSON files found in {annotations_dir}")

    stats = {
        "total_samples": len(annot_files),
        "total_objects": 0,
        "class_counts": {c: 0 for c in CLASS_NAMES},
        "errors": [],
        "warnings": [],
        "passed_samples": 0
    }

    debug_dir = os.path.join("results", "predictions", "validation_debug")
    if save_visual_debug:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"Validating {len(annot_files)} annotation files from {data_dir}...")

    for a_file in annot_files:
        sample_errors = []
        try:
            with open(a_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            stats["errors"].append(f"{a_file}: JSON parse error: {str(e)}")
            continue

        # Check required fields
        for req in ["frame_id", "image_filename", "depth_filename", "camera", "objects"]:
            if req not in data:
                sample_errors.append(f"Missing required field '{req}'")

        if sample_errors:
            stats["errors"].extend([f"{a_file}: {err}" for err in sample_errors])
            continue

        # Check RGB file
        rgb_path = os.path.join(data_dir, data["image_filename"])
        if not os.path.isfile(rgb_path):
            sample_errors.append(f"Referenced RGB file missing: {rgb_path}")
            img = None
        else:
            try:
                img = Image.open(rgb_path)
                if img.size != (data["camera"]["image_width"], data["camera"]["image_height"]):
                    sample_errors.append(f"Image dimension mismatch: {img.size} vs {data['camera']['image_width']}x{data['camera']['image_height']}")
            except Exception as e:
                sample_errors.append(f"Failed to load RGB image: {str(e)}")
                img = None

        # Check depth file
        depth_path = os.path.join(data_dir, data["depth_filename"])
        if not os.path.isfile(depth_path):
            sample_errors.append(f"Referenced depth file missing: {depth_path}")
            depth_map = None
        else:
            try:
                depth_map = np.load(depth_path)
                if depth_map.shape != (data["camera"]["image_height"], data["camera"]["image_width"]):
                    sample_errors.append(f"Depth shape mismatch: {depth_map.shape}")
                if np.isnan(depth_map).any() or np.isinf(depth_map).any():
                    sample_errors.append("Depth map contains NaN or Inf values")
                if (depth_map < 0).any():
                    sample_errors.append("Depth map contains negative values")
            except Exception as e:
                sample_errors.append(f"Failed to load depth array: {str(e)}")
                depth_map = None

        # Camera intrinsics check
        cam = data["camera"]
        if cam["fx"] <= 0 or cam["fy"] <= 0:
            sample_errors.append(f"Invalid focal lengths: fx={cam['fx']}, fy={cam['fy']}")

        # Objects validation
        w_img = cam["image_width"]
        h_img = cam["image_height"]

        if save_visual_debug and img is not None:
            vis_img = img.copy()
            draw = ImageDraw.Draw(vis_img)
        else:
            vis_img, draw = None, None

        for obj in data["objects"]:
            stats["total_objects"] += 1
            c_name = obj.get("class_name")
            if c_name in stats["class_counts"]:
                stats["class_counts"][c_name] += 1
            else:
                sample_errors.append(f"Unknown class name: {c_name}")

            # BBox checks
            box = obj.get("bbox_2d", {})
            xmin, ymin, xmax, ymax = box.get("xmin"), box.get("ymin"), box.get("xmax"), box.get("ymax")

            if any(v is None for v in [xmin, ymin, xmax, ymax]):
                sample_errors.append(f"Incomplete bbox in object {obj.get('object_id')}")
                continue

            if xmin < 0 or ymin < 0 or xmax > w_img or ymax > h_img:
                sample_errors.append(f"BBox out of bounds: [{xmin}, {ymin}, {xmax}, {ymax}] for {w_img}x{h_img}")
            if xmin >= xmax or ymin >= ymax:
                sample_errors.append(f"Inverted bbox coordinates: [{xmin}, {ymin}, {xmax}, {ymax}]")

            # Check 3D back-projection consistency
            pos_cam = obj.get("position_camera_3d")
            if pos_cam and len(pos_cam) == 3:
                x_cam, y_cam, z_cam = pos_cam
                u_center = (xmin + xmax) / 2.0
                # Computed X from pinhole model: X = (u - cx) * Z / fx
                x_expected = (u_center - cam["cx"]) * z_cam / cam["fx"]
                if abs(x_cam - x_expected) > 0.4:
                    stats["warnings"].append(f"{data['frame_id']}: 3D X discrepancy for {c_name}: annotated {x_cam} vs backprojected {round(x_expected, 2)}")

            # Visual overlay
            if draw is not None:
                draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0), width=2)
                label = f"{c_name} (Z={round(obj.get('distance_to_camera', 0), 1)}m)"
                draw.text((xmin + 3, max(0, ymin - 12)), label, fill=(255, 255, 0))

        if vis_img is not None:
            debug_out = os.path.join(debug_dir, f"vis_{data['frame_id']}.png")
            vis_img.save(debug_out)

        if sample_errors:
            stats["errors"].extend([f"{a_file}: {e}" for e in sample_errors])
        else:
            stats["passed_samples"] += 1

    print("=" * 60)
    print("Dataset Validation Report:")
    print(f"Total Samples Inspected: {stats['total_samples']}")
    print(f"Passed Validation:        {stats['passed_samples']} / {stats['total_samples']}")
    print(f"Total Objects Annotated: {stats['total_objects']}")
    print("Class Distribution:")
    for c, cnt in stats["class_counts"].items():
        print(f"  - {c:18s}: {cnt}")
    print(f"Warnings: {len(stats['warnings'])}")
    print(f"Errors:   {len(stats['errors'])}")
    if stats["errors"]:
        print("First 3 Errors:")
        for err in stats["errors"][:3]:
            print("  *", err)
    print("=" * 60)

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SpatialAI dataset.")
    parser.add_argument("--data_dir", type=str, default="data/generated/sample_smoke")
    args = parser.parse_args()

    results = validate_dataset(args.data_dir)
    if results["errors"]:
        exit(1)
    else:
        print("All dataset checks PASSED successfully.")
        exit(0)
