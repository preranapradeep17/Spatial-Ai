"""Standalone Synthetic Dataset Generator for SpatialAI.

Generates synthetic laboratory RGB frames, depth arrays (.npy), and ground-truth
annotations matching the exact schema exported by the Unity SyntheticDataCapture module.
Allows reproducible offline testing, CI verification, and smoke tests.
"""
import os
import json
import math
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw

CLASS_MAP = {
    0: "table",
    1: "chair",
    2: "laptop",
    3: "monitor",
    4: "door",
    5: "fire_extinguisher"
}

CLASS_COLORS = {
    0: (180, 130, 80),   # Wood Brown Table
    1: (60, 60, 80),      # Dark Grey Chair
    2: (190, 195, 200),  # Silver Laptop
    3: (30, 30, 35),     # Black Monitor
    4: (220, 225, 220),  # White/Grey Door (with green sign)
    5: (220, 30, 30)     # Red Fire Extinguisher
}

def generate_sample(
    sample_idx: int,
    output_dir: str,
    width: int = 640,
    height: int = 480,
    fx: float = 525.0,
    fy: float = 525.0,
    cx: float = 320.0,
    cy: float = 240.0
) -> dict:
    """Generates one synthetic RGB-D frame and corresponding annotation."""
    # Initialize background canvas (lab walls and floor)
    img = Image.new("RGB", (width, height), color=(230, 235, 240))
    draw = ImageDraw.Draw(img)

    # Floor plane
    horizon_y = int(height * 0.45)
    draw.rectangle([0, horizon_y, width, height], fill=(210, 210, 215))
    # Wall baseline
    draw.line([0, horizon_y, width, horizon_y], fill=(160, 160, 170), width=2)

    # Initialize depth map with background plane distances
    depth_map = np.ones((height, width), dtype=np.float32) * 8.0
    for y in range(horizon_y, height):
        # Floor depth gradient from 8.0m at horizon down to 1.5m at bottom
        depth_val = 8.0 - ((y - horizon_y) / (height - horizon_y)) * 6.5
        depth_map[y, :] = depth_val

    objects_data = []

    # Camera pose
    cam_pos = [0.0, 1.5, -random.uniform(2.5, 3.5)]
    cam_rot = [0.0, 0.0, 0.0, 1.0]

    # Spawn 3 to 6 objects per scene
    num_objs = random.randint(3, 6)
    selected_classes = random.sample(list(CLASS_MAP.keys()), min(num_objs, len(CLASS_MAP)))

    # Sort objects by depth so foreground renders over background
    scene_objects = []
    for c_id in selected_classes:
        c_name = CLASS_MAP[c_id]
        if c_id == 4:  # Door on back wall
            z = random.uniform(5.5, 7.0)
            x = random.choice([-2.0, 2.0])
            y = 1.2
            w_3d, h_3d = 1.1, 2.1
        elif c_id == 0:  # Table
            z = random.uniform(2.6, 3.5)
            x = random.uniform(-0.5, 0.5)
            y = 0.75
            w_3d, h_3d = 1.6, 0.75
        elif c_id == 1:  # Chair
            z = random.uniform(2.2, 3.0)
            x = random.uniform(-1.5, -0.8)
            y = 0.5
            w_3d, h_3d = 0.6, 0.9
        elif c_id == 2:  # Laptop
            z = random.uniform(2.4, 3.2)
            x = random.uniform(-0.4, 0.2)
            y = 0.82
            w_3d, h_3d = 0.35, 0.25
        elif c_id == 3:  # Monitor
            z = random.uniform(2.7, 3.4)
            x = random.uniform(0.1, 0.6)
            y = 0.95
            w_3d, h_3d = 0.55, 0.45
        elif c_id == 5:  # Fire Extinguisher
            z = random.uniform(4.0, 5.5)
            x = random.uniform(1.8, 2.4)
            y = 0.9
            w_3d, h_3d = 0.25, 0.7
        else:
            z = random.uniform(2.5, 4.0)
            x = random.uniform(-1.0, 1.0)
            y = 0.8
            w_3d, h_3d = 0.5, 0.5

        scene_objects.append((z, x, y, w_3d, h_3d, c_id, c_name))

    # Sort by depth descending (back to front)
    scene_objects.sort(key=lambda item: item[0], reverse=True)

    for idx, (z, x, y, w_3d, h_3d, c_id, c_name) in enumerate(scene_objects):
        # Optical projection
        u_center = (x * fx) / z + cx
        v_center = (- (y - cam_pos[1]) * fy) / z + cy

        box_w_px = (w_3d * fx) / z
        box_h_px = (h_3d * fy) / z

        xmin = max(0.0, u_center - box_w_px / 2.0)
        xmax = min(float(width), u_center + box_w_px / 2.0)
        ymin = max(0.0, v_center - box_h_px / 2.0)
        ymax = min(float(height), v_center + box_h_px / 2.0)

        if (xmax - xmin) < 8 or (ymax - ymin) < 8:
            continue

        # Draw object on RGB
        color = CLASS_COLORS[c_id]
        draw.rectangle([xmin, ymin, xmax, ymax], fill=color, outline=(20, 20, 20), width=2)

        # Draw green sign if door
        if c_id == 4:
            sign_h = (ymax - ymin) * 0.15
            draw.rectangle([xmin + 5, ymin + 5, xmax - 5, ymin + sign_h], fill=(30, 200, 50))

        # Stamp depth into depth map
        depth_map[int(ymin):int(ymax), int(xmin):int(xmax)] = z

        obj_id = f"{c_name}_{idx:02d}"
        objects_data.append({
            "object_id": obj_id,
            "class_id": c_id,
            "class_name": c_name,
            "bbox_2d": {
                "xmin": round(float(xmin), 1),
                "ymin": round(float(ymin), 1),
                "xmax": round(float(xmax), 1),
                "ymax": round(float(ymax), 1)
            },
            "position_world_3d": [round(x, 3), round(y, 3), round(z, 3)],
            "position_camera_3d": [round(x, 3), round(y - cam_pos[1], 3), round(z, 3)],
            "distance_to_camera": round(float(math.sqrt(x**2 + (y - cam_pos[1])**2 + z**2)), 3),
            "is_visible": True
        })

    frame_id = f"frame_{sample_idx:05d}"
    rgb_rel = f"rgb/{frame_id}.png"
    depth_rel = f"depth/{frame_id}.npy"
    annot_rel = f"annotations/{frame_id}.json"

    rgb_full = os.path.join(output_dir, rgb_rel)
    depth_full = os.path.join(output_dir, depth_rel)
    annot_full = os.path.join(output_dir, annot_rel)

    # Save RGB
    img.save(rgb_full)

    # Save depth array
    np.save(depth_full, depth_map)

    # Build annotation
    annotation = {
        "frame_id": frame_id,
        "image_filename": rgb_rel,
        "depth_filename": depth_rel,
        "camera": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "image_width": width,
            "image_height": height,
            "position": cam_pos,
            "rotation_quaternion": cam_rot
        },
        "objects": objects_data
    }

    with open(annot_full, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)

    return annotation

def generate_dataset(output_dir: str, num_samples: int = 10, seed: int = 42):
    """Generates a dataset of synthetic laboratory frames."""
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    print(f"Generating {num_samples} synthetic samples in {output_dir}...")
    annotations = []
    for i in range(num_samples):
        ann = generate_sample(i, output_dir)
        annotations.append(ann)

    print(f"Successfully generated {len(annotations)} samples.")
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic laboratory dataset.")
    parser.add_argument("--output_dir", type=str, default="data/generated/sample_smoke")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.num_samples, args.seed)
