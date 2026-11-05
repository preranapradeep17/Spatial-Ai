"""Label Visualizer for SUN RGB-D Real Samples.

Renders 2D bounding boxes, class names, and 3D centroid distance tags onto real
photographs and saves depth visualization heatmaps for empirical inspection.
"""
import os
import glob
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

COLOR_PALETTE = {
    "table": (255, 165, 0),       # Orange
    "chair": (30, 144, 255),      # Dodger Blue
    "door": (50, 205, 50),        # Lime Green
    "bookshelf": (186, 85, 211),  # Medium Orchid
    "sofa": (220, 20, 60),        # Crimson
    "cabinet": (255, 215, 0)      # Gold
}

def depth_to_colormap(depth_meters: np.ndarray, max_depth: float = 6.0) -> Image.Image:
    """Converts a single-channel metric depth map to an RGB colormap for inspection."""
    clipped = np.clip(depth_meters, 0.1, max_depth)
    norm = ((clipped - 0.1) / (max_depth - 0.1) * 255.0).astype(np.uint8)

    # Simple turbo-like pseudo colormap
    r = norm
    g = 255 - np.abs(norm - 128) * 2
    b = 255 - norm

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)

def visualize_sunrgbd_dataset(data_dir: str = "data/sunrgbd", output_dir: str = "results/predictions/sunrgbd_debug"):
    os.makedirs(output_dir, exist_ok=True)

    annot_files = sorted(glob.glob(os.path.join(data_dir, "annotations", "*.json")))
    print(f"Visualizing {len(annot_files)} real SUN RGB-D samples from {data_dir}...")

    for af in annot_files:
        with open(af, "r", encoding="utf-8") as f:
            meta = json.load(f)

        rgb_path = os.path.join(data_dir, meta["image_filename"])
        depth_path = os.path.join(data_dir, meta["depth_filename"])

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue

        img = Image.open(rgb_path).convert("RGB")
        depth = np.load(depth_path)

        draw = ImageDraw.Draw(img)
        w, h = img.size

        for obj in meta.get("objects", []):
            cname = obj["class_name"]
            color = COLOR_PALETTE.get(cname, (255, 255, 255))
            b = obj["bbox_2d"]

            xmin = max(0, min(w, b["xmin"]))
            ymin = max(0, min(h, b["ymin"]))
            xmax = max(0, min(w, b["xmax"]))
            ymax = max(0, min(h, b["ymax"]))

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

            c3d = obj.get("position_camera_3d")
            dist_str = f" Z={c3d[2]:.1f}m" if c3d is not None else ""
            label = f"{cname}{dist_str}"

            # Tag label background
            draw.rectangle([xmin, max(0, ymin - 16), xmin + len(label) * 8 + 6, max(0, ymin)], fill=color)
            draw.text((xmin + 3, max(0, ymin - 14)), label, fill=(0, 0, 0))

        # Create side-by-side composite: [RGB with BBoxes | Depth Heatmap]
        depth_vis = depth_to_colormap(depth).resize((w, h))
        composite = Image.new("RGB", (w * 2, h))
        composite.paste(img, (0, 0))
        composite.paste(depth_vis, (w, 0))

        # Add title banner
        comp_draw = ImageDraw.Draw(composite)
        title = f"{meta['frame_id']} | Split: {meta['split']} | Sensor: {meta['camera']['sensor_type']} | Objects: {len(meta.get('objects', []))}"
        comp_draw.rectangle([0, 0, w * 2, 20], fill=(0, 0, 0))
        comp_draw.text((10, 4), title, fill=(255, 255, 255))

        out_file = os.path.join(output_dir, f"vis_{meta['frame_id']}.png")
        composite.save(out_file)
        print(f"Saved: {out_file}")

    print("SUN RGB-D visualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SUN RGB-D annotations.")
    parser.add_argument("--data_dir", type=str, default="data/sunrgbd")
    parser.add_argument("--output_dir", type=str, default="results/predictions/sunrgbd_debug")
    args = parser.parse_args()

    visualize_sunrgbd_dataset(args.data_dir, args.output_dir)
