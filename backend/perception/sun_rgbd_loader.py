"""SUN RGB-D PyTorch Dataset Loader for Spatial Quest.

Loads authentic indoor RGB images, registered depth maps, 2D bounding boxes,
3D centroids, and camera intrinsics from both real SUN RGB-D and rendered Three.js splits.
Preserves strict train / val / test splits without data leakage.
"""
import glob
import json
import os
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import DataLoader, Dataset

TARGET_CLASSES = ["table", "chair", "door", "bookshelf", "sofa", "cabinet"]
NUM_CLASSES = len(TARGET_CLASSES)


class SUNRGBDDataset(Dataset):
    """PyTorch Dataset loading real-world SUN RGB-D and rendered 3D facility frames."""

    def __init__(
        self,
        root_dir: str = "data/sunrgbd",
        split: str = "train",
        target_size: Tuple[int, int] = (640, 480),
        max_depth: float = 8.0,
        augment: bool = False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.target_w, self.target_h = target_size
        self.max_depth = max_depth
        self.augment = augment and (split == "train")

        annot_dir = os.path.join(root_dir, "annotations")
        if not os.path.isdir(annot_dir):
            raise FileNotFoundError(f"Annotations directory not found: {annot_dir}")

        all_annot_files = sorted(glob.glob(os.path.join(annot_dir, "*.json")))
        self.samples = []

        for af in all_annot_files:
            with open(af, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if split == "all" or meta.get("split") == split:
                self.samples.append(meta)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {annot_dir} matching split='{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]

        rgb_path = os.path.join(self.root_dir, meta["image_filename"])
        depth_rel = meta.get("depth_filename")
        depth_path = os.path.join(self.root_dir, depth_rel) if depth_rel else None

        pil_img = Image.open(rgb_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        depth_arr = None
        if depth_path and os.path.exists(depth_path):
            depth_arr = np.load(depth_path).astype(np.float32)

        # Optional mild spatial random crop augmentation on train split only
        crop_x0, crop_y0 = 0.0, 0.0
        crop_w, crop_h = float(orig_w), float(orig_h)
        if self.augment and random.random() < 0.35 and len(meta.get("objects", [])) > 0:
            zoom = random.uniform(0.88, 0.98)
            new_w = max(64, int(round(orig_w * zoom)))
            new_h = max(48, int(round(orig_h * zoom)))
            x0 = random.randint(0, max(0, orig_w - new_w))
            y0 = random.randint(0, max(0, orig_h - new_h))
            # Only apply crop if at least one GT object keeps >= 40% area
            kept_any = False
            for obj in meta.get("objects", []):
                bb = obj["bbox_2d"]
                ix1 = max(float(x0), bb["xmin"])
                iy1 = max(float(y0), bb["ymin"])
                ix2 = min(float(x0 + new_w), bb["xmax"])
                iy2 = min(float(y0 + new_h), bb["ymax"])
                orig_area = max(1.0, (bb["xmax"] - bb["xmin"]) * (bb["ymax"] - bb["ymin"]))
                if max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) >= 0.4 * orig_area:
                    kept_any = True
                    break
            if kept_any:
                crop_x0, crop_y0 = float(x0), float(y0)
                crop_w, crop_h = float(new_w), float(new_h)
                pil_img = pil_img.crop((x0, y0, x0 + new_w, y0 + new_h))
                if depth_arr is not None:
                    dh, dw = depth_arr.shape[:2]
                    dx0 = int(round(x0 * dw / orig_w))
                    dy0 = int(round(y0 * dh / orig_h))
                    dx1 = int(round((x0 + new_w) * dw / orig_w))
                    dy1 = int(round((y0 + new_h) * dh / orig_h))
                    depth_arr = depth_arr[max(0, dy0):min(dh, dy1), max(0, dx0):min(dw, dx1)]

        scale_x = self.target_w / crop_w
        scale_y = self.target_h / crop_h

        img_resized = pil_img.resize((self.target_w, self.target_h), Image.BILINEAR)
        do_hflip = self.augment and random.random() < 0.5
        if do_hflip:
            img_resized = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
        if self.augment:
            if random.random() < 0.6:
                img_resized = ImageEnhance.Brightness(img_resized).enhance(random.uniform(0.80, 1.20))
            if random.random() < 0.5:
                img_resized = ImageEnhance.Contrast(img_resized).enhance(random.uniform(0.82, 1.18))
            if random.random() < 0.4:
                img_resized = ImageEnhance.Color(img_resized).enhance(random.uniform(0.85, 1.15))

        rgb_tensor = torch.from_numpy(np.array(img_resized, dtype=np.float32) / 255.0).permute(2, 0, 1)

        if depth_arr is not None and depth_arr.size > 0:
            depth_pil = Image.fromarray(depth_arr)
            if do_hflip:
                depth_pil = depth_pil.transpose(Image.FLIP_LEFT_RIGHT)
            depth_resized = np.array(depth_pil.resize((self.target_w, self.target_h), Image.NEAREST), dtype=np.float32)
            depth_resized[depth_resized > self.max_depth] = self.max_depth
        else:
            depth_resized = np.full((self.target_h, self.target_w), 2.5, dtype=np.float32)
        depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0)

        cam_meta = meta.get("camera", {})
        fx = float(cam_meta.get("fx", 525.0)) * scale_x
        fy = float(cam_meta.get("fy", 525.0)) * scale_y
        cx = (float(cam_meta.get("cx", orig_w / 2.0)) - crop_x0) * scale_x
        cy = (float(cam_meta.get("cy", orig_h / 2.0)) - crop_y0) * scale_y
        if do_hflip:
            cx = float(self.target_w) - cx

        boxes_list = []
        labels_list = []
        centroids_3d_list = []

        for obj in meta.get("objects", []):
            cid = obj["class_id"]
            box = obj["bbox_2d"]

            xmin = max(0.0, min(float(self.target_w), (box["xmin"] - crop_x0) * scale_x))
            ymin = max(0.0, min(float(self.target_h), (box["ymin"] - crop_y0) * scale_y))
            xmax = max(0.0, min(float(self.target_w), (box["xmax"] - crop_x0) * scale_x))
            ymax = max(0.0, min(float(self.target_h), (box["ymax"] - crop_y0) * scale_y))
            if do_hflip:
                xmin, xmax = float(self.target_w) - xmax, float(self.target_w) - xmin

            if (xmax - xmin) > 6.0 and (ymax - ymin) > 6.0:
                boxes_list.append([xmin, ymin, xmax, ymax])
                labels_list.append(cid)
                pos3d = obj.get("position_camera_3d") or [0.0, 0.0, 0.0]
                centroids_3d_list.append(pos3d)

        boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long) if labels_list else torch.zeros((0,), dtype=torch.long)
        c3d_tensor = torch.tensor(centroids_3d_list, dtype=torch.float32) if centroids_3d_list else torch.zeros((0, 3), dtype=torch.float32)

        return {
            "frame_id": meta["frame_id"],
            "split": meta["split"],
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "boxes_2d": boxes_tensor,
            "labels": labels_tensor,
            "centroids_3d": c3d_tensor,
            "intrinsics": torch.tensor([fx, fy, cx, cy], dtype=torch.float32),
        }


def sunrgbd_collate_fn(batch: list) -> dict:
    """Custom collator handling variable number of bounding boxes per frame."""
    return {
        "frame_ids": [b["frame_id"] for b in batch],
        "splits": [b["split"] for b in batch],
        "rgb": torch.stack([b["rgb"] for b in batch], dim=0),
        "depth": torch.stack([b["depth"] for b in batch], dim=0),
        "intrinsics": torch.stack([b["intrinsics"] for b in batch], dim=0),
        "boxes_2d": [b["boxes_2d"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "centroids_3d": [b["centroids_3d"] for b in batch],
    }


def get_sunrgbd_dataloaders(
    root_dir: str = "data/sunrgbd",
    batch_size: int = 4,
    target_size: Tuple[int, int] = (640, 480),
) -> Tuple[DataLoader, DataLoader]:
    """Creates PyTorch train and test DataLoaders strictly respecting official splits."""
    train_ds = SUNRGBDDataset(root_dir=root_dir, split="train", target_size=target_size, augment=True)
    test_ds = SUNRGBDDataset(root_dir=root_dir, split="test", target_size=target_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=sunrgbd_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=sunrgbd_collate_fn)
    return train_loader, test_loader
