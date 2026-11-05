"""SUN RGB-D PyTorch Dataset Loader for Spatial Quest.

Loads authentic indoor RGB images, registered depth maps, 2D bounding boxes,
3D centroids, and camera intrinsics from the official SUN RGB-D benchmark.
Preserves official train/test splits without data leakage.
"""
import os
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

TARGET_CLASSES = ["table", "chair", "door", "bookshelf", "sofa", "cabinet"]
NUM_CLASSES = len(TARGET_CLASSES)

class SUNRGBDDataset(Dataset):
    """PyTorch Dataset loading real-world SUN RGB-D frames."""

    def __init__(
        self,
        root_dir: str = "data/sunrgbd",
        split: str = "train",
        target_size: tuple = (640, 480), # (width, height)
        max_depth: float = 8.0
    ):
        self.root_dir = root_dir
        self.split = split
        self.target_w, self.target_h = target_size
        self.max_depth = max_depth

        annot_dir = os.path.join(root_dir, "annotations")
        if not os.path.isdir(annot_dir):
            raise FileNotFoundError(f"Annotations directory not found: {annot_dir}")

        all_annot_files = sorted(glob.glob(os.path.join(annot_dir, "*.json")))
        self.samples = []

        for af in all_annot_files:
            with open(af, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Filter by official split tag ("train" vs "test")
            if split == "all" or meta.get("split") == split:
                self.samples.append(meta)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {annot_dir} matching split='{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]

        rgb_path = os.path.join(self.root_dir, meta["image_filename"])
        depth_path = os.path.join(self.root_dir, meta["depth_filename"])

        # 1. Load and resize RGB
        pil_img = Image.open(rgb_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Scaling factors
        scale_x = self.target_w / orig_w
        scale_y = self.target_h / orig_h

        img_resized = pil_img.resize((self.target_w, self.target_h), Image.BILINEAR)
        # Convert to (3, H, W) float tensor normalized to [0, 1]
        rgb_tensor = torch.from_numpy(np.array(img_resized, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # 2. Load and resize Depth array
        depth_arr = np.load(depth_path).astype(np.float32)
        depth_pil = Image.fromarray(depth_arr)
        depth_resized = np.array(depth_pil.resize((self.target_w, self.target_h), Image.NEAREST), dtype=np.float32)
        # Clamp depth to max_depth
        depth_resized[depth_resized > self.max_depth] = self.max_depth
        depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0) # (1, H, W)

        # 3. Scale Camera Intrinsics
        cam_meta = meta.get("camera", {})
        fx = cam_meta.get("fx", 525.0) * scale_x
        fy = cam_meta.get("fy", 525.0) * scale_y
        cx = cam_meta.get("cx", 320.0) * scale_x
        cy = cam_meta.get("cy", 240.0) * scale_y

        # 4. Scale 2D Bounding Boxes and format targets
        boxes_list = []
        labels_list = []
        centroids_3d_list = []

        for obj in meta.get("objects", []):
            cid = obj["class_id"]
            box = obj["bbox_2d"]

            xmin = max(0.0, min(float(self.target_w), box["xmin"] * scale_x))
            ymin = max(0.0, min(float(self.target_h), box["ymin"] * scale_y))
            xmax = max(0.0, min(float(self.target_w), box["xmax"] * scale_x))
            ymax = max(0.0, min(float(self.target_h), box["ymax"] * scale_y))

            if (xmax - xmin) > 4 and (ymax - ymin) > 4:
                boxes_list.append([xmin, ymin, xmax, ymax])
                labels_list.append(cid)
                c3d = obj.get("position_camera_3d")
                centroids_3d_list.append(c3d if c3d is not None else [0.0, 0.0, 0.0])

        if len(boxes_list) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            c3d_tensor = torch.zeros((0, 3), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
            c3d_tensor = torch.tensor(centroids_3d_list, dtype=torch.float32)

        return {
            "frame_id": meta["frame_id"],
            "split": meta["split"],
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "boxes_2d": boxes_tensor,
            "labels": labels_tensor,
            "centroids_3d": c3d_tensor,
            "intrinsics": torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        }

def sunrgbd_collate_fn(batch: list) -> dict:
    """Custom collator handling variable number of bounding boxes per frame."""
    frame_ids = [b["frame_id"] for b in batch]
    splits = [b["split"] for b in batch]
    rgb_batch = torch.stack([b["rgb"] for b in batch], dim=0)
    depth_batch = torch.stack([b["depth"] for b in batch], dim=0)
    intrinsics_batch = torch.stack([b["intrinsics"] for b in batch], dim=0)

    # Boxes, labels, and 3D centroids kept as lists of tensors (per frame)
    boxes_batch = [b["boxes_2d"] for b in batch]
    labels_batch = [b["labels"] for b in batch]
    centroids_batch = [b["centroids_3d"] for b in batch]

    return {
        "frame_ids": frame_ids,
        "splits": splits,
        "rgb": rgb_batch,
        "depth": depth_batch,
        "intrinsics": intrinsics_batch,
        "boxes_2d": boxes_batch,
        "labels": labels_batch,
        "centroids_3d": centroids_batch
    }

def get_sunrgbd_dataloaders(root_dir: str = "data/sunrgbd", batch_size: int = 4):
    """Helper returning (train_loader, test_loader)."""
    train_ds = SUNRGBDDataset(root_dir=root_dir, split="train")
    test_ds = SUNRGBDDataset(root_dir=root_dir, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=sunrgbd_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=sunrgbd_collate_fn)

    return train_loader, test_loader
