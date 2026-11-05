"""Dual Domain-Transfer Evaluation Script for Spatial Quest.

Evaluates the from-scratch SpatialDetector on:
1. Benchmark A: Held-out real-world SUN RGB-D test samples.
2. Benchmark B: Unity simulated camera frames.

Measures quantitative domain gap (mAP@0.5 drop, precision, recall, latency)
without ever polluting the training benchmark.
Traceable to CS4231 (Computer Vision - Generalization, Domain Shift & Evaluation).
"""
import os
import sys
import glob
import json
import time
import yaml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.perception.device import get_optimal_device
from backend.perception.model import SpatialDetector, TARGET_CLASSES
from backend.perception.sun_rgbd_loader import get_sunrgbd_dataloaders
from backend.perception.metrics import evaluate_detector

class UnitySimulationDataset(Dataset):
    """Loads simulated camera frames and annotations from Unity synthetic generator."""

    def __init__(self, root_dir: str = "data/generated/sample_smoke", target_size: tuple = (640, 480)):
        self.root_dir = root_dir
        self.target_w, self.target_h = target_size
        annot_dir = os.path.join(root_dir, "annotations")
        self.annot_files = sorted(glob.glob(os.path.join(annot_dir, "*.json")))

    def __len__(self) -> int:
        return len(self.annot_files)

    def __getitem__(self, idx: int):
        with open(self.annot_files[idx], "r", encoding="utf-8") as f:
            meta = json.load(f)

        rgb_path = os.path.join(self.root_dir, meta["image_filename"])
        depth_path = os.path.join(self.root_dir, meta["depth_filename"])

        img = Image.open(rgb_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.target_w, self.target_h), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        # ImageNet standardization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_arr = (img_arr - mean) / std
        rgb_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()

        # Depth
        depth_arr = np.load(depth_path).astype(np.float32)
        depth_tensor = torch.from_numpy(depth_arr).float()

        # Intrinsics
        cam_info = meta.get("camera", {})
        fx = cam_info.get("fx", 525.0) * (self.target_w / orig_w)
        fy = cam_info.get("fy", 525.0) * (self.target_h / orig_h)
        cx = cam_info.get("cx", 320.0) * (self.target_w / orig_w)
        cy = cam_info.get("cy", 240.0) * (self.target_h / orig_h)
        intrinsics_mat = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

        # Boxes and labels
        boxes = []
        labels = []
        centroids = []
        for obj in meta.get("objects", []):
            cid = obj.get("class_id", -1)
            # Map into valid class index (modulo 6)
            cid = cid % len(TARGET_CLASSES) if cid >= 0 else 0
            b = obj.get("box_2d", [0, 0, 10, 10])
            sx = self.target_w / orig_w
            sy = self.target_h / orig_h
            scaled_box = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
            boxes.append(scaled_box)
            labels.append(cid)
            centroids.append(obj.get("position_3d", [0.0, 0.0, 2.0]))

        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "intrinsics": intrinsics_mat,
            "boxes_2d": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "centroids_3d": torch.tensor(centroids, dtype=torch.float32) if centroids else torch.zeros((0, 3), dtype=torch.float32)
        }

def unity_collate_fn(batch):
    rgb_batch = torch.stack([b["rgb"] for b in batch], dim=0)
    depth_batch = torch.stack([b["depth"] for b in batch], dim=0)
    intrinsics_batch = torch.stack([b["intrinsics"] for b in batch], dim=0)
    boxes_batch = [b["boxes_2d"] for b in batch]
    labels_batch = [b["labels"] for b in batch]
    centroids_batch = [b["centroids_3d"] for b in batch]
    return {
        "rgb": rgb_batch,
        "depth": depth_batch,
        "intrinsics": intrinsics_batch,
        "boxes_2d": boxes_batch,
        "labels": labels_batch,
        "centroids_3d": centroids_batch
    }

def run_dual_evaluation():
    print("=" * 65)
    print("Spatial Quest: Dual Domain-Transfer Evaluation")
    print("=" * 65)

    with open("configs/vision.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_optimal_device(cfg["training"]["device"])
    print(f"Device: {device}")

    ckpt_path = "models/vision/spatial_detector_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    model = SpatialDetector(
        num_classes=cfg["model"]["num_classes"],
        input_size=tuple(cfg["model"]["input_resolution"]),
        grid_size=tuple(cfg["model"]["grid_size"]),
        backbone_channels=cfg["model"]["backbone_channels"]
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded trained SpatialDetector weights from checkpoint.")

    # 1. Benchmark A: SUN RGB-D Real Test Split
    print("\n--- Evaluating Benchmark A: Real SUN RGB-D Held-out Test Split ---")
    _, test_loader = get_sunrgbd_dataloaders(root_dir=cfg["dataset"]["data_dir"], batch_size=2)
    bench_a = evaluate_detector(model, test_loader, device, conf_threshold=0.20, iou_threshold=0.50)
    print(f"Benchmark A (Real SUN RGB-D): mAP@0.5 = {bench_a['mAP_50']:.4f} | Recall = {bench_a['overall_recall']:.4f} | Latency = {bench_a['mean_latency_ms']:.2f} ms")

    # 2. Benchmark B: Unity Simulated Game Camera
    print("\n--- Evaluating Benchmark B: Unity Simulated Game Camera Frames ---")
    unity_dir = "data/generated/sample_smoke"
    unity_ds = UnitySimulationDataset(root_dir=unity_dir)
    unity_loader = DataLoader(unity_ds, batch_size=2, shuffle=False, collate_fn=unity_collate_fn)
    bench_b = evaluate_detector(model, unity_loader, device, conf_threshold=0.20, iou_threshold=0.50)
    print(f"Benchmark B (Unity Transfer): mAP@0.5 = {bench_b['mAP_50']:.4f} | Recall = {bench_b['overall_recall']:.4f} | Latency = {bench_b['mean_latency_ms']:.2f} ms")

    # 3. Domain Gap Analysis
    domain_gap = round(bench_a["mAP_50"] - bench_b["mAP_50"], 4)
    print(f"\nDomain Shift Delta (Real mAP - Unity mAP): {domain_gap:+.4f}")

    results = {
        "benchmark_a_sunrgbd_real": bench_a,
        "benchmark_b_unity_simulated": bench_b,
        "domain_gap_delta_mAP": domain_gap,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs("results/metrics", exist_ok=True)
    metrics_file = "results/metrics/dual_domain_transfer_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_file}")

    # Plot comparison
    os.makedirs("results/plots", exist_ok=True)
    plot_file = "results/plots/domain_transfer_comparison.png"
    categories = ["mAP@0.5", "Precision", "Recall"]
    real_vals = [bench_a["mAP_50"], bench_a["overall_precision"], bench_a["overall_recall"]]
    sim_vals = [bench_b["mAP_50"], bench_b["overall_precision"], bench_b["overall_recall"]]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width/2, real_vals, width, label="Real (SUN RGB-D)", color="#2b5c8f")
    plt.bar(x + width/2, sim_vals, width, label="Sim (Unity Transfer)", color="#e07a5f")
    plt.ylabel("Score")
    plt.title("Spatial Quest: Real Benchmark vs Unity Domain Transfer")
    plt.xticks(x, categories)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved domain comparison plot to {plot_file}")

if __name__ == "__main__":
    run_dual_evaluation()
