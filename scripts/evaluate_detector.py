"""Evaluation Benchmark for Custom Detector on Real-World SUN RGB-D Test Split.

Loads the trained SpatialDetector checkpoint and performs rigorous, un-mocked
evaluation of mAP@0.5, IoU, per-class Precision/Recall, and inference latency.
"""
import os
import json
import argparse
import yaml
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.perception.device import get_optimal_device
from backend.perception.model import SpatialDetector, TARGET_CLASSES
from backend.perception.sun_rgbd_loader import get_sunrgbd_dataloaders
from backend.perception.metrics import evaluate_detector

def run_evaluation(
    checkpoint_path: str = "models/vision/spatial_detector_best.pt",
    config_path: str = "configs/vision.yaml",
    conf_thresh: float = 0.20,
    iou_thresh: float = 0.50
) -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_optimal_device(cfg["training"]["device"])
    print(f"Evaluating checkpoint: {checkpoint_path} on {device}")

    # Load test dataloader
    _, test_loader = get_sunrgbd_dataloaders(root_dir=cfg["dataset"]["data_dir"], batch_size=cfg["training"]["batch_size"])
    print(f"Test Split Size: {len(test_loader.dataset)} samples.")

    # Instantiate model
    model = SpatialDetector(
        num_classes=cfg["model"]["num_classes"],
        input_size=tuple(cfg["model"]["input_resolution"]),
        grid_size=tuple(cfg["model"]["grid_size"]),
        backbone_channels=cfg["model"]["backbone_channels"]
    ).to(device)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Successfully loaded checkpoint trained up to epoch {ckpt.get('epoch', 'unknown')}.")

    # Evaluate
    results = evaluate_detector(model, test_loader, device, conf_threshold=conf_thresh, iou_threshold=iou_thresh)

    print("=" * 60)
    print("🎯 Real-World SUN RGB-D Test Benchmark Results (Evaluation A):")
    print(f"  Overall mAP@0.5:       {results['mAP_50']:.4f}")
    print(f"  Overall Precision:     {results['overall_precision']:.4f}")
    print(f"  Overall Recall:        {results['overall_recall']:.4f}")
    print(f"  Mean Latency:          {results['mean_latency_ms']:.2f} ms / frame")
    print("-" * 60)
    print("Per-Class Average Precision (AP@0.5):")
    for cname, ap in results["per_class_ap"].items():
        gt_count = results["gt_counts"].get(cname, 0)
        print(f"  - {cname:15s}: AP={ap:.4f} (Ground-truth instances: {gt_count})")
    print("=" * 60)

    # Save results
    os.makedirs("results/metrics", exist_ok=True)
    out_path = "results/metrics/real_benchmark_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to: {out_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Custom Detector on SUN RGB-D.")
    parser.add_argument("--checkpoint", type=str, default="models/vision/spatial_detector_best.pt")
    parser.add_argument("--config", type=str, default="configs/vision.yaml")
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.config)
