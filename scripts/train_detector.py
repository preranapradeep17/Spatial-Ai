"""Training Pipeline for Custom CNN Object Detector (Spatial Quest).

Trains SpatialDetector from scratch on SUN RGB-D indoor benchmark using
AdamW, multi-task losses, and Apple Silicon MPS hardware acceleration.
Evaluates mAP@0.5 on held-out test split and saves checkpoints.
"""
import os
import json
import time
import yaml
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.perception.device import get_optimal_device
from backend.perception.model import SpatialDetector, TARGET_CLASSES
from backend.perception.losses import DetectionLoss
from backend.perception.sun_rgbd_loader import get_sunrgbd_dataloaders
from backend.perception.metrics import evaluate_detector

def train_custom_detector(config_path: str = "configs/vision.yaml", epochs_override: int = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_optimal_device(cfg["training"]["device"])
    print(f"🚀 Training SpatialDetector on device: {device}")

    # Set random seeds for reproducibility
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)

    # Output directories
    os.makedirs("models/vision", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # 1. Dataset & Loaders
    batch_size = cfg["training"]["batch_size"]
    train_loader, test_loader = get_sunrgbd_dataloaders(root_dir=cfg["dataset"]["data_dir"], batch_size=batch_size)
    print(f"Loaded SUN RGB-D: {len(train_loader.dataset)} train samples, {len(test_loader.dataset)} test samples.")

    # 2. Model & Loss (from random initialization)
    model = SpatialDetector(
        num_classes=cfg["model"]["num_classes"],
        input_size=tuple(cfg["model"]["input_resolution"]),
        grid_size=tuple(cfg["model"]["grid_size"]),
        backbone_channels=cfg["model"]["backbone_channels"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture Initialized from Scratch. Trainable Parameters: {total_params:,}")

    criterion = DetectionLoss(
        grid_size=tuple(cfg["model"]["grid_size"]),
        input_size=tuple(cfg["model"]["input_resolution"]),
        num_classes=cfg["model"]["num_classes"],
        lambda_obj=cfg["training"]["loss_weights"]["objectness"],
        lambda_box=cfg["training"]["loss_weights"]["bounding_box"],
        lambda_cls=cfg["training"]["loss_weights"]["classification"]
    )

    # 3. Optimizer & Scheduler
    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = epochs_override or cfg["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epochs": [],
        "train_loss": [],
        "loss_obj": [],
        "loss_box": [],
        "loss_cls": [],
        "test_mAP": []
    }

    best_map = -1.0
    best_checkpoint_path = "models/vision/spatial_detector_best.pt"

    print("=" * 65)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Obj Loss':<9} | {'Box Loss':<9} | {'Cls Loss':<9} | {'Test mAP':<8}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_loss_obj = 0.0
        epoch_loss_box = 0.0
        epoch_loss_cls = 0.0
        batches = 0

        for batch in train_loader:
            imgs = batch["rgb"].to(device)
            boxes = [b.to(device) for b in batch["boxes_2d"]]
            labels = [l.to(device) for l in batch["labels"]]

            optimizer.zero_grad()
            preds = model(imgs)

            loss_dict = criterion(preds, boxes, labels)
            loss = loss_dict["total_loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_loss_obj += loss_dict["loss_obj"].item()
            epoch_loss_box += loss_dict["loss_box"].item()
            epoch_loss_cls += loss_dict["loss_cls"].item()
            batches += 1

        scheduler.step()

        avg_loss = epoch_total_loss / max(1, batches)
        avg_obj = epoch_loss_obj / max(1, batches)
        avg_box = epoch_loss_box / max(1, batches)
        avg_cls = epoch_loss_cls / max(1, batches)

        # Evaluate on test set
        eval_results = evaluate_detector(model, test_loader, device)
        cur_map = eval_results["mAP_50"]

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_loss, 4))
        history["loss_obj"].append(round(avg_obj, 4))
        history["loss_box"].append(round(avg_box, 4))
        history["loss_cls"].append(round(avg_cls, 4))
        history["test_mAP"].append(round(cur_map, 4))

        print(f"{epoch:<6} | {avg_loss:<11.4f} | {avg_obj:<9.4f} | {avg_box:<9.4f} | {avg_cls:<9.4f} | {cur_map:<8.4f}")

        # Save best checkpoint
        if cur_map > best_map or epoch == epochs:
            best_map = cur_map
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mAP": best_map,
                "config": cfg,
                "classes": TARGET_CLASSES
            }, best_checkpoint_path)

    print("=" * 65)
    print(f"Training Complete. Best Test mAP@0.5: {best_map:.4f}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")

    # Save metrics JSON
    metrics_path = "results/metrics/detector_training_log.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training metrics saved to: {metrics_path}")

    # Generate loss curve plot
    plt.figure(figsize=(10, 5))
    plt.plot(history["epochs"], history["train_loss"], label="Total Loss", linewidth=2, color="#1f77b4")
    plt.plot(history["epochs"], history["loss_obj"], label="Objectness Loss", linestyle="--", color="#ff7f0e")
    plt.plot(history["epochs"], history["loss_box"], label="Bounding Box Loss", linestyle=":", color="#2ca02c")
    plt.plot(history["epochs"], history["loss_cls"], label="Classification Loss", linestyle="-.", color="#d62728")
    plt.title("SpatialDetector Training Loss (From Scratch on SUN RGB-D)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = "results/plots/detector_loss_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to: {plot_path}")

    return history

if __name__ == "__main__":
    train_custom_detector("configs/vision.yaml", epochs_override=10)
