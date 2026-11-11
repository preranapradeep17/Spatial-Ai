"""Training Pipeline for Custom CNN Object Detector (Spatial Quest).

Trains SpatialDetector from random weights using Focal+OHEM objectness,
Smooth L1 + GIoU bounding-box loss, and label-smoothed cross entropy.
Strictly separates train / val / test splits:
  - Updates gradients ONLY on train splits
  - Tracks val loss & selects best checkpoint ONLY on held-out val split
  - Calibrates per-class confidence thresholds strictly on val split
  - Evaluates locked held-out test splits ONCE after calibration
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.perception.device import get_optimal_device
from backend.perception.losses import DetectionLoss
from backend.perception.metrics import calibrate_class_thresholds, evaluate_detector
from backend.perception.model import SpatialDetector, TARGET_CLASSES
from backend.perception.sun_rgbd_loader import SUNRGBDDataset, sunrgbd_collate_fn


def _compute_loader_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
) -> float:
    """Computes average total loss on a held-out validation DataLoader without gradients."""
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["rgb"].to(device)
            boxes = [b.to(device) for b in batch["boxes_2d"]]
            labels = [lbl.to(device) for lbl in batch["labels"]]
            preds = model(imgs)
            ld = criterion(preds, boxes, labels)
            total += float(ld["total_loss"].item())
            batches += 1
    return total / max(1, batches)


def train_custom_detector(config_path: str = "configs/vision.yaml", epochs_override: int = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)

    device = get_optimal_device()
    print(f"Training SpatialDetector from scratch on device: {device}")

    batch_size = max(12, int(cfg["training"].get("batch_size", 12)))
    base_lr = 0.0016
    weight_decay = 1e-4
    epochs = int(epochs_override if epochs_override is not None else 90)
    target_size = tuple(cfg["model"].get("input_resolution", [640, 480]))

    sun_dir = cfg["dataset"].get("data_dir", "data/sunrgbd")
    threejs_dir = "data/generated/threejs_sector4"

    # Build strictly isolated train / val / test datasets (naturally balanced ~1:1 without duplicate oversampling)
    train_parts = []
    val_parts = []
    eval_loaders = {}

    if os.path.isdir(os.path.join(sun_dir, "annotations")):
        sun_train = SUNRGBDDataset(root_dir=sun_dir, split="train", target_size=target_size, augment=True)
        train_parts.append(sun_train)
        try:
            sun_val = SUNRGBDDataset(root_dir=sun_dir, split="val", target_size=target_size, augment=False)
            val_parts.append(sun_val)
            eval_loaders["sun_val"] = DataLoader(sun_val, batch_size=batch_size, shuffle=False, collate_fn=sunrgbd_collate_fn)
        except ValueError:
            pass
        sun_test = SUNRGBDDataset(root_dir=sun_dir, split="test", target_size=target_size, augment=False)
        eval_loaders["sun_test"] = DataLoader(sun_test, batch_size=batch_size, shuffle=False, collate_fn=sunrgbd_collate_fn)

    if os.path.isdir(os.path.join(threejs_dir, "annotations")):
        tjs_train = SUNRGBDDataset(root_dir=threejs_dir, split="train", target_size=target_size, augment=True)
        train_parts.append(tjs_train)
        try:
            tjs_val = SUNRGBDDataset(root_dir=threejs_dir, split="val", target_size=target_size, augment=False)
            val_parts.append(tjs_val)
            eval_loaders["tjs_val"] = DataLoader(tjs_val, batch_size=batch_size, shuffle=False, collate_fn=sunrgbd_collate_fn)
        except ValueError:
            pass
        tjs_test = SUNRGBDDataset(root_dir=threejs_dir, split="test", target_size=target_size, augment=False)
        eval_loaders["tjs_test"] = DataLoader(tjs_test, batch_size=batch_size, shuffle=False, collate_fn=sunrgbd_collate_fn)

    if not train_parts:
        raise RuntimeError("No training datasets available.")
    if not val_parts:
        raise RuntimeError("No validation split available; val split is required to avoid tuning on test.")

    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sunrgbd_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=sunrgbd_collate_fn)

    print(
        f"Loaded splits -> train items: {len(train_dataset)}, "
        f"val items: {len(val_dataset)}, "
        f"held-out test loaders: { {k: len(v.dataset) for k, v in eval_loaders.items() if k.endswith('_test')} }"
    )

    model = SpatialDetector(
        num_classes=len(TARGET_CLASSES),
        input_size=target_size,
        grid_size=tuple(cfg["model"].get("grid_size", [20, 15])),
        backbone_channels=cfg["model"].get("backbone_channels", [32, 64, 128, 256, 256]),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized from random weights. Trainable parameters: {n_params:,}")

    criterion = DetectionLoss(
        grid_size=tuple(cfg["model"].get("grid_size", [20, 15])),
        input_size=target_size,
        num_classes=len(TARGET_CLASSES),
        lambda_obj=1.5,
        lambda_box=float(cfg["training"]["loss_weights"].get("bounding_box", 5.0)),
        lambda_cls=float(cfg["training"]["loss_weights"].get("classification", 1.0)),
    )

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.04)

    os.makedirs("models/vision", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "loss_obj": [],
        "loss_box": [],
        "loss_cls": [],
        "val_map": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "lr": [],
    }

    best_val_score = -1.0
    best_epoch = 0
    best_ckpt_path = "models/vision/spatial_detector_best.pt"

    print("=" * 84)
    print(f"{'Ep':<4} | {'TrainLoss':<9} | {'ValLoss':<9} | {'Obj':<7} | {'Box':<7} | {'Cls':<7} | {'ValmAP':<7} | {'ValP':<6} | {'ValR':<6}")
    print("-" * 84)

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = ep_obj = ep_box = ep_cls = 0.0
        n_batches = 0

        for batch in train_loader:
            imgs = batch["rgb"].to(device)
            boxes = [b.to(device) for b in batch["boxes_2d"]]
            labels = [lbl.to(device) for lbl in batch["labels"]]

            optimizer.zero_grad()
            grid_pred = model(imgs)
            ld = criterion(grid_pred, boxes, labels)
            loss = ld["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            ep_loss += float(loss.item())
            ep_obj += float(ld["loss_obj"].item())
            ep_box += float(ld["loss_box"].item())
            ep_cls += float(ld["loss_cls"].item())
            n_batches += 1

        scheduler.step()
        avg_loss = ep_loss / max(1, n_batches)
        avg_obj = ep_obj / max(1, n_batches)
        avg_box = ep_box / max(1, n_batches)
        avg_cls = ep_cls / max(1, n_batches)

        val_loss = _compute_loader_loss(model, val_loader, criterion, device)
        val_eval = evaluate_detector(
            model,
            val_loader,
            device,
            conf_threshold=0.30,
            iou_threshold=0.50,
            cross_class_iou_threshold=0.55,
        )
        val_map = val_eval["mAP_50"]
        val_p = val_eval["overall_precision"]
        val_r = val_eval["overall_recall"]
        val_f1 = val_eval["overall_f1"]

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["loss_obj"].append(round(avg_obj, 4))
        history["loss_box"].append(round(avg_box, 4))
        history["loss_cls"].append(round(avg_cls, 4))
        history["val_map"].append(val_map)
        history["val_precision"].append(val_p)
        history["val_recall"].append(val_r)
        history["val_f1"].append(val_f1)
        history["lr"].append(round(optimizer.param_groups[0]["lr"], 6))

        print(
            f"{epoch:<4} | {avg_loss:<9.4f} | {val_loss:<9.4f} | {avg_obj:<7.4f} | "
            f"{avg_box:<7.4f} | {avg_cls:<7.4f} | {val_map:<7.4f} | {val_p:<6.4f} | {val_r:<6.4f}"
        )

        # Select checkpoint strictly from combined validation mAP@0.5 + F1
        val_score = (0.75 * val_map) + (0.25 * val_f1)
        if val_score >= best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mAP": val_map,
                    "val_f1": val_f1,
                    "val_loss": val_loss,
                    "config": cfg,
                    "classes": TARGET_CLASSES,
                },
                best_ckpt_path,
            )

    print("=" * 84)
    print(f"Training complete. Best checkpoint selected strictly from VAL epoch {best_epoch} (val_score={best_val_score:.4f}).")

    # Load best validation checkpoint & calibrate per-class thresholds strictly on VAL
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    class_thresholds, val_sweep = calibrate_class_thresholds(model, val_loader, device)
    best_ckpt["class_thresholds"] = class_thresholds
    torch.save(best_ckpt, best_ckpt_path)

    with open("results/metrics/detector_val_calibration.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "class_thresholds": class_thresholds,
                "val_sweep_per_class": val_sweep,
            },
            f,
            indent=2,
        )
    print("Calibrated validation class thresholds:", json.dumps(class_thresholds, indent=2))

    # Untouched held-out TEST evaluations
    if "tjs_test" in eval_loaders:
        tjs_test_eval = evaluate_detector(
            model,
            eval_loaders["tjs_test"],
            device,
            conf_threshold=0.32,
            iou_threshold=0.50,
            class_thresholds=class_thresholds,
            cross_class_iou_threshold=0.55,
        )
        with open("results/metrics/threejs_benchmark_eval.json", "w", encoding="utf-8") as f:
            json.dump(tjs_test_eval, f, indent=2)
        print("Held-out Three.js Test Eval:", json.dumps({
            "mAP_50": tjs_test_eval["mAP_50"],
            "overall_precision": tjs_test_eval["overall_precision"],
            "overall_recall": tjs_test_eval["overall_recall"],
            "overall_f1": tjs_test_eval["overall_f1"],
            "per_class_ap": tjs_test_eval["per_class_ap"],
        }, indent=2))

    if "sun_test" in eval_loaders:
        sun_test_eval = evaluate_detector(
            model,
            eval_loaders["sun_test"],
            device,
            conf_threshold=0.32,
            iou_threshold=0.50,
            class_thresholds=class_thresholds,
            cross_class_iou_threshold=0.55,
        )
        with open("results/metrics/real_benchmark_eval.json", "w", encoding="utf-8") as f:
            json.dump(sun_test_eval, f, indent=2)
        print("Held-out SUN RGB-D Test Eval:", json.dumps({
            "mAP_50": sun_test_eval["mAP_50"],
            "overall_precision": sun_test_eval["overall_precision"],
            "overall_recall": sun_test_eval["overall_recall"],
            "overall_f1": sun_test_eval["overall_f1"],
            "per_class_ap": sun_test_eval["per_class_ap"],
        }, indent=2))

    with open("results/metrics/detector_training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Generate train vs validation loss curve
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["epochs"], history["train_loss"], label="Train Loss", linewidth=2, color="#2563eb")
    ax1.plot(history["epochs"], history["val_loss"], label="Validation Loss", linewidth=2, color="#f59e0b")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Multi-Task Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(history["epochs"], history["val_map"], label="Val mAP@0.5", linestyle="--", linewidth=1.8, color="#10b981")
    ax2.set_ylabel("Validation mAP@0.5")
    ax2.set_ylim(0.0, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("SpatialDetector From-Scratch Training & Validation Convergence")
    plt.savefig("results/plots/detector_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.savefig("results/plots/cv_train_val_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    return history


if __name__ == "__main__":
    train_custom_detector()
