"""Comprehensive Scientific Evaluation for Spatial Quest Computer Vision Subsystem.

Generates strictly held-out measurements (zero test manipulation) and visual artifacts:
  1. Split & data-leakage audit across train / val / test (SUN RGB-D & Three.js)
  2. Uniform vs Val-Calibrated 2D Detection metrics (mAP@0.5, Precision, Recall, F1, AP)
  3. Confusion matrix across 6 target classes + background
  4. 3D Localization validation (Measured WebGL depth vs Estimated BBox-height depth vs SUN RGB-D depth)
  5. End-to-End Integration test (RGB-D -> CNN -> Depth -> 3D -> Scene Graph -> Tool Reasoning -> Action)
  6. Publication-grade PNG plots saved to results/plots/
"""
import base64
import glob
import io
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.api.routes import _TYPICAL_CLASS_HEIGHT_M
from backend.perception.device import get_optimal_device
from backend.perception.metrics import compute_box_iou, evaluate_detector
from backend.perception.model import SpatialDetector, TARGET_CLASSES
from backend.perception.sun_rgbd_loader import SUNRGBDDataset, sunrgbd_collate_fn
from backend.spatial.geometry import (
    backproject_pixel_to_camera_point,
    quaternion_to_rotation_matrix,
    sample_robust_depth_from_box,
)
from backend.spatial.localization import benchmark_localization_on_dataset, evaluate_3d_localization_error


def audit_dataset_splits(root_dir: str, domain_name: str) -> dict:
    """Verifies split counts, per-class balance, and zero group/MD5/viewpoint leakage."""
    ann_files = sorted(glob.glob(os.path.join(root_dir, "annotations", "*.json")))
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_counts = {sp: {c: 0 for c in TARGET_CLASSES} for sp in ("train", "val", "test")}
    groups = {"train": set(), "val": set(), "test": set()}
    md5s = {"train": set(), "val": set(), "test": set()}
    xzs = {"train": set(), "val": set(), "test": set()}

    for af in ann_files:
        with open(af, "r", encoding="utf-8") as f:
            meta = json.load(f)
        sp = meta.get("split", "train")
        if sp not in split_counts:
            continue
        split_counts[sp] += 1
        if meta.get("scene_group"):
            groups[sp].add(meta["scene_group"])
        if meta.get("image_md5"):
            md5s[sp].add(meta["image_md5"])
        if meta.get("viewpoint_xz"):
            xzs[sp].add(tuple(meta["viewpoint_xz"]))
        for o in meta.get("objects", []):
            cname = o.get("class_name")
            if cname in class_counts[sp]:
                class_counts[sp][cname] += 1

    return {
        "domain": domain_name,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "leakage": {
            "group_train_val": len(groups["train"] & groups["val"]),
            "group_train_test": len(groups["train"] & groups["test"]),
            "group_val_test": len(groups["val"] & groups["test"]),
            "md5_train_val": len(md5s["train"] & md5s["val"]),
            "md5_train_test": len(md5s["train"] & md5s["test"]),
            "md5_val_test": len(md5s["val"] & md5s["test"]),
            "xz_train_val": len(xzs["train"] & xzs["val"]),
            "xz_train_test": len(xzs["train"] & xzs["test"]),
            "xz_val_test": len(xzs["val"] & xzs["test"]),
        },
    }


def evaluate_threejs_3d_localization(
    threejs_dir: str,
    model: SpatialDetector,
    device: torch.device,
    class_thresholds: dict,
) -> dict:
    """Measures 3D localization error on Three.js held-out test split (measured WebGL depth vs height estimate)."""
    ann_files = sorted(glob.glob(os.path.join(threejs_dir, "annotations", "*.json")))
    test_anns = []
    for af in ann_files:
        with open(af, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("split") == "test":
            test_anns.append(m)

    oracle_measured_errs = []
    oracle_est_errs = []
    cnn_measured_errs = []
    cnn_est_errs = []
    depth_abs_errs_measured = []
    depth_abs_errs_est = []
    per_class_measured = {c: [] for c in TARGET_CLASSES}
    per_class_est = {c: [] for c in TARGET_CLASSES}

    model.eval()
    with torch.no_grad():
        for meta in test_anns:
            depth_rel = meta.get("depth_filename")
            if not depth_rel:
                continue
            depth_path = os.path.join(threejs_dir, depth_rel)
            rgb_path = os.path.join(threejs_dir, meta["image_filename"])
            if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
                continue

            depth_map = np.load(depth_path).astype(np.float32)
            cam = meta["camera"]
            fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
            pose = meta.get("camera_pose", {"position": [0.0, 1.5, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]})
            cam_pos = pose["position"]
            R = quaternion_to_rotation_matrix(pose["rotation"])
            cam_t = np.array(cam_pos, dtype=np.float32)

            def _box_to_world_pos(box_dict_or_xyxy, class_name, use_measured: bool):
                if isinstance(box_dict_or_xyxy, dict):
                    xmin, ymin = box_dict_or_xyxy["xmin"], box_dict_or_xyxy["ymin"]
                    xmax, ymax = box_dict_or_xyxy["xmax"], box_dict_or_xyxy["ymax"]
                else:
                    xmin, ymin, xmax, ymax = box_dict_or_xyxy
                uc = (xmin + xmax) / 2.0
                vc = (ymin + ymax) / 2.0
                if use_measured:
                    d = sample_robust_depth_from_box(
                        depth_map, xmin, ymin, xmax, ymax, sampling_ratio=0.55, reject_background=True
                    )
                    if d is None:
                        return None, None
                else:
                    bh = max(16.0, ymax - ymin)
                    nominal_h = _TYPICAL_CLASS_HEIGHT_M.get(class_name, 1.0)
                    d = float(np.clip((nominal_h * abs(fy)) / bh, 0.8, 9.5))
                xc, yc, zc = backproject_pixel_to_camera_point(uc, vc, d, fx, fy, cx, cy)
                p_local = np.array([xc, -yc, -zc], dtype=np.float32)
                world = (R @ p_local + cam_t).tolist()
                return world, d

            # 1. Oracle 2D GT boxes -> isolate pure depth + geometry error
            for obj in meta.get("objects", []):
                gt_world = obj.get("position_world_3d")
                gt_cam = obj.get("position_camera_3d")
                if not gt_world or not gt_cam:
                    continue
                gt_depth = abs(float(gt_cam[2]))
                wm, dm = _box_to_world_pos(obj["bbox_2d"], obj["class_name"], use_measured=True)
                we, de = _box_to_world_pos(obj["bbox_2d"], obj["class_name"], use_measured=False)
                if wm is not None:
                    err_m = evaluate_3d_localization_error(wm, gt_world)
                    oracle_measured_errs.append(err_m)
                    depth_abs_errs_measured.append(abs(dm - gt_depth))
                    per_class_measured[obj["class_name"]].append(err_m)
                if we is not None:
                    err_e = evaluate_3d_localization_error(we, gt_world)
                    oracle_est_errs.append(err_e)
                    depth_abs_errs_est.append(abs(de - gt_depth))
                    per_class_est[obj["class_name"]].append(err_e)

            # 2. Predicted CNN 2D boxes matched to GT at IoU >= 0.50 -> full pipeline 3D error
            pil_img = Image.open(rgb_path).convert("RGB").resize((640, 480), Image.BILINEAR)
            x_tensor = torch.from_numpy(np.array(pil_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            preds = model.decode_predictions(
                model(x_tensor),
                conf_threshold=0.30,
                nms_iou_threshold=0.45,
                class_thresholds=class_thresholds,
                cross_class_iou_threshold=0.55,
            )[0]
            for p in preds:
                p_box = [p["xmin"], p["ymin"], p["xmax"], p["ymax"]]
                best_obj = None
                best_iou = 0.0
                for obj in meta.get("objects", []):
                    if obj["class_id"] != p["class_id"] or not obj.get("position_world_3d"):
                        continue
                    bb = obj["bbox_2d"]
                    iou = compute_box_iou(p_box, [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]])
                    if iou >= 0.50 and iou > best_iou:
                        best_iou = iou
                        best_obj = obj
                if best_obj:
                    gt_world = best_obj["position_world_3d"]
                    wm, _ = _box_to_world_pos(p_box, p["class_name"], use_measured=True)
                    we, _ = _box_to_world_pos(p_box, p["class_name"], use_measured=False)
                    if wm is not None:
                        cnn_measured_errs.append(evaluate_3d_localization_error(wm, gt_world))
                    if we is not None:
                        cnn_est_errs.append(evaluate_3d_localization_error(we, gt_world))

    def _stats(arr):
        if not arr:
            return {"n": 0, "mean_m": None, "median_m": None, "rmse_m": None}
        a = np.array(arr, dtype=np.float64)
        return {
            "n": int(len(a)),
            "mean_m": round(float(np.mean(a)), 3),
            "median_m": round(float(np.median(a)), 3),
            "rmse_m": round(float(np.sqrt(np.mean(a ** 2))), 3),
        }

    return {
        "oracle_2d_measured_webgl_depth": _stats(oracle_measured_errs),
        "oracle_2d_estimated_bbox_height": _stats(oracle_est_errs),
        "cnn_pred_measured_webgl_depth": _stats(cnn_measured_errs),
        "cnn_pred_estimated_bbox_height": _stats(cnn_est_errs),
        "depth_mae_measured_m": round(float(np.mean(depth_abs_errs_measured)), 3) if depth_abs_errs_measured else None,
        "depth_mae_estimated_m": round(float(np.mean(depth_abs_errs_est)), 3) if depth_abs_errs_est else None,
        "per_class_measured_mean_m": {
            c: (round(float(np.mean(v)), 3) if v else None) for c, v in per_class_measured.items()
        },
        "per_class_estimated_mean_m": {
            c: (round(float(np.mean(v)), 3) if v else None) for c, v in per_class_est.items()
        },
    }


def verify_end_to_end_pipeline(threejs_dir: str) -> dict:
    """Runs real FastAPI integration: RGB + RG16 Depth -> CNN -> Depth -> 3D -> SceneGraph -> Tool -> Nav."""
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.game.facility import seed_facility_scene
    from backend.spatial.scene_graph import global_scene_graph
    from backend.tools.registry import global_tool_registry

    seed_facility_scene()
    client = TestClient(app)

    ann_files = sorted(glob.glob(os.path.join(threejs_dir, "annotations", "*.json")))
    sample_meta = None
    for af in ann_files:
        with open(af, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("split") == "test" and m.get("depth_filename"):
            sample_meta = m
            break
    if sample_meta is None:
        return {"passed": False, "error": "No Three.js test sample with depth found"}

    with Image.open(os.path.join(threejs_dir, sample_meta["image_filename"])) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=92)
    rgb_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Encode saved metric float depth cleanly into RG16 PNG exactly matching browser frontend
    depth_m = np.load(os.path.join(threejs_dir, sample_meta["depth_filename"])).astype(np.float32)
    mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    rgba = np.zeros((depth_m.shape[0], depth_m.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = (mm >> 8).astype(np.uint8)
    rgba[..., 1] = (mm & 0xFF).astype(np.uint8)
    rgba[..., 3] = 255
    d_buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(d_buf, format="PNG")
    depth_b64 = base64.b64encode(d_buf.getvalue()).decode("ascii")

    cam = sample_meta["camera"]
    pose = sample_meta.get("camera_pose", {"position": [0.0, 1.5, -15.0], "rotation": [0.0, 0.0, 0.0, 1.0]})

    resp = client.post(
        "/perception",
        json={
            "image_base64": rgb_b64,
            "depth_base64": depth_b64,
            "depth_encoding": "webgl_rg16_mm",
            "camera_pose": pose,
            "camera_intrinsics": {"fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"]},
            "coordinate_system": "threejs",
        },
    )
    perc_data = resp.json()
    live_objs = global_tool_registry.execute("get_visible_objects", {"live_only": True})
    all_objs = global_tool_registry.execute("get_visible_objects", {"live_only": False})
    scene_data = client.get("/scene").json()
    tel_data = client.get("/telemetry/latest").json()

    live_count = len(live_objs.get("result", []))
    measured_count = sum(1 for d in perc_data.get("detections", []) if d.get("depth_source") == "measured_webgl_depth")
    nav_action = {"success": False}
    if perc_data.get("detections"):
        target_cls = perc_data["detections"][0]["class_name"]
        nav_action = global_tool_registry.execute("navigate_to", {"destination": target_cls})

    return {
        "passed": resp.status_code == 200 and live_count > 0 and measured_count > 0,
        "test_frame_id": sample_meta["frame_id"],
        "perception_status": resp.status_code,
        "inference_time_ms": perc_data.get("inference_time_ms"),
        "detection_count": len(perc_data.get("detections", [])),
        "measured_webgl_depth_count": measured_count,
        "live_only_tool_count": live_count,
        "total_scene_nodes": len(scene_data.get("nodes", [])),
        "live_objects_sample": live_objs.get("result", [])[:4],
        "navigation_action_success": bool(nav_action.get("success")),
        "telemetry_depth_sources": (tel_data.get("cv", {}).get("last_live_scan") or {}).get("depth_sources_used", []),
        "scene_graph_edges_count": len(scene_data.get("edges", [])),
    }


def _plot_confusion_matrices(tjs_cm: dict, sun_cm: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, cm_data, title in [
        (axes[0], tjs_cm, "Three.js Sector 4 Held-Out Test Confusion Matrix"),
        (axes[1], sun_cm, "SUN RGB-D Held-Out Test Confusion Matrix"),
    ]:
        labels = cm_data["labels"]
        mat = np.array(cm_data["matrix"], dtype=float)
        row_sums = np.maximum(1.0, mat.sum(axis=1, keepdims=True))
        norm = mat / row_sums

        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Ground Truth Class")
        ax.set_title(title, fontsize=11)

        for i in range(len(labels)):
            for j in range(len(labels)):
                cnt = int(mat[i, j])
                val_color = "white" if norm[i, j] > 0.5 else "#111827"
                ax.text(j, i, f"{cnt}", ha="center", va="center", color=val_color, fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Row-Normalized Proportion")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_per_class_metrics(tjs_eval: dict, sun_eval: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    x = np.arange(len(TARGET_CLASSES))
    w = 0.25

    for ax, ev, title in [
        (axes[0], tjs_eval, "Three.js Held-Out Test: Precision, Recall & AP@0.5"),
        (axes[1], sun_eval, "SUN RGB-D Held-Out Test: Precision, Recall & AP@0.5"),
    ]:
        p_vals = [ev["per_class_metrics"][c]["precision"] for c in TARGET_CLASSES]
        r_vals = [ev["per_class_metrics"][c]["recall"] for c in TARGET_CLASSES]
        ap_vals = [ev["per_class_metrics"][c]["ap_50"] for c in TARGET_CLASSES]

        ax.bar(x - w, p_vals, width=w, label="Precision", color="#2563eb")
        ax.bar(x, r_vals, width=w, label="Recall", color="#10b981")
        ax.bar(x + w, ap_vals, width=w, label="AP@0.5", color="#f59e0b")
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_CLASSES, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(title)
        ax.legend(loc="upper right")

    axes[0].set_ylabel("Score [0.0 – 1.0]")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_map_comparison(report: dict, out_path: str):
    categories = ["Three.js Precision", "Three.js mAP@0.5", "SUN RGB-D Precision", "SUN RGB-D mAP@0.5"]
    baseline_vals = [0.1090, 0.4747, 0.0227, 0.1212]
    uniform_vals = [
        report["threejs_test_uniform_025"]["overall_precision"],
        report["threejs_test_uniform_025"]["mAP_50"],
        report["sunrgbd_test_uniform_025"]["overall_precision"],
        report["sunrgbd_test_uniform_025"]["mAP_50"],
    ]
    calibrated_vals = [
        report["threejs_test_calibrated"]["overall_precision"],
        report["threejs_test_calibrated"]["mAP_50"],
        report["sunrgbd_test_calibrated"]["overall_precision"],
        report["sunrgbd_test_calibrated"]["mAP_50"],
    ]

    x = np.arange(len(categories))
    w = 0.25
    plt.figure(figsize=(10.5, 5.2))
    b1 = plt.bar(x - w, baseline_vals, width=w, label="Prior Baseline (BCE Mean Loss, Uncalibrated)", color="#94a3b8")
    b2 = plt.bar(x, uniform_vals, width=w, label="Improved CNN + Dual NMS (Uniform 0.25)", color="#3b82f6")
    b3 = plt.bar(x + w, calibrated_vals, width=w, label="Improved CNN + Dual NMS + Val Thresholds", color="#10b981")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    plt.xticks(x, categories, fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Metric Score")
    plt.title("Computer Vision Subsystem Comparison: Baseline vs Focal+OHEM+GIoU + Val Calibration")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_detection_examples(
    root_dir: str,
    model: SpatialDetector,
    device: torch.device,
    class_thresholds: dict,
    out_path: str,
    title: str,
):
    ds = SUNRGBDDataset(root_dir=root_dir, split="test", augment=False)
    indices = np.linspace(0, len(ds) - 1, min(4, len(ds)), dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    axes = axes.ravel()

    model.eval()
    with torch.no_grad():
        for ax_idx, ds_idx in enumerate(indices):
            sample = ds[int(ds_idx)]
            rgb = sample["rgb"]
            img_np = (rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            preds = model.decode_predictions(
                model(rgb.unsqueeze(0).to(device)),
                conf_threshold=0.30,
                nms_iou_threshold=0.45,
                class_thresholds=class_thresholds,
                cross_class_iou_threshold=0.55,
            )[0]

            ax = axes[ax_idx]
            ax.imshow(img_np)
            ax.set_title(f"{sample['frame_id']} (GT: {len(sample['boxes_2d'])}, Pred: {len(preds)})", fontsize=10)
            ax.axis("off")

            for box, lbl in zip(sample["boxes_2d"].tolist(), sample["labels"].tolist()):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="#22c55e", facecolor="none", linestyle="--")
                ax.add_patch(rect)
                ax.text(x1 + 2, max(10, y1 - 3), f"GT:{TARGET_CLASSES[lbl]}", color="#22c55e", fontsize=8, bbox=dict(facecolor="black", alpha=0.55, pad=1))

            for p in preds:
                x1, y1, x2, y2 = p["xmin"], p["ymin"], p["xmax"], p["ymax"]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="#38bdf8", facecolor="none")
                ax.add_patch(rect)
                ax.text(x1 + 2, min(470, y2 + 10), f"{p['class_name']} {p['confidence']:.2f}", color="#38bdf8", fontsize=8, bbox=dict(facecolor="black", alpha=0.65, pad=1))

    plt.suptitle(f"{title} (Green Dashed = Ground Truth, Cyan Solid = SpatialDetector CNN)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_3d_localization(loc_threejs: dict, loc_sun: dict, out_path: str):
    labels = [
        "Three.js Oracle 2D\n+ Measured WebGL Depth",
        "Three.js Oracle 2D\n+ Estimated BBox Height",
        "Three.js CNN Pred\n+ Measured WebGL Depth",
        "Three.js CNN Pred\n+ Estimated BBox Height",
        "SUN RGB-D Oracle 2D\n+ Real Sensor Depth",
    ]
    means = [
        loc_threejs["oracle_2d_measured_webgl_depth"]["mean_m"] or 0.0,
        loc_threejs["oracle_2d_estimated_bbox_height"]["mean_m"] or 0.0,
        loc_threejs["cnn_pred_measured_webgl_depth"]["mean_m"] or 0.0,
        loc_threejs["cnn_pred_estimated_bbox_height"]["mean_m"] or 0.0,
        loc_sun.get("mean_error_meters", 0.0) or 0.0,
    ]
    medians = [
        loc_threejs["oracle_2d_measured_webgl_depth"]["median_m"] or 0.0,
        loc_threejs["oracle_2d_estimated_bbox_height"]["median_m"] or 0.0,
        loc_threejs["cnn_pred_measured_webgl_depth"]["median_m"] or 0.0,
        loc_threejs["cnn_pred_estimated_bbox_height"]["median_m"] or 0.0,
        loc_sun.get("median_error_meters", 0.0) or 0.0,
    ]

    x = np.arange(len(labels))
    w = 0.34
    plt.figure(figsize=(11, 5.2))
    b1 = plt.bar(x - w / 2, means, width=w, label="Mean 3D Euclidean Error (m)", color="#2563eb")
    b2 = plt.bar(x + w / 2, medians, width=w, label="Median 3D Euclidean Error (m)", color="#10b981")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}m", ha="center", va="bottom", fontsize=8.5)

    plt.xticks(x, labels, fontsize=9)
    plt.ylabel("3D Euclidean Localization Error (Meters)")
    plt.title("3D Localization Error: Measured Depth vs Height-Ratio Depth Estimation")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_comprehensive_evaluation():
    device = get_optimal_device()
    ckpt_path = "models/vision/spatial_detector_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    class_thresholds = ckpt.get("class_thresholds") or {c: 0.35 for c in TARGET_CLASSES}

    model = SpatialDetector().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sun_dir = "data/sunrgbd"
    tjs_dir = "data/generated/threejs_sector4"

    sun_audit = audit_dataset_splits(sun_dir, "SUN RGB-D")
    tjs_audit = audit_dataset_splits(tjs_dir, "Three.js Sector 4")

    sun_test_loader = DataLoader(SUNRGBDDataset(sun_dir, split="test", augment=False), batch_size=8, shuffle=False, collate_fn=sunrgbd_collate_fn)
    tjs_test_loader = DataLoader(SUNRGBDDataset(tjs_dir, split="test", augment=False), batch_size=8, shuffle=False, collate_fn=sunrgbd_collate_fn)

    tjs_uniform = evaluate_detector(model, tjs_test_loader, device, conf_threshold=0.25, iou_threshold=0.50, cross_class_iou_threshold=0.55)
    tjs_cal = evaluate_detector(model, tjs_test_loader, device, conf_threshold=0.32, iou_threshold=0.50, class_thresholds=class_thresholds, cross_class_iou_threshold=0.55)

    sun_uniform = evaluate_detector(model, sun_test_loader, device, conf_threshold=0.25, iou_threshold=0.50, cross_class_iou_threshold=0.55)
    sun_cal = evaluate_detector(model, sun_test_loader, device, conf_threshold=0.32, iou_threshold=0.50, class_thresholds=class_thresholds, cross_class_iou_threshold=0.55)

    loc_threejs = evaluate_threejs_3d_localization(tjs_dir, model, device, class_thresholds)
    loc_sun = benchmark_localization_on_dataset(sun_dir)
    e2e_flow = verify_end_to_end_pipeline(tjs_dir)

    report = {
        "checkpoint_epoch": ckpt.get("epoch"),
        "val_calibrated_thresholds": class_thresholds,
        "dataset_audits": {
            "sunrgbd": sun_audit,
            "threejs_sector4": tjs_audit,
        },
        "threejs_test_uniform_025": tjs_uniform,
        "threejs_test_calibrated": tjs_cal,
        "sunrgbd_test_uniform_025": sun_uniform,
        "sunrgbd_test_calibrated": sun_cal,
        "localization_3d": {
            "threejs": loc_threejs,
            "sunrgbd": loc_sun,
        },
        "end_to_end_pipeline_test": e2e_flow,
    }

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    with open("results/metrics/threejs_benchmark_eval.json", "w", encoding="utf-8") as f:
        json.dump(tjs_cal, f, indent=2)
    with open("results/metrics/real_benchmark_eval.json", "w", encoding="utf-8") as f:
        json.dump(sun_cal, f, indent=2)
    with open("results/metrics/cv_comprehensive_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _plot_confusion_matrices(tjs_cal["confusion_matrix"], sun_cal["confusion_matrix"], "results/plots/cv_confusion_matrix.png")
    _plot_per_class_metrics(tjs_cal, sun_cal, "results/plots/cv_per_class_precision_recall_ap.png")
    _plot_map_comparison(report, "results/plots/cv_map_comparison.png")
    _plot_detection_examples(tjs_dir, model, device, class_thresholds, "results/plots/cv_detection_examples_threejs.png", "Three.js Held-Out Test Detections")
    _plot_detection_examples(sun_dir, model, device, class_thresholds, "results/plots/cv_detection_examples_sunrgbd.png", "SUN RGB-D Held-Out Test Detections")
    _plot_3d_localization(loc_threejs, loc_sun, "results/plots/cv_3d_localization_error.png")

    print("\n" + "=" * 72)
    print("COMPREHENSIVE CV EVALUATION COMPLETE")
    print("=" * 72)
    print("Three.js Calibrated Test:", json.dumps({k: tjs_cal[k] for k in ("mAP_50", "overall_precision", "overall_recall", "overall_f1", "total_tp", "total_fp", "total_fn")}, indent=2))
    print("SUN RGB-D Calibrated Test:", json.dumps({k: sun_cal[k] for k in ("mAP_50", "overall_precision", "overall_recall", "overall_f1", "total_tp", "total_fp", "total_fn")}, indent=2))
    print("3D Localization Summary:", json.dumps({"threejs": loc_threejs, "sunrgbd_mean_m": loc_sun.get("mean_error_meters"), "sunrgbd_median_m": loc_sun.get("median_error_meters")}, indent=2))
    print("End-to-End Integration:", json.dumps(e2e_flow, indent=2))


if __name__ == "__main__":
    run_comprehensive_evaluation()
