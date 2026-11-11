"""Evaluation Metrics for Custom Object Detector (mAP@0.5, IoU, Precision, Recall, Confusion Matrix).

Implements standard PASCAL VOC style 11-point interpolated Average Precision,
per-class F1 calibration on validation splits, and class + background confusion matrix
without third-party black-box metric libraries.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

TARGET_CLASSES = ["table", "chair", "door", "bookshelf", "sofa", "cabinet"]


def compute_box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
    ixmin = max(box_a[0], box_b[0])
    iymin = max(box_a[1], box_b[1])
    ixmax = min(box_a[2], box_b[2])
    iymax = min(box_a[3], box_b[3])

    iw = max(0.0, ixmax - ixmin)
    ih = max(0.0, iymax - iymin)
    inter_area = iw * ih

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_detector(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    class_thresholds: Optional[Dict[str, float]] = None,
    cross_class_iou_threshold: float = 0.55,
) -> Dict[str, Any]:
    """Evaluates a trained detector on a split, computing mAP@0.5, P/R/F1, and confusion matrix."""
    model.eval()
    num_c = len(TARGET_CLASSES)
    bg_idx = num_c

    predictions_per_class: Dict[int, List[Tuple[float, bool]]] = {c: [] for c in range(num_c)}
    gt_count_per_class: Dict[int, int] = {c: 0 for c in range(num_c)}
    # confusion_matrix[gt_cls][pred_cls], last row/col is 'background'
    conf_mat = np.zeros((num_c + 1, num_c + 1), dtype=int)
    latencies = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["rgb"].to(device)
            B = imgs.shape[0]

            t0 = time.time()
            grid_out = model(imgs)
            batch_preds = model.decode_predictions(
                grid_out,
                conf_threshold=conf_threshold,
                nms_iou_threshold=0.45,
                class_thresholds=class_thresholds,
                cross_class_iou_threshold=cross_class_iou_threshold,
            )
            elapsed_ms = ((time.time() - t0) * 1000.0) / B
            latencies.append(elapsed_ms)

            for b in range(B):
                preds = batch_preds[b]
                gt_boxes = batch["boxes_2d"][b].tolist()
                gt_labels = batch["labels"][b].tolist()

                for lbl in gt_labels:
                    gt_count_per_class[lbl] += 1

                # 1. Standard per-class AP matching
                matched_gt_same_cls = set()
                sorted_preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)
                for p in sorted_preds:
                    p_box = [p["xmin"], p["ymin"], p["xmax"], p["ymax"]]
                    p_cls = p["class_id"]
                    score = p["confidence"]

                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, (g_box, g_lbl) in enumerate(zip(gt_boxes, gt_labels)):
                        if g_lbl == p_cls and g_idx not in matched_gt_same_cls:
                            iou = compute_box_iou(p_box, g_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx

                    if best_iou >= iou_threshold and best_gt_idx != -1:
                        predictions_per_class[p_cls].append((score, True))
                        matched_gt_same_cls.add(best_gt_idx)
                    else:
                        predictions_per_class[p_cls].append((score, False))

                # 2. Class-agnostic matching for True Confusion Matrix (detects misclassified objects vs background FP/FN)
                matched_gt_any = set()
                for p in sorted_preds:
                    p_box = [p["xmin"], p["ymin"], p["xmax"], p["ymax"]]
                    p_cls = p["class_id"]

                    best_any_iou = 0.0
                    best_any_idx = -1
                    best_any_lbl = bg_idx
                    # Prefer exact class if both overlap >= iou_threshold
                    for g_idx, (g_box, g_lbl) in enumerate(zip(gt_boxes, gt_labels)):
                        if g_idx in matched_gt_any:
                            continue
                        iou = compute_box_iou(p_box, g_box)
                        effective = iou + (0.05 if g_lbl == p_cls else 0.0)
                        if iou >= iou_threshold and effective > best_any_iou:
                            best_any_iou = effective
                            best_any_idx = g_idx
                            best_any_lbl = g_lbl

                    if best_any_idx != -1:
                        matched_gt_any.add(best_any_idx)
                        conf_mat[best_any_lbl, p_cls] += 1
                    else:
                        # Background falsely predicted as p_cls
                        conf_mat[bg_idx, p_cls] += 1

                for g_idx, g_lbl in enumerate(gt_labels):
                    if g_idx not in matched_gt_any:
                        # Ground truth object missed -> predicted as background (FN)
                        conf_mat[g_lbl, bg_idx] += 1

    ap_per_class = {}
    per_class_metrics = {}
    total_tp = 0
    total_fp = 0

    for c in range(num_c):
        c_name = TARGET_CLASSES[c]
        preds = predictions_per_class[c]
        n_gt = gt_count_per_class[c]
        tp_c = sum(1 for _, is_tp in preds if is_tp)
        fp_c = len(preds) - tp_c
        fn_c = max(0, n_gt - tp_c)

        prec_c = round(tp_c / (tp_c + fp_c), 4) if (tp_c + fp_c) > 0 else 0.0
        rec_c = round(tp_c / n_gt, 4) if n_gt > 0 else 0.0
        f1_c = round((2 * prec_c * rec_c) / (prec_c + rec_c), 4) if (prec_c + rec_c) > 0 else 0.0

        total_tp += tp_c
        total_fp += fp_c

        if n_gt == 0 or len(preds) == 0:
            ap_per_class[c_name] = 0.0
        else:
            preds.sort(key=lambda x: x[0], reverse=True)
            tp_cum = np.cumsum([1 if p[1] else 0 for p in preds])
            fp_cum = np.cumsum([0 if p[1] else 1 for p in preds])
            recalls = tp_cum / n_gt
            precisions = tp_cum / (tp_cum + fp_cum)

            ap = 0.0
            for t in np.linspace(0.0, 1.0, 11):
                p_above = precisions[recalls >= t]
                max_p = np.max(p_above) if len(p_above) > 0 else 0.0
                ap += max_p / 11.0
            ap_per_class[c_name] = round(float(ap), 4)

        per_class_metrics[c_name] = {
            "ap_50": ap_per_class[c_name],
            "precision": prec_c,
            "recall": rec_c,
            "f1": f1_c,
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "gt_count": n_gt,
            "pred_count": len(preds),
        }

    mean_ap = round(float(np.mean(list(ap_per_class.values()))), 4)
    overall_precision = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
    total_gt = sum(gt_count_per_class.values())
    overall_recall = round(total_tp / total_gt, 4) if total_gt > 0 else 0.0
    overall_f1 = (
        round((2 * overall_precision * overall_recall) / (overall_precision + overall_recall), 4)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    return {
        "mAP_50": mean_ap,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": max(0, total_gt - total_tp),
        "per_class_ap": ap_per_class,
        "per_class_metrics": per_class_metrics,
        "gt_counts": {TARGET_CLASSES[c]: gt_count_per_class[c] for c in range(num_c)},
        "confusion_matrix": {
            "labels": TARGET_CLASSES + ["background"],
            "matrix": conf_mat.tolist(),
        },
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
    }


def calibrate_class_thresholds(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    candidate_thresholds: Optional[List[float]] = None,
    iou_threshold: float = 0.50,
    cross_class_iou_threshold: float = 0.55,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Sweeps per-class confidence thresholds strictly on validation data maximizing F1."""
    if candidate_thresholds is None:
        candidate_thresholds = [round(0.18 + 0.04 * i, 2) for i in range(14)]  # 0.18 .. 0.70

    per_class_sweep: Dict[str, List[Dict[str, float]]] = {c: [] for c in TARGET_CLASSES}
    best_thresholds: Dict[str, float] = {c: 0.35 for c in TARGET_CLASSES}

    for thr in candidate_thresholds:
        res = evaluate_detector(
            model,
            val_loader,
            device,
            conf_threshold=thr,
            iou_threshold=iou_threshold,
            cross_class_iou_threshold=cross_class_iou_threshold,
        )
        for c in TARGET_CLASSES:
            m = res["per_class_metrics"][c]
            per_class_sweep[c].append({
                "threshold": thr,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "tp": m["tp"],
                "fp": m["fp"],
            })

    for c in TARGET_CLASSES:
        rows = per_class_sweep[c]
        # Maximize F1, tie-breaking toward higher precision
        best_row = max(rows, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
        # Guard against zero-GT classes on small validation slice
        best_thresholds[c] = best_row["threshold"] if best_row["f1"] > 0 else 0.38

    return best_thresholds, per_class_sweep
