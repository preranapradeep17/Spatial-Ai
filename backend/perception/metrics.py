"""Evaluation Metrics for Custom Object Detector (mAP@0.5, IoU, Precision, Recall).

Implements standard PASCAL VOC style 11-point interpolated Average Precision
without third-party metric black-boxes.
"""
import time
from typing import List, Dict, Any, Tuple
import torch
import numpy as np

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
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.50
) -> Dict[str, Any]:
    """Evaluates a trained detector on a test split, computing mAP@0.5 and class metrics."""
    model.eval()

    # Track detections and ground truths per class
    # class_id -> list of (confidence, is_true_positive)
    predictions_per_class: Dict[int, List[Tuple[float, bool]]] = {c: [] for c in range(len(TARGET_CLASSES))}
    gt_count_per_class: Dict[int, int] = {c: 0 for c in range(len(TARGET_CLASSES))}

    latencies = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["rgb"].to(device)
            B = imgs.shape[0]

            t0 = time.time()
            grid_out = model(imgs)
            batch_preds = model.decode_predictions(grid_out, conf_threshold=conf_threshold, nms_iou_threshold=0.45)
            elapsed_ms = ((time.time() - t0) * 1000.0) / B
            latencies.append(elapsed_ms)

            # Match against ground truth per image
            for b in range(B):
                preds = batch_preds[b]
                gt_boxes = batch["boxes_2d"][b].tolist()
                gt_labels = batch["labels"][b].tolist()

                # Count ground truths
                for lbl in gt_labels:
                    gt_count_per_class[lbl] += 1

                # Track which GT boxes have already been matched
                matched_gt = set()

                # Sort predictions descending by confidence
                sorted_preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)

                for p in sorted_preds:
                    p_box = [p["xmin"], p["ymin"], p["xmax"], p["ymax"]]
                    p_cls = p["class_id"]
                    score = p["confidence"]

                    best_iou = 0.0
                    best_gt_idx = -1

                    for g_idx, (g_box, g_lbl) in enumerate(zip(gt_boxes, gt_labels)):
                        if g_lbl == p_cls and g_idx not in matched_gt:
                            iou = compute_box_iou(p_box, g_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx

                    if best_iou >= iou_threshold and best_gt_idx != -1:
                        predictions_per_class[p_cls].append((score, True))
                        matched_gt.add(best_gt_idx)
                    else:
                        predictions_per_class[p_cls].append((score, False))

    # Compute Average Precision (AP) per class using 11-point interpolation
    ap_per_class = {}
    total_tp = 0
    total_fp = 0

    for c in range(len(TARGET_CLASSES)):
        c_name = TARGET_CLASSES[c]
        preds = predictions_per_class[c]
        n_gt = gt_count_per_class[c]

        if n_gt == 0:
            ap_per_class[c_name] = 0.0
            continue
        if len(preds) == 0:
            ap_per_class[c_name] = 0.0
            continue

        # Sort by confidence descending
        preds.sort(key=lambda x: x[0], reverse=True)
        tp_cum = np.cumsum([1 if p[1] else 0 for p in preds])
        fp_cum = np.cumsum([0 if p[1] else 1 for p in preds])

        total_tp += int(tp_cum[-1]) if len(tp_cum) > 0 else 0
        total_fp += int(fp_cum[-1]) if len(fp_cum) > 0 else 0

        recalls = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum)

        # 11-point interpolation (0.0, 0.1, ..., 1.0)
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 11):
            p_above = precisions[recalls >= t]
            max_p = np.max(p_above) if len(p_above) > 0 else 0.0
            ap += max_p / 11.0

        ap_per_class[c_name] = round(float(ap), 4)

    mean_ap = round(float(np.mean(list(ap_per_class.values()))), 4)
    overall_precision = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
    total_gt = sum(gt_count_per_class.values())
    overall_recall = round(total_tp / total_gt, 4) if total_gt > 0 else 0.0

    return {
        "mAP_50": mean_ap,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "per_class_ap": ap_per_class,
        "gt_counts": {TARGET_CLASSES[c]: gt_count_per_class[c] for c in range(len(TARGET_CLASSES))},
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0
    }
