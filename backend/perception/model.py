"""Custom Single-Stage CNN Object Detector for Spatial Quest (from scratch).

Implemented from random initialization without using pre-trained weights.
Grid-based anchor-free single-stage prediction architecture downsampling
an input image (3, 480, 640) by 32x to a spatial grid (15, 20, 11).

Output channels per cell:
  [0]     : Objectness logit (sigmoid -> probability in [0, 1])
  [1:3]   : Center offsets (tx, ty) relative to cell top-left (sigmoid -> [0, 1])
  [3:5]   : Box dimensions (tw, th) relative to cell size
  [5:11]  : Class logits across 6 classes (table, chair, door, bookshelf, sofa, cabinet)
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_CLASSES = ["table", "chair", "door", "bookshelf", "sofa", "cabinet"]
NUM_CLASSES = len(TARGET_CLASSES)


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + LeakyReLU activation block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SpatialDetector(nn.Module):
    """Custom Single-Stage CNN Object Detector built from tensor mathematics."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_size: Tuple[int, int] = (640, 480),
        grid_size: Tuple[int, int] = (20, 15),
        backbone_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_w, self.input_h = input_size
        self.grid_w, self.grid_h = grid_size
        self.cell_w = self.input_w / self.grid_w  # 32.0 pixels
        self.cell_h = self.input_h / self.grid_h  # 32.0 pixels

        channels = backbone_channels or [32, 64, 128, 256, 256]

        # Convolutional Feature Backbone (Downsampling factor 32x: 2^5 = 32)
        self.layer1 = ConvBlock(3, channels[0], stride=2)            # -> (B, 32, 240, 320)
        self.layer2 = ConvBlock(channels[0], channels[1], stride=2)  # -> (B, 64, 120, 160)
        self.layer3 = ConvBlock(channels[1], channels[2], stride=2)  # -> (B, 128, 60, 80)
        self.layer4 = ConvBlock(channels[2], channels[3], stride=2)  # -> (B, 256, 30, 40)
        self.layer5 = ConvBlock(channels[3], channels[4], stride=2)  # -> (B, 256, 15, 20)

        # Multi-scale skip path fusing 16x (layer4: 30x40) + 32x (layer5: 15x20) features
        self.skip4 = ConvBlock(channels[3], 128, kernel_size=3, stride=2, padding=1)
        self.neck = nn.Sequential(
            ConvBlock(channels[4], channels[4], kernel_size=3, padding=1),
            ConvBlock(channels[4], channels[4], kernel_size=3, padding=1),
        )
        self.fuse = ConvBlock(channels[4] + 128, channels[4], kernel_size=1, padding=0)

        # Prediction Head: 1x1 conv mapping 256 -> (1 + 4 + num_classes)
        self.out_channels = 1 + 4 + self.num_classes
        self.head = nn.Conv2d(channels[4], self.out_channels, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        """Initializes all weights from random distributions."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Objectness prior = 0.01 -> bias ~= -4.5
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[0] = -4.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, 3, 480, 640) -> (B, 15, 20, 11)."""
        feat = self.layer1(x)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        f4 = self.layer4(feat)
        f5 = self.layer5(f4)
        f_ctx = f5 + self.neck(f5)
        f_fused = self.fuse(torch.cat([f_ctx, self.skip4(f4)], dim=1))

        out = self.head(f_fused)
        return out.permute(0, 2, 3, 1).contiguous()

    def decode_predictions(
        self,
        grid_pred: torch.Tensor,
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        class_thresholds: Optional[Dict[str, float]] = None,
        cross_class_iou_threshold: float = 0.55,
        min_box_pixels: float = 14.0,
        max_detections: int = 16,
    ) -> List[List[Dict[str, Any]]]:
        """Decodes raw grid predictions into boxes with optional per-class calibration and dual NMS."""
        B, H, W, _ = grid_pred.shape
        batch_detections = []

        thresh_vec = None
        if class_thresholds:
            thresh_vec = torch.tensor(
                [float(class_thresholds.get(c, conf_threshold)) for c in TARGET_CLASSES],
                device=grid_pred.device,
                dtype=grid_pred.dtype,
            )

        for b in range(B):
            pred_b = grid_pred[b]
            obj_prob = torch.sigmoid(pred_b[..., 0])
            cls_prob = F.softmax(pred_b[..., 5:5 + self.num_classes], dim=-1)
            max_cls_prob, max_cls_id = torch.max(cls_prob, dim=-1)
            total_conf = obj_prob * max_cls_prob

            if thresh_vec is not None:
                cell_thresh = thresh_vec[max_cls_id]
                mask = total_conf >= cell_thresh
            else:
                mask = total_conf > conf_threshold

            y_indices, x_indices = torch.where(mask)
            tx_raw = pred_b[..., 1]
            ty_raw = pred_b[..., 2]
            tw_raw = pred_b[..., 3]
            th_raw = pred_b[..., 4]

            candidates = []
            for i, j in zip(y_indices, x_indices):
                score = float(total_conf[i, j].item())
                class_id = int(max_cls_id[i, j].item())

                cx_norm = torch.sigmoid(tx_raw[i, j]).item()
                cy_norm = torch.sigmoid(ty_raw[i, j]).item()
                u_center = (float(j) + cx_norm) * self.cell_w
                v_center = (float(i) + cy_norm) * self.cell_h

                box_w = torch.exp(torch.clamp(tw_raw[i, j], -3.0, 3.5)).item() * self.cell_w
                box_h = torch.exp(torch.clamp(th_raw[i, j], -3.0, 3.5)).item() * self.cell_h

                xmin = max(0.0, u_center - box_w / 2.0)
                ymin = max(0.0, v_center - box_h / 2.0)
                xmax = min(float(self.input_w), u_center + box_w / 2.0)
                ymax = min(float(self.input_h), v_center + box_h / 2.0)

                if (xmax - xmin) < min_box_pixels or (ymax - ymin) < min_box_pixels:
                    continue

                candidates.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "confidence": score,
                    "class_id": class_id,
                    "class_name": TARGET_CLASSES[class_id],
                    "objectness": round(float(obj_prob[i, j].item()), 4),
                    "class_prob": round(float(max_cls_prob[i, j].item()), 4),
                })

            nms_detections = self._apply_nms(
                candidates,
                iou_thresh=nms_iou_threshold,
                cross_class_iou_thresh=cross_class_iou_threshold,
            )
            if max_detections and len(nms_detections) > max_detections:
                nms_detections = nms_detections[:max_detections]
            batch_detections.append(nms_detections)

        return batch_detections

    @staticmethod
    def _box_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        ixmin = max(a["xmin"], b["xmin"])
        iymin = max(a["ymin"], b["ymin"])
        ixmax = min(a["xmax"], b["xmax"])
        iymax = min(a["ymax"], b["ymax"])
        inter = max(0.0, ixmax - ixmin) * max(0.0, iymax - iymin)
        area_a = max(0.0, a["xmax"] - a["xmin"]) * max(0.0, a["ymax"] - a["ymin"])
        area_b = max(0.0, b["xmax"] - b["xmin"]) * max(0.0, b["ymax"] - b["ymin"])
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def _apply_nms(
        boxes: List[Dict[str, Any]],
        iou_thresh: float,
        cross_class_iou_thresh: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """Dual-stage arithmetic NMS: tight same-class suppression + high-IoU cross-class dedup."""
        if len(boxes) == 0:
            return []

        sorted_boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
        keep = []

        while sorted_boxes:
            best = sorted_boxes.pop(0)
            keep.append(best)

            remaining = []
            for candidate in sorted_boxes:
                iou = SpatialDetector._box_iou(best, candidate)
                if candidate["class_id"] == best["class_id"]:
                    if iou < iou_thresh:
                        remaining.append(candidate)
                else:
                    if cross_class_iou_thresh is None or iou < cross_class_iou_thresh:
                        remaining.append(candidate)
            sorted_boxes = remaining

        return keep
