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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

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
        input_size: Tuple[int, int] = (640, 480), # (width, height)
        grid_size: Tuple[int, int] = (20, 15),     # (grid_w, grid_h)
        backbone_channels: List[int] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_w, self.input_h = input_size
        self.grid_w, self.grid_h = grid_size
        self.cell_w = self.input_w / self.grid_w  # 32.0 pixels
        self.cell_h = self.input_h / self.grid_h  # 32.0 pixels

        channels = backbone_channels or [32, 64, 128, 256, 256]

        # Convolutional Feature Backbone (Downsampling factor 32x: 2^5 = 32)
        # Input: (B, 3, 480, 640)
        self.layer1 = ConvBlock(3, channels[0], stride=2)            # -> (B, 32, 240, 320)
        self.layer2 = ConvBlock(channels[0], channels[1], stride=2)  # -> (B, 64, 120, 160)
        self.layer3 = ConvBlock(channels[1], channels[2], stride=2)  # -> (B, 128, 60, 80)
        self.layer4 = ConvBlock(channels[2], channels[3], stride=2)  # -> (B, 256, 30, 40)
        self.layer5 = ConvBlock(channels[3], channels[4], stride=2)  # -> (B, 256, 15, 20)

        # Neck: Context refinement
        self.neck = nn.Sequential(
            ConvBlock(channels[4], channels[4], kernel_size=3, padding=1),
            ConvBlock(channels[4], channels[4], kernel_size=3, padding=1)
        )

        # Prediction Head: 1x1 conv mapping 256 -> (1 + 4 + num_classes)
        # 1 (obj) + 4 (tx, ty, tw, th) + 6 (class logits) = 11 channels
        self.out_channels = 1 + 4 + self.num_classes
        self.head = nn.Conv2d(channels[4], self.out_channels, kernel_size=1, stride=1, padding=0)

        # Initialize weights from scratch (He/Kaiming normal for LeakyReLU)
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

        # Initialize objectness bias to prevent early divergence (prior = 0.01)
        # bias = -log((1 - p)/p) = -log(99) ~= -4.6
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[0] = -4.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: RGB tensor of shape (B, 3, 480, 640)
        Returns:
            Grid output tensor of shape (B, 15, 20, 11)
        """
        feat = self.layer1(x)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        feat = self.layer5(feat)
        feat = self.neck(feat)

        out = self.head(feat)  # (B, 11, 15, 20)
        # Permute to (B, 15, 20, 11): (B, H_grid, W_grid, C)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out

    def decode_predictions(
        self,
        grid_pred: torch.Tensor,
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45
    ) -> List[List[Dict[str, Any]]]:
        """Decodes raw grid predictions into bounding boxes, class labels, and confidences.

        Args:
            grid_pred: (B, 15, 20, 11) raw model outputs
            conf_threshold: minimum detection confidence
            nms_iou_threshold: IoU threshold for Non-Maximum Suppression

        Returns:
            List of detections per image in batch.
        """
        B, H, W, C = grid_pred.shape
        batch_detections = []

        for b in range(B):
            pred_b = grid_pred[b]  # (15, 20, 11)

            # Extract raw predictions
            obj_logits = pred_b[..., 0]
            tx_raw = pred_b[..., 1]
            ty_raw = pred_b[..., 2]
            tw_raw = pred_b[..., 3]
            th_raw = pred_b[..., 4]
            cls_logits = pred_b[..., 5:5 + self.num_classes]

            # Activations
            obj_prob = torch.sigmoid(obj_logits)
            cls_prob = F.softmax(cls_logits, dim=-1)
            max_cls_prob, max_cls_id = torch.max(cls_prob, dim=-1)

            total_conf = obj_prob * max_cls_prob  # (15, 20)

            # Filter candidates above confidence threshold
            mask = total_conf > conf_threshold
            y_indices, x_indices = torch.where(mask)

            candidates = []
            for i, j in zip(y_indices, x_indices):
                score = float(total_conf[i, j].item())
                class_id = int(max_cls_id[i, j].item())

                # Decode center offsets
                # sigmoid(tx) gives offset within cell [0, 1]
                cx_norm = torch.sigmoid(tx_raw[i, j]).item()
                cy_norm = torch.sigmoid(ty_raw[i, j]).item()

                u_center = (float(j) + cx_norm) * self.cell_w
                v_center = (float(i) + cy_norm) * self.cell_h

                # Box width and height: exp(tw) * cell_size
                # Clamped to prevent exponential explosion
                box_w = torch.exp(torch.clamp(tw_raw[i, j], -3.0, 3.5)).item() * self.cell_w
                box_h = torch.exp(torch.clamp(th_raw[i, j], -3.0, 3.5)).item() * self.cell_h

                xmin = max(0.0, u_center - box_w / 2.0)
                ymin = max(0.0, v_center - box_h / 2.0)
                xmax = min(float(self.input_w), u_center + box_w / 2.0)
                ymax = min(float(self.input_h), v_center + box_h / 2.0)

                candidates.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "confidence": score,
                    "class_id": class_id,
                    "class_name": TARGET_CLASSES[class_id]
                })

            # Apply Non-Maximum Suppression (from scratch)
            nms_detections = self._apply_nms(candidates, nms_iou_threshold)
            batch_detections.append(nms_detections)

        return batch_detections

    @staticmethod
    def _apply_nms(boxes: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
        """Non-Maximum Suppression implemented from raw arithmetic."""
        if len(boxes) == 0:
            return []

        # Sort descending by confidence
        sorted_boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
        keep = []

        while len(sorted_boxes) > 0:
            best = sorted_boxes.pop(0)
            keep.append(best)

            remaining = []
            for candidate in sorted_boxes:
                # Same-class suppression
                if candidate["class_id"] != best["class_id"]:
                    remaining.append(candidate)
                    continue

                # Compute IoU
                ixmin = max(best["xmin"], candidate["xmin"])
                iymin = max(best["ymin"], candidate["ymin"])
                ixmax = min(best["xmax"], candidate["xmax"])
                iymax = min(best["ymax"], candidate["ymax"])

                iw = max(0.0, ixmax - ixmin)
                ih = max(0.0, iymax - iymin)
                inter = iw * ih

                area_a = (best["xmax"] - best["xmin"]) * (best["ymax"] - best["ymin"])
                area_b = (candidate["xmax"] - candidate["xmin"]) * (candidate["ymax"] - candidate["ymin"])
                union = area_a + area_b - inter

                iou = inter / union if union > 0 else 0.0
                if iou < iou_thresh:
                    remaining.append(candidate)

            sorted_boxes = remaining

        return keep
