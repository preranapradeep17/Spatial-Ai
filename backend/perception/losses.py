"""Multi-Task Loss Functions and Target Encoders for Custom Object Detector.

Implements:
1. Grid Target Assignment (anchor-free center cell matching)
2. Objectness BCE Loss with negative down-weighting
3. Box Center & Scale Smooth L1 Loss
4. Multi-Class Cross-Entropy Loss
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class DetectionLoss(nn.Module):
    """Computes combined multi-task detection loss."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 15),     # (grid_w, grid_h)
        input_size: Tuple[int, int] = (640, 480),   # (width, height)
        num_classes: int = 6,
        lambda_obj: float = 1.0,
        lambda_box: float = 5.0,
        lambda_cls: float = 1.0,
        neg_obj_weight: float = 0.1
    ):
        super().__init__()
        self.grid_w, self.grid_h = grid_size
        self.input_w, self.input_h = input_size
        self.cell_w = self.input_w / self.grid_w  # 32.0
        self.cell_h = self.input_h / self.grid_h  # 32.0
        self.num_classes = num_classes

        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.neg_obj_weight = neg_obj_weight

    def build_targets(
        self,
        boxes_batch: List[torch.Tensor],
        labels_batch: List[torch.Tensor],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes variable ground-truth bounding boxes into fixed spatial grid targets.

        Returns:
            obj_target: (B, 15, 20) float tensor (1 for positive cell, 0 for negative)
            box_target: (B, 15, 20, 4) float tensor (tx, ty, tw, th)
            cls_target: (B, 15, 20) long tensor (class index 0..5)
            pos_mask: (B, 15, 20) bool tensor indicating positive cells
        """
        B = len(boxes_batch)
        obj_target = torch.zeros((B, self.grid_h, self.grid_w), dtype=torch.float32, device=device)
        box_target = torch.zeros((B, self.grid_h, self.grid_w, 4), dtype=torch.float32, device=device)
        cls_target = torch.zeros((B, self.grid_h, self.grid_w), dtype=torch.long, device=device)
        pos_mask = torch.zeros((B, self.grid_h, self.grid_w), dtype=torch.bool, device=device)

        for b in range(B):
            boxes = boxes_batch[b]
            labels = labels_batch[b]

            if len(boxes) == 0:
                continue

            for obj_idx in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[obj_idx].tolist()
                cid = int(labels[obj_idx].item())

                # Object center
                xc = (xmin + xmax) / 2.0
                yc = (ymin + ymax) / 2.0
                w = max(1.0, xmax - xmin)
                h = max(1.0, ymax - ymin)

                # Responsible grid cell
                j = int(min(self.grid_w - 1, max(0, math.floor(xc / self.cell_w))))
                i = int(min(self.grid_h - 1, max(0, math.floor(yc / self.cell_h))))

                # Normalized offsets within cell [0, 1]
                tx = (xc / self.cell_w) - j
                ty = (yc / self.cell_h) - i

                # Log-scale box size relative to cell size
                tw = math.log(max(0.1, w / self.cell_w))
                th = math.log(max(0.1, h / self.cell_h))

                obj_target[b, i, j] = 1.0
                box_target[b, i, j] = torch.tensor([tx, ty, tw, th], dtype=torch.float32, device=device)
                cls_target[b, i, j] = cid
                pos_mask[b, i, j] = True

        return obj_target, box_target, cls_target, pos_mask

    def forward(
        self,
        grid_pred: torch.Tensor,
        boxes_batch: List[torch.Tensor],
        labels_batch: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Computes multi-task loss between predictions and batch targets.

        Args:
            grid_pred: (B, 15, 20, 11) model output
            boxes_batch: list of (N, 4) ground-truth boxes per image
            labels_batch: list of (N,) class IDs per image
        """
        device = grid_pred.device
        obj_target, box_target, cls_target, pos_mask = self.build_targets(boxes_batch, labels_batch, device)

        # 1. Objectness Loss (BCEWithLogits)
        obj_logits = grid_pred[..., 0]  # (B, 15, 20)
        # Weight negative cells to prevent background dominance
        weight_mask = torch.where(pos_mask, torch.ones_like(obj_target), torch.full_like(obj_target, self.neg_obj_weight))
        loss_obj = F.binary_cross_entropy_with_logits(obj_logits, obj_target, weight=weight_mask, reduction="mean")

        # 2. Box Regression Loss (Smooth L1 on positive cells only)
        if pos_mask.sum() > 0:
            pred_boxes_pos = grid_pred[pos_mask][:, 1:5]
            # tx, ty activations use sigmoid to constrain to [0, 1]
            pred_tx_ty = torch.sigmoid(pred_boxes_pos[:, 0:2])
            pred_tw_th = pred_boxes_pos[:, 2:4]
            pred_box_activated = torch.cat([pred_tx_ty, pred_tw_th], dim=-1)

            target_boxes_pos = box_target[pos_mask]
            loss_box = F.smooth_l1_loss(pred_box_activated, target_boxes_pos, reduction="mean")

            # 3. Classification Loss (Cross Entropy on positive cells only)
            pred_cls_logits = grid_pred[pos_mask][:, 5:5 + self.num_classes]
            target_cls_pos = cls_target[pos_mask]
            loss_cls = F.cross_entropy(pred_cls_logits, target_cls_pos, reduction="mean")
        else:
            loss_box = torch.tensor(0.0, device=device, requires_grad=True)
            loss_cls = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = (self.lambda_obj * loss_obj) + (self.lambda_box * loss_box) + (self.lambda_cls * loss_cls)

        return {
            "total_loss": total_loss,
            "loss_obj": loss_obj,
            "loss_box": loss_box,
            "loss_cls": loss_cls,
            "num_pos": float(pos_mask.sum().item())
        }
