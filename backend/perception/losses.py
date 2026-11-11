"""Multi-Task Loss Functions and Target Encoders for Custom Object Detector.

Implements:
1. Grid Target Assignment (anchor-free center cell matching)
2. Focal Objectness Loss + Hard Negative Mining (slashes background false positives)
3. Direct Generalized IoU (GIoU) + Smooth L1 Bounding-Box Regression
4. Label-Smoothed Multi-Class Cross-Entropy Loss
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bbox_giou_loss(
    pred_boxes_xyxy: torch.Tensor,
    target_boxes_xyxy: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes mean (1 - GIoU) loss directly in pixel box coordinates."""
    px1, py1, px2, py2 = pred_boxes_xyxy.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = target_boxes_xyxy.unbind(dim=-1)

    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)

    area_p = torch.clamp(px2 - px1, min=0.0) * torch.clamp(py2 - py1, min=0.0)
    area_t = torch.clamp(tx2 - tx1, min=0.0) * torch.clamp(ty2 - ty1, min=0.0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    cx1 = torch.min(px1, tx1)
    cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2)
    cy2 = torch.max(py2, ty2)
    area_c = torch.clamp(cx2 - cx1, min=0.0) * torch.clamp(cy2 - cy1, min=0.0) + eps

    giou = iou - (area_c - union) / area_c
    return (1.0 - giou).mean()


class DetectionLoss(nn.Module):
    """Computes balanced multi-task detection loss with Focal OHEM objectness and GIoU."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 15),
        input_size: Tuple[int, int] = (640, 480),
        num_classes: int = 6,
        lambda_obj: float = 1.5,
        lambda_box: float = 5.0,
        lambda_cls: float = 1.0,
        neg_obj_weight: float = 0.25,
        focal_gamma: float = 2.0,
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
        self.focal_gamma = focal_gamma

    def build_targets(
        self,
        boxes_batch: List[torch.Tensor],
        labels_batch: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes variable ground-truth bounding boxes into fixed spatial grid targets.

        Returns:
            obj_target: (B, 15, 20) float tensor (1 positive cell per object)
            box_target: (B, 15, 20, 4) encoded (tx, ty, tw, th)
            cls_target: (B, 15, 20) long class index
            pos_mask: (B, 15, 20) bool positive cells
            xyxy_target: (B, 15, 20, 4) exact pixel coordinates [xmin, ymin, xmax, ymax]
        """
        B = len(boxes_batch)
        obj_target = torch.zeros((B, self.grid_h, self.grid_w), dtype=torch.float32, device=device)
        box_target = torch.zeros((B, self.grid_h, self.grid_w, 4), dtype=torch.float32, device=device)
        cls_target = torch.zeros((B, self.grid_h, self.grid_w), dtype=torch.long, device=device)
        pos_mask = torch.zeros((B, self.grid_h, self.grid_w), dtype=torch.bool, device=device)
        xyxy_target = torch.zeros((B, self.grid_h, self.grid_w, 4), dtype=torch.float32, device=device)

        for b in range(B):
            boxes = boxes_batch[b]
            labels = labels_batch[b]
            if len(boxes) == 0:
                continue

            order = sorted(
                range(len(boxes)),
                key=lambda idx: max(1.0, (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])).item(),
                reverse=True,
            )
            for obj_idx in order:
                xmin, ymin, xmax, ymax = boxes[obj_idx].tolist()
                cid = int(labels[obj_idx].item())

                xc = (xmin + xmax) / 2.0
                yc = (ymin + ymax) / 2.0
                w = max(1.0, xmax - xmin)
                h = max(1.0, ymax - ymin)

                j = int(min(self.grid_w - 1, max(0, math.floor(xc / self.cell_w))))
                i = int(min(self.grid_h - 1, max(0, math.floor(yc / self.cell_h))))

                tx = max(0.01, min(0.99, (xc / self.cell_w) - j))
                ty = max(0.01, min(0.99, (yc / self.cell_h) - i))
                tw = max(-3.0, min(3.5, math.log(max(0.1, w / self.cell_w))))
                th = max(-3.0, min(3.5, math.log(max(0.1, h / self.cell_h))))

                obj_target[b, i, j] = 1.0
                box_target[b, i, j] = torch.tensor([tx, ty, tw, th], dtype=torch.float32, device=device)
                xyxy_target[b, i, j] = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32, device=device)
                cls_target[b, i, j] = cid
                pos_mask[b, i, j] = True

        return obj_target, box_target, cls_target, pos_mask, xyxy_target

    def forward(
        self,
        grid_pred: torch.Tensor,
        boxes_batch: List[torch.Tensor],
        labels_batch: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Computes Focal+OHEM objectness loss, SmoothL1+GIoU box loss, and label-smoothed CE."""
        device = grid_pred.device
        B, H, W, _ = grid_pred.shape
        obj_target, box_target, cls_target, pos_mask, xyxy_target = self.build_targets(boxes_batch, labels_batch, device)

        # 1. Balanced Objectness BCE + Focal Hard-Negative Mining
        obj_logits = grid_pred[..., 0]  # (B, H, W)
        obj_prob = torch.sigmoid(obj_logits)
        bce_raw = F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction="none")

        neg_mask = ~pos_mask
        num_pos = int(pos_mask.sum().item())
        loss_obj_pos = bce_raw[pos_mask].mean() if num_pos > 0 else torch.tensor(0.0, device=device)

        if neg_mask.any():
            neg_probs = obj_prob[neg_mask]
            neg_bce = bce_raw[neg_mask]
            focal_neg = (neg_probs.pow(self.focal_gamma)) * neg_bce
            k_hard = min(focal_neg.numel(), max(16 * B, 6 * num_pos))
            hard_neg_focal, _ = torch.topk(focal_neg, k=k_hard, largest=True, sorted=False)
            loss_obj_neg = (0.35 * neg_bce.mean()) + (1.0 * hard_neg_focal.mean())
        else:
            loss_obj_neg = torch.tensor(0.0, device=device)

        loss_obj = loss_obj_pos + (0.6 * loss_obj_neg)

        # 2. Box Regression Loss (Smooth L1 + Pixel GIoU on positive cells)
        if num_pos > 0:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )
            gy_exp = grid_y.unsqueeze(0).expand(B, H, W)[pos_mask]
            gx_exp = grid_x.unsqueeze(0).expand(B, H, W)[pos_mask]

            pred_boxes_pos = grid_pred[pos_mask][:, 1:5]
            pred_tx = torch.sigmoid(pred_boxes_pos[:, 0])
            pred_ty = torch.sigmoid(pred_boxes_pos[:, 1])
            pred_tw = torch.clamp(pred_boxes_pos[:, 2], -3.0, 3.5)
            pred_th = torch.clamp(pred_boxes_pos[:, 3], -3.0, 3.5)

            pred_box_activated = torch.stack([pred_tx, pred_ty, pred_tw, pred_th], dim=-1)
            target_boxes_pos = box_target[pos_mask]
            loss_l1 = F.smooth_l1_loss(pred_box_activated, target_boxes_pos, reduction="mean")

            # Decode pixel coords cleanly in differentiable PyTorch
            u_c = (gx_exp + pred_tx) * self.cell_w
            v_c = (gy_exp + pred_ty) * self.cell_h
            bw = torch.exp(pred_tw) * self.cell_w
            bh = torch.exp(pred_th) * self.cell_h
            pred_xyxy = torch.stack(
                [u_c - 0.5 * bw, v_c - 0.5 * bh, u_c + 0.5 * bw, v_c + 0.5 * bh],
                dim=-1,
            )
            loss_giou = _bbox_giou_loss(pred_xyxy, xyxy_target[pos_mask])
            loss_box = (0.5 * loss_l1) + (0.5 * loss_giou)

            # 3. Label-smoothed Cross Entropy
            pred_cls_logits = grid_pred[pos_mask][:, 5:5 + self.num_classes]
            target_cls_pos = cls_target[pos_mask]
            loss_cls = F.cross_entropy(pred_cls_logits, target_cls_pos, reduction="mean", label_smoothing=0.05)
        else:
            loss_box = torch.tensor(0.0, device=device, requires_grad=True)
            loss_cls = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = (self.lambda_obj * loss_obj) + (self.lambda_box * loss_box) + (self.lambda_cls * loss_cls)
        return {
            "total_loss": total_loss,
            "loss_obj": loss_obj,
            "loss_box": loss_box,
            "loss_cls": loss_cls,
            "num_pos": float(num_pos),
        }
