"""Unit tests for Custom CNN Detector forward pass, loss computation, and NMS decoding."""
import pytest
import torch
from backend.perception.model import SpatialDetector, NUM_CLASSES
from backend.perception.losses import DetectionLoss

def test_detector_forward_pass_shape():
    """Verify forward pass transforms (B, 3, 480, 640) into spatial grid (B, 15, 20, 11)."""
    model = SpatialDetector(num_classes=NUM_CLASSES)
    x = torch.randn(2, 3, 480, 640)

    out = model(x)
    assert out.shape == (2, 15, 20, 11), f"Expected (2, 15, 20, 11), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"

def test_loss_computation_and_backward():
    """Verify DetectionLoss computes finite loss and backward pass computes non-zero gradients."""
    model = SpatialDetector(num_classes=NUM_CLASSES)
    criterion = DetectionLoss(num_classes=NUM_CLASSES)

    x = torch.randn(2, 3, 480, 640)
    preds = model(x)

    boxes = [
        torch.tensor([[100.0, 150.0, 300.0, 400.0]], dtype=torch.float32),
        torch.tensor([[50.0, 80.0, 220.0, 310.0], [400.0, 200.0, 580.0, 450.0]], dtype=torch.float32)
    ]
    labels = [
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.long)
    ]

    loss_dict = criterion(preds, boxes, labels)
    loss = loss_dict["total_loss"]

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0

    loss.backward()

    # Verify gradients flowed to backbone layers
    conv1_grad = model.layer1.conv.weight.grad
    assert conv1_grad is not None
    assert torch.abs(conv1_grad).sum() > 0.0, "Zero gradient in initial convolution layer"

def test_prediction_decoding():
    """Verify decode_predictions produces valid bounding box coordinates and scores."""
    model = SpatialDetector(num_classes=NUM_CLASSES)
    grid_pred = torch.zeros(1, 15, 20, 11)

    # Inject high-confidence detection in cell (7, 10)
    # obj_logit = 5.0 (sigmoid ~ 0.993)
    grid_pred[0, 7, 10, 0] = 5.0
    # Center offsets: 0.5, 0.5 (sigmoid = 0.5)
    grid_pred[0, 7, 10, 1] = 0.0
    grid_pred[0, 7, 10, 2] = 0.0
    # Scale: log(2.0)
    grid_pred[0, 7, 10, 3] = 0.693
    grid_pred[0, 7, 10, 4] = 0.693
    # Class 1 (chair): logit = 5.0
    grid_pred[0, 7, 10, 5 + 1] = 5.0

    detections = model.decode_predictions(grid_pred, conf_threshold=0.5)[0]
    assert len(detections) == 1
    det = detections[0]

    assert det["class_id"] == 1
    assert det["class_name"] == "chair"
    assert det["confidence"] > 0.90
    assert 0 <= det["xmin"] < det["xmax"] <= 640
    assert 0 <= det["ymin"] < det["ymax"] <= 480
