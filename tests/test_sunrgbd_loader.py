"""Unit tests for SUN RGB-D Real-World Dataset Loader."""
import pytest
import os
import torch
from backend.perception.sun_rgbd_loader import SUNRGBDDataset, get_sunrgbd_dataloaders

def test_sunrgbd_dataset_splits():
    """Verify official train and test splits load separate samples with no data leakage."""
    train_ds = SUNRGBDDataset(root_dir="data/sunrgbd", split="train")
    test_ds = SUNRGBDDataset(root_dir="data/sunrgbd", split="test")

    assert len(train_ds) > 0, "Train dataset should contain samples"
    assert len(test_ds) > 0, "Test dataset should contain samples"

    train_ids = set(train_ds.samples[i]["frame_id"] for i in range(len(train_ds)))
    test_ids = set(test_ds.samples[i]["frame_id"] for i in range(len(test_ds)))

    # Zero data leakage
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Data leakage detected between train and test splits: {overlap}"

def test_sunrgbd_item_shapes_and_types():
    """Verify tensors returned by __getitem__ adhere to strict dimensions and value ranges."""
    ds = SUNRGBDDataset(root_dir="data/sunrgbd", split="train", target_size=(640, 480))
    sample = ds[0]

    assert "rgb" in sample
    assert "depth" in sample
    assert "boxes_2d" in sample
    assert "labels" in sample
    assert "intrinsics" in sample

    rgb = sample["rgb"]
    depth = sample["depth"]
    boxes = sample["boxes_2d"]
    labels = sample["labels"]
    intrinsics = sample["intrinsics"]

    # RGB: (3, 480, 640), float32, range [0, 1]
    assert rgb.shape == (3, 480, 640)
    assert rgb.dtype == torch.float32
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0

    # Depth: (1, 480, 640), float32, positive metric depth
    assert depth.shape == (1, 480, 640)
    assert depth.dtype == torch.float32
    valid_depth = depth[depth > 0.1]
    assert len(valid_depth) > 0
    assert valid_depth.min() >= 0.1 and valid_depth.max() <= 8.0

    # Intrinsics: (4,) -> fx, fy, cx, cy
    assert intrinsics.shape == (4,)
    assert intrinsics[0] > 0 and intrinsics[1] > 0

    # Bboxes: (N, 4) within [0, 640] and [0, 480]
    if len(boxes) > 0:
        assert boxes.shape[1] == 4
        assert (boxes[:, 0] >= 0).all()
        assert (boxes[:, 1] >= 0).all()
        assert (boxes[:, 2] <= 640).all()
        assert (boxes[:, 3] <= 480).all()
        assert (boxes[:, 0] < boxes[:, 2]).all()
        assert (boxes[:, 1] < boxes[:, 3]).all()

        # Labels: (N,) between 0 and 5
        assert (labels >= 0).all() and (labels < 6).all()

def test_sunrgbd_dataloader_batching():
    """Verify PyTorch DataLoader collates batches with variable object counts."""
    train_loader, test_loader = get_sunrgbd_dataloaders(root_dir="data/sunrgbd", batch_size=2)

    batch = next(iter(train_loader))
    assert batch["rgb"].shape == (2, 3, 480, 640)
    assert batch["depth"].shape == (2, 1, 480, 640)
    assert batch["intrinsics"].shape == (2, 4)
    assert len(batch["boxes_2d"]) == 2
    assert len(batch["labels"]) == 2
