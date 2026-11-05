"""Unit and smoke tests for synthetic dataset generation and validation."""
import pytest
import os
import shutil
import numpy as np
from scripts.generate_synthetic_dataset import generate_sample, generate_dataset
from scripts.validate_dataset import validate_dataset

@pytest.fixture
def temp_dataset_dir(tmp_path):
    test_dir = str(tmp_path / "test_dataset")
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)

def test_generate_single_sample(temp_dataset_dir):
    """Verify single frame generation creates valid RGB, depth, and JSON."""
    os.makedirs(os.path.join(temp_dataset_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_dir, "annotations"), exist_ok=True)

    sample = generate_sample(0, temp_dataset_dir, width=640, height=480)
    assert sample["frame_id"] == "frame_00000"
    assert len(sample["objects"]) >= 1

    rgb_file = os.path.join(temp_dataset_dir, sample["image_filename"])
    depth_file = os.path.join(temp_dataset_dir, sample["depth_filename"])

    assert os.path.isfile(rgb_file)
    assert os.path.isfile(depth_file)

    depth_array = np.load(depth_file)
    assert depth_array.shape == (480, 640)
    assert (depth_array > 0).all()

def test_dataset_validator_clean(temp_dataset_dir):
    """Verify validate_dataset passes with zero errors on freshly generated dataset."""
    generate_dataset(temp_dataset_dir, num_samples=3, seed=123)
    report = validate_dataset(temp_dataset_dir, save_visual_debug=False)

    assert report["total_samples"] == 3
    assert report["passed_samples"] == 3
    assert len(report["errors"]) == 0
    assert report["total_objects"] > 0
