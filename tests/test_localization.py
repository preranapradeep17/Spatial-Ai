"""Unit tests for 3D localization accuracy against real SUN RGB-D physical ground truth."""
import pytest
from backend.spatial.localization import benchmark_localization_on_dataset

def test_real_world_3d_localization_accuracy():
    """Verify calibrated depth back-projection achieves sub-meter error on physical benchmark."""
    stats = benchmark_localization_on_dataset("data/sunrgbd")
    assert stats["num_evaluated"] >= 10, "Should evaluate at least 10 physical objects"
    assert stats["mean_error_meters"] < 1.0, f"Expected mean error < 1.0m, got {stats['mean_error_meters']}m"
    assert stats["median_error_meters"] < 0.8, f"Expected median error < 0.8m, got {stats['median_error_meters']}m"
    assert "door" in stats["per_class_mean_error_meters"]
    assert "table" in stats["per_class_mean_error_meters"]
