"""Test hardware detection, PyTorch acceleration, and core dependencies."""
import pytest
import sys
import torch
import numpy as np
from backend.perception.device import get_optimal_device, get_device_info

def test_python_version():
    """Verify Python runtime version >= 3.9."""
    assert sys.version_info >= (3, 9), f"Python 3.9+ required, got {sys.version}"

def test_pytorch_tensor_operations():
    """Verify PyTorch tensor creation, math, and autograd functionality."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2.0 + 1.0
    loss = y.sum()
    loss.backward()

    assert torch.allclose(x.grad, torch.tensor([2.0, 2.0, 2.0]))

def test_device_detection():
    """Verify optimal device detection identifies MPS on Apple Silicon or CPU fallback."""
    dev = get_optimal_device("auto")
    info = get_device_info()

    assert dev is not None
    assert isinstance(info, dict)
    assert "selected_device" in info

    # Test tensor allocation on detected device
    t = torch.ones((3, 3), device=dev)
    res = (t @ t).cpu().numpy()
    assert np.allclose(res, np.ones((3, 3)) * 3.0)
