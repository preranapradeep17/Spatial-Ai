"""Hardware device detection and selection for PyTorch.

Supports:
- Apple Silicon Metal Performance Shaders ('mps')
- NVIDIA CUDA ('cuda')
- Standard CPU fallback ('cpu')
"""
import torch

def get_optimal_device(preference: str = "auto") -> torch.device:
    """Returns the best available torch.device according to preference and hardware."""
    if preference != "auto":
        try:
            dev = torch.device(preference)
            # Simple test to confirm device is usable
            _ = torch.zeros(1, device=dev)
            return dev
        except Exception:
            pass  # Fall back to auto-detection

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def get_device_info() -> dict:
    """Returns a dictionary summary of hardware accelerator status."""
    device = get_optimal_device("auto")
    info = {
        "selected_device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
    }
    return info

if __name__ == "__main__":
    print(get_device_info())
