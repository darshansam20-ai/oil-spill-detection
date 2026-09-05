"""
Hardware and device selection utility with automatic CUDA verification and fallback.
Handles newer architectures (e.g. sm_120 Blackwell) gracefully.
"""
import torch
from src.utils.logger import get_logger

logger = get_logger("utils.device")


def get_default_device() -> torch.device:
    """
    Determine the optimal device (CUDA or CPU) by verifying functional tensor execution.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Verify if CUDA kernel images are available for the installed compute capability
    try:
        x = torch.zeros(1, 1, 4, 4, device="cuda")
        conv = torch.nn.Conv2d(1, 1, 1).to("cuda")
        _ = conv(x)
        return torch.device("cuda")
    except Exception as e:
        logger.warning(
            f"CUDA device detected ({torch.cuda.get_device_name(0)}) but compute capability "
            f"is not compiled into current PyTorch build ({e}). Falling back to CPU."
        )
        return torch.device("cpu")
