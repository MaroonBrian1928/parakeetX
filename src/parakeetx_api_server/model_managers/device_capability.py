from __future__ import annotations

# Volta (sm_70) is the floor for fast FP16 math and for Triton/torch.compile
# GPU kernels; Ampere (sm_80) is the floor for bfloat16 and flash attention.
MIN_FP16_CAPABILITY = (7, 0)
MIN_TORCH_COMPILE_CAPABILITY = (7, 0)
MIN_BF16_CAPABILITY = (8, 0)
MIN_FLASH_ATTENTION_CAPABILITY = (8, 0)


def cuda_compute_capability(device: str) -> tuple[int, int] | None:
    """Return the (major, minor) compute capability for a CUDA device string,
    or None when the device is not CUDA or the capability cannot be queried."""
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return tuple(torch.cuda.get_device_capability(torch.device(device)))
    except Exception:
        return None


def meets_capability(device: str, minimum: tuple[int, int]) -> bool:
    """True when the device meets the minimum compute capability.

    Unknown capability (CPU device, CUDA unavailable, query failure) returns
    True so configured behavior is only downgraded on a confirmed old GPU.
    """
    capability = cuda_compute_capability(device)
    return capability is None or capability >= minimum
