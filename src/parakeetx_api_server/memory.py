from __future__ import annotations

import ctypes
import gc
import logging
import sys

logger = logging.getLogger(__name__)


def release_memory_to_os(*, clear_cuda: bool = False) -> None:
    if clear_cuda:
        _clear_cuda_cache()

    gc.collect()
    if not sys.platform.startswith("linux"):
        return

    try:
        libc = ctypes.CDLL("libc.so.6")
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
    except Exception as exc:
        logger.debug("Unable to trim libc allocator arenas: %s", exc)


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("Unable to clear CUDA cache: %s", exc)
