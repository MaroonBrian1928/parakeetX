from __future__ import annotations

from contextlib import contextmanager
import logging
import sys
from collections.abc import Iterator
from typing import TextIO
import warnings

_SUPPRESSED_MESSAGE_FRAGMENTS = (
    "You are using a non-tarred dataset and requested tokenization during data sampling",
    "The following configuration keys are ignored by Lhotse dataloader: use_start_end_token",
    "If you intend to do training, please call the ModelPT.setup_training_data()",
    "If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data()",
    "If you intend to do validation, please call the ModelPT.setup_validation_data()",
    "Timestamps requested, setting decoding timestamps to True",
    "Using RNNT Loss : tdt",
    "Loss tdt_kwargs:",
    "Megatron num_microbatches_calculator not found, using Apex version.",
    "TensorFloat-32 (TF32) has been disabled",
    "OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter",
    "No exporters were provided. This means that no telemetry data will be collected.",
    "SyntaxWarning: invalid escape sequence",
    "escape sequence '\\('",
    "m = re.match('([su]([0-9]{1,2})p?) \\(([0-9]{1,2}) bit\\)$', token)",
    "m2 = re.match('([su]([0-9]{1,2})p?)( \\(default\\))?$', token)",
    "elif re.match('(flt)p?( \\(default\\))?$', token):",
    "elif re.match('(dbl)p?( \\(default\\))?$', token):",
)


class _MessageFragmentFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in _SUPPRESSED_MESSAGE_FRAGMENTS)


_filter = _MessageFragmentFilter()


class _SuppressingTextStream:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        if any(fragment in text for fragment in _SUPPRESSED_MESSAGE_FRAGMENTS):
            return len(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


@contextmanager
def suppress_noisy_dependency_streams() -> Iterator[None]:
    """Drop known direct stdout/stderr dependency noise during import-time setup."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _SuppressingTextStream(original_stdout)  # type: ignore[assignment]
    sys.stderr = _SuppressingTextStream(original_stderr)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def install_noisy_dependency_log_filters() -> None:
    """Hide known noisy dependency warnings without muting unrelated warnings."""
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*invalid escape sequence '\\\('.*",
        category=SyntaxWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*escape sequence '\\\('.*",
        category=SyntaxWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=SyntaxWarning,
        module=r"pydub\.utils",
    )

    candidate_loggers = (
        logging.getLogger(),
        logging.getLogger("nemo"),
        logging.getLogger("nemo_logger"),
        logging.getLogger("NeMo"),
        logging.getLogger("pyannote"),
        logging.getLogger("uvicorn.error"),
    )

    for logger in candidate_loggers:
        _add_filter_once(logger.filters)
        for handler in logger.handlers:
            _add_filter_once(handler.filters)


def _add_filter_once(filters: list[logging.Filter]) -> None:
    if not any(isinstance(existing, _MessageFragmentFilter) for existing in filters):
        filters.append(_filter)
