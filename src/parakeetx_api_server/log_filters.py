from __future__ import annotations

import logging
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
    "TensorFloat-32 (TF32) has been disabled",
)


class _MessageFragmentFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in _SUPPRESSED_MESSAGE_FRAGMENTS)


_filter = _MessageFragmentFilter()


def install_noisy_dependency_log_filters() -> None:
    """Hide known noisy dependency warnings without muting unrelated warnings."""
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
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
