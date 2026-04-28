from __future__ import annotations

import logging

from parakeetx_api_server.log_filters import install_noisy_dependency_log_filters


def test_suppresses_specific_nemo_pretokenize_warning(caplog) -> None:
    logger = logging.getLogger("nemo_logger")
    logger.setLevel(logging.WARNING)
    logger.propagate = True

    install_noisy_dependency_log_filters()

    noisy_message = (
        "You are using a non-tarred dataset and requested tokenization during "
        "data sampling (pretokenize=True)."
    )
    logger.warning(noisy_message)
    logger.warning("A different NeMo warning")

    assert noisy_message not in caplog.text
    assert "A different NeMo warning" in caplog.text


def test_suppresses_lhotse_ignored_key_warning(caplog) -> None:
    logger = logging.getLogger("nemo_logger")
    logger.setLevel(logging.WARNING)
    logger.propagate = True

    install_noisy_dependency_log_filters()

    noisy_message = (
        "The following configuration keys are ignored by Lhotse dataloader: "
        "use_start_end_token"
    )
    logger.warning(noisy_message)
    logger.warning("Another warning survives")

    assert noisy_message not in caplog.text
    assert "Another warning survives" in caplog.text
