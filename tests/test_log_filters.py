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


def test_suppresses_nemo_training_and_validation_config_warnings(caplog) -> None:
    logger = logging.getLogger("nemo_logger")
    logger.setLevel(logging.WARNING)
    logger.propagate = True

    install_noisy_dependency_log_filters()

    training_message = (
        "If you intend to do training, please call the ModelPT.setup_training_data() "
        "method and provide a valid configuration file to setup the train data loader."
    )
    validation_message = (
        "If you intend to do validation, please call the ModelPT.setup_validation_data() "
        "or ModelPT.setup_multiple_validation_data() method and provide a valid "
        "configuration file to setup the validation data loader(s)."
    )

    logger.warning(training_message)
    logger.warning(validation_message)
    logger.warning("A real model load warning")

    assert training_message not in caplog.text
    assert validation_message not in caplog.text
    assert "A real model load warning" in caplog.text
