from __future__ import annotations

import logging
import warnings

from parakeetx_api_server.log_filters import (
    install_noisy_dependency_log_filters,
    suppress_noisy_dependency_streams,
)


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


def test_suppresses_nemo_rnnt_timestamp_and_loss_noise(caplog) -> None:
    logger = logging.getLogger("nemo_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = True

    install_noisy_dependency_log_filters()

    timestamp_message = (
        "Timestamps requested, setting decoding timestamps to True. Capture them in "
        "Hypothesis object, with output[0][idx].timestep['word'/'segment'/'char']"
    )
    loss_message = "Using RNNT Loss : tdt"
    kwargs_message = (
        "Loss tdt_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0, "
        "'durations': [0, 1, 2, 3, 4], 'sigma': 0.02, 'omega': 0.1}"
    )

    logger.info(timestamp_message)
    logger.info(loss_message)
    logger.info(kwargs_message)
    logger.info("A useful NeMo info line")

    assert timestamp_message not in caplog.text
    assert loss_message not in caplog.text
    assert kwargs_message not in caplog.text
    assert "A useful NeMo info line" in caplog.text


def test_suppresses_nemo_startup_dependency_noise(caplog) -> None:
    logger = logging.getLogger("nemo_logger")
    logger.setLevel(logging.WARNING)
    logger.propagate = True

    install_noisy_dependency_log_filters()

    noisy_messages = (
        "Megatron num_microbatches_calculator not found, using Apex version.",
        (
            "OneLogger: Setting error_handling_strategy to "
            "DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with "
            "OneLogger disabled. To override: explicitly set "
            "error_handling_strategy parameter."
        ),
        "No exporters were provided. This means that no telemetry data will be collected.",
        (
            "/app/.venv/lib/python3.12/site-packages/pydub/utils.py:300: "
            "SyntaxWarning: invalid escape sequence '\\('"
        ),
        "escape sequence '\\('",
        "m = re.match('([su]([0-9]{1,2})p?) \\(([0-9]{1,2}) bit\\)$', token)",
        "m2 = re.match('([su]([0-9]{1,2})p?)( \\(default\\))?$', token)",
        "elif re.match('(flt)p?( \\(default\\))?$', token):",
        "elif re.match('(dbl)p?( \\(default\\))?$', token):",
    )

    for message in noisy_messages:
        logger.warning(message)
    logger.warning("A startup warning worth seeing")

    for message in noisy_messages:
        assert message not in caplog.text
    assert "A startup warning worth seeing" in caplog.text


def test_suppresses_pydub_invalid_escape_warnings() -> None:
    install_noisy_dependency_log_filters()

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("invalid escape sequence '\\('", SyntaxWarning)
        warnings.warn("escape sequence '\\('", SyntaxWarning)
        warnings.warn("Different syntax warning", SyntaxWarning)

    messages = [str(warning.message) for warning in caught]
    assert "invalid escape sequence '\\('" not in messages
    assert "escape sequence '\\('" not in messages
    assert "Different syntax warning" in messages


def test_suppresses_direct_dependency_stream_noise(capsys) -> None:
    with suppress_noisy_dependency_streams():
        print(
            "OneLogger: Setting error_handling_strategy to "
            "DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with "
            "OneLogger disabled. To override: explicitly set "
            "error_handling_strategy parameter."
        )
        print("No exporters were provided. This means that no telemetry data will be collected.")
        print("A useful direct stream line")

    captured = capsys.readouterr()
    assert "OneLogger: Setting error_handling_strategy" not in captured.out
    assert "No exporters were provided" not in captured.out
    assert "A useful direct stream line" in captured.out


def test_suppresses_pyannote_tf32_reproducibility_warning() -> None:
    install_noisy_dependency_log_filters()

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn(
            "TensorFloat-32 (TF32) has been disabled as it might lead to "
            "reproducibility issues and lower accuracy.",
            UserWarning,
        )
        warnings.warn("Different dependency warning", UserWarning)

    messages = [str(warning.message) for warning in caught]
    assert not any("TensorFloat-32 (TF32) has been disabled" in message for message in messages)
    assert "Different dependency warning" in messages
