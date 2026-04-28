from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_PARAKEET_INTEGRATION") != "1",
    reason="Set RUN_PARAKEET_INTEGRATION=1 to run live Parakeet integration tests.",
)
def test_parakeet_integration_guard_placeholder():
    assert True


@pytest.mark.skipif(
    os.getenv("RUN_DIARIZATION_INTEGRATION") != "1",
    reason="Set RUN_DIARIZATION_INTEGRATION=1 to run live diarization integration tests.",
)
def test_diarization_integration_guard_placeholder():
    assert True
