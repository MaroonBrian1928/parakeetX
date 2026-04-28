from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..config import DiarizationSettings


class DiarizationModelManager:
    def __init__(self, settings: DiarizationSettings, hf_token: str | None) -> None:
        self._settings = settings
        self._hf_token = hf_token
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    @property
    def configured_model_name(self) -> str:
        return self._settings.model_name

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._pipeline is not None,
            "model_name": self._settings.model_name,
            "device": self._settings.device,
            "requires_hf_token": True,
        }

    def load_model(self) -> dict[str, Any]:
        if not self._hf_token:
            raise RuntimeError("HF_TOKEN is required to load diarization model")

        with self._lock:
            if self._pipeline is not None:
                return self.status()

            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "pyannote-audio is not installed. Install with `uv sync --extra diarization`."
                ) from exc

            try:
                pipeline = Pipeline.from_pretrained(
                    self._settings.model_name,
                    token=self._hf_token,
                )
            except TypeError:
                # Backward compatibility with older pyannote versions.
                pipeline = Pipeline.from_pretrained(
                    self._settings.model_name,
                    use_auth_token=self._hf_token,
                )

            if self._settings.device.startswith("cuda"):
                try:
                    import torch

                    pipeline.to(torch.device(self._settings.device))
                except Exception:
                    pass

            self._pipeline = pipeline

        return self.status()

    def unload_model(self) -> dict[str, Any]:
        with self._lock:
            self._pipeline = None
            if self._settings.device.startswith("cuda"):
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return self.status()

    def diarize(
        self,
        audio_path: Path,
        *,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        num_speakers: int | None = None,
    ) -> list[dict[str, Any]]:
        pipeline = self._pipeline
        if pipeline is None:
            self.load_model()
            pipeline = self._pipeline

        if pipeline is None:
            raise RuntimeError("Diarization model failed to load")

        kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        mono = np.asarray(waveform, dtype=np.float32).mean(axis=1)
        import torch

        audio_input = {
            "waveform": torch.from_numpy(mono).unsqueeze(0),
            "sample_rate": int(sample_rate),
        }

        annotation = pipeline(audio_input, **kwargs)

        iterable_annotation = annotation
        if not hasattr(iterable_annotation, "itertracks"):
            wrapped = getattr(annotation, "speaker_diarization", None)
            if wrapped is not None and hasattr(wrapped, "itertracks"):
                iterable_annotation = wrapped

        if not hasattr(iterable_annotation, "itertracks"):
            raise RuntimeError(
                f"Unsupported diarization output type: {type(annotation).__name__}"
            )

        diarization_segments: list[dict[str, Any]] = []
        for segment, _, speaker in iterable_annotation.itertracks(yield_label=True):
            diarization_segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": str(speaker),
                }
            )

        return diarization_segments
