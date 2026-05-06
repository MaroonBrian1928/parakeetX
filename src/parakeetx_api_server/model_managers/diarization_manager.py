from __future__ import annotations

import threading
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..config import DiarizationSettings
from ..memory import release_memory_to_os
from .idle_eviction import IdleModelEvictor
from .model_worker import ModelWorkerClient


class DiarizationModelManager:
    def __init__(
        self,
        settings: DiarizationSettings,
        hf_token: str | None,
        *,
        idle_evict_minutes: float | None = None,
        worker_client: ModelWorkerClient | None = None,
    ) -> None:
        self._settings = settings
        self._hf_token = hf_token
        self._pipeline: Any | None = None
        self._worker_client = worker_client
        self._worker_loaded = False
        self._lock = threading.Lock()
        self._idle_evictor = IdleModelEvictor(
            model_label="diarization",
            idle_minutes=idle_evict_minutes,
            is_loaded=self._is_loaded,
            unload=self.unload_model,
        )

    @property
    def configured_model_name(self) -> str:
        return self._settings.model_name

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._is_loaded(),
            "model_name": self._settings.model_name,
            "device": self._settings.device,
            "idle_evict_minutes": self._idle_evictor.idle_minutes,
            "requires_hf_token": True,
        }

    def load_model(self) -> dict[str, Any]:
        if self._worker_client is not None:
            status = self._worker_client.load_diarization()
            self._worker_loaded = True
            self._idle_evictor.note_loaded()
            status["idle_evict_minutes"] = self._idle_evictor.idle_minutes
            return status

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

        self._idle_evictor.note_loaded()
        return self.status()

    def unload_model(self) -> dict[str, Any]:
        if self._worker_client is not None:
            self._worker_loaded = False
            self._idle_evictor.cancel()
            status = self._worker_client.unload_diarization()
            release_memory_to_os()
            status["idle_evict_minutes"] = self._idle_evictor.idle_minutes
            return status

        with self._lock:
            self._pipeline = None
            self._idle_evictor.cancel()
            if self._settings.device.startswith("cuda"):
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
        release_memory_to_os()
        return self.status()

    def diarize(
        self,
        audio_path: Path,
        *,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        num_speakers: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._worker_client is not None:
            with self._idle_evictor.use():
                result = self._worker_client.diarize(
                    audio_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    num_speakers=num_speakers,
                )
                self._worker_loaded = True
                return result

        with self._idle_evictor.use():
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

            annotation = self._run_pipeline(pipeline, audio_path, kwargs)

            iterable_annotation = annotation
            if not hasattr(iterable_annotation, "itertracks"):
                wrapped = getattr(annotation, "speaker_diarization", None)
                if wrapped is not None and hasattr(wrapped, "itertracks"):
                    iterable_annotation = wrapped

            if not hasattr(iterable_annotation, "itertracks"):
                raise RuntimeError(
                    f"Unsupported diarization output type: {type(annotation).__name__}"
                )

            try:
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
            finally:
                del annotation
                del iterable_annotation
                release_memory_to_os(clear_cuda=self._settings.device.startswith("cuda"))

    def diarize_regions(
        self,
        audio_path: Path,
        regions: list[dict[str, Any]],
        *,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        num_speakers: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._worker_client is not None:
            with self._idle_evictor.use():
                result = self._worker_client.diarize_regions(
                    audio_path,
                    regions,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    num_speakers=num_speakers,
                )
                self._worker_loaded = True
                return result

        return self._diarize_regions_local(
            audio_path,
            regions,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
        )

    def _diarize_regions_local(
        self,
        audio_path: Path,
        regions: list[dict[str, Any]],
        *,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        num_speakers: int | None = None,
    ) -> list[dict[str, Any]]:
        speech_intervals = _speech_intervals_from_vad_regions(regions)
        if not speech_intervals:
            return []

        info = sf.info(str(audio_path))
        sample_rate = int(info.samplerate)
        total_frames = int(info.frames)
        if _single_interval_covers_whole_file(
            speech_intervals,
            total_frames=total_frames,
            sample_rate=sample_rate,
        ):
            return self.diarize(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
            )

        compact_chunks: list[np.ndarray] = []
        mapping: list[dict[str, float]] = []
        compact_cursor = 0.0

        for start_seconds, end_seconds in speech_intervals:
            start_frame = min(total_frames, max(0, int(start_seconds * sample_rate)))
            end_frame = min(total_frames, max(start_frame, int(end_seconds * sample_rate)))
            if end_frame <= start_frame:
                continue

            audio_chunk, _ = sf.read(
                str(audio_path),
                start=start_frame,
                stop=end_frame,
                dtype="float32",
                always_2d=True,
            )
            if audio_chunk.size == 0:
                continue

            duration_seconds = float(end_frame - start_frame) / float(sample_rate)
            original_start = float(start_frame) / float(sample_rate)
            compact_chunks.append(np.asarray(audio_chunk, dtype=np.float32))
            mapping.append(
                {
                    "compact_start": compact_cursor,
                    "compact_end": compact_cursor + duration_seconds,
                    "original_start": original_start,
                }
            )
            compact_cursor += duration_seconds

        if not compact_chunks:
            return []

        with tempfile.TemporaryDirectory(prefix="parakeetx-diarization-vad-", dir=str(audio_path.parent)) as tmpdir:
            compact_path = Path(tmpdir) / "speech_only.wav"
            compact_audio = np.concatenate(compact_chunks, axis=0)
            sf.write(
                str(compact_path),
                compact_audio,
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
            compact_segments = self.diarize(
                compact_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
            )

        return _map_compact_diarization_to_original(compact_segments, mapping)

    def _is_loaded(self) -> bool:
        if self._worker_client is not None:
            return self._worker_loaded
        return self._pipeline is not None

    def _run_pipeline(
        self,
        pipeline: Any,
        audio_path: Path,
        kwargs: dict[str, Any],
    ) -> Any:
        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        mono = np.asarray(waveform, dtype=np.float32).mean(axis=1)
        import torch

        audio_input = {
            "waveform": torch.from_numpy(mono).unsqueeze(0),
            "sample_rate": int(sample_rate),
        }
        try:
            return pipeline(audio_input, **kwargs)
        finally:
            del audio_input
            del mono
            del waveform
            release_memory_to_os(clear_cuda=self._settings.device.startswith("cuda"))


def _speech_intervals_from_vad_regions(
    regions: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for region in regions:
        child_segments = region.get("segments")
        if child_segments:
            for child in child_segments:
                start, end = child
                if float(end) > float(start):
                    intervals.append((float(start), float(end)))
            continue

        start = float(region.get("start", 0.0))
        end = float(region.get("end", start))
        if end > start:
            intervals.append((start, end))

    return sorted(intervals, key=lambda item: item[0])


def _single_interval_covers_whole_file(
    intervals: list[tuple[float, float]],
    *,
    total_frames: int,
    sample_rate: int,
) -> bool:
    if len(intervals) != 1:
        return False

    start_seconds, end_seconds = intervals[0]
    start_frame = max(0, int(start_seconds * sample_rate))
    end_frame = int(end_seconds * sample_rate)
    frame_tolerance = max(1, int(sample_rate * 0.01))
    return start_frame <= frame_tolerance and end_frame >= total_frames - frame_tolerance


def _map_compact_diarization_to_original(
    compact_segments: list[dict[str, Any]],
    mapping: list[dict[str, float]],
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for segment in compact_segments:
        compact_start = float(segment.get("start", 0.0))
        compact_end = float(segment.get("end", compact_start))
        if compact_end <= compact_start:
            continue

        for interval in mapping:
            overlap_start = max(compact_start, interval["compact_start"])
            overlap_end = min(compact_end, interval["compact_end"])
            if overlap_end <= overlap_start:
                continue

            original_start = interval["original_start"] + (
                overlap_start - interval["compact_start"]
            )
            original_end = interval["original_start"] + (
                overlap_end - interval["compact_start"]
            )
            remapped.append(
                {
                    "start": original_start,
                    "end": original_end,
                    "speaker": str(segment["speaker"]),
                }
            )

    return remapped
