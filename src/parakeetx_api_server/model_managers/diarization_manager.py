from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..config import DiarizationSettings
from ..memory import release_memory_to_os
from .device_capability import MIN_FP16_CAPABILITY, cuda_compute_capability, meets_capability
from .idle_eviction import IdleModelEvictor
from .model_worker import ModelWorkerClient

logger = logging.getLogger(__name__)


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

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._is_loaded(),
            "backend": self._settings.backend,
            "model_name": self._settings.model_name,
            "device": self._settings.device,
            "segmentation_batch_size": self._settings.segmentation_batch_size,
            "embedding_batch_size": self._settings.embedding_batch_size,
            "idle_evict_minutes": self._idle_evictor.idle_minutes,
            "requires_hf_token": self._settings.backend == "pyannote",
        }

    def load_model(self) -> dict[str, Any]:
        if self._worker_client is not None:
            status = self._worker_client.load_diarization()
            self._worker_loaded = True
            self._idle_evictor.note_loaded()
            status["idle_evict_minutes"] = self._idle_evictor.idle_minutes
            return status

        if self._settings.backend == "speakrs":
            # speakrs runs as a subprocess per request; there is nothing to keep resident.
            if not self._speakrs_binary().is_file():
                raise RuntimeError(f"speakrs-diarize not found at {self._speakrs_binary()}")
            return self.status()

        if not self._hf_token:
            raise RuntimeError("HF_TOKEN is required to load diarization model")

        with self._lock:
            if self._pipeline is not None:
                return self.status()

            load_started = time.perf_counter()
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

            self._configure_pipeline_batch_sizes(pipeline)
            if (
                self._settings.cuda_half_precision
                and self._settings.device.startswith("cuda")
            ):
                if meets_capability(self._settings.device, MIN_FP16_CAPABILITY):
                    self._apply_half_precision(pipeline)
                else:
                    logger.warning(
                        "Skipping diarization half precision: compute capability %s on %s is below %s, "
                        "where FP16 runs slower than FP32.",
                        cuda_compute_capability(self._settings.device),
                        self._settings.device,
                        MIN_FP16_CAPABILITY,
                    )
            self._pipeline = pipeline
            print(f"Model load: diarization elapsed={time.perf_counter() - load_started:.2f}s", file=sys.stderr, flush=True)

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
        release_memory_to_os(clear_cuda=self._settings.device.startswith("cuda"))
        return self.status()

    def diarize(
        self,
        audio_path: Path,
    ) -> list[dict[str, Any]]:
        if self._worker_client is not None:
            with self._idle_evictor.use():
                result = self._worker_client.diarize(audio_path)
                self._worker_loaded = True
                return result

        if self._settings.backend == "speakrs":
            return self._run_speakrs(audio_path)

        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        return self._diarize_waveform(
            waveform,
            int(sample_rate),
        )

    def diarize_regions(
        self,
        audio_path: Path,
        regions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._worker_client is not None:
            with self._idle_evictor.use():
                result = self._worker_client.diarize_regions(audio_path, regions)
                self._worker_loaded = True
                return result

        return self._diarize_regions_local(audio_path, regions)

    def _diarize_regions_local(
        self,
        audio_path: Path,
        regions: list[dict[str, Any]],
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
            return self.diarize(audio_path)

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

        compact_segments = self._diarize_waveform(
            np.concatenate(compact_chunks, axis=0),
            sample_rate,
        )
        return _map_compact_diarization_to_original(compact_segments, mapping)

    def _diarize_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> list[dict[str, Any]]:
        if self._settings.backend == "speakrs":
            with tempfile.TemporaryDirectory(prefix="parakeetx-speakrs-") as tmpdir:
                wav_path = Path(tmpdir) / "audio.wav"
                sf.write(str(wav_path), waveform, sample_rate, format="WAV", subtype="PCM_16")
                return self._run_speakrs(wav_path)

        with self._idle_evictor.use():
            pipeline = self._pipeline
            if pipeline is None:
                self.load_model()
                pipeline = self._pipeline
            if pipeline is None:
                raise RuntimeError("Diarization model failed to load")

            annotation = self._run_pipeline(pipeline, waveform, sample_rate)
            # pyannote 4 wraps the Annotation in a DiarizeOutput.
            if not hasattr(annotation, "itertracks"):
                annotation = getattr(annotation, "speaker_diarization", annotation)
            if not hasattr(annotation, "itertracks"):
                raise RuntimeError(
                    f"Unsupported diarization output type: {type(annotation).__name__}"
                )

            return [
                {"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)}
                for segment, _, speaker in annotation.itertracks(yield_label=True)
            ]

    def _speakrs_binary(self) -> Path:
        return Path(self._settings.speakrs_home) / "bin" / "speakrs-diarize"

    def _run_speakrs(
        self,
        audio_path: Path,
    ) -> list[dict[str, Any]]:
        home = Path(self._settings.speakrs_home)
        ort_lib = home / "ort" / "lib"
        mode = "cuda" if self._settings.device.startswith("cuda") else "cpu"
        env = {
            **os.environ,
            "ORT_DYLIB_PATH": str(ort_lib / "libonnxruntime.so"),
            "LD_LIBRARY_PATH": os.pathsep.join(
                filter(None, [str(ort_lib), os.environ.get("LD_LIBRARY_PATH")])
            ),
        }
        command = [str(self._speakrs_binary()), mode, str(home / "models"), str(audio_path)]

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=env,
                timeout=self._settings.speakrs_timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"speakrs-diarize not found at {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"speakrs diarization timed out after {self._settings.speakrs_timeout_seconds:.0f}s"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(f"speakrs diarization failed: {completed.stderr.strip()[-2000:]}")
        print(
            f"Diarization timing: backend=speakrs mode={mode} elapsed={time.perf_counter() - started:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        return _parse_rttm(completed.stdout)

    def _is_loaded(self) -> bool:
        if self._worker_client is not None:
            return self._worker_loaded
        return self._pipeline is not None

    def _configure_pipeline_batch_sizes(self, pipeline: Any) -> None:
        for attr, value in (
            ("segmentation_batch_size", self._settings.segmentation_batch_size),
            ("embedding_batch_size", self._settings.embedding_batch_size),
        ):
            try:
                setattr(pipeline, attr, value)
            except Exception as exc:
                logger.warning(
                    "Unable to set pyannote %s=%s; continuing with pipeline default: %s",
                    attr,
                    value,
                    exc,
                )

    def _run_pipeline(
        self,
        pipeline: Any,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> Any:
        import torch

        mono = np.asarray(waveform, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        audio_input = {
            "waveform": torch.from_numpy(mono).unsqueeze(0),
            "sample_rate": int(sample_rate),
        }
        timer = _PipelineStageTimer()
        result = pipeline(audio_input, hook=timer)
        timer.log()
        return result

    def _apply_half_precision(self, pipeline: Any) -> None:
        """Convert pyannote's embedding model to FP16 in place and wrap its forward
        to auto-cast float inputs.

        Pyannote's Inference wrapper moves inputs to the device but doesn't cast
        dtype, so a half model with float32 inputs crashes inside batch/instance
        norm. We wrap forward to cast any float-tensor inputs to the model's dtype.

        We deliberately skip the segmentation model: SincNet's instance_norm path
        is dtype-sensitive and the convert+wrap dance there has rough edges; the
        embedding pass is the dominant cost anyway.
        """
        candidate_paths = (
            ("_embedding", "model_"),
            ("_embedding", "model"),
            ("embedding_model",),
        )
        converted = 0
        for path in candidate_paths:
            obj: Any = pipeline
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is None or not hasattr(obj, "half") or not hasattr(obj, "forward"):
                continue
            try:
                obj.half()
                _wrap_forward_with_dtype_cast(obj)
                converted += 1
            except Exception as exc:
                logger.warning("FP16 conversion failed for pyannote %s: %s", ".".join(path), exc)
        logger.info("Pyannote FP16 conversion: %d submodules", converted)


class _PipelineStageTimer:
    """pyannote progress hook that logs how long each pipeline step took.

    pyannote calls the hook as each step finishes (and during segmentation and
    embedding progress), so a step's duration runs from the previous step's
    last call to its own last call. Clustering happens between "embeddings"
    and "discrete_diarization"; "finalize" covers exclusive-diarization
    reconstruction and conversion after the last hook call.
    """

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._last_call: dict[str, float] = {}

    def __call__(self, step_name: str, step_artifact: Any, **kwargs: Any) -> None:
        self._last_call[step_name] = time.perf_counter()

    def log(self) -> None:
        parts = []
        previous = self._started
        for step_name, finished in self._last_call.items():
            label = "clustering" if step_name == "discrete_diarization" else step_name
            parts.append(f"{label}={finished - previous:.2f}s")
            previous = finished
        parts.append(f"finalize={time.perf_counter() - previous:.2f}s")
        print(f"Diarization timing: {' '.join(parts)}", file=sys.stderr, flush=True)


def _wrap_forward_with_dtype_cast(model: Any) -> None:
    """Replace model.forward with a wrapper that casts float-tensor args to the
    model's parameter dtype, so callers (like pyannote's Inference) that don't
    handle dtype themselves still work after a .half() conversion."""
    import torch

    original_forward = model.forward

    def _target_dtype() -> Any:
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return None

    def _cast(value: Any, dtype: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.to(dtype)
        return value

    def _wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        dtype = _target_dtype()
        if dtype is None:
            return original_forward(*args, **kwargs)
        cast_args = tuple(_cast(a, dtype) for a in args)
        cast_kwargs = {k: _cast(v, dtype) for k, v in kwargs.items()}
        return original_forward(*cast_args, **cast_kwargs)

    model.forward = _wrapped_forward


def _parse_rttm(rttm: str) -> list[dict[str, Any]]:
    # SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    segments = []
    for line in rttm.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0] == "SPEAKER":
            start = float(fields[3])
            segments.append({"start": start, "end": start + float(fields[4]), "speaker": fields[7]})
    return segments


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
