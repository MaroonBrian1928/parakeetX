# ParakeetX API Server

Standalone FastAPI transcription service inspired by WhisperX API conventions, backed by NVIDIA Parakeet (`nvidia/parakeet-tdt-0.6b-v2`) with optional pyannote diarization.

## Features

- `POST /v1/audio/transcriptions` with OpenAI-style multipart fields.
- Native Parakeet timestamps (no Whisper forced-alignment stage).
- WhisperX-compatible word timestamp fields for downstream tools (`word_segments` and `segments[].words`).
- Optional Silero VAD (`vad_filter=true`) that transcribes speech-only chunks with WhisperX-style VAD knobs.
- Optional diarization via `pyannote/speaker-diarization-community-1` (`diarize=true`).
- Speaker labels assigned to words/segments by maximum timestamp overlap.
- `GET /health` readiness check.
- Explicit unsupported-feature behavior:
  - `/v1/audio/translations` returns `501`.
  - Non-English language, streaming, prompt biasing, temperature sampling, hotwords, and forced-alignment return `422`.

## Quickstart

1. Copy env values:

```bash
cp .env.example .env
```

2. Sync dependencies:

```bash
mise run sync
```

3. Run tests:

```bash
mise run test
```

4. Start server:

```bash
mise run serve
```

Server default: `http://0.0.0.0:7474`

If `API_KEY` is set, authenticated endpoints require a bearer token:

```bash
Authorization: Bearer $API_KEY
```

## API Notes

### `POST /v1/audio/transcriptions`

Supported multipart fields:

- `file`
- `model`
- `language`
- `response_format`
- `timestamp_granularities[]`
- `stream`
- `diarize`
- `min_speakers`
- `max_speakers`
- `num_speakers`
- `speaker_embeddings`
- `highlight_words`
- `prompt`
- `temperature`
- `hotwords`
- `forced_alignment`
- `vad_filter`
- `vad_method`
- `vad_onset`
- `vad_offset`
- `chunk_size`
- `min_speech_duration_ms`
- `min_silence_duration_ms`
- `speech_pad_ms`

Supported response formats:

- `json`
- `text`
- `srt`
- `vtt`
- `verbose_json`
- `diarized_json` (alias of `verbose_json`)

When `response_format=verbose_json` or `diarized_json`, response includes:

- `text`
- `language`
- `duration`
- `segments[]`
- `segments[].words[]` in WhisperX-compatible shape
- `word_segments[]` in WhisperX-compatible shape
- `words[]` as a compatibility alias for `word_segments[]`
- `segments[].speaker`
- `words[].speaker`
- raw `diarization` segments
- raw merged `vad` speech chunks
- `model`

Word timestamp entries use WhisperX-style `score` when confidence is available:

```json
{
  "segments": [
    {
      "start": 0.42,
      "end": 3.18,
      "text": "Hello, this is a test.",
      "words": [
        { "word": "Hello,", "start": 0.42, "end": 0.86, "score": 0.91 }
      ]
    }
  ],
  "word_segments": [
    { "word": "Hello,", "start": 0.42, "end": 0.86, "score": 0.91 }
  ],
  "language": "en"
}
```

Subtitle formats (`srt` / `vtt`) prefix cues with speaker labels when available.

## Model Controls

- `GET /v1/models/status`
- `POST /v1/models/parakeet/load`
- `POST /v1/models/parakeet/unload`
- `POST /v1/models/diarization/load`
- `POST /v1/models/diarization/unload`
- `POST /v1/models/vad/load`
- `POST /v1/models/vad/unload`

CUDA unload attempts `torch.cuda.empty_cache()`.
Model unload also runs Python GC and, on Linux, asks glibc to trim allocator arenas.
Set `MODEL_PROCESS_ISOLATION=true` to run ASR/diarization model work in a child process. When both models are unloaded or idle-evicted, the worker exits so native PyTorch/NeMo/pyannote RSS can return to the OS instead of staying mapped in uvicorn.
`PARAKEET__USE_EXTRACTED_NEMO_CACHE=false` by default because Parakeet's restore peak is dominated by checkpoint loading, and pre-extraction can increase first-load RSS.
`PARAKEET__TORCH_LOAD_MMAP=false` by default; PyTorch mmap loading is available as an experiment, but it did not reduce the measured Parakeet restore peak on the CUDA legacy image.
Set `UNLOAD_ASR_BEFORE_DIARIZATION=true` only if you want lower ASR/diarization overlap at the cost of forcing Parakeet to reload for the next request.
When `PARAKEET__DEVICE` is CUDA, the ASR model attempts `to(cuda)` + FP16 (`half()`), and transcription can auto-chunk audio based on currently available GPU memory.
Adaptive chunking uses a conservative memory-based ladder, caps chunks at 600 seconds by default, and logs the chosen chunk plan at transcription start.
Set `VAD__ENABLED=true` or pass `vad_filter=true` per request to run Silero VAD before ASR. VAD cuts the normalized audio into speech-only chunks, transcribes those chunks, then offsets word and segment timestamps back to the original timeline. Silero loads through ONNX Runtime by default (`VAD__USE_ONNX=true`) and can fall back to JIT if ONNX Runtime is unavailable or incompatible. `chunk_size`, `vad_onset`, and `vad_offset` follow the same shape as WhisperX's VAD controls.
If CUDA reports `device not ready`, lower `PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE` to a value such as `120` or reduce `PARAKEET__CUDA_CHUNK_MAX_SECONDS`.
Set `PARAKEET__CUDA_FORCE_GREEDY_DECODING=true` to switch NeMo decoding from `greedy_batch` to `greedy` if a Maxwell/TITAN-era CUDA runtime hits decoder compatibility failures.
The default CUDA Docker image uses a CUDA 12.8 runtime and PyTorch CUDA 12.8 wheels so RTX 50-series / Blackwell GPUs can run kernels for their newer compute capability.

## Environment Variables

Core env vars:

- `API_KEY`
- `HF_TOKEN`
- `PARAKEET__MODEL_NAME`
- `PARAKEET__DEVICE`
- `PARAKEET__PRELOAD_MODEL`
- `PARAKEET__LOCAL_FILES_ONLY`
- `PARAKEET__CUDA_HALF_PRECISION`
- `PARAKEET__CUDA_ADAPTIVE_CHUNKING`
- `PARAKEET__CUDA_FORCE_GREEDY_DECODING`
- `PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE`
- `PARAKEET__CUDA_CHUNK_MIN_SECONDS`
- `PARAKEET__CUDA_CHUNK_MAX_SECONDS`
- `PARAKEET__CUDA_CHUNK_OVERLAP_SECONDS`
- `PARAKEET__USE_EXTRACTED_NEMO_CACHE`
- `PARAKEET__TORCH_LOAD_MMAP`
- `DIARIZATION__MODEL_NAME`
- `DIARIZATION__DEVICE`
- `DIARIZATION__PRELOAD_MODEL`
- `VAD__ENABLED`
- `VAD__METHOD`
- `VAD__PRELOAD_MODEL`
- `VAD__USE_ONNX`
- `VAD__ONNX_FALLBACK_TO_JIT`
- `VAD__ONNX_OPSET_VERSION`
- `VAD__VAD_ONSET`
- `VAD__VAD_OFFSET`
- `VAD__CHUNK_SIZE`
- `VAD__MIN_SPEECH_DURATION_MS`
- `VAD__MIN_SILENCE_DURATION_MS`
- `VAD__SPEECH_PAD_MS`
- `MAX_CONCURRENT_TRANSCRIPTIONS`
- `DEBUG_LOG_TRANSCRIPTION_PAYLOAD`
- `MODEL_IDLE_EVICT_MINUTES`
- `MODEL_PROCESS_ISOLATION`
- `UNLOAD_ASR_BEFORE_DIARIZATION`
- `UVICORN_HOST`
- `UVICORN_PORT`

Set `DEBUG_LOG_TRANSCRIPTION_PAYLOAD=true` to log parsed incoming transcription request fields (metadata only, not raw audio bytes).

## Integration Test Flags

- `RUN_PARAKEET_INTEGRATION=1`
- `RUN_DIARIZATION_INTEGRATION=1`

These are skipped by default in normal local tests.

## Docker

Published image tags:

- `ghcr.io/maroonbrian1928/parakeetx:cpu`: CPU-only runtime.
- `ghcr.io/maroonbrian1928/parakeetx:cuda`: CUDA 12.8 / PyTorch cu128 runtime for RTX 50-series / Blackwell and newer supported CUDA GPUs.
- `ghcr.io/maroonbrian1928/parakeetx:cuda-legacy`: CUDA 12.4 / PyTorch cu118 runtime for older GPUs such as TITAN X / Maxwell that are not covered by newer PyTorch cu128 wheels.

Build CPU image:

```bash
docker compose -f compose.cpu.yaml build
```

Build Modern CUDA image:

```bash
docker compose -f compose.cuda.yaml build
```

Build legacy CUDA image:

```bash
docker compose -f compose.yaml build
```

Run CPU profile:

```bash
docker compose -f compose.cpu.yaml up
```

Run Modern CUDA profile:

```bash
docker compose -f compose.cuda.yaml up
```

Run legacy CUDA profile:

```bash
docker compose up
```

Default host port: `7474`.

All Compose profiles use `HOST_PORT`. Only one profile can bind the default host port at a time. Override `HOST_PORT` when starting a second profile if you need multiple profiles running simultaneously.

CUDA profile defaults are tuned for lower VRAM pressure:

- `PARAKEET__DEVICE_CUDA=cuda`
- `DIARIZATION__DEVICE_CUDA=cpu` when unset in `.env`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

If you set `DIARIZATION__DEVICE_CUDA=cuda`, diarization will also run on the GPU and share VRAM with Parakeet.
