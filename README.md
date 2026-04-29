# ParakeetX API Server

Standalone FastAPI transcription service inspired by WhisperX API conventions, backed by NVIDIA Parakeet (`nvidia/parakeet-tdt-0.6b-v2`) with optional pyannote diarization.

## Features

- `POST /v1/audio/transcriptions` with OpenAI-style multipart fields.
- Native Parakeet timestamps (no Whisper forced-alignment stage).
- WhisperX-compatible word timestamp fields for downstream tools (`word_segments` and `segments[].words`).
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

CUDA unload attempts `torch.cuda.empty_cache()`.
When `PARAKEET__DEVICE` is CUDA, the ASR model attempts `to(cuda)` + FP16 (`half()`), and transcription can auto-chunk audio based on currently available GPU memory.
Adaptive chunking uses a conservative memory-based ladder, caps chunks at 600 seconds by default, and logs the chosen chunk plan at transcription start.
If CUDA reports `device not ready`, lower `PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE` to a value such as `120` or reduce `PARAKEET__CUDA_CHUNK_MAX_SECONDS`.
Maxwell/TITAN-era CUDA runs switch NeMo decoding from `greedy_batch` to `greedy` to avoid CUDA graph decoder compatibility failures while keeping ASR on GPU.
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
- `PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE`
- `PARAKEET__CUDA_CHUNK_MIN_SECONDS`
- `PARAKEET__CUDA_CHUNK_MAX_SECONDS`
- `PARAKEET__CUDA_CHUNK_OVERLAP_SECONDS`
- `DIARIZATION__MODEL_NAME`
- `DIARIZATION__DEVICE`
- `DIARIZATION__PRELOAD_MODEL`
- `MAX_CONCURRENT_TRANSCRIPTIONS`
- `DEBUG_LOG_TRANSCRIPTION_PAYLOAD`
- `MODEL_IDLE_EVICT_MINUTES`
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

Build CUDA image:

```bash
docker compose -f compose.yaml build
```

Build legacy CUDA image:

```bash
docker compose -f compose.cuda-legacy.yaml build
```

Run CPU profile:

```bash
docker compose -f compose.cpu.yaml up
```

Run CUDA profile:

```bash
docker compose -f compose.yaml up
```

Run legacy CUDA profile:

```bash
docker compose -f compose.cuda-legacy.yaml up
```

Default host port: `7474`.

All Compose profiles use `HOST_PORT`. Only one profile can bind the default host port at a time. Override `HOST_PORT` when starting a second profile if you need multiple profiles running simultaneously.

CUDA profile defaults are tuned for lower VRAM pressure:

- `PARAKEET__DEVICE_CUDA=cuda`
- `DIARIZATION__DEVICE_CUDA=cpu` when unset in `.env`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

If you set `DIARIZATION__DEVICE_CUDA=cuda`, diarization will also run on the GPU and share VRAM with Parakeet.
