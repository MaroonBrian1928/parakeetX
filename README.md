# ParakeetX API Server

Standalone FastAPI transcription service inspired by WhisperX API conventions, backed by NVIDIA Parakeet (`nvidia/parakeet-tdt-0.6b-v2`) with optional pyannote diarization.

## Features

- `POST /v1/audio/transcriptions` with OpenAI-style multipart fields.
- Native Parakeet timestamps (no Whisper forced-alignment stage).
- Optional diarization via `pyannote/speaker-diarization-community-1` (`diarize=true`).
- Speaker labels assigned to words/segments by maximum timestamp overlap.
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

Server default: `http://0.0.0.0:7317`

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

When `response_format=verbose_json`, response includes:

- `segments[].speaker`
- `words[].speaker`
- raw `diarization` segments

Subtitle formats (`srt` / `vtt`) prefix cues with speaker labels when available.

## Model Controls

- `GET /v1/models/status`
- `POST /v1/models/parakeet/load`
- `POST /v1/models/parakeet/unload`
- `POST /v1/models/diarization/load`
- `POST /v1/models/diarization/unload`

CUDA unload attempts `torch.cuda.empty_cache()`.

## Environment Variables

Core env vars:

- `API_KEY`
- `HF_TOKEN`
- `PARAKEET__MODEL_NAME`
- `PARAKEET__DEVICE`
- `PARAKEET__PRELOAD_MODEL`
- `PARAKEET__LOCAL_FILES_ONLY`
- `DIARIZATION__MODEL_NAME`
- `DIARIZATION__DEVICE`
- `DIARIZATION__PRELOAD_MODEL`
- `MAX_CONCURRENT_TRANSCRIPTIONS`
- `DEBUG_LOG_TRANSCRIPTION_PAYLOAD`
- `UVICORN_HOST`
- `UVICORN_PORT`

Set `DEBUG_LOG_TRANSCRIPTION_PAYLOAD=true` to log parsed incoming transcription request fields (metadata only, not raw audio bytes).

## Integration Test Flags

- `RUN_PARAKEET_INTEGRATION=1`
- `RUN_DIARIZATION_INTEGRATION=1`

These are skipped by default in normal local tests.

## Docker

Build CPU image:

```bash
docker compose --profile cpu build
```

Build CUDA image:

```bash
docker compose --profile cuda build
```

Default host ports:

- CPU profile: `7474`
- CUDA profile: `7373`

CUDA profile defaults are tuned for lower VRAM pressure:

- `PARAKEET__DEVICE_CUDA=cuda`
- `DIARIZATION__DEVICE_CUDA=cpu`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
