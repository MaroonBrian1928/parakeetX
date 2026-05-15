from __future__ import annotations

from parakeetx_api_server.deps import get_transcription_service
from parakeetx_api_server.main import app
from parakeetx_api_server.routers.transcriptions import _friendly_runtime_error_detail


class FakeTranscriptionService:
    async def transcribe_upload(
        self,
        *,
        upload,
        language,
        diarize,
        min_speakers,
        max_speakers,
        num_speakers,
        vad_options,
        forced_alignment,
    ):
        _ = upload
        _ = language
        _ = min_speakers
        _ = max_speakers
        _ = num_speakers
        self.vad_options = vad_options
        self.forced_alignment = forced_alignment
        diarization = (
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}] if diarize else []
        )
        return {
            "text": "hello world",
            "language": "en",
            "model": "nvidia/parakeet-tdt-0.6b-v2",
            "words": [
                {
                    "word": "hello",
                    "start": 0.0,
                    "end": 0.4,
                    "confidence": 0.91,
                    "speaker": "SPEAKER_00" if diarize else None,
                }
            ],
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello world",
                    "speaker": "SPEAKER_00" if diarize else None,
                }
            ],
            "diarization": diarization,
        }


def _post_transcription(client, wav_bytes: bytes, **extra_form):
    form_data = {
        "model": "whisper-1",
        "response_format": "json",
    }
    form_data.update(extra_form)
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
        data=form_data,
    )


def test_transcription_json_success(client, wav_bytes):
    app.dependency_overrides[get_transcription_service] = lambda: FakeTranscriptionService()

    response = _post_transcription(client, wav_bytes)

    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}


def test_transcription_verbose_json_includes_speakers(client, wav_bytes):
    app.dependency_overrides[get_transcription_service] = lambda: FakeTranscriptionService()

    response = _post_transcription(
        client,
        wav_bytes,
        response_format="verbose_json",
        diarize="true",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["segments"][0]["speaker"] == "SPEAKER_00"
    assert body["words"][0]["speaker"] == "SPEAKER_00"
    assert body["word_segments"][0]["score"] == 0.91
    assert body["segments"][0]["words"] == body["word_segments"]


def test_unsupported_feature_failures(client, wav_bytes):
    app.dependency_overrides[get_transcription_service] = lambda: FakeTranscriptionService()

    stream_response = _post_transcription(client, wav_bytes, stream="true")
    assert stream_response.status_code == 422

    language_response = _post_transcription(client, wav_bytes, language="es")
    assert language_response.status_code == 422

    hotwords_response = _post_transcription(client, wav_bytes, hotwords="hello")
    assert hotwords_response.status_code == 422


def test_vad_options_are_accepted(client, wav_bytes):
    fake_service = FakeTranscriptionService()
    app.dependency_overrides[get_transcription_service] = lambda: fake_service

    response = _post_transcription(
        client,
        wav_bytes,
        vad_filter="true",
        vad_method="silero",
        vad_onset="0.4",
        vad_offset="0.25",
        chunk_size="20",
        min_speech_duration_ms="100",
        min_silence_duration_ms="200",
        speech_pad_ms="50",
    )

    assert response.status_code == 200
    assert fake_service.vad_options.enabled is True
    assert fake_service.vad_options.method == "silero"
    assert fake_service.vad_options.vad_onset == 0.4
    assert fake_service.vad_options.vad_offset == 0.25
    assert fake_service.vad_options.chunk_size == 20
    assert fake_service.vad_options.min_speech_duration_ms == 100
    assert fake_service.vad_options.min_silence_duration_ms == 200
    assert fake_service.vad_options.speech_pad_ms == 50


def test_forced_alignment_flag_is_accepted(client, wav_bytes):
    fake_service = FakeTranscriptionService()
    app.dependency_overrides[get_transcription_service] = lambda: fake_service

    response = _post_transcription(client, wav_bytes, forced_alignment="true")

    assert response.status_code == 200
    assert fake_service.forced_alignment is True


def test_global_forced_alignment_setting_is_accepted(client, wav_bytes, monkeypatch):
    monkeypatch.setenv("FORCED_ALIGNMENT__ENABLED", "true")
    fake_service = FakeTranscriptionService()
    app.dependency_overrides[get_transcription_service] = lambda: fake_service

    response = _post_transcription(client, wav_bytes)

    assert response.status_code == 200
    assert fake_service.forced_alignment is True


def test_invalid_vad_method_is_rejected(client, wav_bytes):
    app.dependency_overrides[get_transcription_service] = lambda: FakeTranscriptionService()

    response = _post_transcription(client, wav_bytes, vad_filter="true", vad_method="pyannote")

    assert response.status_code == 422
    assert "Only silero VAD is supported" in response.json()["detail"]


def test_cuda_kernel_image_error_mentions_blackwell_image_rebuild():
    detail = _friendly_runtime_error_detail(
        RuntimeError("CUDA error: no kernel image is available for execution on the device")
    )

    assert "PyTorch CUDA 12.8+ wheels" in detail
    assert "PARAKEET__CUDA_FORCE_GREEDY_DECODING" in detail


def test_cuda_device_not_ready_error_mentions_smaller_chunks():
    detail = _friendly_runtime_error_detail(
        RuntimeError("CUDA driver error: device not ready")
    )

    assert "PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE" in detail
    assert "120" in detail


def test_translations_return_501(client):
    response = client.post("/v1/audio/translations")
    assert response.status_code == 501


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "whisper-1" in ids
    assert "nvidia/parakeet-tdt-0.6b-v2" in ids
