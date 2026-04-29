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
    ):
        _ = upload
        _ = language
        _ = min_speakers
        _ = max_speakers
        _ = num_speakers
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


def test_cuda_kernel_image_error_mentions_blackwell_image_rebuild():
    detail = _friendly_runtime_error_detail(
        RuntimeError("CUDA error: no kernel image is available for execution on the device")
    )

    assert "PyTorch CUDA 12.8+ wheels" in detail


def test_translations_return_501(client):
    response = client.post("/v1/audio/translations")
    assert response.status_code == 501
