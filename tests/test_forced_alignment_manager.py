from __future__ import annotations

from types import SimpleNamespace

from parakeetx_api_server.model_managers.forced_alignment_manager import (
    _alignment_item_to_word,
    _coalesce_text_segments,
    _offset_words,
    _language_name,
)


def test_language_name_normalizes_qwen_supported_languages() -> None:
    assert _language_name(None) == "English"
    assert _language_name("en") == "English"
    assert _language_name("Spanish") == "Spanish"


def test_alignment_item_to_word_accepts_qwen_objects_and_dicts() -> None:
    item = SimpleNamespace(text="hello", start_time=0.1, end_time=0.4, score=0.9)
    assert _alignment_item_to_word(item) == {
        "word": "hello",
        "start": 0.1,
        "end": 0.4,
        "score": 0.9,
    }

    assert _alignment_item_to_word({"word": "world", "start": 0.5, "end": 0.8}) == {
        "word": "world",
        "start": 0.5,
        "end": 0.8,
    }


def test_coalesce_text_segments_respects_max_chunk_seconds() -> None:
    chunks = _coalesce_text_segments(
        [
            {"start": 0.0, "end": 10.0, "text": "first"},
            {"start": 10.0, "end": 20.0, "text": "second"},
            {"start": 20.0, "end": 35.0, "text": "third"},
        ],
        max_chunk_seconds=25,
    )

    assert chunks == [
        {"start": 0.0, "end": 20.0, "text": "first second"},
        {"start": 20.0, "end": 35.0, "text": "third"},
    ]


def test_offset_words_maps_chunk_words_back_to_original_timeline() -> None:
    assert _offset_words([{"word": "hello", "start": 0.2, "end": 0.5}], offset=10.0) == [
        {"word": "hello", "start": 10.2, "end": 10.5}
    ]
