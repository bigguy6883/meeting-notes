import os
import pytest
from unittest.mock import patch, MagicMock
from transcriber import transcribe, Segment


def _make_mock_response(segments_data):
    """
    segments_data: list of (text, start, end)
    Returns a mock Groq verbose_json transcription response.
    """
    response = MagicMock()
    segs = []
    texts = []
    for text, start, end in segments_data:
        segs.append({'text': text, 'start': start, 'end': end})
        texts.append(text.strip())
    response.text = " ".join(texts)
    response.segments = segs
    return response


def test_transcribe_returns_text(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([("Hello world.", 0.0, 2.0)])
        result = transcribe(audio_file)
    assert result == "Hello world."


def test_transcribe_saves_to_file(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    out_file = str(tmp_path / "transcript.txt")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([("Meeting content here.", 0.0, 3.0)])
        transcribe(audio_file, output_path=out_file)
    assert os.path.exists(out_file)
    with open(out_file) as f:
        assert f.read() == "Meeting content here."


def test_transcribe_returns_segments_when_requested(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([("Hello world.", 0.0, 2.5)])
        text, segments = transcribe(audio_file, return_segments=True)
    assert text == "Hello world."
    assert len(segments) == 1
    assert isinstance(segments[0], Segment)
    assert segments[0].start == 0.0
    assert segments[0].end == 2.5
    assert segments[0].text == "Hello world."


def test_transcribe_returns_multiple_segments(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([
                ("First segment.", 0.0, 2.0),
                ("Second segment.", 2.0, 4.0),
            ])
        text, segments = transcribe(audio_file, return_segments=True)
    assert text == "First segment. Second segment."
    assert len(segments) == 2
    assert segments[1].start == 2.0
    assert segments[1].end == 4.0


def test_transcribe_uses_correct_model_and_format(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.audio.transcriptions.create.return_value = \
            _make_mock_response([("text", 0.0, 1.0)])
        transcribe(audio_file)
    call_kwargs = mock_instance.audio.transcriptions.create.call_args[1]
    assert call_kwargs["model"] == "whisper-large-v3-turbo"
    assert call_kwargs["response_format"] == "verbose_json"


def test_transcribe_still_returns_string_by_default(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([("Hello", 0.0, 1.0)])
        result = transcribe(audio_file)
    assert isinstance(result, str)
