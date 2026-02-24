import os
import pytest
from unittest.mock import patch, MagicMock
from transcriber import transcribe

def _make_mock_model(texts):
    """Return a mock WhisperModel whose transcribe() yields segments with .text."""
    mock_model = MagicMock()
    segments = []
    for t in texts:
        seg = MagicMock()
        seg.text = t
        segments.append(seg)
    mock_model.transcribe.return_value = (segments, MagicMock())
    return mock_model

def test_transcribe_returns_text(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()  # dummy file
    with patch("transcriber.WhisperModel") as mock_cls:
        mock_cls.return_value = _make_mock_model(["Hello this is a test meeting transcript."])
        result = transcribe(audio_file, model_name="tiny")
    assert result == "Hello this is a test meeting transcript."

def test_transcribe_saves_to_file(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    out_file = str(tmp_path / "transcript.txt")
    open(audio_file, "w").close()
    with patch("transcriber.WhisperModel") as mock_cls:
        mock_cls.return_value = _make_mock_model(["Meeting content here."])
        transcribe(audio_file, model_name="tiny", output_path=out_file)
    assert os.path.exists(out_file)
    assert open(out_file).read() == "Meeting content here."

def test_transcribe_joins_multiple_segments(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()
    with patch("transcriber.WhisperModel") as mock_cls:
        mock_cls.return_value = _make_mock_model(["First segment.", "Second segment."])
        result = transcribe(audio_file, model_name="tiny")
    assert result == "First segment. Second segment."

def test_transcribe_uses_correct_model_params(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()
    with patch("transcriber.WhisperModel") as mock_cls:
        mock_cls.return_value = _make_mock_model(["text"])
        transcribe(audio_file, model_name="medium")
    mock_cls.assert_called_once_with("medium", device="cpu", compute_type="int8")

def test_transcribe_returns_segments_when_requested(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()
    seg = MagicMock()
    seg.text = "Hello world"
    seg.start = 0.0
    seg.end = 3.0
    with patch("transcriber.WhisperModel") as mock_cls:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock())
        mock_cls.return_value = mock_model
        text, segments = transcribe(audio_file, model_name="tiny", return_segments=True)
    assert text == "Hello world"
    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].end == 3.0

def test_transcribe_still_returns_string_by_default(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()
    with patch("transcriber.WhisperModel") as mock_cls:
        mock_cls.return_value = _make_mock_model(["Hello world"])
        result = transcribe(audio_file, model_name="tiny")
    assert isinstance(result, str)
    assert result == "Hello world"
