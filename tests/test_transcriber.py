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
