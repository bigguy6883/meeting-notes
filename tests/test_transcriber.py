import os
import pytest
from unittest.mock import patch, MagicMock
from transcriber import transcribe

def test_transcribe_returns_text(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()  # dummy file
    mock_result = {"text": "Hello this is a test meeting transcript."}
    with patch("transcriber.whisper.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_result
        mock_load.return_value = mock_model
        result = transcribe(audio_file, model_name="tiny")
    assert result == "Hello this is a test meeting transcript."

def test_transcribe_saves_to_file(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    out_file = str(tmp_path / "transcript.txt")
    open(audio_file, "w").close()
    mock_result = {"text": "Meeting content here."}
    with patch("transcriber.whisper.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_result
        mock_load.return_value = mock_model
        transcribe(audio_file, model_name="tiny", output_path=out_file)
    assert os.path.exists(out_file)
    assert open(out_file).read() == "Meeting content here."
