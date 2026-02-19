import os
import time
import pytest
from unittest.mock import patch, MagicMock
from recorder import Recorder

def test_recorder_starts_and_creates_file(tmp_path):
    rec = Recorder(mic_device="hw:0,0", output_dir=str(tmp_path))
    with patch("recorder.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        filepath = rec.start()
    assert filepath.endswith(".mp3")
    assert rec.is_recording()

def test_recorder_stop_returns_filepath(tmp_path):
    rec = Recorder(mic_device="hw:0,0", output_dir=str(tmp_path))
    with patch("recorder.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        filepath = rec.start()
        result = rec.stop()
    assert result == filepath
    mock_proc.terminate.assert_called_once()

def test_recorder_not_recording_initially(tmp_path):
    rec = Recorder(mic_device="hw:0,0", output_dir=str(tmp_path))
    assert not rec.is_recording()

def test_recorder_stop_when_not_recording(tmp_path):
    rec = Recorder(mic_device="hw:0,0", output_dir=str(tmp_path))
    result = rec.stop()
    assert result is None
