import sys
import pytest
from unittest.mock import MagicMock, patch
from diarizer import diarize, _find_speaker


def _seg(start, end, text):
    """Create a fake faster-whisper segment object."""
    s = MagicMock()
    s.start = start
    s.end = end
    s.text = text
    return s


def _mock_pyannote(turns):
    """
    Build a mock pyannote.audio module.
    turns: list of (speaker_label, start, end)
    """
    mock_audio = MagicMock()
    mock_diarization = MagicMock()
    mock_turns = []
    for speaker, start, end in turns:
        turn = MagicMock()
        turn.start = start
        turn.end = end
        mock_turns.append((turn, None, speaker))
    mock_diarization.itertracks.return_value = mock_turns
    mock_audio.Pipeline.from_pretrained.return_value.return_value = mock_diarization
    return mock_audio


def test_find_speaker_returns_matching_segment():
    segments = [("Speaker_00", 0.0, 5.0), ("Speaker_01", 5.0, 10.0)]
    assert _find_speaker(2.5, segments) == "Speaker_00"
    assert _find_speaker(7.5, segments) == "Speaker_01"


def test_find_speaker_falls_back_to_nearest_midpoint():
    # Gap between 5.0 and 6.0 — no segment contains 5.4
    # Speaker_00 midpoint=2.5 (dist 2.9), Speaker_01 midpoint=8.0 (dist 2.6) → nearest
    segments = [("Speaker_00", 0.0, 5.0), ("Speaker_01", 6.0, 10.0)]
    assert _find_speaker(5.4, segments) == "Speaker_01"


def test_diarize_returns_plain_text_without_token():
    segs = [_seg(0.0, 3.0, "Hello world"), _seg(3.0, 6.0, "Goodbye world")]
    text, diarized = diarize("/fake/audio.mp3", segs, hf_token="")
    assert diarized is False
    assert "Hello world" in text
    assert "Goodbye world" in text


def test_diarize_labels_two_speakers():
    segs = [_seg(0.0, 3.0, "Hello world"), _seg(5.0, 8.0, "Goodbye world")]
    mock_audio = _mock_pyannote([("Speaker_00", 0.0, 4.0), ("Speaker_01", 4.0, 9.0)])

    with patch.dict(sys.modules, {"pyannote": MagicMock(), "pyannote.audio": mock_audio}):
        text, diarized = diarize("/fake/audio.mp3", segs, hf_token="fake_token")

    assert diarized is True
    assert "Speaker_00:" in text
    assert "Speaker_01:" in text
    assert "Hello world" in text
    assert "Goodbye world" in text


def test_diarize_falls_back_on_single_speaker():
    segs = [_seg(0.0, 3.0, "Hello world")]
    mock_audio = _mock_pyannote([("Speaker_00", 0.0, 3.0)])

    with patch.dict(sys.modules, {"pyannote": MagicMock(), "pyannote.audio": mock_audio}):
        text, diarized = diarize("/fake/audio.mp3", segs, hf_token="fake_token")

    assert diarized is False


def test_diarize_falls_back_on_import_error():
    segs = [_seg(0.0, 3.0, "Hello")]
    mock_audio = MagicMock()
    mock_audio.Pipeline = MagicMock(side_effect=ImportError("No module named 'pyannote'"))

    with patch.dict(sys.modules, {"pyannote": MagicMock(), "pyannote.audio": mock_audio}):
        text, diarized = diarize("/fake/audio.mp3", segs, hf_token="fake_token")

    assert diarized is False
    assert "Hello" in text


def test_diarize_falls_back_on_exception():
    segs = [_seg(0.0, 3.0, "Hello")]
    mock_audio = MagicMock()
    mock_audio.Pipeline.from_pretrained.side_effect = Exception("model error")

    with patch.dict(sys.modules, {"pyannote": MagicMock(), "pyannote.audio": mock_audio}):
        text, diarized = diarize("/fake/audio.mp3", segs, hf_token="fake_token")

    assert diarized is False
    assert "Hello" in text
