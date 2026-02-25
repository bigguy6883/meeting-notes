import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from diarizer import diarize


def _seg(start, end, text):
    s = MagicMock()
    s.start = start
    s.end = end
    s.text = text
    return s


def _mock_resemblyzer(n_segments):
    """Mock resemblyzer with a 10s WAV and distinct embeddings per segment."""
    mock = MagicMock()
    mock.preprocess_wav.return_value = np.zeros(160000)  # 10s at 16kHz
    encoder = MagicMock()
    # Each call returns a unique unit vector so clustering has signal
    encoder.embed_utterance.side_effect = [
        np.eye(n_segments)[i] for i in range(n_segments)
    ]
    mock.VoiceEncoder.return_value = encoder
    return mock


def _mock_sklearn(labels):
    """Mock sklearn clustering to return fixed labels."""
    mock = MagicMock()
    clustering = MagicMock()
    clustering.fit_predict.return_value = np.array(labels)
    mock.cluster.AgglomerativeClustering.return_value = clustering
    return mock


def _patch_resemblyzer(mock_resemblyzer, mock_sklearn):
    return patch.dict(sys.modules, {
        "resemblyzer": mock_resemblyzer,
        "sklearn": mock_sklearn,
        "sklearn.cluster": mock_sklearn.cluster,
    })


def test_diarize_falls_back_on_import_error():
    segs = [_seg(0.0, 3.0, "Hello world"), _seg(5.0, 8.0, "Goodbye world")]
    with patch.dict(sys.modules, {"resemblyzer": None}):
        text, diarized = diarize("/fake/audio.mp3", segs)
    assert diarized is False
    assert "Hello world" in text
    assert "Goodbye world" in text


def test_diarize_labels_two_speakers():
    segs = [_seg(0.0, 3.0, "Hello world"), _seg(5.0, 8.0, "Goodbye world")]
    mock_r = _mock_resemblyzer(2)
    mock_sk = _mock_sklearn([0, 1])

    with patch("diarizer._convert_to_wav", return_value="/tmp/fake.wav"), \
         patch("os.unlink"), \
         _patch_resemblyzer(mock_r, mock_sk):
        text, diarized = diarize("/fake/audio.mp3", segs)

    assert diarized is True
    assert "Speaker_00: Hello world" in text
    assert "Speaker_01: Goodbye world" in text


def test_diarize_falls_back_on_single_speaker():
    segs = [_seg(0.0, 3.0, "Hello"), _seg(5.0, 8.0, "World")]
    mock_r = _mock_resemblyzer(2)
    mock_sk = _mock_sklearn([0, 0])  # both assigned to same cluster

    with patch("diarizer._convert_to_wav", return_value="/tmp/fake.wav"), \
         patch("os.unlink"), \
         _patch_resemblyzer(mock_r, mock_sk):
        text, diarized = diarize("/fake/audio.mp3", segs)

    assert diarized is False
    assert "Hello" in text
    assert "World" in text


def test_diarize_falls_back_on_exception():
    segs = [_seg(0.0, 3.0, "Hello")]
    mock_r = _mock_resemblyzer(1)
    mock_sk = _mock_sklearn([0])

    with patch("diarizer._convert_to_wav", side_effect=Exception("ffmpeg error")), \
         _patch_resemblyzer(mock_r, mock_sk):
        text, diarized = diarize("/fake/audio.mp3", segs)

    assert diarized is False
    assert "Hello" in text


def test_diarize_skips_very_short_segments():
    # seg 0 is 0.05s → 800 samples → skipped; seg 1 is 3s → included
    # Only 1 valid embedding → falls back
    segs = [_seg(0.0, 0.05, "Uh"), _seg(5.0, 8.0, "Full sentence")]
    mock_r = _mock_resemblyzer(1)  # only 1 embed_utterance call expected
    mock_sk = _mock_sklearn([0])

    with patch("diarizer._convert_to_wav", return_value="/tmp/fake.wav"), \
         patch("os.unlink"), \
         _patch_resemblyzer(mock_r, mock_sk):
        text, diarized = diarize("/fake/audio.mp3", segs)

    assert diarized is False


def test_diarize_merges_consecutive_same_speaker():
    # Three segments: A, A, B — first two should merge into one line
    segs = [
        _seg(0.0, 3.0, "Hello"),
        _seg(3.0, 6.0, "world"),
        _seg(6.0, 9.0, "Goodbye"),
    ]
    mock_r = _mock_resemblyzer(3)
    mock_sk = _mock_sklearn([0, 0, 1])

    with patch("diarizer._convert_to_wav", return_value="/tmp/fake.wav"), \
         patch("os.unlink"), \
         _patch_resemblyzer(mock_r, mock_sk):
        text, diarized = diarize("/fake/audio.mp3", segs)

    assert diarized is True
    assert "Speaker_00: Hello world" in text
    assert "Speaker_01: Goodbye" in text
