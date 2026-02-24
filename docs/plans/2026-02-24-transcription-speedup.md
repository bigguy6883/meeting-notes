# Transcription Speedup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Speed up local Whisper transcription by adding VAD filtering, greedy decoding, CPU thread tuning, and model caching.

**Architecture:** All changes are isolated to `transcriber.py`. A module-level `_model_cache` dict holds loaded `WhisperModel` instances keyed by model name so the model is loaded from disk only once per process. `model.transcribe()` gains `vad_filter=True` and `beam_size=1`. `WhisperModel()` gains `cpu_threads=4`.

**Tech Stack:** faster-whisper, pytest, unittest.mock

---

### Task 1: Update `transcriber.py`

**Files:**
- Modify: `transcriber.py`

**Step 1: Replace the file contents**

```python
from faster_whisper import WhisperModel

_model_cache = {}

def _get_model(model_name):
    if model_name not in _model_cache:
        _model_cache[model_name] = WhisperModel(
            model_name, device="cpu", compute_type="int8", cpu_threads=4
        )
    return _model_cache[model_name]

def transcribe(audio_path, model_name="small", output_path=None, return_segments=False):
    model = _get_model(model_name)
    segments_gen, _ = model.transcribe(audio_path, beam_size=1, vad_filter=True)
    segments_list = list(segments_gen)
    text = " ".join(s.text.strip() for s in segments_list).strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    if return_segments:
        return text, segments_list
    return text
```

**Step 2: Verify the file looks right**

Run: `cat transcriber.py`
Expected: shows `_model_cache`, `_get_model`, `vad_filter=True`, `beam_size=1`, `cpu_threads=4`

---

### Task 2: Update `tests/test_transcriber.py`

The existing tests patch `transcriber.WhisperModel` at the class level. With model caching, if two tests use the same `model_name`, the second test skips the constructor call and reuses the cached instance. Tests must clear `transcriber._model_cache` before each test.

Also:
- `test_transcribe_uses_correct_model_params` needs `cpu_threads=4` added to the assertion
- Add assertions that `vad_filter=True` and `beam_size=1` are passed to `model.transcribe()`
- Add a test that the model is cached (constructor called only once across two `transcribe()` calls)

**Files:**
- Modify: `tests/test_transcriber.py`

**Step 1: Write the updated test file**

```python
import os
import pytest
from unittest.mock import patch, MagicMock
import transcriber
from transcriber import transcribe

@pytest.fixture(autouse=True)
def clear_model_cache():
    transcriber._model_cache.clear()
    yield
    transcriber._model_cache.clear()

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
    open(audio_file, "w").close()
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
    mock_cls.assert_called_once_with("medium", device="cpu", compute_type="int8", cpu_threads=4)

def test_transcribe_uses_vad_and_beam_size(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()
    with patch("transcriber.WhisperModel") as mock_cls:
        mock_model = _make_mock_model(["text"])
        mock_cls.return_value = mock_model
        transcribe(audio_file, model_name="tiny")
    mock_model.transcribe.assert_called_once_with(audio_file, beam_size=1, vad_filter=True)

def test_transcribe_caches_model(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()
    with patch("transcriber.WhisperModel") as mock_cls:
        mock_cls.return_value = _make_mock_model(["a", "b"])
        transcribe(audio_file, model_name="tiny")
        # reset return value for second call
        mock_cls.return_value = _make_mock_model(["c"])
        transcribe(audio_file, model_name="tiny")
    # WhisperModel constructor should only be called once — second call uses cache
    assert mock_cls.call_count == 1

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
```

**Step 2: Run tests to verify all pass**

Run: `pytest tests/test_transcriber.py -v`
Expected: all tests PASS

**Step 3: Run full test suite to verify no regressions**

Run: `pytest -v`
Expected: all tests PASS

**Step 4: Commit**

```bash
git add transcriber.py tests/test_transcriber.py
git commit -m "perf: speed up transcription with VAD filter, beam_size=1, cpu_threads=4, model cache"
```
