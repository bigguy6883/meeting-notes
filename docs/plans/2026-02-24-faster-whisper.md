# faster-whisper Switch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `openai-whisper` with `faster-whisper` in the meeting-notes transcriber for ~3× faster CPU transcription with no API changes.

**Architecture:** Three-file change: swap the package in `requirements.txt`, update `transcriber.py` to use `WhisperModel` with `compute_type="int8"`, and update the test mocks to match the new API. The `transcribe()` function signature stays identical so no other files change.

**Tech Stack:** Python 3.13, faster-whisper (CTranslate2), pytest

---

### Task 1: Install faster-whisper and update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Install faster-whisper into the system Python environment**

```bash
pip install faster-whisper
```

Expected: Successfully installed faster-whisper and its dependencies (ctranslate2, tokenizers, etc.)

**Step 2: Verify it's importable**

```bash
python3 -c "from faster_whisper import WhisperModel; print('ok')"
```

Expected: `ok`

**Step 3: Update requirements.txt**

Replace `openai-whisper` with `faster-whisper`. The file should look like:

```
Flask==3.1.1
faster-whisper
ollama
python-dotenv==1.0.1
pytest
pytest-flask
```

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: replace openai-whisper with faster-whisper in requirements"
```

---

### Task 2: Update tests to use faster-whisper mocks (TDD — write failing tests first)

**Files:**
- Modify: `tests/test_transcriber.py`

**Background — API difference:**

| openai-whisper | faster-whisper |
|---|---|
| `whisper.load_model(name)` | `WhisperModel(name, device="cpu", compute_type="int8")` |
| `model.transcribe(path)` → `{"text": "..."}` | `model.transcribe(path, beam_size=5)` → `(segments_iter, info)` |
| result is a dict | result is a tuple; text comes from joining `s.text` for each segment |

**Step 1: Replace the entire contents of `tests/test_transcriber.py`**

```python
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
```

**Step 2: Run tests to verify they FAIL (transcriber.py still uses old API)**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_transcriber.py -v
```

Expected: All 4 tests FAIL — `ImportError` or `AssertionError` because `transcriber.py` still imports `whisper`, not `WhisperModel`.

**Step 3: Commit the failing tests**

```bash
git add tests/test_transcriber.py
git commit -m "test: update transcriber tests for faster-whisper API"
```

---

### Task 3: Update transcriber.py to use faster-whisper

**Files:**
- Modify: `transcriber.py`

**Step 1: Replace the entire contents of `transcriber.py`**

```python
from faster_whisper import WhisperModel

def transcribe(audio_path, model_name="small", output_path=None):
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    return text
```

**Step 2: Run transcriber tests to verify they pass**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_transcriber.py -v
```

Expected:
```
tests/test_transcriber.py::test_transcribe_returns_text PASSED
tests/test_transcriber.py::test_transcribe_saves_to_file PASSED
tests/test_transcriber.py::test_transcribe_joins_multiple_segments PASSED
tests/test_transcriber.py::test_transcribe_uses_correct_model_params PASSED

4 passed
```

**Step 3: Run full test suite to verify nothing else broke**

```bash
cd /home/pi/meeting-notes && python3 -m pytest --tb=short
```

Expected: All 24 tests pass (22 original + 2 new).

**Step 4: Commit**

```bash
git add transcriber.py
git commit -m "feat: switch transcriber to faster-whisper with int8 CPU quantization"
```

---

### Task 4: Restart service and pre-download model

**Step 1: Restart the service**

```bash
sudo systemctl restart meeting-notes
sudo systemctl status meeting-notes --no-pager | head -5
```

Expected: `active (running)`

**Step 2: Pre-download the medium model (optional but avoids delay on first real meeting)**

This downloads the CTranslate2-format model (~1.5GB) to `~/.cache/huggingface/`:

```bash
python3 -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8'); print('model ready')"
```

Expected: Downloads progress bars, then `model ready`. Takes a few minutes on first run.

**Step 3: Push to remote**

```bash
cd /home/pi/meeting-notes && git push
```
