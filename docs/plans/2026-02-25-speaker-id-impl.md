# Speaker ID Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace pyannote (never installed) with resemblyzer for speaker diarization, and fix the bug where the labeled transcript is never saved to disk.

**Architecture:** Rewrite `diarizer.py` to use resemblyzer speaker embeddings + agglomerative clustering instead of pyannote. Fix `jobs.py` to overwrite `transcript_path` with the labeled transcript when diarization succeeds. All other pipeline stages (transcribe, summarize, email) are unchanged.

**Tech Stack:** resemblyzer (~17MB model, no PyTorch), scikit-learn AgglomerativeClustering, ffmpeg (already installed) for MP3→WAV conversion.

---

### Task 1: Install dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Install packages**

```bash
cd /home/pi/meeting-notes
pip install resemblyzer scikit-learn
```

Expected: both install successfully. resemblyzer will download its ~17MB model on first use (not now).

**Step 2: Add to requirements.txt**

Add these two lines to `requirements.txt`:
```
resemblyzer
scikit-learn
```

**Step 3: Verify imports work**

```bash
python3 -c "from resemblyzer import VoiceEncoder; from sklearn.cluster import AgglomerativeClustering; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add resemblyzer and scikit-learn for speaker diarization"
```

---

### Task 2: Rewrite diarizer.py

**Files:**
- Modify: `diarizer.py`

Replace the entire file with the following implementation:

**Step 1: Write the new diarizer.py**

```python
import os
import subprocess
import tempfile

import numpy as np


def _convert_to_wav(audio_path):
    """Convert audio to 16kHz mono WAV via ffmpeg. Returns temp WAV path (caller must delete)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp.name],
        capture_output=True,
        check=True,
    )
    return tmp.name


def diarize(audio_path, whisper_segments):
    """
    Assign speaker labels to Whisper segments using resemblyzer embeddings.

    whisper_segments: list of segment objects with .start, .end, .text

    Returns (transcript: str, diarized: bool).
    Falls back to plain transcript (diarized=False) when:
    - resemblyzer or scikit-learn is not installed
    - fewer than 2 speakers are detected
    - any exception occurs
    """
    plain_text = " ".join(seg.text.strip() for seg in whisper_segments)

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from sklearn.cluster import AgglomerativeClustering

        wav_path = _convert_to_wav(audio_path)
        try:
            wav = preprocess_wav(wav_path)
        finally:
            os.unlink(wav_path)

        encoder = VoiceEncoder()
        embeddings = []
        valid_indices = []
        for i, seg in enumerate(whisper_segments):
            start = int(seg.start * 16000)
            end = int(seg.end * 16000)
            chunk = wav[start:end]
            if len(chunk) < 1600:  # skip segments shorter than 0.1s
                continue
            embeddings.append(encoder.embed_utterance(chunk))
            valid_indices.append(i)

        if len(embeddings) < 2:
            return plain_text, False

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.6,
            metric="cosine",
            linkage="complete",
        ).fit_predict(np.array(embeddings))

        if len(set(labels)) < 2:
            return plain_text, False

        speaker_map = {c: f"Speaker_{i:02d}" for i, c in enumerate(sorted(set(labels)))}
        segment_speakers = ["Speaker_00"] * len(whisper_segments)
        for idx, label in zip(valid_indices, labels):
            segment_speakers[idx] = speaker_map[label]

        lines = []
        current_speaker = None
        current_texts = []
        for i, seg in enumerate(whisper_segments):
            speaker = segment_speakers[i]
            if speaker != current_speaker:
                if current_texts and current_speaker:
                    lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                current_speaker = speaker
                current_texts = [seg.text.strip()]
            else:
                current_texts.append(seg.text.strip())

        if current_texts and current_speaker:
            lines.append(f"{current_speaker}: {' '.join(current_texts)}")

        return "\n".join(lines), True

    except Exception:
        return plain_text, False
```

**Step 2: Commit**

```bash
git add diarizer.py
git commit -m "feat: replace pyannote with resemblyzer for speaker diarization"
```

---

### Task 3: Rewrite test_diarizer.py

**Files:**
- Modify: `tests/test_diarizer.py`

The old tests mock `pyannote.audio` and test `_find_speaker` (now removed). Replace the entire file.

**Step 1: Write the new test file**

```python
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
```

**Step 2: Run tests to verify they pass**

```bash
cd /home/pi/meeting-notes
pytest tests/test_diarizer.py -v
```

Expected: 6 tests pass.

**Step 3: Commit**

```bash
git add tests/test_diarizer.py
git commit -m "test: update diarizer tests for resemblyzer implementation"
```

---

### Task 4: Fix jobs.py — save labeled transcript to disk

**Files:**
- Modify: `jobs.py:95-97`

**Step 1: Write the failing test first**

In `tests/test_jobs.py`, add this test after `test_process_job_sets_transcript_path_and_summary`:

```python
def test_process_job_writes_diarized_transcript(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake audio")

    jm = JobManager(":memory:")
    job_id = jm.create_job("meeting_20260218_1030")

    with patch("jobs.transcribe", return_value=("plain text", [])), \
         patch("jobs.diarize", return_value=("Speaker_00: Hello\nSpeaker_01: World", True)), \
         patch("jobs.summarize", return_value="summary"), \
         patch("jobs.send_notes"):
        jm.process(
            job_id=job_id,
            audio_path=str(audio),
            transcript_dir=str(tmp_path),
            gmail_user="u@g.com",
            gmail_password="pw",
            to_address="u@g.com",
            summary_model="llama-3.3-70b-versatile"
        )

    job = jm.get_job(job_id)
    transcript_contents = open(job["transcript_path"]).read()
    assert "Speaker_00: Hello" in transcript_contents
    assert "Speaker_01: World" in transcript_contents
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_jobs.py::test_process_job_writes_diarized_transcript -v
```

Expected: FAIL — transcript file contains plain text, not speaker labels.

**Step 3: Fix jobs.py**

In `jobs.py`, find the block after `diarize()` is called (around line 96) and add the transcript overwrite:

Old code:
```python
            self._set_status(job_id, JobStatus.DIARIZING)
            transcript, diarized = diarize(audio_path, whisper_segments)

            self._set_status(job_id, JobStatus.SUMMARIZING)
```

New code:
```python
            self._set_status(job_id, JobStatus.DIARIZING)
            transcript, diarized = diarize(audio_path, whisper_segments)
            if diarized:
                with open(transcript_path, "w") as f:
                    f.write(transcript)

            self._set_status(job_id, JobStatus.SUMMARIZING)
```

**Step 4: Run the new test to verify it passes**

```bash
pytest tests/test_jobs.py::test_process_job_writes_diarized_transcript -v
```

Expected: PASS.

**Step 5: Run all tests**

```bash
pytest -v
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add jobs.py tests/test_jobs.py
git commit -m "fix: save diarized transcript to disk when speaker labels detected"
```
