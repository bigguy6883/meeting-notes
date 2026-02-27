# Summary & Highlights Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional speaker diarization and a richer 6-section summarization prompt — the service works without pyannote.audio installed, and diarization activates automatically when `HF_TOKEN` is set.

**Architecture:** Five code changes in sequence: (1) extend `transcriber.py` to return segment objects, (2) create `diarizer.py` with lazy pyannote import gated on HF_TOKEN, (3) add dual prompts to `summarizer.py`, (4) wire all three into `jobs.py`, (5) add UI status label. All faster-whisper compatible — segments are objects with `.start`/`.end`/`.text` attributes.

**Tech Stack:** Python 3.13, faster-whisper (already installed), pyannote.audio (optional, install manually), Ollama/llama3, pytest

---

### Task 1: Extend transcriber.py with return_segments mode

**Files:**
- Modify: `transcriber.py`
- Modify: `tests/test_transcriber.py`

**Background:** `transcriber.py` currently iterates the faster-whisper generator once to build text. For diarization we need the segment objects (with `.start`, `.end`, `.text`). The fix: collect the generator into a list first, then optionally return it alongside the text. Existing callers are unaffected — `return_segments` defaults to `False`.

**Step 1: Add two failing tests to `tests/test_transcriber.py`**

Append to the end of the existing file:

```python
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

**Step 2: Run to verify they fail**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_transcriber.py::test_transcribe_returns_segments_when_requested tests/test_transcriber.py::test_transcribe_still_returns_string_by_default -v
```

Expected: FAIL — `TypeError: cannot unpack non-iterable str object`

**Step 3: Replace `transcriber.py` with:**

```python
from faster_whisper import WhisperModel

def transcribe(audio_path, model_name="small", output_path=None, return_segments=False):
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments_gen, _ = model.transcribe(audio_path, beam_size=5)
    segments_list = list(segments_gen)
    text = " ".join(s.text.strip() for s in segments_list).strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    if return_segments:
        return text, segments_list
    return text
```

**Step 4: Run all transcriber tests**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_transcriber.py -v
```

Expected: 6 passed

**Step 5: Run full suite to check no regressions**

```bash
cd /home/pi/meeting-notes && python3 -m pytest --tb=short
```

Expected: all passed

**Step 6: Commit**

```bash
git add transcriber.py tests/test_transcriber.py
git commit -m "feat: add return_segments option to transcribe()"
```

---

### Task 2: Create diarizer.py with lazy optional pyannote import

**Files:**
- Create: `diarizer.py`
- Create: `tests/test_diarizer.py`

**Background:** `diarizer.py` merges pyannote speaker segments with Whisper segment timing. It checks for `HF_TOKEN` first — if absent, returns immediately without touching pyannote at all. The `from pyannote.audio import Pipeline` import is inside the function body, so pyannote never needs to be installed for the service to run. Segments are faster-whisper objects with `.start`, `.end`, `.text` attributes.

Mocking strategy for tests: inject a mock into `sys.modules` for `pyannote.audio` so the lazy import resolves without pyannote installed.

**Step 1: Create `tests/test_diarizer.py`**

```python
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
    Returns the mock module to inject into sys.modules.
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


# --- _find_speaker unit tests ---

def test_find_speaker_returns_matching_segment():
    segments = [("Speaker_00", 0.0, 5.0), ("Speaker_01", 5.0, 10.0)]
    assert _find_speaker(2.5, segments) == "Speaker_00"
    assert _find_speaker(7.5, segments) == "Speaker_01"


def test_find_speaker_falls_back_to_nearest_midpoint():
    # Gap between 5.0 and 6.0 — no segment contains 5.4
    # Speaker_00 midpoint=2.5 (dist 2.9), Speaker_01 midpoint=8.0 (dist 2.6) → nearest
    segments = [("Speaker_00", 0.0, 5.0), ("Speaker_01", 6.0, 10.0)]
    assert _find_speaker(5.4, segments) == "Speaker_01"


# --- diarize() integration tests ---

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
    # Simulate pyannote not installed: ImportError on import
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
```

**Step 2: Run to verify they fail**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_diarizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'diarizer'`

**Step 3: Create `diarizer.py`**

```python
import os


def _find_speaker(midpoint, pyannote_segments):
    """Return speaker whose segment contains midpoint, or nearest by midpoint distance."""
    for speaker, start, end in pyannote_segments:
        if start <= midpoint <= end:
            return speaker
    if pyannote_segments:
        return min(pyannote_segments, key=lambda s: abs((s[1] + s[2]) / 2 - midpoint))[0]
    return "Speaker_00"


def diarize(audio_path, whisper_segments, hf_token=None):
    """
    Merge pyannote speaker diarization with faster-whisper segment timing.

    whisper_segments: list of faster-whisper segment objects (.start, .end, .text)

    Returns (transcript: str, diarized: bool).
    Falls back to plain transcript (diarized=False) when:
    - HF_TOKEN is absent
    - pyannote.audio is not installed
    - only one speaker is detected
    - any exception occurs
    """
    plain_text = " ".join(seg.text.strip() for seg in whisper_segments)

    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        return plain_text, False

    try:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        diarization = pipeline(audio_path)

        pyannote_segs = [
            (speaker, turn.start, turn.end)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        if len(set(s[0] for s in pyannote_segs)) < 2:
            return plain_text, False

        lines = []
        current_speaker = None
        current_texts = []

        for seg in whisper_segments:
            midpoint = (seg.start + seg.end) / 2
            speaker = _find_speaker(midpoint, pyannote_segs)

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

**Step 4: Run diarizer tests**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_diarizer.py -v
```

Expected: 7 passed

**Step 5: Run full suite**

```bash
cd /home/pi/meeting-notes && python3 -m pytest --tb=short
```

Expected: all passed

**Step 6: Commit**

```bash
git add diarizer.py tests/test_diarizer.py
git commit -m "feat: add speaker diarization module with optional pyannote.audio"
```

---

### Task 3: Add dual prompts to summarizer.py

**Files:**
- Modify: `summarizer.py`
- Modify: `tests/test_summarizer.py`

**Background:** Add a `diarized=False` parameter. When `True`, use the rich 6-section prompt. Existing callers pass no argument and get the current 3-section behaviour.

**Step 1: Add three failing tests to `tests/test_summarizer.py`**

Append to the existing file:

```python
def test_summarize_uses_diarized_prompt_when_diarized():
    mock_response = MagicMock()
    mock_response.message.content = "Summary"
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize("Speaker_00: Hello", model="llama3", diarized=True)
    messages = mock_chat.call_args[1]["messages"]
    prompt = " ".join(m["content"] for m in messages)
    assert "SPEAKERS" in prompt
    assert "KEY HIGHLIGHTS" in prompt
    assert "OPEN QUESTIONS" in prompt

def test_summarize_uses_simple_prompt_when_not_diarized():
    mock_response = MagicMock()
    mock_response.message.content = "Summary"
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize("Hello world", model="llama3", diarized=False)
    messages = mock_chat.call_args[1]["messages"]
    prompt = " ".join(m["content"] for m in messages)
    assert "SPEAKERS" not in prompt

def test_summarize_defaults_to_simple_prompt():
    mock_response = MagicMock()
    mock_response.message.content = "Summary"
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize("Hello world", model="llama3")
    messages = mock_chat.call_args[1]["messages"]
    prompt = " ".join(m["content"] for m in messages)
    assert "SPEAKERS" not in prompt
```

**Step 2: Run to verify they fail**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_summarizer.py -v
```

Expected: 3 new tests FAIL — `TypeError: summarize() got unexpected keyword argument 'diarized'`

**Step 3: Replace `summarizer.py` with:**

```python
import ollama

PROMPT_SIMPLE = """You are a meeting assistant. Analyze this transcript and provide:

1. SUMMARY (3-5 bullet points of main topics discussed)
2. ACTION ITEMS (person: task, or "None identified" if none)
3. KEY DECISIONS (or "None identified" if none)

Transcript:
{transcript}"""

PROMPT_DIARIZED = """You are a meeting assistant. Analyze this transcript and provide:

1. SUMMARY (3-5 bullets — include specific numbers, dates, names, and commitments)

2. SPEAKERS — for each speaker:
   - Apparent role/name if identifiable
   - Interaction type: decision-maker | facilitator | questioner | contributor | dissenter
   - 1-line characterization of their style

3. KEY HIGHLIGHTS — concrete moments worth noting (specific quotes, commitments,
   surprises, or disagreements — not just topic labels)

4. ACTION ITEMS — person: task, deadline if mentioned (or "None identified")

5. KEY DECISIONS — what was decided and the brief rationale behind it

6. OPEN QUESTIONS — unresolved issues or follow-ups with no clear owner

Transcript:
{transcript}"""


def summarize(transcript, model="llama3", diarized=False):
    template = PROMPT_DIARIZED if diarized else PROMPT_SIMPLE
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": template.format(transcript=transcript)}]
    )
    return response.message.content.strip()
```

**Step 4: Run all summarizer tests**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_summarizer.py -v
```

Expected: 5 passed

**Step 5: Run full suite**

```bash
cd /home/pi/meeting-notes && python3 -m pytest --tb=short
```

Expected: all passed

**Step 6: Commit**

```bash
git add summarizer.py tests/test_summarizer.py
git commit -m "feat: add diarized prompt with speaker roles and key highlights"
```

---

### Task 4: Wire diarization into the jobs pipeline

**Files:**
- Modify: `jobs.py`
- Modify: `tests/test_jobs.py`

**Background:** Four changes to `jobs.py`: (1) import `diarize`, (2) add `DIARIZING` to `JobStatus` and `_INTERRUPTED_STATUSES`, (3) call `transcribe(..., return_segments=True)` to get `(text, segments)`, (4) insert diarize step before summarize. Tests that mock `jobs.transcribe` must be updated from `return_value="transcript text"` to `return_value=("transcript text", [])`, and `jobs.diarize` must be mocked too.

**Step 1: Update mocks in `tests/test_jobs.py`**

Three tests currently patch `jobs.transcribe` with `return_value="transcript text"`. Update all three to return a tuple and add a `jobs.diarize` mock:

Find and replace each occurrence of:
```python
with patch("jobs.transcribe", return_value="transcript text"), \
     patch("jobs.summarize", return_value="summary text"), \
     patch("jobs.send_notes"):
```

Replace with:
```python
with patch("jobs.transcribe", return_value=("transcript text", [])), \
     patch("jobs.diarize", return_value=("transcript text", False)), \
     patch("jobs.summarize", return_value="summary text"), \
     patch("jobs.send_notes"):
```

There are 3 such blocks in the file (in `test_process_job_runs_pipeline`, `test_process_job_sets_transcript_path_and_summary`, and a third). Update all of them.

Also add to `test_startup_marks_interrupted_jobs_as_error`: after `jm._recover()`, assert the DIARIZING status is also recovered. Find the existing test and update the manual status injection to also test `"diarizing"`:

```python
def test_startup_marks_interrupted_jobs_as_error():
    jm = JobManager(":memory:")
    job_id = jm.create_job("meeting_interrupted")
    jm._db.execute("UPDATE jobs SET status='diarizing' WHERE id=?", (job_id,))
    jm._db.commit()
    jm._recover()
    job = jm.get_job(job_id)
    assert job["status"] == JobStatus.ERROR
    assert job["error"] == "interrupted by restart"
```

**Step 2: Run jobs tests to verify they fail**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_jobs.py -v
```

Expected: failures because `jobs.transcribe` still returns a string and `jobs.diarize` doesn't exist yet.

**Step 3: Update `jobs.py`**

Make four changes:

**(a)** Add import after existing imports at top:
```python
from diarizer import diarize
```

**(b)** Add `DIARIZING` to `JobStatus` enum:
```python
class JobStatus(str, Enum):
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    SUMMARIZING = "summarizing"
    EMAILING = "emailing"
    DONE = "done"
    ERROR = "error"
```

**(c)** Add `"diarizing"` to `_INTERRUPTED_STATUSES`:
```python
_INTERRUPTED_STATUSES = ("pending", "transcribing", "diarizing", "summarizing", "emailing")
```

**(d)** Replace the transcription + summarization lines in `process()`:

Old:
```python
self._set_status(job_id, JobStatus.TRANSCRIBING)
transcript = transcribe(audio_path, model_name=whisper_model, output_path=transcript_path)

self._set_status(job_id, JobStatus.SUMMARIZING)
summary = summarize(transcript, model=ollama_model)
```

New:
```python
self._set_status(job_id, JobStatus.TRANSCRIBING)
transcript_text, whisper_segments = transcribe(
    audio_path, model_name=whisper_model,
    output_path=transcript_path, return_segments=True
)

self._set_status(job_id, JobStatus.DIARIZING)
transcript, diarized = diarize(audio_path, whisper_segments)

self._set_status(job_id, JobStatus.SUMMARIZING)
summary = summarize(transcript, model=ollama_model, diarized=diarized)
```

**Step 4: Run jobs tests**

```bash
cd /home/pi/meeting-notes && python3 -m pytest tests/test_jobs.py -v
```

Expected: all passed

**Step 5: Run full suite**

```bash
cd /home/pi/meeting-notes && python3 -m pytest --tb=short
```

Expected: all passed

**Step 6: Commit**

```bash
git add jobs.py tests/test_jobs.py
git commit -m "feat: wire speaker diarization into processing pipeline"
```

---

### Task 5: Add diarizing status to the UI + env example

**Files:**
- Modify: `templates/index.html`
- Modify: `.env.example` (if it exists, or note to create it)

**Step 1: Find `STATUS_LABELS` in `templates/index.html`**

Search for `STATUS_LABELS` — it's a JS object near the top of the `<script>` section. It looks like:
```javascript
const STATUS_LABELS = {
  pending: 'Pending',
  transcribing: 'Transcribing...',
  summarizing: 'Summarizing...',
  ...
};
```

Add one line after `transcribing`:
```javascript
  diarizing: 'Diarizing...',
```

**Step 2: Check for .env.example**

```bash
ls /home/pi/meeting-notes/.env.example 2>/dev/null || echo "not found"
```

If it exists, append `HF_TOKEN=your_huggingface_token_here`.
If it doesn't exist, skip — just note in the commit message that HF_TOKEN must be added to `.env` manually.

**Step 3: Run full suite one final time**

```bash
cd /home/pi/meeting-notes && python3 -m pytest --tb=short
```

Expected: all passed

**Step 4: Restart service and verify**

```bash
sudo systemctl restart meeting-notes && sudo systemctl status meeting-notes --no-pager | head -5
```

Expected: `active (running)`

**Step 5: Commit and push**

```bash
git add templates/index.html
git add .env.example 2>/dev/null || true
git commit -m "feat: show Diarizing status in UI, document HF_TOKEN env var"
git push
```
