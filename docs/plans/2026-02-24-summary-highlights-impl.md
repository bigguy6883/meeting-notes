# Summary & Highlights Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pyannote.audio speaker diarization and a richer 6-section summarization prompt so meeting notes identify who spoke, how they engaged, and capture concrete highlights instead of vague topic labels.

**Architecture:** Add a `diarizer.py` module between transcription and summarization. Transcriber gains a `return_segments=True` mode to expose Whisper's timing data (needed for the speaker-to-text merge). Diarizer runs pyannote, merges speaker segments with Whisper segments by midpoint, and returns a speaker-labeled transcript. Summarizer picks between two prompts based on whether diarization succeeded.

**Tech Stack:** pyannote.audio 3.x, openai-whisper (already installed), Ollama/llama3 (already running), HuggingFace free account for model download.

---

### Task 1: Add pyannote.audio to requirements and HF_TOKEN to env

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`

**Step 1: Add pyannote.audio to requirements**

```
# requirements.txt — add after ollama:
pyannote.audio
```

**Step 2: Add HF_TOKEN to env example**

```
# .env.example — add at bottom:
HF_TOKEN=your_huggingface_token_here
```

**Step 3: Commit**

```bash
git add requirements.txt .env.example
git commit -m "feat: add pyannote.audio dependency and HF_TOKEN env var"
```

> **Note for the developer:** Before running the app, create a free account at huggingface.co, generate a token at huggingface.co/settings/tokens, and add it to your `.env` file. Also accept the model license at huggingface.co/pyannote/speaker-diarization-3.1. pyannote downloads ~600MB of model weights on first run, cached in `~/.cache/torch/`.

---

### Task 2: Extend transcriber.py to return Whisper segments

The merge step in diarizer needs Whisper's segment-level timing (`start`, `end`, `text` per phrase). Add an opt-in `return_segments` parameter — existing callers are unaffected.

**Files:**
- Modify: `transcriber.py`
- Modify: `tests/test_transcriber.py`

**Step 1: Write the failing tests**

Add to `tests/test_transcriber.py`:

```python
def test_transcribe_returns_segments_when_requested(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()
    mock_result = {
        "text": " Hello world",
        "segments": [{"start": 0.0, "end": 3.0, "text": "Hello world"}]
    }
    with patch("transcriber.whisper.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_result
        mock_load.return_value = mock_model
        text, segments = transcribe(audio_file, model_name="tiny", return_segments=True)
    assert text == "Hello world"
    assert segments == [{"start": 0.0, "end": 3.0, "text": "Hello world"}]

def test_transcribe_still_returns_string_by_default(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "w").close()
    mock_result = {"text": " Hello world", "segments": []}
    with patch("transcriber.whisper.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_result
        mock_load.return_value = mock_model
        result = transcribe(audio_file, model_name="tiny")
    assert isinstance(result, str)
    assert result == "Hello world"
```

**Step 2: Run to verify they fail**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_transcriber.py -v
```

Expected: `TypeError` — `transcribe()` doesn't accept `return_segments`

**Step 3: Update transcriber.py**

```python
import whisper

def transcribe(audio_path, model_name="medium", output_path=None, return_segments=False):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    if return_segments:
        return text, result.get("segments", [])
    return text
```

**Step 4: Run all transcriber tests**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_transcriber.py -v
```

Expected: all PASS (4 tests)

**Step 5: Commit**

```bash
git add transcriber.py tests/test_transcriber.py
git commit -m "feat: add return_segments option to transcribe()"
```

---

### Task 3: Create diarizer.py

Runs pyannote on the audio file, merges speaker segments with Whisper segments by midpoint, returns a labeled transcript string and a `diarized` boolean. Falls back gracefully on missing token, single speaker, or any exception.

**Files:**
- Create: `diarizer.py`
- Create: `tests/test_diarizer.py`

**Step 1: Write the failing tests**

Create `tests/test_diarizer.py`:

```python
from unittest.mock import patch, MagicMock
from diarizer import diarize, _find_speaker


def test_find_speaker_returns_matching_segment():
    segments = [("Speaker_00", 0.0, 5.0), ("Speaker_01", 5.0, 10.0)]
    assert _find_speaker(2.5, segments) == "Speaker_00"
    assert _find_speaker(7.5, segments) == "Speaker_01"


def test_find_speaker_falls_back_to_nearest_midpoint():
    # Gap between 5.0 and 6.0 — no segment contains 5.4
    # Speaker_00 midpoint = 2.5, distance = 2.9
    # Speaker_01 midpoint = 8.0, distance = 2.6 → nearest
    segments = [("Speaker_00", 0.0, 5.0), ("Speaker_01", 6.0, 10.0)]
    assert _find_speaker(5.4, segments) == "Speaker_01"


def test_diarize_returns_plain_text_without_token():
    whisper_segs = [
        {"start": 0.0, "end": 3.0, "text": "Hello world"},
        {"start": 3.0, "end": 6.0, "text": "Goodbye world"},
    ]
    text, diarized = diarize("/fake/audio.mp3", whisper_segs, hf_token="")
    assert diarized is False
    assert "Hello world" in text
    assert "Goodbye world" in text


def test_diarize_labels_two_speakers():
    whisper_segs = [
        {"start": 0.0, "end": 3.0, "text": "Hello world"},
        {"start": 5.0, "end": 8.0, "text": "Goodbye world"},
    ]

    turn_0 = MagicMock(); turn_0.start = 0.0; turn_0.end = 4.0
    turn_1 = MagicMock(); turn_1.start = 4.0; turn_1.end = 9.0

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [
        (turn_0, None, "Speaker_00"),
        (turn_1, None, "Speaker_01"),
    ]
    mock_pipeline = MagicMock(return_value=mock_diarization)

    with patch("diarizer.Pipeline.from_pretrained", return_value=mock_pipeline):
        text, diarized = diarize("/fake/audio.mp3", whisper_segs, hf_token="fake_token")

    assert diarized is True
    assert "Speaker_00:" in text
    assert "Speaker_01:" in text
    assert "Hello world" in text
    assert "Goodbye world" in text


def test_diarize_falls_back_on_single_speaker():
    whisper_segs = [{"start": 0.0, "end": 3.0, "text": "Hello world"}]

    turn = MagicMock(); turn.start = 0.0; turn.end = 3.0
    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [(turn, None, "Speaker_00")]
    mock_pipeline = MagicMock(return_value=mock_diarization)

    with patch("diarizer.Pipeline.from_pretrained", return_value=mock_pipeline):
        text, diarized = diarize("/fake/audio.mp3", whisper_segs, hf_token="fake_token")

    assert diarized is False


def test_diarize_falls_back_on_exception():
    whisper_segs = [{"start": 0.0, "end": 3.0, "text": "Hello"}]
    with patch("diarizer.Pipeline.from_pretrained", side_effect=Exception("model error")):
        text, diarized = diarize("/fake/audio.mp3", whisper_segs, hf_token="fake_token")
    assert diarized is False
    assert "Hello" in text
```

**Step 2: Run to verify they fail**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_diarizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'diarizer'`

**Step 3: Create diarizer.py**

```python
import os
from pyannote.audio import Pipeline


def _find_speaker(midpoint, pyannote_segments):
    """Return the speaker whose segment contains midpoint, or the nearest by midpoint distance."""
    for speaker, start, end in pyannote_segments:
        if start <= midpoint <= end:
            return speaker
    if pyannote_segments:
        return min(pyannote_segments, key=lambda s: abs((s[1] + s[2]) / 2 - midpoint))[0]
    return "Speaker_00"


def diarize(audio_path, whisper_segments, hf_token=None):
    """
    Merge pyannote speaker diarization with Whisper segment timing.

    Returns (labeled_transcript: str, diarized: bool).
    Falls back to plain transcript (diarized=False) if:
    - no HF token provided
    - only one speaker detected
    - any exception occurs
    """
    plain_text = " ".join(seg["text"].strip() for seg in whisper_segments)

    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        return plain_text, False

    try:
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
            midpoint = (seg["start"] + seg["end"]) / 2
            speaker = _find_speaker(midpoint, pyannote_segs)

            if speaker != current_speaker:
                if current_texts and current_speaker:
                    lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                current_speaker = speaker
                current_texts = [seg["text"].strip()]
            else:
                current_texts.append(seg["text"].strip())

        if current_texts and current_speaker:
            lines.append(f"{current_speaker}: {' '.join(current_texts)}")

        return "\n".join(lines), True

    except Exception:
        return plain_text, False
```

**Step 4: Run tests**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_diarizer.py -v
```

Expected: all 6 PASS

**Step 5: Commit**

```bash
git add diarizer.py tests/test_diarizer.py
git commit -m "feat: add speaker diarization module with pyannote.audio"
```

---

### Task 4: Update summarizer.py with dual prompts

Add a `diarized` parameter. When `True`, use the rich 6-section prompt that captures speaker interaction types, concrete highlights, and open questions.

**Files:**
- Modify: `summarizer.py`
- Modify: `tests/test_summarizer.py`

**Step 1: Write the failing tests**

Add to `tests/test_summarizer.py`:

```python
def test_summarize_uses_diarized_prompt_when_diarized():
    mock_response = MagicMock()
    mock_response.message.content = "Summary"
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize("Speaker_00: Hello", model="llama3", diarized=True)
    messages = mock_chat.call_args[1]["messages"]
    assert any("SPEAKERS" in m["content"] for m in messages)
    assert any("KEY HIGHLIGHTS" in m["content"] for m in messages)
    assert any("OPEN QUESTIONS" in m["content"] for m in messages)


def test_summarize_uses_simple_prompt_when_not_diarized():
    mock_response = MagicMock()
    mock_response.message.content = "Summary"
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize("Hello world", model="llama3", diarized=False)
    messages = mock_chat.call_args[1]["messages"]
    assert not any("SPEAKERS" in m["content"] for m in messages)


def test_summarize_defaults_to_simple_prompt():
    mock_response = MagicMock()
    mock_response.message.content = "Summary"
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize("Hello world", model="llama3")
    messages = mock_chat.call_args[1]["messages"]
    assert not any("SPEAKERS" in m["content"] for m in messages)
```

**Step 2: Run to verify they fail**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_summarizer.py -v
```

Expected: FAIL — `summarize()` doesn't accept `diarized` kwarg

**Step 3: Update summarizer.py**

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
cd /home/pi/meeting-notes && python -m pytest tests/test_summarizer.py -v
```

Expected: all 5 PASS

**Step 5: Commit**

```bash
git add summarizer.py tests/test_summarizer.py
git commit -m "feat: add diarized prompt with speaker roles and key highlights"
```

---

### Task 5: Wire diarization into the jobs pipeline

Add `DIARIZING` status to `JobStatus` and `_INTERRUPTED_STATUSES`. Insert the diarization step between transcription and summarization in `process()`.

**Files:**
- Modify: `jobs.py`
- Modify: `tests/test_jobs.py` (check existing tests still pass — no new test needed, behavior is covered by diarizer tests)

**Step 1: Check existing jobs tests still pass before touching anything**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_jobs.py -v
```

Expected: all PASS (baseline)

**Step 2: Update jobs.py**

Four changes — mark each with a comment as you go:

1. Add import at top:
```python
from diarizer import diarize
```

2. Add `DIARIZING` to `JobStatus`:
```python
class JobStatus(str, Enum):
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"       # ← new
    SUMMARIZING = "summarizing"
    EMAILING = "emailing"
    DONE = "done"
    ERROR = "error"
```

3. Add `"diarizing"` to `_INTERRUPTED_STATUSES`:
```python
_INTERRUPTED_STATUSES = ("pending", "transcribing", "diarizing", "summarizing", "emailing")
```

4. Insert diarization between transcription and summarization in `process()`:
```python
# replace the two lines:
#   self._set_status(job_id, JobStatus.SUMMARIZING)
#   summary = summarize(transcript, model=ollama_model)
# with:

self._set_status(job_id, JobStatus.DIARIZING)
transcript, diarized = diarize(audio_path, whisper_segments)

self._set_status(job_id, JobStatus.SUMMARIZING)
summary = summarize(transcript, model=ollama_model, diarized=diarized)
```

And update the transcription call above it to use `return_segments=True`:
```python
# replace:
#   transcript = transcribe(audio_path, model_name=whisper_model, output_path=transcript_path)
# with:
transcript_text, whisper_segments = transcribe(
    audio_path, model_name=whisper_model, output_path=transcript_path, return_segments=True
)
```

The full updated `process()` method:

```python
def process(self, job_id, audio_path, transcript_dir, gmail_user,
            gmail_password, to_address, whisper_model, ollama_model):
    try:
        transcript_path = os.path.join(
            transcript_dir,
            os.path.basename(audio_path).replace(".mp3", ".txt")
        )
        self._set_status(job_id, JobStatus.TRANSCRIBING)
        transcript_text, whisper_segments = transcribe(
            audio_path, model_name=whisper_model,
            output_path=transcript_path, return_segments=True
        )

        self._set_status(job_id, JobStatus.DIARIZING)
        transcript, diarized = diarize(audio_path, whisper_segments)

        self._set_status(job_id, JobStatus.SUMMARIZING)
        summary = summarize(transcript, model=ollama_model, diarized=diarized)

        self._set_status(job_id, JobStatus.EMAILING)
        label = self._db.execute(
            "SELECT label FROM jobs WHERE id=?", (job_id,)
        ).fetchone()[0]
        send_notes(
            gmail_user=gmail_user,
            gmail_password=gmail_password,
            to_address=to_address,
            meeting_label=label,
            summary=summary,
            transcript_path=transcript_path
        )
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, summary=?, transcript_path=? WHERE id=?",
                (JobStatus.DONE, summary, transcript_path, job_id)
            )
            self._db.commit()

    except Exception as e:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, error=? WHERE id=?",
                (JobStatus.ERROR, str(e), job_id)
            )
            self._db.commit()
```

**Step 3: Run all jobs tests**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_jobs.py -v
```

Expected: all PASS (jobs tests mock transcribe and summarize, so they should still pass — if any fail, check that the mocks match the new signatures)

**Step 4: Run full test suite**

```bash
cd /home/pi/meeting-notes && python -m pytest -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add jobs.py
git commit -m "feat: wire speaker diarization into processing pipeline"
```

---

### Task 6: Add diarizing status to the UI

One-line change to `index.html` so the job list shows "Diarizing..." instead of the raw status string.

**Files:**
- Modify: `templates/index.html`

**Step 1: Find the STATUS_LABELS object in index.html (around line 56) and add the new entry**

```javascript
const STATUS_LABELS = {
  pending: 'Pending',
  transcribing: 'Transcribing...',
  diarizing: 'Diarizing...',        // ← add this line
  summarizing: 'Summarizing...',
  emailing: 'Sending email...',
  done: 'Email sent',
  error: 'Error'
};
```

**Step 2: Run the full test suite one final time**

```bash
cd /home/pi/meeting-notes && python -m pytest -v
```

Expected: all PASS

**Step 3: Commit**

```bash
git add templates/index.html
git commit -m "feat: show Diarizing status in job list UI"
```

---

### Task 7: Install pyannote.audio and smoke test

**Step 1: Install the new dependency**

```bash
cd /home/pi/meeting-notes && pip install pyannote.audio
```

Expected: installs cleanly (pulls in torch, torchaudio — this will take a few minutes)

**Step 2: Verify HF_TOKEN is in .env**

```bash
grep HF_TOKEN /home/pi/meeting-notes/.env
```

Expected: `HF_TOKEN=hf_...` (your token)

**Step 3: Smoke test diarizer import**

```bash
cd /home/pi/meeting-notes && python -c "from diarizer import diarize; print('OK')"
```

Expected: `OK`

**Step 4: Run full test suite one final time**

```bash
cd /home/pi/meeting-notes && python -m pytest -v
```

Expected: all PASS

**Step 5: Restart the service and verify**

```bash
sudo systemctl restart meeting-notes && sudo systemctl status meeting-notes
```

Expected: `active (running)`
