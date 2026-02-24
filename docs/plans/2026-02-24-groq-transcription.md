# Groq Transcription Switch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace faster_whisper with the Groq API for near-instant transcription while keeping the diarization pipeline intact.

**Architecture:** `transcriber.py` calls Groq's `whisper-large-v3-turbo` with `verbose_json` response format to get segment timestamps, wraps them in a `Segment` dataclass compatible with `diarizer.py`. `whisper_model` parameter is removed from `jobs.py` and `app.py` since the Groq model is hardcoded. `faster-whisper` dependency removed.

**Tech Stack:** groq Python SDK, pytest, unittest.mock

---

### Task 1: Rewrite `transcriber.py` and its tests

**Files:**
- Modify: `transcriber.py`
- Modify: `tests/test_transcriber.py`

**Step 1: Write the failing tests first**

Replace `tests/test_transcriber.py` with:

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from transcriber import transcribe, Segment


def _make_mock_response(segments_data):
    """
    segments_data: list of (text, start, end)
    Returns a mock Groq verbose_json transcription response.
    """
    response = MagicMock()
    segs = []
    texts = []
    for text, start, end in segments_data:
        s = MagicMock()
        s.text = text
        s.start = start
        s.end = end
        segs.append(s)
        texts.append(text.strip())
    response.text = " ".join(texts)
    response.segments = segs
    return response


def test_transcribe_returns_text(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([("Hello world.", 0.0, 2.0)])
        result = transcribe(audio_file)
    assert result == "Hello world."


def test_transcribe_saves_to_file(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    out_file = str(tmp_path / "transcript.txt")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([("Meeting content here.", 0.0, 3.0)])
        transcribe(audio_file, output_path=out_file)
    assert os.path.exists(out_file)
    with open(out_file) as f:
        assert f.read() == "Meeting content here."


def test_transcribe_returns_segments_when_requested(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([("Hello world.", 0.0, 2.5)])
        text, segments = transcribe(audio_file, return_segments=True)
    assert text == "Hello world."
    assert len(segments) == 1
    assert isinstance(segments[0], Segment)
    assert segments[0].start == 0.0
    assert segments[0].end == 2.5
    assert segments[0].text == "Hello world."


def test_transcribe_returns_multiple_segments(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([
                ("First segment.", 0.0, 2.0),
                ("Second segment.", 2.0, 4.0),
            ])
        text, segments = transcribe(audio_file, return_segments=True)
    assert text == "First segment. Second segment."
    assert len(segments) == 2
    assert segments[1].start == 2.0
    assert segments[1].end == 4.0


def test_transcribe_uses_correct_model_and_format(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.audio.transcriptions.create.return_value = \
            _make_mock_response([("text", 0.0, 1.0)])
        transcribe(audio_file)
    call_kwargs = mock_instance.audio.transcriptions.create.call_args[1]
    assert call_kwargs["model"] == "whisper-large-v3-turbo"
    assert call_kwargs["response_format"] == "verbose_json"


def test_transcribe_still_returns_string_by_default(tmp_path):
    audio_file = str(tmp_path / "test.mp3")
    open(audio_file, "wb").close()
    with patch("transcriber.Groq") as mock_cls:
        mock_cls.return_value.audio.transcriptions.create.return_value = \
            _make_mock_response([("Hello", 0.0, 1.0)])
        result = transcribe(audio_file)
    assert isinstance(result, str)
```

**Step 2: Run tests to confirm they fail**

Run: `cd /home/pi/meeting-notes && python -m pytest tests/test_transcriber.py -v`
Expected: FAIL — `ImportError: cannot import name 'Segment'` or similar

**Step 3: Write the new `transcriber.py`**

```python
import dataclasses
from groq import Groq


@dataclasses.dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(audio_path, output_path=None, return_segments=False):
    client = Groq()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
    text = response.text.strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    if return_segments:
        segments = [Segment(start=s.start, end=s.end, text=s.text) for s in response.segments]
        return text, segments
    return text
```

**Step 4: Run tests to confirm they pass**

Run: `cd /home/pi/meeting-notes && python -m pytest tests/test_transcriber.py -v`
Expected: all 7 tests PASS

**Step 5: Commit**

```bash
cd /home/pi/meeting-notes && git add transcriber.py tests/test_transcriber.py
git commit -m "feat: replace faster_whisper with Groq API for transcription"
```

---

### Task 2: Update `jobs.py` and `tests/test_jobs.py`

Remove `whisper_model` parameter from `process()` and `process_async()`. Update the `transcribe()` call to match the new signature.

**Files:**
- Modify: `jobs.py`
- Modify: `tests/test_jobs.py`

**Step 1: Write failing tests**

In `tests/test_jobs.py`, remove `whisper_model="tiny"` from all `jm.process()` calls. The updated file:

```python
import time
import pytest
from unittest.mock import patch, MagicMock
from jobs import JobManager, JobStatus


def test_new_job_is_pending():
    jm = JobManager(":memory:")
    job_id = jm.create_job("meeting_20260218_1030")
    job = jm.get_job(job_id)
    assert job["status"] == JobStatus.PENDING
    assert job["label"] == "meeting_20260218_1030"


def test_jobs_list_returns_all():
    jm = JobManager(":memory:")
    jm.create_job("meeting_a")
    jm.create_job("meeting_b")
    assert len(jm.list_jobs()) == 2


def test_process_job_runs_pipeline(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake audio")

    jm = JobManager(":memory:")
    job_id = jm.create_job("meeting_20260218_1030")

    mock_summarize = MagicMock(return_value="summary text")
    with patch("jobs.transcribe", return_value=("transcript text", [])), \
         patch("jobs.diarize", return_value=("transcript text", False)), \
         patch("jobs.summarize", mock_summarize), \
         patch("jobs.send_notes"):
        jm.process(
            job_id=job_id,
            audio_path=str(audio),
            transcript_dir=str(tmp_path),
            gmail_user="u@g.com",
            gmail_password="pw",
            to_address="u@g.com",
            ollama_model="llama3"
        )

    job = jm.get_job(job_id)
    assert job["status"] == JobStatus.DONE
    mock_summarize.assert_called_once_with("transcript text", model="llama3", diarized=False)


def test_process_job_sets_error_on_failure(tmp_path):
    jm = JobManager(":memory:")
    job_id = jm.create_job("meeting_fail")
    with patch("jobs.transcribe", side_effect=Exception("groq failed")):
        jm.process(
            job_id=job_id,
            audio_path="/nonexistent.mp3",
            transcript_dir=str(tmp_path),
            gmail_user="u@g.com",
            gmail_password="pw",
            to_address="u@g.com",
            ollama_model="llama3"
        )
    job = jm.get_job(job_id)
    assert job["status"] == JobStatus.ERROR
    assert "groq failed" in job["error"]


def test_get_job_unknown_id_returns_none():
    jm = JobManager(":memory:")
    result = jm.get_job("nonexistent-id")
    assert result is None


def test_process_job_sets_transcript_path_and_summary(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake audio")

    jm = JobManager(":memory:")
    job_id = jm.create_job("meeting_20260218_1030")

    mock_summarize = MagicMock(return_value="summary text")
    with patch("jobs.transcribe", return_value=("transcript text", [])), \
         patch("jobs.diarize", return_value=("transcript text", False)), \
         patch("jobs.summarize", mock_summarize), \
         patch("jobs.send_notes"):
        jm.process(
            job_id=job_id,
            audio_path=str(audio),
            transcript_dir=str(tmp_path),
            gmail_user="u@g.com",
            gmail_password="pw",
            to_address="u@g.com",
            ollama_model="llama3"
        )

    job = jm.get_job(job_id)
    assert job["transcript_path"] is not None
    assert job["summary"] == "summary text"


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

**Step 2: Run tests to confirm they fail**

Run: `cd /home/pi/meeting-notes && python -m pytest tests/test_jobs.py -v`
Expected: FAIL — `TypeError: process() got an unexpected keyword argument 'whisper_model'` or similar (once jobs.py is not yet updated)

**Step 3: Update `jobs.py`**

Remove `whisper_model` from `process()` signature and its use in the `transcribe()` call. The updated `process()` and `process_async()`:

```python
def process(self, job_id, audio_path, transcript_dir, gmail_user,
            gmail_password, to_address, ollama_model):
    try:
        transcript_path = os.path.join(
            transcript_dir,
            os.path.basename(audio_path).replace(".mp3", ".txt")
        )
        self._set_status(job_id, JobStatus.TRANSCRIBING)
        transcript_text, whisper_segments = transcribe(
            audio_path,
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

def process_async(self, **kwargs):
    t = threading.Thread(target=self.process, kwargs=kwargs, daemon=True)
    t.start()
```

**Step 4: Run tests to confirm they pass**

Run: `cd /home/pi/meeting-notes && python -m pytest tests/test_jobs.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
cd /home/pi/meeting-notes && git add jobs.py tests/test_jobs.py
git commit -m "refactor: remove whisper_model param from jobs pipeline"
```

---

### Task 3: Update `app.py` and `tests/test_routes.py`

Remove `whisper_model` from `process_async()` calls. Add `GROQ_API_KEY` to required env vars. Remove `WHISPER_MODEL` reads.

**Files:**
- Modify: `app.py`
- Modify: `tests/test_routes.py`

**Step 1: Write failing tests**

Replace the fixture in `tests/test_routes.py` — remove `WHISPER_MODEL`, add `GROQ_API_KEY`:

```python
import pytest
from unittest.mock import patch, MagicMock
import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "test@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    monkeypatch.setenv("GMAIL_TO", "test@gmail.com")
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.setenv("MIC_DEVICE", "hw:0,0")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setattr(app_module, "RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "TRANSCRIPTS_DIR", str(tmp_path))
    from jobs import JobManager
    monkeypatch.setattr(app_module, "job_manager", JobManager(":memory:"))
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_start_recording(client):
    with patch.object(app_module.recorder, "start", return_value="/tmp/meeting.mp3"), \
         patch.object(app_module.recorder, "is_recording", return_value=False):
        resp = client.post("/api/start")
    assert resp.status_code == 200
    assert resp.json["status"] == "recording"


def test_stop_recording_creates_job(client):
    with patch.object(app_module.recorder, "stop", return_value="/tmp/meeting_20260218_1030.mp3"), \
         patch.object(app_module.job_manager, "process_async"):
        resp = client.post("/api/stop")
    assert resp.status_code == 200
    assert "job_id" in resp.json


def test_jobs_returns_list(client):
    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json, list)


def test_start_when_already_recording_returns_409(client):
    with patch.object(app_module.recorder, "is_recording", return_value=True):
        resp = client.post("/api/start")
    assert resp.status_code == 409
```

**Step 2: Run tests to confirm they fail**

Run: `cd /home/pi/meeting-notes && python -m pytest tests/test_routes.py -v`
Expected: FAIL (app.py still references WHISPER_MODEL / old process_async signature)

**Step 3: Update `app.py`**

Changes:
- Add `"GROQ_API_KEY"` to `_required_env`
- Remove `"WHISPER_MODEL"` reads
- Remove `whisper_model=` kwarg from both `process_async()` calls

Updated relevant sections:

```python
_required_env = ["GMAIL_USER", "GMAIL_APP_PASSWORD", "GMAIL_TO", "GROQ_API_KEY"]
```

Updated `stop_recording` route:

```python
@app.route("/api/stop", methods=["POST"])
def stop_recording():
    filepath = recorder.stop()
    if not filepath:
        return jsonify({"error": "Not recording"}), 409

    basename = os.path.basename(filepath).replace("meeting_", "").replace(".mp3", "")
    label = f"{basename[:8]} {basename[9:13]}" if len(basename) >= 13 else basename

    job_id = job_manager.create_job(label, audio_path=filepath)
    job_manager.process_async(
        job_id=job_id,
        audio_path=filepath,
        transcript_dir=TRANSCRIPTS_DIR,
        gmail_user=os.getenv("GMAIL_USER"),
        gmail_password=os.getenv("GMAIL_APP_PASSWORD"),
        to_address=os.getenv("GMAIL_TO"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
    )
    return jsonify({"status": "processing", "job_id": job_id})
```

Updated `retry_job` route:

```python
@app.route("/api/jobs/<job_id>/retry", methods=["POST"])
def retry_job(job_id):
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    if not job_manager.retry_job(job_id):
        return jsonify({"error": "Job is not in error state"}), 409
    job_manager.process_async(
        job_id=job_id,
        audio_path=job["audio_path"],
        transcript_dir=TRANSCRIPTS_DIR,
        gmail_user=os.getenv("GMAIL_USER"),
        gmail_password=os.getenv("GMAIL_APP_PASSWORD"),
        to_address=os.getenv("GMAIL_TO"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
    )
    return jsonify({"status": "retrying", "job_id": job_id})
```

**Step 4: Run tests to confirm they pass**

Run: `cd /home/pi/meeting-notes && python -m pytest tests/test_routes.py -v`
Expected: all tests PASS

**Step 5: Run full suite**

Run: `cd /home/pi/meeting-notes && python -m pytest -v`
Expected: all tests PASS

**Step 6: Commit**

```bash
cd /home/pi/meeting-notes && git add app.py tests/test_routes.py
git commit -m "refactor: remove WHISPER_MODEL, add GROQ_API_KEY to required env"
```

---

### Task 4: Update `requirements.txt` and `.env.example`

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`

**Step 1: Update `requirements.txt`**

Replace `faster-whisper` with `groq`:

```
Flask==3.1.1
groq
ollama
python-dotenv==1.0.1
pytest
pytest-flask
```

**Step 2: Update `.env.example`**

Remove `WHISPER_MODEL`, add `GROQ_API_KEY`:

```
GMAIL_USER=you@gmail.com
GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
GMAIL_TO=you@gmail.com
MIC_DEVICE=hw:1,0
OLLAMA_MODEL=llama3
HF_TOKEN=your_huggingface_token_here
GROQ_API_KEY=your_groq_api_key_here
```

**Step 3: Install `groq` package**

Run: `pip install groq`
Expected: Successfully installed groq

**Step 4: Run full test suite one more time**

Run: `cd /home/pi/meeting-notes && python -m pytest -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
cd /home/pi/meeting-notes && git add requirements.txt .env.example
git commit -m "chore: swap faster-whisper for groq in dependencies"
```
