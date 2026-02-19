# Meeting Note Taker — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Flask web app on homelab Pi that records USB mic audio, transcribes with Whisper, summarizes with Ollama, and emails results — supporting concurrent meetings.

**Architecture:** Flask app on port 5001 with ffmpeg for recording, background threads for processing pipeline (Whisper → Ollama → Gmail). UI polls `/api/jobs` for status. Recording and processing are fully independent — new recording can start while previous is processing.

**Tech Stack:** Flask 3.1, openai-whisper, ollama (Python client), smtplib (stdlib), python-dotenv, pytest, ALSA (ffmpeg -f alsa)

---

## Prerequisites (do before Task 1)

Install missing dependencies:
```bash
pip3 install openai-whisper ollama pytest-flask
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

Find USB mic device name (plug it in first):
```bash
arecord -l
# Look for your USB mic, note "card N, device M" → use hw:N,M
```

---

### Task 1: Project scaffold + config

**Files:**
- Create: `~/meeting-notes/requirements.txt`
- Create: `~/meeting-notes/.env.example`
- Create: `~/meeting-notes/.gitignore`
- Create: `~/meeting-notes/app.py` (skeleton only)
- Create: `~/meeting-notes/recordings/.gitkeep`
- Create: `~/meeting-notes/transcripts/.gitkeep`
- Create: `~/meeting-notes/templates/` (empty dir)

**Step 1: Create requirements.txt**

```
Flask==3.1.1
openai-whisper
ollama
python-dotenv==1.0.1
pytest
pytest-flask
```

**Step 2: Create .env.example**

```
GMAIL_USER=you@gmail.com
GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
GMAIL_TO=you@gmail.com
MIC_DEVICE=hw:1,0
WHISPER_MODEL=medium
OLLAMA_MODEL=llama3
```

**Step 3: Create .gitignore**

```
.env
recordings/
transcripts/
__pycache__/
*.pyc
.pytest_cache/
```

**Step 4: Create app.py skeleton**

```python
import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
```

**Step 5: Create directory structure**
```bash
mkdir -p ~/meeting-notes/recordings ~/meeting-notes/transcripts ~/meeting-notes/templates
touch ~/meeting-notes/recordings/.gitkeep ~/meeting-notes/transcripts/.gitkeep
```

**Step 6: Verify Flask runs**
```bash
cd ~/meeting-notes && python3 app.py
# Expected: * Running on http://0.0.0.0:5001
# Ctrl+C to stop
```

**Step 7: Commit**
```bash
cd ~/meeting-notes
git add requirements.txt .env.example .gitignore app.py recordings/.gitkeep transcripts/.gitkeep
git commit -m "feat: project scaffold and config"
```

---

### Task 2: Recording module (with tests)

**Files:**
- Create: `~/meeting-notes/recorder.py`
- Create: `~/meeting-notes/tests/test_recorder.py`

**Step 1: Create tests/test_recorder.py**

```python
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
```

**Step 2: Run tests — expect FAIL**
```bash
cd ~/meeting-notes && python3 -m pytest tests/test_recorder.py -v
# Expected: ImportError — recorder not defined yet
```

**Step 3: Create recorder.py**

```python
import subprocess
import os
from datetime import datetime

class Recorder:
    def __init__(self, mic_device, output_dir):
        self.mic_device = mic_device
        self.output_dir = output_dir
        self._process = None
        self._filepath = None

    def start(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(self.output_dir, f"meeting_{timestamp}.mp3")
        self._process = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "alsa",
            "-i", self.mic_device,
            "-ar", "16000",
            "-ac", "1",
            self._filepath
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self._filepath

    def stop(self):
        if self._process is None:
            return None
        self._process.terminate()
        self._process.wait()
        filepath = self._filepath
        self._process = None
        self._filepath = None
        return filepath

    def is_recording(self):
        return self._process is not None and self._process.poll() is None
```

**Step 4: Run tests — expect PASS**
```bash
cd ~/meeting-notes && python3 -m pytest tests/test_recorder.py -v
# Expected: 4 passed
```

**Step 5: Commit**
```bash
git add recorder.py tests/test_recorder.py
git commit -m "feat: audio recorder with ffmpeg"
```

---

### Task 3: Transcription module (with tests)

**Files:**
- Create: `~/meeting-notes/transcriber.py`
- Create: `~/meeting-notes/tests/test_transcriber.py`

**Step 1: Create tests/test_transcriber.py**

```python
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
```

**Step 2: Run tests — expect FAIL**
```bash
python3 -m pytest tests/test_transcriber.py -v
```

**Step 3: Create transcriber.py**

```python
import whisper

def transcribe(audio_path, model_name="medium", output_path=None):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    return text
```

**Step 4: Run tests — expect PASS**
```bash
python3 -m pytest tests/test_transcriber.py -v
```

**Step 5: Commit**
```bash
git add transcriber.py tests/test_transcriber.py
git commit -m "feat: whisper transcription module"
```

---

### Task 4: Summarization module (with tests)

**Files:**
- Create: `~/meeting-notes/summarizer.py`
- Create: `~/meeting-notes/tests/test_summarizer.py`

**Step 1: Create tests/test_summarizer.py**

```python
from unittest.mock import patch, MagicMock
from summarizer import summarize

def test_summarize_returns_string():
    mock_response = MagicMock()
    mock_response.message.content = "- Discussed roadmap\nAction: Alice to follow up"
    with patch("summarizer.ollama.chat", return_value=mock_response):
        result = summarize("This is a transcript about the Q1 roadmap.", model="llama3")
    assert isinstance(result, str)
    assert len(result) > 0

def test_summarize_sends_transcript_in_prompt():
    mock_response = MagicMock()
    mock_response.message.content = "Summary here"
    transcript = "We talked about deadlines and deliverables."
    with patch("summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = mock_response
        summarize(transcript, model="llama3")
    call_args = mock_chat.call_args
    messages = call_args[1]["messages"]
    assert any(transcript in m["content"] for m in messages)
```

**Step 2: Run tests — expect FAIL**
```bash
python3 -m pytest tests/test_summarizer.py -v
```

**Step 3: Create summarizer.py**

```python
import ollama

PROMPT_TEMPLATE = """You are a meeting assistant. Analyze this transcript and provide:

1. SUMMARY (3-5 bullet points of main topics discussed)
2. ACTION ITEMS (person: task, or "None identified" if none)
3. KEY DECISIONS (or "None identified" if none)

Transcript:
{transcript}"""

def summarize(transcript, model="llama3"):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(transcript=transcript)}]
    )
    return response.message.content.strip()
```

**Step 4: Run tests — expect PASS**
```bash
python3 -m pytest tests/test_summarizer.py -v
```

**Step 5: Commit**
```bash
git add summarizer.py tests/test_summarizer.py
git commit -m "feat: ollama summarization module"
```

---

### Task 5: Email module (with tests)

**Files:**
- Create: `~/meeting-notes/emailer.py`
- Create: `~/meeting-notes/tests/test_emailer.py`

**Step 1: Create tests/test_emailer.py**

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from emailer import send_notes

def test_send_notes_connects_to_gmail(tmp_path):
    transcript_file = tmp_path / "transcript.txt"
    transcript_file.write_text("Full meeting transcript here.")
    with patch("emailer.smtplib.SMTP") as mock_smtp_class:
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)
        send_notes(
            gmail_user="test@gmail.com",
            gmail_password="app-password",
            to_address="test@gmail.com",
            meeting_label="2026-02-18 10:30",
            summary="- Discussed roadmap",
            transcript_path=str(transcript_file)
        )
    mock_smtp_class.assert_called_with("smtp.gmail.com", 587)

def test_send_notes_attaches_transcript(tmp_path):
    transcript_file = tmp_path / "transcript.txt"
    transcript_file.write_text("Full transcript content.")
    sent_messages = []
    with patch("emailer.smtplib.SMTP") as mock_smtp_class:
        mock_smtp = MagicMock()
        mock_smtp.send_message.side_effect = lambda msg: sent_messages.append(msg)
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)
        send_notes(
            gmail_user="test@gmail.com",
            gmail_password="app-password",
            to_address="test@gmail.com",
            meeting_label="2026-02-18 10:30",
            summary="Summary here",
            transcript_path=str(transcript_file)
        )
    assert len(sent_messages) == 1
    msg = sent_messages[0]
    assert "Meeting Notes" in msg["Subject"]
    payloads = msg.get_payload()
    assert len(payloads) == 2  # body + attachment
```

**Step 2: Run tests — expect FAIL**
```bash
python3 -m pytest tests/test_emailer.py -v
```

**Step 3: Create emailer.py**

```python
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_notes(gmail_user, gmail_password, to_address, meeting_label, summary, transcript_path):
    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = to_address
    msg["Subject"] = f"Meeting Notes — {meeting_label}"

    msg.attach(MIMEText(summary, "plain"))

    with open(transcript_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    filename = os.path.basename(transcript_path)
    part.add_header("Content-Disposition", f"attachment; filename={filename}")
    msg.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
```

**Step 4: Run tests — expect PASS**
```bash
python3 -m pytest tests/test_emailer.py -v
```

**Step 5: Commit**
```bash
git add emailer.py tests/test_emailer.py
git commit -m "feat: gmail email module with transcript attachment"
```

---

### Task 6: Job manager — background processing pipeline

**Files:**
- Create: `~/meeting-notes/jobs.py`
- Create: `~/meeting-notes/tests/test_jobs.py`

**Step 1: Create tests/test_jobs.py**

```python
import time
import pytest
from unittest.mock import patch, MagicMock
from jobs import JobManager, JobStatus

def test_new_job_is_pending():
    jm = JobManager()
    job_id = jm.create_job("meeting_20260218_1030")
    job = jm.get_job(job_id)
    assert job["status"] == JobStatus.PENDING
    assert job["label"] == "meeting_20260218_1030"

def test_jobs_list_returns_all():
    jm = JobManager()
    jm.create_job("meeting_a")
    jm.create_job("meeting_b")
    assert len(jm.list_jobs()) == 2

def test_process_job_runs_pipeline(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake audio")

    jm = JobManager()
    job_id = jm.create_job("meeting_20260218_1030")

    with patch("jobs.transcribe", return_value="transcript text"), \
         patch("jobs.summarize", return_value="summary text"), \
         patch("jobs.send_notes"):
        jm.process(
            job_id=job_id,
            audio_path=str(audio),
            transcript_dir=str(tmp_path),
            gmail_user="u@g.com",
            gmail_password="pw",
            to_address="u@g.com",
            whisper_model="tiny",
            ollama_model="llama3"
        )

    job = jm.get_job(job_id)
    assert job["status"] == JobStatus.DONE

def test_process_job_sets_error_on_failure(tmp_path):
    jm = JobManager()
    job_id = jm.create_job("meeting_fail")
    with patch("jobs.transcribe", side_effect=Exception("whisper failed")):
        jm.process(
            job_id=job_id,
            audio_path="/nonexistent.mp3",
            transcript_dir=str(tmp_path),
            gmail_user="u@g.com",
            gmail_password="pw",
            to_address="u@g.com",
            whisper_model="tiny",
            ollama_model="llama3"
        )
    job = jm.get_job(job_id)
    assert job["status"] == JobStatus.ERROR
    assert "whisper failed" in job["error"]
```

**Step 2: Run tests — expect FAIL**
```bash
python3 -m pytest tests/test_jobs.py -v
```

**Step 3: Create jobs.py**

```python
import os
import uuid
import threading
from enum import Enum
from datetime import datetime
from transcriber import transcribe
from summarizer import summarize
from emailer import send_notes

class JobStatus(str, Enum):
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    SUMMARIZING = "summarizing"
    EMAILING = "emailing"
    DONE = "done"
    ERROR = "error"

class JobManager:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create_job(self, label):
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "label": label,
                "status": JobStatus.PENDING,
                "summary": None,
                "transcript_path": None,
                "error": None,
                "created_at": datetime.now().isoformat()
            }
        return job_id

    def get_job(self, job_id):
        with self._lock:
            return dict(self._jobs[job_id])

    def list_jobs(self):
        with self._lock:
            return [dict(j) for j in self._jobs.values()]

    def _set_status(self, job_id, status):
        with self._lock:
            self._jobs[job_id]["status"] = status

    def process(self, job_id, audio_path, transcript_dir, gmail_user,
                gmail_password, to_address, whisper_model, ollama_model):
        try:
            transcript_path = os.path.join(
                transcript_dir,
                os.path.basename(audio_path).replace(".mp3", ".txt")
            )
            self._set_status(job_id, JobStatus.TRANSCRIBING)
            transcript = transcribe(audio_path, model_name=whisper_model, output_path=transcript_path)

            self._set_status(job_id, JobStatus.SUMMARIZING)
            summary = summarize(transcript, model=ollama_model)

            self._set_status(job_id, JobStatus.EMAILING)
            label = self._jobs[job_id]["label"]
            send_notes(
                gmail_user=gmail_user,
                gmail_password=gmail_password,
                to_address=to_address,
                meeting_label=label,
                summary=summary,
                transcript_path=transcript_path
            )
            with self._lock:
                self._jobs[job_id]["status"] = JobStatus.DONE
                self._jobs[job_id]["summary"] = summary
                self._jobs[job_id]["transcript_path"] = transcript_path

        except Exception as e:
            with self._lock:
                self._jobs[job_id]["status"] = JobStatus.ERROR
                self._jobs[job_id]["error"] = str(e)

    def process_async(self, **kwargs):
        t = threading.Thread(target=self.process, kwargs=kwargs, daemon=True)
        t.start()
```

**Step 4: Run tests — expect PASS**
```bash
python3 -m pytest tests/test_jobs.py -v
```

**Step 5: Commit**
```bash
git add jobs.py tests/test_jobs.py
git commit -m "feat: job manager with background processing pipeline"
```

---

### Task 7: Flask routes

**Files:**
- Modify: `~/meeting-notes/app.py`
- Create: `~/meeting-notes/tests/test_routes.py`

**Step 1: Create tests/test_routes.py**

```python
import pytest
from unittest.mock import patch, MagicMock
import app as app_module

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "test@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    monkeypatch.setenv("GMAIL_TO", "test@gmail.com")
    monkeypatch.setenv("MIC_DEVICE", "hw:0,0")
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setattr(app_module, "RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "TRANSCRIPTS_DIR", str(tmp_path))
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

**Step 2: Run tests — expect FAIL**
```bash
python3 -m pytest tests/test_routes.py -v
```

**Step 3: Replace app.py with full implementation**

```python
import os
from flask import Flask, jsonify, render_template, send_file
from dotenv import load_dotenv
from recorder import Recorder
from jobs import JobManager

load_dotenv()

app = Flask(__name__)

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "transcripts")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

recorder = Recorder(
    mic_device=os.getenv("MIC_DEVICE", "hw:1,0"),
    output_dir=RECORDINGS_DIR
)
job_manager = JobManager()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_recording():
    if recorder.is_recording():
        return jsonify({"error": "Already recording"}), 409
    filepath = recorder.start()
    return jsonify({"status": "recording", "file": filepath})


@app.route("/api/stop", methods=["POST"])
def stop_recording():
    filepath = recorder.stop()
    if not filepath:
        return jsonify({"error": "Not recording"}), 409

    basename = os.path.basename(filepath).replace("meeting_", "").replace(".mp3", "")
    label = f"{basename[:8]} {basename[9:13]}" if len(basename) >= 13 else basename

    job_id = job_manager.create_job(label)
    job_manager.process_async(
        job_id=job_id,
        audio_path=filepath,
        transcript_dir=TRANSCRIPTS_DIR,
        gmail_user=os.getenv("GMAIL_USER"),
        gmail_password=os.getenv("GMAIL_APP_PASSWORD"),
        to_address=os.getenv("GMAIL_TO"),
        whisper_model=os.getenv("WHISPER_MODEL", "medium"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
    )
    return jsonify({"status": "processing", "job_id": job_id})


@app.route("/api/status")
def recording_status():
    return jsonify({"recording": recorder.is_recording()})


@app.route("/api/jobs")
def list_jobs():
    return jsonify(job_manager.list_jobs())


@app.route("/api/jobs/<job_id>/transcript")
def view_transcript(job_id):
    job = job_manager.get_job(job_id)
    if not job or not job.get("transcript_path"):
        return jsonify({"error": "Not found"}), 404
    return send_file(job["transcript_path"], mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
```

**Step 4: Run tests — expect PASS**
```bash
python3 -m pytest tests/test_routes.py -v
```

**Step 5: Run all tests**
```bash
python3 -m pytest tests/ -v
# Expected: all passing
```

**Step 6: Commit**
```bash
git add app.py tests/test_routes.py
git commit -m "feat: flask routes for record/stop/jobs"
```

---

### Task 8: Web UI

**Files:**
- Create: `~/meeting-notes/templates/index.html`

**Step 1: Create templates/index.html**

Note: all dynamic content is set via `textContent` (never `innerHTML`) to prevent XSS.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Meeting Notes</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f5f5f5; padding: 2rem; }
    .card { background: white; border-radius: 12px; padding: 2rem; max-width: 600px; margin: 0 auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    h1 { font-size: 1.4rem; color: #333; margin-bottom: 1.5rem; }
    .btn { display: inline-block; padding: 0.75rem 1.5rem; border-radius: 8px; border: none; font-size: 1rem; cursor: pointer; font-weight: 600; }
    .btn-start { background: #e53e3e; color: white; }
    .btn-stop  { background: #2d3748; color: white; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .timer { font-size: 1.1rem; color: #e53e3e; margin: 0.75rem 0; font-variant-numeric: tabular-nums; }
    .jobs { margin-top: 2rem; }
    .jobs h2 { font-size: 1rem; color: #666; margin-bottom: 0.75rem; }
    .job { padding: 0.6rem 0.75rem; border-radius: 6px; background: #f9f9f9; margin-bottom: 0.5rem; font-size: 0.9rem; display: flex; justify-content: space-between; align-items: center; }
    .job.done { background: #f0fff4; }
    .job.error { background: #fff5f5; }
    .job-label { font-weight: 500; }
    .job-status { color: #666; font-size: 0.8rem; }
    .job-status.done { color: #276749; }
    .job-status.error { color: #c53030; }
    .view-link { font-size: 0.8rem; color: #3182ce; text-decoration: none; margin-left: 0.5rem; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #e53e3e; margin-right: 6px; animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .error-msg { color: #c53030; font-size: 0.75rem; display: block; margin-top: 2px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Meeting Note Taker</h1>

    <div id="idle-state">
      <button class="btn btn-start" onclick="startRecording()">&#9679; Start Recording</button>
    </div>

    <div id="recording-state" style="display:none">
      <div class="timer"><span class="dot"></span> Recording... <span id="timer">00:00:00</span></div>
      <button class="btn btn-stop" onclick="stopRecording()">&#9632; Stop &amp; Process</button>
    </div>

    <div class="jobs">
      <h2>Processing jobs</h2>
      <div id="jobs-list"></div>
    </div>
  </div>

  <script>
    const STATUS_LABELS = {
      pending: 'Pending',
      transcribing: 'Transcribing...',
      summarizing: 'Summarizing...',
      emailing: 'Sending email...',
      done: 'Email sent',
      error: 'Error'
    };

    let timerInterval = null;
    let startTime = null;

    function formatTime(ms) {
      const s = Math.floor(ms / 1000);
      const h = Math.floor(s / 3600).toString().padStart(2, '0');
      const m = Math.floor((s % 3600) / 60).toString().padStart(2, '0');
      const sec = (s % 60).toString().padStart(2, '0');
      return `${h}:${m}:${sec}`;
    }

    async function startRecording() {
      const resp = await fetch('/api/start', { method: 'POST' });
      if (!resp.ok) { alert('Could not start recording'); return; }
      document.getElementById('idle-state').style.display = 'none';
      document.getElementById('recording-state').style.display = 'block';
      startTime = Date.now();
      timerInterval = setInterval(() => {
        document.getElementById('timer').textContent = formatTime(Date.now() - startTime);
      }, 1000);
    }

    async function stopRecording() {
      clearInterval(timerInterval);
      document.getElementById('recording-state').style.display = 'none';
      document.getElementById('idle-state').style.display = 'block';
      await fetch('/api/stop', { method: 'POST' });
      refreshJobs();
    }

    function buildJobRow(job) {
      const row = document.createElement('div');
      row.className = 'job' + (job.status === 'done' ? ' done' : job.status === 'error' ? ' error' : '');

      const left = document.createElement('div');
      const labelSpan = document.createElement('span');
      labelSpan.className = 'job-label';
      labelSpan.textContent = job.label;
      left.appendChild(labelSpan);

      if (job.status === 'error' && job.error) {
        const errSpan = document.createElement('span');
        errSpan.className = 'error-msg';
        errSpan.textContent = job.error;
        left.appendChild(errSpan);
      }

      const right = document.createElement('div');
      const statusSpan = document.createElement('span');
      statusSpan.className = 'job-status' + (job.status === 'done' ? ' done' : job.status === 'error' ? ' error' : '');
      const prefix = job.status === 'done' ? '\u2713 ' : job.status === 'error' ? '\u2717 ' : '\u23f3 ';
      statusSpan.textContent = prefix + (STATUS_LABELS[job.status] || job.status);
      right.appendChild(statusSpan);

      if (job.status === 'done') {
        const link = document.createElement('a');
        link.className = 'view-link';
        link.href = `/api/jobs/${encodeURIComponent(job.id)}/transcript`;
        link.target = '_blank';
        link.textContent = 'Transcript';
        right.appendChild(link);
      }

      row.appendChild(left);
      row.appendChild(right);
      return row;
    }

    async function refreshJobs() {
      const resp = await fetch('/api/jobs');
      const jobs = await resp.json();
      const container = document.getElementById('jobs-list');
      container.replaceChildren();

      if (jobs.length === 0) {
        const em = document.createElement('em');
        em.style.cssText = 'color:#999;font-size:0.85rem';
        em.textContent = 'None';
        container.appendChild(em);
        return;
      }

      jobs.sort((a, b) => b.created_at.localeCompare(a.created_at));
      jobs.forEach(job => container.appendChild(buildJobRow(job)));
    }

    setInterval(refreshJobs, 5000);
    refreshJobs();
  </script>
</body>
</html>
```

**Step 2: Run the app and verify UI manually**
```bash
cd ~/meeting-notes && python3 app.py
# Open http://homelab:5001 in browser
# Verify: Start button visible, jobs list shows "None"
```

**Step 3: Commit**
```bash
git add templates/index.html
git commit -m "feat: web UI with start/stop and live job status"
```

---

### Task 9: .env setup + smoke test

**Files:**
- Create: `~/meeting-notes/.env` (from .env.example, fill in real values)

**Step 1: Create your .env**
```bash
cp ~/meeting-notes/.env.example ~/meeting-notes/.env
nano ~/meeting-notes/.env
# Fill in:
#   GMAIL_USER=your.email@gmail.com
#   GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx  (from https://myaccount.google.com/apppasswords)
#   GMAIL_TO=your.email@gmail.com
#   MIC_DEVICE=hw:1,0  (check: arecord -l, find your USB mic card/device numbers)
#   WHISPER_MODEL=medium
#   OLLAMA_MODEL=llama3
```

**Step 2: Verify Ollama is running with llama3**
```bash
ollama list
# Should show llama3. If not: ollama pull llama3
```

**Step 3: Test email independently**
```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv()
import os, tempfile
from emailer import send_notes
with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as f:
    f.write('Test transcript'); fname = f.name
send_notes(os.getenv('GMAIL_USER'), os.getenv('GMAIL_APP_PASSWORD'),
           os.getenv('GMAIL_TO'), 'Test 2026-02-18', 'Test summary bullet', fname)
print('Email sent!')
"
# Check your inbox
```

**Step 4: Run full app**
```bash
cd ~/meeting-notes && python3 app.py
```

**Step 5: Full smoke test**
- Open `http://homelab:5001` in browser
- Plug in USB mic, verify it appears: `arecord -l`
- Click Start Recording
- Wait ~10 seconds, click Stop & Process
- Watch job status update every 5 seconds
- Check inbox for email with summary + transcript attachment

**Step 6: Run all tests one final time**
```bash
cd ~/meeting-notes && python3 -m pytest tests/ -v
# All green
```

**Step 7: Final commit**
```bash
git add .
git commit -m "feat: complete meeting note taker v1"
```

---

### Task 10 (Optional): Systemd service

If you want it to auto-start on boot:

**Step 1: Create service file**
```bash
sudo nano /etc/systemd/system/meeting-notes.service
```
```ini
[Unit]
Description=Meeting Note Taker
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/meeting-notes
ExecStart=/usr/bin/python3 /home/pi/meeting-notes/app.py
Restart=on-failure
EnvironmentFile=/home/pi/meeting-notes/.env

[Install]
WantedBy=multi-user.target
```

**Step 2: Enable and start**
```bash
sudo systemctl daemon-reload
sudo systemctl enable meeting-notes
sudo systemctl start meeting-notes
sudo systemctl status meeting-notes
```
