# SQLite Job Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the in-memory `JobManager` dict with SQLite so jobs survive restarts and failures are always surfaced.

**Architecture:** A single shared SQLite connection (`:memory:` in tests, `jobs.db` on disk in prod) replaces the in-memory dict. On startup, any jobs stuck in an in-progress status are immediately marked as `error` with "interrupted by restart". All status writes commit immediately so nothing is lost.

**Tech Stack:** Python `sqlite3` (stdlib, no install needed), Flask, threading.Lock

---

### Task 1: Update test_jobs.py — pass `:memory:` and add recovery test

**Files:**
- Modify: `tests/test_jobs.py`

**Step 1: Change all `JobManager()` calls to `JobManager(":memory:")`**

In `tests/test_jobs.py`, every test that calls `JobManager()` must now pass `":memory:"`.
Replace every occurrence of `JobManager()` with `JobManager(":memory:")`.
There are 5 tests — update all of them.

**Step 2: Add startup recovery test at the bottom of the file**

```python
def test_startup_marks_interrupted_jobs_as_error():
    jm = JobManager(":memory:")
    job_id = jm.create_job("meeting_interrupted")
    # Manually force a mid-run status directly in the DB
    jm._db.execute("UPDATE jobs SET status='transcribing' WHERE id=?", (job_id,))
    jm._db.commit()

    # Simulate a restart by creating a new JobManager on the same DB
    # For :memory: we can't literally reuse the connection, so we call _recover() directly
    jm._recover()

    job = jm.get_job(job_id)
    assert job["status"] == JobStatus.ERROR
    assert job["error"] == "interrupted by restart"
```

**Step 3: Run tests to confirm they fail**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_jobs.py -v
```

Expected: `TypeError: JobManager.__init__() takes 1 positional argument but 2 were given` (or similar). All 6 tests fail.

**Step 4: Commit the failing tests**

```bash
git add tests/test_jobs.py
git commit -m "test: update test_jobs for SQLite-backed JobManager"
```

---

### Task 2: Rewrite jobs.py with SQLite

**Files:**
- Modify: `jobs.py`

**Step 1: Write the new implementation**

Replace the entire contents of `jobs.py` with:

```python
import os
import uuid
import sqlite3
import threading
from enum import Enum
from datetime import datetime
from transcriber import transcribe
from summarizer import summarize
from emailer import send_notes

_INTERRUPTED_STATUSES = ("pending", "transcribing", "summarizing", "emailing")


class JobStatus(str, Enum):
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    SUMMARIZING = "summarizing"
    EMAILING = "emailing"
    DONE = "done"
    ERROR = "error"


class JobManager:
    def __init__(self, db_path="jobs.db"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_db()
        self._recover()

    def _init_db(self):
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id               TEXT PRIMARY KEY,
                label            TEXT NOT NULL,
                status           TEXT NOT NULL,
                summary          TEXT,
                transcript_path  TEXT,
                audio_path       TEXT,
                error            TEXT,
                created_at       TEXT NOT NULL
            )
        """)
        self._db.commit()

    def _recover(self):
        placeholders = ",".join("?" * len(_INTERRUPTED_STATUSES))
        self._db.execute(
            f"UPDATE jobs SET status=?, error=? WHERE status IN ({placeholders})",
            [JobStatus.ERROR, "interrupted by restart"] + list(_INTERRUPTED_STATUSES)
        )
        self._db.commit()

    def create_job(self, label, audio_path=None):
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, label, status, audio_path, created_at) VALUES (?,?,?,?,?)",
                (job_id, label, JobStatus.PENDING, audio_path, datetime.now().isoformat())
            )
            self._db.commit()
        return job_id

    def get_job(self, job_id):
        row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self):
        rows = self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def _set_status(self, job_id, status):
        with self._lock:
            self._db.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))
            self._db.commit()

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

    def retry_job(self, job_id):
        with self._lock:
            row = self._db.execute(
                "SELECT status FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row or row[0] != JobStatus.ERROR:
                return False
            self._db.execute(
                "UPDATE jobs SET status=?, error=NULL, summary=NULL, transcript_path=NULL WHERE id=?",
                (JobStatus.PENDING, job_id)
            )
            self._db.commit()
        return True

    def process_async(self, **kwargs):
        t = threading.Thread(target=self.process, kwargs=kwargs, daemon=True)
        t.start()
```

**Step 2: Run test_jobs.py to verify all 6 tests pass**

```bash
cd /home/pi/meeting-notes && python -m pytest tests/test_jobs.py -v
```

Expected: All 6 tests PASS.

**Step 3: Commit**

```bash
git add jobs.py
git commit -m "feat: persist jobs to SQLite with startup recovery for interrupted jobs"
```

---

### Task 3: Update app.py to pass db_path

**Files:**
- Modify: `app.py:16-20`

**Step 1: Add DB_PATH constant and pass it to JobManager**

Find this block in `app.py`:
```python
recorder = Recorder(
    mic_device=os.getenv("MIC_DEVICE", "hw:1,0"),
    output_dir=RECORDINGS_DIR
)
job_manager = JobManager()
```

Replace with:
```python
DB_PATH = os.path.join(os.path.dirname(__file__), "jobs.db")

recorder = Recorder(
    mic_device=os.getenv("MIC_DEVICE", "hw:1,0"),
    output_dir=RECORDINGS_DIR
)
job_manager = JobManager(db_path=DB_PATH)
```

**Step 2: Run the full test suite**

```bash
cd /home/pi/meeting-notes && python -m pytest -v
```

Expected: All tests pass. (test_routes.py may show a warning about `jobs.db` being created — that's ok for now, fixed in Task 4.)

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: wire app.py to use on-disk SQLite DB path"
```

---

### Task 4: Fix test_routes.py to use in-memory JobManager

**Files:**
- Modify: `tests/test_routes.py:6-16`

**Step 1: Update the `client` fixture to monkeypatch job_manager**

Find the `client` fixture:
```python
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
```

Replace with:
```python
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
    from jobs import JobManager
    monkeypatch.setattr(app_module, "job_manager", JobManager(":memory:"))
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()
```

**Step 2: Run the full test suite**

```bash
cd /home/pi/meeting-notes && python -m pytest -v
```

Expected: All tests pass. No `jobs.db` file created during test run.

**Step 3: Commit**

```bash
git add tests/test_routes.py
git commit -m "test: isolate route tests with in-memory JobManager"
```

---

### Task 5: Verify end-to-end and add jobs.db to .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Add jobs.db to .gitignore**

Open `.gitignore` and add this line:
```
jobs.db
```

**Step 2: Run full test suite one final time**

```bash
cd /home/pi/meeting-notes && python -m pytest -v
```

Expected: All tests pass.

**Step 3: Final commit**

```bash
git add .gitignore
git commit -m "chore: ignore jobs.db SQLite file"
```
