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
