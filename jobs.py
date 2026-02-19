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
