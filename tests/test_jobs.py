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

def test_get_job_unknown_id_returns_none():
    jm = JobManager()
    result = jm.get_job("nonexistent-id")
    assert result is None

def test_process_job_sets_transcript_path_and_summary(tmp_path):
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
    assert job["transcript_path"] is not None
    assert job["summary"] == "summary text"
