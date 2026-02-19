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
