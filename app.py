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

DB_PATH = os.path.join(os.path.dirname(__file__), "jobs.db")

recorder = Recorder(
    mic_device=os.getenv("MIC_DEVICE", "hw:1,0"),
    output_dir=RECORDINGS_DIR
)
job_manager = JobManager(db_path=DB_PATH)

_required_env = ["GMAIL_USER", "GMAIL_APP_PASSWORD", "GMAIL_TO", "GROQ_API_KEY"]
_missing = [k for k in _required_env if not os.getenv(k)]
if _missing:
    import sys
    print(f"ERROR: Missing required env vars: {', '.join(_missing)}", file=sys.stderr)
    print("Copy .env.example to .env and fill in your credentials.", file=sys.stderr)
    sys.exit(1)


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


@app.route("/api/status")
def recording_status():
    return jsonify({"recording": recorder.is_recording()})


@app.route("/api/jobs")
def list_jobs():
    return jsonify(job_manager.list_jobs())


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


@app.route("/api/jobs/<job_id>/transcript")
def view_transcript(job_id):
    job = job_manager.get_job(job_id)
    if not job or not job.get("transcript_path"):
        return jsonify({"error": "Not found"}), 404
    return send_file(job["transcript_path"], mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
