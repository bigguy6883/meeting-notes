# Meeting Note Taker — Design Doc
Date: 2026-02-18

## Overview
A Flask web app on the homelab Pi that records Teams meetings via USB microphone,
transcribes audio with Whisper, summarizes with Ollama, and emails the results to the user.

## Context
- Teams calls happen on a locked-down Windows work computer
- Pi USB mic placed near speakers captures room audio
- Homelab Pi: 7.6GB RAM, aarch64, ffmpeg already installed
- Pattern modeled after existing Baby Tracker Flask app (port 5000)

## Architecture

**Stack:** Flask + ffmpeg + Whisper + Ollama + Gmail SMTP
**Port:** 5001 (avoids conflict with Baby Tracker)

**Folder structure:**
```
~/meeting-notes/
  app.py              # Flask app + pipeline logic
  templates/
    index.html        # Web UI
  recordings/         # Temp audio files
  transcripts/        # Saved transcripts + summaries
  .env                # Gmail credentials (never committed)
```

## Data Flow

```
[Browser] → Start → Flask spawns ffmpeg (USB mic → .mp3)
[Browser] → Stop  → Flask kills ffmpeg
                  → Background thread starts:
                      Whisper transcribes .mp3 → transcript.txt
                      Ollama summarizes → summary.txt
                      Gmail sends (summary in body, transcript attached)
                  → UI returns to Ready immediately
                  → Browser polls for job status
```

## Concurrency Model
- Recording and processing are **independent** — user can start a new recording
  while a previous meeting is still being processed
- Each recording spawns a background thread for processing
- Multiple jobs can be in-flight simultaneously
- User receives a separate email per meeting when each finishes

## Web UI States

**Idle:**
```
[ ● Start Recording ]

Processing jobs: (none)
```

**Recording:**
```
● Recording... 00:14:32
[ ■ Stop & Process ]

Processing jobs: (none)
```

**Processing (background):**
```
[ ● Start Recording ]   ← immediately available again

Processing jobs:
⏳ meeting_10:30 — transcribing...
✓  meeting_09:00 — email sent
```

**Job complete (in jobs list):**
```
✓ meeting_10:30 — email sent
  [View Summary]  [View Transcript]
```

## Email Format
- **Subject:** Meeting Notes — YYYY-MM-DD HH:MM
- **Body:** AI-generated summary + action items (plain text)
- **Attachment:** full transcript as .txt file

## Gmail Setup
- Uses Gmail App Password (not account password) via smtplib TLS
- Credentials stored in `.env` file, never committed to git

## Processing Details
- **Whisper model:** `medium` — good accuracy, fits in 7.6GB RAM
- **Ollama model:** `llama3` — summarization prompt extracts: summary bullets, action items, key decisions
- **Expected processing time:** ~15-20 min per hour of audio

## Out of Scope (v1)
- Speaker diarization (who said what)
- Scheduling / calendar integration
- Multiple email recipients
- Cloud storage of recordings
