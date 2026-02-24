# Groq Transcription Switch — Design Doc
Date: 2026-02-24

## Problem
Local Whisper CPU transcription is too slow even after optimization (VAD filter, beam_size=1,
cpu_threads=4). An 8-minute meeting still takes several minutes to process.

## Solution
Replace `faster_whisper` with the Groq API (`whisper-large-v3-turbo`). Groq's LPU hardware
processes an 8-minute recording in ~2-3 seconds. Audio leaves the local network.

## Constraints
- Diarization pipeline must remain intact (needs segment timestamps)
- `transcribe()` function signature unchanged
- `diarizer.py` unchanged

## Architecture

### `transcriber.py`
- Use `groq.Groq()` client, reading `GROQ_API_KEY` from environment
- Call `client.audio.transcriptions.create()` with `response_format="verbose_json"` to get
  segment-level timestamps (start, end, text) alongside the full transcript text
- Define a `Segment` dataclass with `.start`, `.end`, `.text` to wrap Groq's response objects —
  keeps `diarizer.py` compatible without any changes
- Hardcode model to `whisper-large-v3-turbo` (best speed/accuracy for meeting audio)
- Remove model caching and threading lock (no longer needed for an API call)

### `app.py` / `jobs.py`
- Remove `whisper_model` parameter from `process()` and `process_async()` calls
- Remove `WHISPER_MODEL` env var reads

### `requirements.txt`
- Remove `faster-whisper`
- Add `groq`

### `.env.example`
- Add `GROQ_API_KEY=your_groq_api_key_here`
- Remove `WHISPER_MODEL=medium`

## Files Changed
- `transcriber.py`
- `app.py`
- `jobs.py`
- `requirements.txt`
- `.env.example`

## Files Unchanged
- `diarizer.py`
- `summarizer.py`
- `emailer.py`
- `recorder.py`

## Out of Scope
- Chunking for files >25MB (Groq API limit) — error naturally at that size
- Fallback to local Whisper if Groq is unavailable
