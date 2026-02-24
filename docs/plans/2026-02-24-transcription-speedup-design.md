# Transcription Speedup — Design Doc
Date: 2026-02-24

## Problem
Local Whisper transcription is too slow. The `medium` model runs on CPU with defaults
(beam_size=5, no VAD, single-threaded), resulting in ~15-20 min processing per hour of audio.

## Constraints
- Must stay local (no cloud APIs)
- Must keep `medium` model accuracy (no model downgrade)
- `transcribe()` function signature must remain unchanged

## Solution: Option A — VAD filter + beam_size=1 + cpu_threads

Three changes to `transcriber.py` only:

### 1. Model caching
Move `WhisperModel` instantiation to module-level, keyed by model name. First call loads
from disk; subsequent calls (retries, back-to-back jobs) reuse the cached instance.

### 2. VAD filter
Pass `vad_filter=True` to `model.transcribe()`. faster_whisper uses Silero VAD to detect
and skip non-speech segments before Whisper processes them. Meeting audio has significant
silence/pauses — this is expected to give 30-50% speedup with no quality loss.

### 3. CPU threads + beam_size=1
- `cpu_threads=4` on `WhisperModel()` — uses all Pi cores instead of defaulting to 1-2
- `beam_size=1` on `model.transcribe()` — greedy decode instead of beam search, ~20% faster
  with very minor quality trade-off

## Files Changed
- `transcriber.py` — only file modified

## No Changes To
- `jobs.py`, `app.py`, or any other file
- `transcribe()` function signature
