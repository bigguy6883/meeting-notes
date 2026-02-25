# Speaker Identification — Design Doc
Date: 2026-02-25

## Problem

Two bugs prevent speaker labels from appearing in meeting transcripts:

1. `pyannote.audio` was never installed — `diarizer.py` silently falls back to plain text every time.
2. Even if diarization succeeded, the labeled transcript is never saved to disk. `transcribe()` writes plain Whisper text to `transcript_path`; the diarized version (with `Speaker_00:` labels) is only passed to `summarize()` and then discarded. The email attachment and stored transcript are always unlabeled.

## Solution

Replace `pyannote.audio` with `resemblyzer` (lightweight, ~17MB model, no PyTorch) and fix the transcript write path.

## Architecture

No structural changes to the pipeline. Same stages, same interfaces:

```
transcribe() → (text, segments)
diarize(audio_path, segments) → (transcript, diarized)
summarize(transcript, diarized=diarized) → summary
send_notes(..., transcript_path)
```

### diarizer.py — new implementation

1. Convert MP3 → temporary WAV via `ffmpeg` subprocess (already installed)
2. Load WAV with `resemblyzer.preprocess_wav()`
3. For each Whisper segment, extract a speaker embedding vector via `VoiceEncoder.embed_utterance()`
4. Cluster all embeddings with `AgglomerativeClustering(distance_threshold=0.6, n_clusters=None)` — no need to pre-specify speaker count
5. Map cluster IDs → `Speaker_00`, `Speaker_01`, etc.
6. Merge consecutive same-speaker segments into blocks
7. Return `"\n".join(lines), True` — identical output format to existing code

Fallback behavior preserved:
- Import error → plain text, `diarized=False`
- Single speaker detected → plain text, `diarized=False`
- Any exception → plain text, `diarized=False`

### jobs.py — transcript write fix

After `diarize()` returns, if `diarized=True`, overwrite `transcript_path` with the labeled transcript before calling `send_notes()`. The email attachment and stored transcript will then contain speaker labels.

## Files Changed

| File | Change |
|------|--------|
| `diarizer.py` | Replace pyannote with resemblyzer + ffmpeg conversion |
| `jobs.py` | Write labeled transcript to disk when `diarized=True` |
| `requirements.txt` | Add `resemblyzer`, `scikit-learn` |
| `tests/test_diarizer.py` | Update mocks for new implementation |

## Dependencies

- `resemblyzer` — speaker embedding (~17MB model download on first use)
- `scikit-learn` — `AgglomerativeClustering`
- `ffmpeg` — already installed, used for MP3→WAV conversion

## Out of Scope

- Named speaker profiles (voice enrollment)
- Post-meeting name assignment UI
- LLM name inference
