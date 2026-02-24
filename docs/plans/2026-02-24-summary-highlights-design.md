# Meeting Notes — Summary & Highlights Improvement Design
Date: 2026-02-24

## Problem

Two pain points with the current summarizer:

1. **No speaker identification** — Whisper produces a wall of text with no speaker labels, so the summary can't attribute who said what or characterize how each person engaged.
2. **Highlights are too high-level** — The current 3-bullet summary captures topic labels but misses concrete details: specific numbers, dates, commitments, decisions with rationale, and unresolved issues.

## Approach

Add speaker diarization via `pyannote.audio` between the transcription and summarization steps. The labeled transcript unlocks a richer 6-section prompt that captures speaker interaction types and concrete highlights.

## Data Flow

```
Whisper transcribes .mp3 → raw transcript (with word timestamps)
     ↓
pyannote.audio diarizes .mp3 → speaker segments
     (Speaker_00: 0.0–12.5s, Speaker_01: 12.5–25.0s, ...)
     ↓
Merge: assign each Whisper word to its speaker segment
     ↓
Labeled transcript:
  Speaker_00: We need to finalize the roadmap by end of month.
  Speaker_01: I agree, but the budget needs resolving first.
     ↓
Ollama summarizes with improved prompt → richer structured output
```

A new `"diarizing"` pipeline status appears in the UI between `"transcribing"` and `"summarizing"`.

If diarization fails or produces only one speaker, the pipeline falls back to the current simple prompt — no breakage.

## Prompt Design

### Diarized prompt (new)

```
You are a meeting assistant. Analyze this transcript and provide:

1. SUMMARY (3-5 bullets — include specific numbers, dates, names, and commitments)

2. SPEAKERS — for each speaker:
   - Apparent role/name if identifiable
   - Interaction type: decision-maker | facilitator | questioner | contributor | dissenter
   - 1-line characterization of their style

3. KEY HIGHLIGHTS — concrete moments worth noting (specific quotes, commitments,
   surprises, or disagreements — not just topic labels)

4. ACTION ITEMS — person: task, deadline if mentioned (or "None identified")

5. KEY DECISIONS — what was decided and the brief rationale behind it

6. OPEN QUESTIONS — unresolved issues or follow-ups with no clear owner

Transcript:
{transcript}
```

### Simple prompt (fallback, current behavior)

```
You are a meeting assistant. Analyze this transcript and provide:

1. SUMMARY (3-5 bullet points of main topics discussed)
2. ACTION ITEMS (person: task, or "None identified" if none)
3. KEY DECISIONS (or "None identified" if none)

Transcript:
{transcript}
```

## Implementation Changes

### New file: `diarizer.py`
- Accepts audio file path
- Runs `pyannote.audio` pipeline to get `(speaker_id, start_sec, end_sec)` segments
- Re-runs Whisper with `word_timestamps=True` to get word-level timing
- Merges: assigns each word to its speaker segment
- Returns labeled transcript string

### Modified files

| File | Change |
|------|--------|
| `summarizer.py` | Two prompt templates; `summarize()` picks based on whether transcript has speaker labels |
| `jobs.py` | Adds `"diarizing"` status step; calls diarizer after transcription, passes labeled transcript to summarizer |
| `.env.example` | Adds `HF_TOKEN=` (free HuggingFace account required for pyannote model download) |
| `requirements.txt` | Adds `pyannote.audio` |

### No changes needed
`app.py`, `emailer.py`, `transcriber.py`, `recorder.py`, UI template (STATUS_LABELS in `index.html` gets one new entry: `diarizing: 'Diarizing...'`)

## Updated Architecture (2026-02-24)

Revised after faster-whisper migration and disk space review. Three key changes:

### 1. pyannote.audio is an optional dependency

`pyannote.audio` is **not added to `requirements.txt`**. Install manually when ready:
```bash
pip install pyannote.audio
```
The service works without it. Diarization silently skips when not installed.

### 2. Gated behind HF_TOKEN with lazy import

`diarizer.py` checks for `HF_TOKEN` first. If absent, returns `(plain_text, False)` immediately — pyannote is never imported. If token is present but pyannote is not installed, `ImportError` is caught and falls back gracefully.

```python
def diarize(audio_path, whisper_segments, hf_token=None):
    plain_text = " ".join(seg.text.strip() for seg in whisper_segments)
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        return plain_text, False
    try:
        from pyannote.audio import Pipeline   # lazy — only if token present
        ...
    except (ImportError, Exception):
        return plain_text, False
```

### 3. faster-whisper segment objects (not dicts)

`transcriber.py` gains `return_segments=True` which returns a list of faster-whisper segment objects. `diarizer.py` accesses `.start`, `.end`, `.text` attributes (not `["start"]` dict keys).

`transcriber.py` collects the generator into a list first:
```python
segments_list = list(model.transcribe(audio_path, beam_size=5)[0])
text = " ".join(s.text.strip() for s in segments_list).strip()
if return_segments:
    return text, segments_list
return text
```

## Environment Setup

- One-time: create free account at huggingface.co, generate token, set `HF_TOKEN=` in `.env`
- One-time: accept pyannote model license on HuggingFace (required by pyannote terms)
- One-time: `pip install pyannote.audio` (torch 2.6.0 already installed — ~300–500MB additional)
- pyannote downloads model weights on first run (~600MB), cached in `~/.cache/`

## Performance

- Current pipeline (faster-whisper): ~5–8 min per hour of audio
- Diarization adds: ~10–15 min per hour on aarch64 CPU
- Runs fully in background — no user-facing delay

## Out of Scope

- Speaker name resolution (e.g., mapping Speaker_00 → "Alice") — labels only
- Real-time diarization during recording
- Multiple Ollama model options
