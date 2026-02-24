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

## Environment Setup

- One-time: create free account at huggingface.co, generate token, set `HF_TOKEN=` in `.env`
- One-time: accept pyannote model license on HuggingFace (required by pyannote terms)
- pyannote downloads model weights on first run (~600MB), cached in `~/.cache/torch/`

## Performance

- Current pipeline: ~15–20 min per hour of audio
- Diarization adds: ~10–15 min per hour on aarch64 CPU
- Runs fully in background — no user-facing delay

## Out of Scope

- Speaker name resolution (e.g., mapping Speaker_00 → "Alice") — labels only
- Real-time diarization during recording
- Multiple Ollama model options
