# faster-whisper Switch — Design

**Date:** 2026-02-24
**Status:** Approved

## Problem

`openai-whisper` runs at ~3–5× real time on the Pi 5 CPU. A 30-min meeting takes 1.5–2.5 hours to transcribe. `faster-whisper` uses CTranslate2 with int8 quantization and runs ~3× faster on the same hardware.

## Approach

Straight swap — identical `transcribe()` function signature, no changes outside `transcriber.py`, `requirements.txt`, and its tests.

## Design

**`requirements.txt`:** Replace `openai-whisper` with `faster-whisper`.

**`transcriber.py`:** Replace whisper API with faster-whisper:
- `WhisperModel(model_name, device="cpu", compute_type="int8")`
- `segments, _ = model.transcribe(audio_path, beam_size=5)`
- `text = " ".join(s.text.strip() for s in segments).strip()`

**`tests/test_transcriber.py`:** Update mocks to patch `faster_whisper.WhisperModel` and return mock segment iterables with `.text` attributes.

## Key Decisions

- `compute_type="int8"` — quantized CPU inference, ~3× speedup, negligible accuracy loss
- `beam_size=5` — default, good quality/speed balance
- Model stays `medium` — same accuracy target, just faster
- First run will download CTranslate2 model format (~1.5GB) from HuggingFace

## Files Changed

| File | Change |
|------|--------|
| `requirements.txt` | `openai-whisper` → `faster-whisper` |
| `transcriber.py` | New WhisperModel API, segment iteration |
| `tests/test_transcriber.py` | Updated mocks for new API |

## Out of Scope

- Model size change (staying on medium)
- GPU/device changes
- Any other app files
