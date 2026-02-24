import os


def _find_speaker(midpoint, pyannote_segments):
    """Return speaker whose segment contains midpoint, or nearest by midpoint distance."""
    for speaker, start, end in pyannote_segments:
        if start <= midpoint <= end:
            return speaker
    if pyannote_segments:
        return min(pyannote_segments, key=lambda s: abs((s[1] + s[2]) / 2 - midpoint))[0]
    return "Speaker_00"


def diarize(audio_path, whisper_segments, hf_token=None):
    """
    Merge pyannote speaker diarization with faster-whisper segment timing.

    whisper_segments: list of faster-whisper segment objects (.start, .end, .text)

    Returns (transcript: str, diarized: bool).
    Falls back to plain transcript (diarized=False) when:
    - HF_TOKEN is absent
    - pyannote.audio is not installed
    - only one speaker is detected
    - any exception occurs
    """
    plain_text = " ".join(seg.text.strip() for seg in whisper_segments)

    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        return plain_text, False

    try:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        diarization = pipeline(audio_path)

        pyannote_segs = [
            (speaker, turn.start, turn.end)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        if len(set(s[0] for s in pyannote_segs)) < 2:
            return plain_text, False

        lines = []
        current_speaker = None
        current_texts = []

        for seg in whisper_segments:
            midpoint = (seg.start + seg.end) / 2
            speaker = _find_speaker(midpoint, pyannote_segs)

            if speaker != current_speaker:
                if current_texts and current_speaker:
                    lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                current_speaker = speaker
                current_texts = [seg.text.strip()]
            else:
                current_texts.append(seg.text.strip())

        if current_texts and current_speaker:
            lines.append(f"{current_speaker}: {' '.join(current_texts)}")

        return "\n".join(lines), True

    except Exception:
        return plain_text, False
