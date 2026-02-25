import os
import subprocess
import tempfile

import numpy as np


def _convert_to_wav(audio_path):
    """Convert audio to 16kHz mono WAV via ffmpeg. Returns temp WAV path (caller must delete)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp.name],
            capture_output=True,
            check=True,
        )
    except Exception:
        os.unlink(tmp.name)
        raise
    return tmp.name


def diarize(audio_path, whisper_segments):
    """
    Assign speaker labels to Whisper segments using resemblyzer embeddings.

    whisper_segments: list of segment objects with .start, .end, .text

    Returns (transcript: str, diarized: bool).
    Falls back to plain transcript (diarized=False) when:
    - resemblyzer or scikit-learn is not installed
    - fewer than 2 speakers are detected
    - any exception occurs
    """
    plain_text = " ".join((seg.text or "").strip() for seg in whisper_segments)

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from sklearn.cluster import AgglomerativeClustering

        wav_path = _convert_to_wav(audio_path)
        try:
            wav = preprocess_wav(wav_path)
        finally:
            os.unlink(wav_path)

        encoder = VoiceEncoder()
        embeddings = []
        valid_indices = []
        for i, seg in enumerate(whisper_segments):
            start = int(seg.start * 16000)
            end = int(seg.end * 16000)
            chunk = wav[start:end]
            if len(chunk) < 1600:  # skip segments shorter than 0.1s
                continue
            embeddings.append(encoder.embed_utterance(chunk))
            valid_indices.append(i)

        if len(embeddings) < 2:
            return plain_text, False

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.6,
            metric="cosine",
            linkage="complete",
        ).fit_predict(np.array(embeddings))

        if len(set(labels)) < 2:
            return plain_text, False

        speaker_map = {c: f"Speaker_{i:02d}" for i, c in enumerate(sorted(set(labels)))}
        segment_speakers = ["Speaker_00"] * len(whisper_segments)
        for idx, label in zip(valid_indices, labels):
            segment_speakers[idx] = speaker_map[label]

        lines = []
        current_speaker = None
        current_texts = []
        for i, seg in enumerate(whisper_segments):
            speaker = segment_speakers[i]
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
