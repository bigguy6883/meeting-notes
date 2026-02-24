from faster_whisper import WhisperModel

def transcribe(audio_path, model_name="small", output_path=None):
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    return text
