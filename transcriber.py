from faster_whisper import WhisperModel

def transcribe(audio_path, model_name="small", output_path=None, return_segments=False):
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments_gen, _ = model.transcribe(audio_path, beam_size=5)
    segments_list = list(segments_gen)
    text = " ".join(s.text.strip() for s in segments_list).strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    if return_segments:
        return text, segments_list
    return text
