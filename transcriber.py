import whisper

def transcribe(audio_path, model_name="medium", output_path=None):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    return text
