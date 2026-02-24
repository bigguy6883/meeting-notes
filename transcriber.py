import threading
from faster_whisper import WhisperModel

_model_cache = {}
_model_lock = threading.Lock()

def _get_model(model_name):
    with _model_lock:
        if model_name not in _model_cache:
            _model_cache[model_name] = WhisperModel(
                model_name, device="cpu", compute_type="int8", cpu_threads=4
            )
        return _model_cache[model_name]

def transcribe(audio_path, model_name="small", output_path=None, return_segments=False):
    model = _get_model(model_name)
    segments_gen, _ = model.transcribe(audio_path, beam_size=1, vad_filter=True)
    segments_list = list(segments_gen)
    text = " ".join(s.text.strip() for s in segments_list).strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    if return_segments:
        return text, segments_list
    return text
