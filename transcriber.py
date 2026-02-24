import dataclasses
from groq import Groq


@dataclasses.dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(audio_path, output_path=None, return_segments=False):
    client = Groq()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
    text = response.text.strip()
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    if return_segments:
        segments = [Segment(start=s.start, end=s.end, text=s.text) for s in response.segments]
        return text, segments
    return text
