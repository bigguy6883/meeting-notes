import subprocess
import os
from datetime import datetime

class Recorder:
    def __init__(self, mic_device, output_dir):
        self.mic_device = mic_device
        self.output_dir = output_dir
        self._process = None
        self._filepath = None

    def start(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(self.output_dir, f"meeting_{timestamp}.mp3")
        self._process = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "alsa",
            "-i", self.mic_device,
            "-ar", "16000",
            "-ac", "1",
            self._filepath
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self._filepath

    def stop(self):
        if self._process is None:
            return None
        self._process.terminate()
        self._process.wait()
        filepath = self._filepath
        self._process = None
        self._filepath = None
        return filepath

    def is_recording(self):
        return self._process is not None and self._process.poll() is None
