import wave
import numpy as np
from pathlib import Path
from datetime import datetime

class Debugger:
    def __init__(self, debug_dir: str, enabled: bool) -> None:
        self.enabled = enabled
        self.wakeword_counter = 1
        self.utterance_counter = 1
        if not self.enabled:
            return
        
        timestamp = datetime.now().strftime("%d.%m.%Y_%H:%M")
        self.debug_dir = Path(debug_dir) / timestamp
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def save_text(self, text: str, step: str) -> None:
        if not self.enabled:
            return
        file_path = self.debug_dir / "output.txt"

        with file_path.open("a", encoding="utf-8") as f:
            f.write(f"[{step}] " + text + "\n")

    def _write_wav(
        self,
        audio_data: np.ndarray,
        step: str,
        counter: int,
    ) -> None:
        if not self.enabled:
            return

        if audio_data.size == 0:
            return

        audio_data = np.asarray(audio_data).reshape(-1)

        if audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)

        file_path = self.debug_dir / f"{step}{counter}.wav"

        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data.tobytes())

    def save_ring_buffer(
        self,
        ring_buffer: list[np.ndarray],
        step: str,
    ) -> None:
        if not self.enabled:
            return

        if not ring_buffer:
            return

        audio_data = np.concatenate(ring_buffer)
        self._write_wav(audio_data, step=step, counter=self.wakeword_counter)
        self.wakeword_counter += 1

    def save_utterance(
        self,
        utterance_audio: np.ndarray,
        step: str,
    ) -> None:
        if not self.enabled:
            return

        self._write_wav(utterance_audio, step=step, counter=self.utterance_counter)
        self.utterance_counter += 1
