from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel


class WhisperStt:
    def __init__(
        self,
        model_path: str = "models/stt/faster-whisper-small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "de",
        beam_size: int = 5,
    ):
        self.model = WhisperModel(
            model_size_or_path=model_path,
            device=device,
            compute_type=compute_type,
        )
        self.language = language
        self.beam_size = beam_size

    def transcribe_stream(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return ""
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
        )

        text = "".join(segment.text for segment in segments).strip()
        return text
    
    def transcribe_sample(self, audio_path: str | Path) -> str:
        audio_path = Path(audio_path)

        if not audio_path.exists() or not audio_path.is_file():
            return ""

        segments, _ = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
        )

        text = "".join(segment.text for segment in segments).strip()
        return text
    
    
# --- DEL ---
# in transcribe int16 -> float32 normalisieren
# nvidia Cuda bzw cudNN für Beelink pc?