from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel


class WhisperStt:
    """Speech-to-Text-Modul (faster-whisper)"""

    def __init__(
        self,
        model_path: str = "models/stt/faster-whisper-small",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "de",
        beam_size: int = 5,
    ) -> None:
        """Initialisiert das faster-whisper STT-Modul.

        Args:
            model_path (str, optional): Pfad zum Whisper-Modell. Defaults to "models/stt/faster-whisper-small".
            device (str, optional): Berechnung auf CPU oder CUDA. Defaults to "auto".
            compute_type (str, optional): Berechnungstyp. Defaults to "int8".
            language (str, optional): Sprache für die Transkription. Defaults to "de".
            beam_size (int, optional): Anzahl der parallel verfolgten Hypothesen. Defaults to 5.

        Raises:
            ValueError: Wenn beam_size kleiner oder gleich 0 ist.
        """
        if beam_size <= 0:
            raise ValueError("beam_size muss größer als 0 sein")

        self.model = WhisperModel(
            model_size_or_path=model_path,
            device=device,
            compute_type=compute_type,
        )
        self.language = language
        self.beam_size = beam_size

    def transcribe_stream(self, audio: np.ndarray) -> str:
        """Transkribiert einen Audio-Stream in Text.

        Args:
            audio (np.ndarray): Audio-Chunks als NumPy-Array.

        Returns:
            str: Transkribierter Text.
        """
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
        """Transkribiert eine Audiodatei in Text.

        Args:
            audio_path (str | Path): Pfad zur Audiodatei.

        Returns:
           str: Transkribierter Text.
        """
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
