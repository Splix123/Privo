from pathlib import Path
import numpy as np
from openwakeword.model import Model


class WakewordDetector:
    """Wakeword-detektor Modul (OpenWakeWord)"""

    def __init__(
        self,
        model_path: str = "models/wakeword/alexa_v0.1.onnx",
        threshold: float = 0.5,
        vad_threshold: float = 0.5,
    ) -> None:
        """Initialisiert ein OpenWakeWord-Modul mit einem ONNX-Modell und einem Schwellenwert.
        Args:
            model_path (str, optional): Pfad zum ONNX-Modell. Defaults to "models/wakeword/alexa_v0.1.onnx".
            threshold (float, optional): Schwellenwert für die Wakeword-Erkennung. Defaults to 0.5.
            vad_threshold (float, optional): Schwellenwert für die Sprachaktivitätserkennung. Defaults to 0.5.

        Raises:
            FileNotFoundError: Wenn das Wakeword-Modell nicht gefunden wird.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold muss zwischen 0.0 und 1.0 liegen")

        if not 0.0 <= vad_threshold <= 1.0:
            raise ValueError("vad_threshold muss zwischen 0.0 und 1.0 liegen")

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Wakeword-Modell nicht gefunden: {model_file}")

        self.model = Model(
            wakeword_models=[str(model_file)],
            vad_threshold=vad_threshold,
        )

        self.threshold = threshold
        self.was_above_threshold = False

    def process(self, audio_chunk: np.ndarray) -> tuple[bool, str | None, float | None]:
        """Testet einen Audio-Chunk auf Wakewords, indem der Score mit einem Schwellenwert verglichen wird.

        Args:
            audio_chunk (np.ndarray): Audio-Chunk als NumPy-Array.

        Returns:
            tuple[bool, str | None, float | None]: Tuple mit Erkennungsstatus, Wakeword-Name und Score.
        """
        predictions = self.model.predict(audio_chunk)

        if not predictions:
            return False, None, None

        wakeword, score = next(iter(predictions.items()))

        is_above = score >= self.threshold
        detected = is_above and not self.was_above_threshold

        self.was_above_threshold = is_above

        if detected:
            return True, wakeword, score

        return False, None, None

    def reset(self) -> None:
        """Setzt den internen Zustand und die Edge-Logik des Detektors zurück."""
        self.was_above_threshold = False
        self.model.reset()


# --- DEL ---
# model_path in config
# Threshold mit Edge detection (also nur Triggern wenn vorher unter Threshold war)
