from pathlib import Path
from openwakeword.model import Model


class WakewordDetector:
    def __init__(
        self,
        model_path: str = "models/wakeword/alexa_v0.1.onnx",
        threshold: float = 0.5,
        vad_threshold: float = 0.5,
    ):
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Wakeword-Modell nicht gefunden: {model_file}")

        self.model = Model(
            wakeword_models=[str(model_file)],
            vad_threshold=vad_threshold,
        )

        self.threshold = threshold
        self.was_above_threshold = False
        self.wakeword_name = None

    def process(self, audio_chunk):
        predictions = self.model.predict(audio_chunk)

        wakeword, score = next(iter(predictions.items()))

        is_above = score >= self.threshold
        detected = is_above and not self.was_above_threshold

        self.was_above_threshold = is_above

        if detected:
            return True, wakeword, score

        return False, None, None

    def reset(self):
        self.was_above_threshold = False
        self.model.reset()


# --- DEL ---
# model_path in config
# Threshold mit Edge detection (also nur Triggern wenn vorher unter Threshold war)
