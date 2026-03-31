from pathlib import Path
from openwakeword.model import Model

class WakewordDetector:
    def __init__(self, model_path: str, threshold: float = 0.5, vad_threshold: float = 0.5):

        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Wakeword-Modell nicht gefunden: {model_file}")

        self.model = Model(
            wakeword_models=[str(model_file)],
            vad_threshold=vad_threshold,
        )
        self.threshold = threshold

    def process(self, audio_chunk):
        predictions = self.model.predict(audio_chunk)

        for wakeword, score in predictions.items():
            if score >= self.threshold:
                return True, wakeword, score

        return False, None, None