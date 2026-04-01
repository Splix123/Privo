import numpy as np

class UtteranceRecorder:
    def __init__(self, silence_threshold: float = 500.0, silence_blocks: int = 8):
        self.silence_threshold = silence_threshold
        self.silence_blocks = silence_blocks

        self.chunks = []
        self.silent_count = 0
        self.recording = False

    def start(self, pre_roll_chunks=None):
        self.chunks = []
        self.silent_count = 0
        self.recording = True

        if pre_roll_chunks:
            self.chunks.extend(pre_roll_chunks)

    def process(self, chunk: np.ndarray) -> bool:
        if not self.recording:
            return False

        self.chunks.append(chunk.copy())

        level = self._audio_level(chunk)

        if level < self.silence_threshold:
            self.silent_count += 1
        else:
            self.silent_count = 0

        if self.silent_count >= self.silence_blocks:
            self.recording = False
            return True

        return False

    def get_audio(self) -> np.ndarray:
        if not self.chunks:
            return np.array([], dtype=np.int16)

        return np.concatenate(self.chunks).astype(np.int16)

    def reset(self):
        self.chunks = []
        self.silent_count = 0
        self.recording = False

    @staticmethod
    def _audio_level(chunk: np.ndarray) -> float:
        chunk = chunk.astype(np.float32)
        return float(np.mean(np.abs(chunk)))