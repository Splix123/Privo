import numpy as np


class UtteranceRecorder:
    """Utterance-Recorder Modul, welches Audio chunks aufnimmt und nach anhaltender Stille für die Weiterverarbeitung zusammenfügt."""

    def __init__(
        self, silence_threshold: float = 500.0, silence_blocks: int = 12
    ) -> None:
        """Initialisiert der UtteranceRecorder.

        Args:
            silence_threshold (float, optional): Schwellenwert für Lautstärke, ab der Stille erkannt wird. Defaults to 500.0.
            silence_blocks (int, optional): Anzahl der Stille-Chunks, um das Ende einer Äußerung zu erkennen. Defaults to 12.

        Raises:
            ValueError: Wenn silence_threshold kleiner als 0 ist.
            ValueError: Wenn silence_blocks kleiner oder gleich 0 ist.
        """
        if silence_threshold < 0:
            raise ValueError("silence_threshold muss größer oder gleich 0 sein")

        if silence_blocks <= 0:
            raise ValueError("silence_blocks muss größer als 0 sein")

        self.silence_threshold = silence_threshold
        self.silence_blocks = silence_blocks

        self.chunks: list[np.ndarray] = []
        self.silent_count = 0
        self.recording = False

    def save_pre_roll(self, pre_roll_chunks: list[np.ndarray] | None) -> None:
        """Speichert Chunks, die beim sagen den Wakewords generiert wurden und für die Vervollständigung des Satzes angehängt werden.

        Args:
            pre_roll_chunks (list[np.ndarray] | None): Chunks, die vor der eigentlichen Aufnahme hinzugefügt werden sollen.
        """
        self.chunks = []
        self.silent_count = 0
        self.recording = True

        if pre_roll_chunks:
            self.chunks.extend(pre_roll_chunks)

    def process_chunk(self, chunk: np.ndarray) -> bool:
        """Verarbeitet einen Audio_Chunk und fügt ihn der aktuellen Äußerung hinzu. Erkennt das Ende der Äußerung anhand von Stille.

        Args:
            chunk (np.ndarray): Audio-Chunk, der verarbeitet werden soll.

        Returns:
            bool: True, wenn das Ende der Äußerung erkannt wurde, sonst False.
        """
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
        """Gibt die gesammelten Audio-Chunks als ein zusammenhängendes Array zurück.

        Returns:
            np.ndarray: Zusammenhängendes NumPy-Array der gesammelten Audio-Chunks.
        """
        if not self.chunks:
            return np.array([], dtype=np.int16)

        return np.concatenate(self.chunks).astype(np.int16)

    def reset(self) -> None:
        """Setzt den UtteranceRecorder zurück und löscht alle gesammelten Audio-Chunks."""
        self.chunks = []
        self.silent_count = 0
        self.recording = False

    @staticmethod
    def _audio_level(chunk: np.ndarray) -> float:
        """Berechnet den Audiopegel eines Chunks.

        Args:
            chunk (np.ndarray): Audio-Chunk, dessen Pegel berechnet werden soll.

        Returns:
            float: Berechneter Audiopegel des Chunks.
        """
        chunk = chunk.astype(np.float32)
        return float(np.mean(np.abs(chunk)))


# --- DEL ---
# Um Audio chunks zu einem kompletten satz zusammenzufügen
# Stille am anfang und ende entfernen
