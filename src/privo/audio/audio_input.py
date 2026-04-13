import sounddevice as sd
import numpy as np
from collections import deque


class AudioInput:
    """Audiomodul, das den Mikrofoneingang verwaltet und in konfigurierbaren Blöcken liest. Es verfügt über einen Ringpuffer, um vor dem Erkennen des Wakewords aufgenommenes Audio zu speichern, damit es später in die Spracherkennung einbezogen werden kann."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_ms: int = 80,
        channels: int = 1,
        ring_buffer_chunks: int = 20,
    ) -> None:
        """Initialisiert das AudioInput-Modul.

        Args:
            sample_rate (int, optional): Abtastrate in Hz. Defaults to 16000.
            block_ms (int, optional): Blockgröße in Millisekunden. Defaults to 80.
            channels (int, optional): Anzahl der Audiokanäle (1 = Mono, 2 = Stereo). Defaults to 1.
            ring_buffer_chunks (int, optional): Anzahl der Chunks die im Ringpuffer gespeichert werden. Defaults to 20.

        Raises:
            ValueError: Wenn die sample_rate kleiner oder gleich 0 ist.
            ValueError: Wenn die block_ms kleiner oder gleich 0 ist.
            ValueError: Wenn channels nicht 1 oder 2 ist.
            ValueError: Wenn ring_buffer_chunks kleiner oder gleich 0 ist.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate muss größer als 0 sein")
        if block_ms <= 0:
            raise ValueError("block_ms muss größer als 0 sein")
        if channels not in (1, 2):
            raise ValueError("channels muss 1 (mono) oder 2 (stereo) sein")
        if ring_buffer_chunks <= 0:
            raise ValueError("ring_buffer_chunks muss größer als 0 sein")

        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = int(sample_rate * block_ms / 1000)

        self.stream = None
        self.ring_buffer = deque(maxlen=ring_buffer_chunks)

    def start(self) -> None:
        """Startet den Audiostream."""
        if self.stream is not None:
            raise RuntimeError("Audiostream läuft bereits.")

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.block_size,
            )
            self.stream.start()
        except Exception as e:
            raise RuntimeError(f"Audiostream konnte nicht gestartet werden: {e}") from e

    def read_chunk(self) -> np.ndarray:
        """Liest einen Audio-Chunk aus dem Stream und speichert ihn im Ringbuffer.


        Raises:
            RuntimeError: Wenn der Audiostream nicht gestartet wurde.

        Returns:
            np.ndarray: Der gelesene Audio-Chunk.
        """
        if self.stream is None:
            raise RuntimeError(
                "Der Audiostream wurde noch nicht gestartet. Bitte start() aufrufen, bevor read_chunk() verwendet wird."
            )
        audio_chunk, _ = self.stream.read(self.block_size)
        if self.channels == 1:
            audio_chunk = audio_chunk.reshape(-1)

        self.ring_buffer.append(audio_chunk.copy())
        return audio_chunk

    def get_buffered_audio(self) -> list[np.ndarray]:
        """Gibt den aktuellen Inhalt des Ringpuffers zurück.

        Returns:
            list[np.ndarray]: Liste der im Ringpuffer gespeicherten Audio-Chunks.
        """
        return list(self.ring_buffer)

    def clear_buffer(self) -> None:
        """Löscht den Inhalt des Ringpuffers."""
        self.ring_buffer.clear()

    def stop(self) -> None:
        """Stoppt den Audiostream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# --- DEL ---
# channels 1 = mono
# Ringbuffer
# 16khz -> 1280 samples pro 80ms chunk
# 80ms chunks
# 20 Chunks -> 1.6s Pre-Roll
