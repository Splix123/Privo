import sounddevice as sd
import numpy as np
from collections import deque


class AudioInput:
    def __init__(
        self,
        sample_rate: int = 16000,
        block_ms: int = 80,
        channels: int = 1,
        ring_buffer_chunks: int = 20,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = int(sample_rate * block_ms / 1000)

        self.stream = None
        self.ring_buffer = deque(maxlen=ring_buffer_chunks)

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_size,
        )
        self.stream.start()

    def read_chunk(self):
        if self.stream is None:
            raise RuntimeError(
                "Der Audiostream wurde noch nicht gestartet. Bitte start() aufrufen, bevor read_chunk() verwendet wird."
            )
        audio_chunk, _ = self.stream.read(self.block_size)
        audio_chunk = np.squeeze(audio_chunk)

        self.ring_buffer.append(audio_chunk.copy())
        return audio_chunk

    def get_buffered_audio(self):
        return list(self.ring_buffer)

    def clear_buffer(self):
        self.ring_buffer.clear()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


# --- DEL ---
# channels 1 = mono
# Ringbuffer
# 16khz -> 1280 samples pro 80ms chunk
# 80ms chunks
# 20 Chunks -> 1.6s Pre-Roll
