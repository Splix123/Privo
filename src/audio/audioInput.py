import sounddevice as sd
import numpy as np

class AudioInput:
    def __init__(self, sample_rate=16000, block_ms=80, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = int(sample_rate * block_ms / 1000)

        self.stream = None

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_size,
        )
        self.stream.start()

    def read_chunk(self):
        audio_chunk, _ = self.stream.read(self.block_size)
        return np.squeeze(audio_chunk)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()