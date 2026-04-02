from __future__ import annotations
import json
import subprocess
from pathlib import Path


class PiperTts:
    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        speaker: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
        sentence_silence: float | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else Path(f"{model_path}.json")
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.sentence_silence = sentence_silence

        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper-Modell nicht gefunden: {self.model_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Piper-Config nicht gefunden: {self.config_path}")

        self.sample_rate = self._read_sample_rate()

    def _read_sample_rate(self) -> int:
        with self.config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sample_rate = data.get("audio", {}).get("sample_rate")
        if sample_rate is None:
            raise ValueError("sample_rate konnte nicht aus der Piper-Config gelesen werden.")
        return int(sample_rate)

    def _build_cmd(self) -> list[str]:
        cmd = [
            "piper-tts",
            "--model",
            str(self.model_path),
            "--output-raw",
            "--quiet",
        ]

        if self.config_path is not None:
            cmd.extend(["--config", str(self.config_path)])

        if self.speaker is not None:
            cmd.extend(["--speaker", str(self.speaker)])

        if self.length_scale is not None:
            cmd.extend(["--length_scale", str(self.length_scale)])

        if self.noise_scale is not None:
            cmd.extend(["--noise_scale", str(self.noise_scale)])

        if self.noise_w_scale is not None:
            cmd.extend(["--noise-w", str(self.noise_w_scale)])

        if self.sentence_silence is not None:
            cmd.extend(["--sentence-silence", str(self.sentence_silence)])

        return cmd

    def stream_speak(self, text: str, chunk_size: int = 4096) -> None:
        import sounddevice as sd

        text = text.strip()
        if not text:
            return

        process = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            assert process.stdin is not None
            assert process.stdout is not None

            process.stdin.write(text.encode("utf-8"))
            process.stdin.close()

            with sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=0,
            ) as stream:
                while True:
                    audio_chunk = process.stdout.read(chunk_size)
                    if not audio_chunk:
                        break
                    stream.write(audio_chunk)

            return_code = process.wait()
            if return_code != 0:
                error_output = ""
                if process.stderr is not None:
                    error_output = process.stderr.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"Piper-tts beendet mit Code {return_code}: {error_output}")

        finally:
            if process.poll() is None:
                process.kill()

    def stop(self) -> None:
        import sounddevice as sd
        sd.stop()