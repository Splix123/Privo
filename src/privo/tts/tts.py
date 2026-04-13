from __future__ import annotations
import sys
import json
import subprocess
import sounddevice as sd
from pathlib import Path


class PiperTts:
    """Text-to-Speech-Modul (Piper)"""

    def __init__(
        self,
        model_path: str = "models/tts/de_DE-thorsten-high.onnx",
        config_path: str = "models/tts/de_DE-thorsten-high.onnx.json",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w_scale: float = 0.8,
        sentence_silence: float = 0.2,
    ) -> None:
        """Initialisiert ein Piper-Text-to-Speech-Modul mit einem ONNX-Modell und einer Konfigurationsdatei.

        Args:
            model_path (str, optional): Pfad zum Piper-Modell. Defaults to "models/tts/de_DE-thorsten-high.onnx".
            config_path (str, optional): Pfad zur Piper-Konfigurationsdatei. Defaults to "models/tts/de_DE-thorsten-high.onnx.json".
            length_scale (float, optional): Wie schnell gesprochen wird (Höher = langsamer). Defaults to 1.0.
            noise_scale (float, optional): Mehr Audiovariation. Defaults to 0.667.
            noise_w_scale (float, optional): Mehr Sprachvariation. Defaults to 0.8.
            sentence_silence (float, optional): Stille zwischen Sätzen. Defaults to 0.2.

        Raises:
            FileNotFoundError: Wenn das Piper-Modell nicht gefunden wird.
            FileNotFoundError: Wenn die Piper-Konfigurationsdatei nicht gefunden wird.
        """
        self.model_path = Path(model_path)
        self.config_path = (
            Path(config_path) if config_path else Path(f"{model_path}.json")
        )

        if length_scale <= 0:
            raise ValueError("length_scale muss größer als 0 sein")
        if noise_scale < 0:
            raise ValueError("noise_scale muss größer oder gleich 0 sein")
        if noise_w_scale < 0:
            raise ValueError("noise_w_scale muss größer oder gleich 0 sein")
        if sentence_silence < 0:
            raise ValueError("sentence_silence muss größer oder gleich 0 sein")

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
        """Liest die Sample-Rate aus der Piper-Konfigurationsdatei.

        Raises:
            ValueError: Wenn die Sample-Rate nicht aus der Piper-Config gelesen werden kann.

        Returns:
            int: Die Sample-Rate des Modells.
        """
        with self.config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sample_rate = data.get("audio", {}).get("sample_rate")
        if sample_rate is None:
            raise ValueError(
                "sample_rate konnte nicht aus der Piper-Config gelesen werden."
            )
        return int(sample_rate)

    def _build_cmd(self) -> list[str]:
        """Erstellt den Befehl zum Ausführen des Piper-Text-to-Speech-Prozesses.

        Returns:
            list[str]: Der Befehl als Liste von Argumenten.
        """
        cmd = [
            sys.executable,
            "-m",
            "piper",
            "--model",
            str(self.model_path),
            "--output-raw",
        ]

        if self.config_path is not None:
            cmd.extend(["--config", str(self.config_path)])

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
        # TODO: PiperVoice.synthesize benutzen? siehe https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
        """Synthetisiert Text mit Piper und gibt das Audiosignal als stream aus..

        Args:
            text (str): Der zu sprechende Text.
            chunk_size (int, optional): Größe der Audio-Chunks. Defaults to 4096.

        Raises:
            ValueError: Wenn chunk_size kleiner oder gleich 0 ist.
            RuntimeError: Wenn der Piper-Prozess nicht korrekt initialisiert werden konnte.
            RuntimeError: Wenn der Piper-Prozess mit einem Fehler beendet wurde.
        """

        text = text.strip()
        if not text:
            return

        if chunk_size <= 0:
            raise ValueError("chunk_size muss größer als 0 sein")

        process = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            if process.stdin is None or process.stdout is None:
                raise RuntimeError(
                    "Piper-Prozess konnte nicht korrekt initialisiert werden."
                )

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
                    error_output = process.stderr.read().decode(
                        "utf-8", errors="ignore"
                    )
                raise RuntimeError(
                    f"Piper-tts beendet mit Code {return_code}: {error_output}"
                )

        finally:
            if process.poll() is None:
                process.kill()
