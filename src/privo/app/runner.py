import time
from enum import Enum, auto
from rich.console import Console
from privo.app.config_loader import ConfigLoader
from privo.audio import AudioInput
from privo.wakeword import WakewordDetector
from privo.stt import UtteranceRecorder

class State(Enum):
    LISTENING = auto()
    RECORDING = auto()
    PROCESSING = auto()
    COOLDOWN = auto()

def run() -> None:
    console = Console() 
    console.print("\n\nStarte Privo...\n")

    # Config laden
    config_loader = ConfigLoader()
    config = config_loader.load()

    # Audio Initialisieren
    audio = AudioInput(
        **{
            k: v for k, v in {
                "sample_rate": config.get("sample_rate"),
                "block_ms": config.get("block_size"),
                "channels": config.get("channels"),
                "ring_buffer_chunks": config.get("ring_buffer_chunks"),
            }.items()
            if v is not None
        }
    )

    # Wakeword-Detector Initialisieren
    detector = WakewordDetector(
        **{
            k: v for k, v in {
                "model_path": config.get("model_path"),
                "threshold": config.get("threshold"),
                "vad_threshold": config.get("vad_threshold"),
            }.items()
            if v is not None
        }
    )

    # Utterance-Recorder Initialisieren
    recorder = UtteranceRecorder(
        **{
            k: v for k, v in {
                "silence_threshold": config.get("silence_threshold"),
                "silence_blocks": config.get("silence_blocks"),
            }.items()
            if v is not None
        }
    )

     # Whisper / STT initialisieren
    # stt = WhisperStt(
    #     WhisperSttConfig(
    #         model_name=config["stt"]["model_name"],
    #         device=config["stt"].get("device", "cpu"),
    #         compute_type=config["stt"].get("compute_type", "int8"),
    #         language=config["stt"].get("language", "de"),
    #         beam_size=config["stt"].get("beam_size", 5),
    #     )
    # )

    state = State.LISTENING
    # TODO: Cooldown abchecken und vllt in config packen
    cooldown_seconds = config.get("cooldown_seconds", 1.5)
    cooldown_until = 0.0

    utterance_audio = None

    audio.start()
    with console.status("Höre auf Wakeword...", spinner="arc") as status:
        try:
            while True:
                chunk = audio.read_chunk()

                if state == State.LISTENING:
                    detected, wakeword, score = detector.process(chunk)

                    if detected:
                        console.print(f"Wakeword erkannt: {wakeword} ({score:.2f})")

                        pre_roll_chunks = audio.get_buffered_audio()
                        recorder.start(pre_roll_chunks=pre_roll_chunks)

                        detector.reset()

                        state = State.RECORDING

                elif state == State.RECORDING:
                    status.update("Nehme Sprache auf...")
                    finished = recorder.process(chunk)

                    if finished:
                        utterance_audio = recorder.get_audio()
                        console.print(f"Aufnahme beendet. Samples: {len(utterance_audio)}\n")

                        recorder.reset()
                        state = State.PROCESSING

                elif state == State.PROCESSING:
                    status.update("Verarbeite Eingabe...")
                    audio.clear_buffer()

                    cooldown_until = time.monotonic() + cooldown_seconds
                    state = State.COOLDOWN

                elif state == State.COOLDOWN:
                    if time.monotonic() >= cooldown_until:
                        status.update("Höre wieder auf Wakeword...")
                        state = State.LISTENING

        except KeyboardInterrupt:
            console.print("Beende Privo...")
        finally:
            audio.stop()