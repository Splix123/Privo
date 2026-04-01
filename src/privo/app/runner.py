import time
from enum import Enum, auto
from rich.console import Console
from privo.app.config_loader import ConfigLoader
from privo.audio import AudioInput
from privo.wakeword import WakewordDetector
from privo.stt import UtteranceRecorder
from privo.stt import WhisperStt

class State(Enum):
    LISTENING = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    GENERATING = auto()
    SPEAKING = auto()

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
    stt = WhisperStt(
        **{
            k: v for k, v in {
                "model_path": config.get("stt_model_path"),
                "device": config.get("stt_device"),
                "compute_type": config.get("stt_compute_type"),
                "language": config.get("stt_language"),
                "beam_size": config.get("stt_beam_size"),
            }.items()
            if v is not None
        }
    )

    state = State.LISTENING
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
                        audio.clear_buffer()
                        recorder.reset()
                        state = State.TRANSCRIBING

                elif state == State.TRANSCRIBING:
                    status.update("Verarbeite Eingabe...")
                    if utterance_audio is not None and len(utterance_audio) > 0:
                        text = stt.transcribe(utterance_audio)
                        console.print(f"\n[bold green]Transkript:[/bold green] {text}")
                    else:
                        console.print("\n[yellow]Keine Audioaufnahme zum Transkribieren vorhanden.[/yellow]")

                    utterance_audio = None
                    state = State.GENERATING

                elif state == State.GENERATING:
                    status.update("Generiere Antwort...")
                    time.sleep(0.1)  # Platzhalter für tatsächliche Verarbeitung
                    state = State.SPEAKING

                elif state == State.SPEAKING:
                    status.update("Gebe Antwort aus...")
                    time.sleep(0.1)  # Platzhalter für tatsächliche Sprachausgabe
                    status.update("Höre wieder auf Wakeword...")
                    state = State.LISTENING

        except KeyboardInterrupt:
            console.print("Beende Privo...")
        finally:
            audio.stop()


# --- DEL ---
# Statemachine
# Rich-text verarbeitung