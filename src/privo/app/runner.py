import time
import numpy as np
from enum import Enum, auto
from rich.console import Console
from privo.app.config_loader import ConfigLoader
from privo.app.debugger import Debugger
from privo.audio import AudioInput
from privo.wakeword import WakewordDetector
from privo.stt import UtteranceRecorder
from privo.stt import WhisperStt
from privo.llm import LocalLLM
from privo.tts import PiperTts

class State(Enum):
    LISTENING = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    GENERATING = auto()
    SPEAKING = auto()
    FOLLOWUP = auto()

def run(debug: bool = False) -> None:
    console = Console() 
    console.print("\n\nStarte Privo...\n")
    

    # Config laden
    config_loader = ConfigLoader()
    config = config_loader.load()

    # Debugger initialisieren (optional)
    debugger = Debugger(debug_dir=config.get("debug_dir", "debug"), enabled=debug)

    # Audio Initialisieren
    audio = AudioInput(
        **{
            k: v for k, v in {
                "sample_rate": config.get("au_sample_rate"),
                "block_ms": config.get("au_block_size"),
                "channels": config.get("au_channels"),
                "ring_buffer_chunks": config.get("au_ring_buffer_chunks"),
            }.items()
            if v is not None
        }
    )
    console.print("Audio-Modul geladen")

    # Wakeword-Detector Initialisieren (OpenWakeWord)
    detector = WakewordDetector(
        **{
            k: v for k, v in {
                "model_path": config.get("wwd_model_path"),
                "threshold": config.get("wwd_threshold"),
                "vad_threshold": config.get("wwd_vad_threshold"),
            }.items()
            if v is not None
        }
    )
    console.print("Wakeword-Detector geladen")

    # Utterance-Recorder Initialisieren
    recorder = UtteranceRecorder(
        **{
            k: v for k, v in {
                "silence_threshold": config.get("stt_silence_threshold"),
                "silence_blocks": config.get("stt_silence_blocks"),
            }.items()
            if v is not None
        }
    )
    console.print("Utterance-Recorder-Modul geladen")

    # STT initialisieren (Whisper)
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
    console.print("STT-Modul geladen")

    # LLM backend initialisieren (llama.cpp)
    llm = LocalLLM(
        **{
            k: v for k, v in {
                "model_path": config.get("llm_model_path"),
                "n_ctx": config.get("llm_n_ctx"),
                "n_gpu_layers": config.get("llm_n_gpu_layers"),
                "verbose": config.get("llm_verbose"),
                "max_tokens": config.get("llm_max_tokens"),
                "temperature": config.get("llm_temperature"),
                "history_limit": config.get("llm_history_limit"),
            }.items()
            if v is not None
        }
    )
    console.print("LLM-Modul geladen")

    # TTS backend initialisieren (Piper)
    tts = PiperTts(
        **{
            k: v for k, v in {
                "model_path": config.get("tts_model_path"),
                "config_path": config.get("tts_config_path"),
                "speaker": config.get("tts_speaker"),
                "length_scale": config.get("tts_length_scale"),
                "noise_scale": config.get("tts_noise_scale"),
                "noise_w_scale": config.get("tts_noise_w_scale"),
                "sentence_silence": config.get("tts_sentence_silence"),
            }.items()
            if v is not None
        }
    )
    console.print("TTS-Modul geladen\n")

    state = State.LISTENING
    utterance_audio = None
    transcript = ""
    conversation_active = False
    last_interaction_time = 0.0
    silence_threshold = config.get("stt_silence_threshold", 500.0)
    conversation_timeout = config.get("llm_conversation_timeout", 8.0)
    answer = ""

    audio.start()
    with console.status("Höre auf Wakeword...", spinner="arc") as status:
        try:
            while True:
                chunk = audio.read_chunk()

                if state == State.LISTENING:
                    detected, wakeword, score = detector.process(chunk)

                    if detected:
                        llm.reset_history()
                        conversation_active = True
                        last_interaction_time = time.time()

                        pre_roll_chunks = audio.get_buffered_audio()
                        debugger.save_ring_buffer(pre_roll_chunks, "Wakeword")
                        recorder.start(pre_roll_chunks=pre_roll_chunks)

                        detector.reset()
                        state = State.RECORDING

                elif state == State.RECORDING:
                    status.update("Nehme Sprache auf...")
                    finished = recorder.process(chunk)

                    if finished:
                        utterance_audio = recorder.get_audio()
                        debugger.save_utterance(utterance_audio, "Eingabe")
                        audio.clear_buffer()
                        recorder.reset()
                        state = State.TRANSCRIBING

                elif state == State.TRANSCRIBING:
                    status.update("Verarbeite Eingabe...")
                    if utterance_audio is not None and len(utterance_audio) > 0:
                        transcript = stt.transcribe(utterance_audio)
                        debugger.save_text(transcript, "Transkript")
                        cleaned = transcript.strip()
                        lower = cleaned.lower()

                        for wakeword in ["hey alexa", "alexa", "alex"]:
                            if lower.startswith(wakeword):
                                cleaned = cleaned[len(wakeword):].lstrip(" ,.!?:;")
                                break

                        if not cleaned:
                            console.print("\n[bold red]Nur Wakeword erkannt, keine eigentliche Eingabe.[/bold red]")
                            utterance_audio = None
                            state = State.LISTENING
                            continue
                        transcript = cleaned
                        debugger.save_text(transcript + "\n", "Bereinigtes Transkript")
                    else:
                        console.print("\n[bold red]Es konnte kein Audio transkribiert werden.[/bold red]")
                        state = State.LISTENING
                        continue
                    utterance_audio = None
                    last_interaction_time = time.time()
                    state = State.GENERATING

                elif state == State.GENERATING:
                    status.update("Generiere Antwort...")

                    if not transcript:
                        console.print("\n[bold red]Es wurde nur Stille aufgenommen – keine Antwort generiert.[/bold red]")
                        state = State.FOLLOWUP if conversation_active else State.LISTENING
                        continue

                    answer = llm.generate(
                        user_text=transcript,
                        system_prompt=config.get("llm_system_prompt", "Du bist ein hilfreicher Sprachassistent"),
                    )
                    debugger.save_text(answer + "\n", "LLM-Antwort")
                    last_interaction_time = time.time()
                    state = State.SPEAKING

                elif state == State.SPEAKING:
                    status.update("[bold green]Antwort:[/bold green] " + answer)
                    try:
                        audio.clear_buffer()
                        tts.stream_speak(answer)
                    except Exception as e:
                        console.print(f"\n[bold red]TTS-Fehler:[/bold red] {e}")

                    last_interaction_time = time.time()

                    if conversation_active:
                        status.update("Höre weiter zu...")
                        state = State.FOLLOWUP
                    else:
                        status.update("Höre auf Wakeword...")
                        state = State.LISTENING

                elif state == State.FOLLOWUP:
                    if time.time() - last_interaction_time > conversation_timeout:
                        conversation_active = False
                        llm.reset_history()
                        recorder.reset()
                        audio.clear_buffer()
                        detector.reset()
                        status.update("Höre auf Wakeword...")
                        state = State.LISTENING
                        continue

                    if not recorder.recording:
                        # TODO:Silero-VAD überall benutzen!!! 
                        rms = np.sqrt(np.mean(np.square(chunk.astype(np.float32))))

                        if rms > silence_threshold:
                            pre_roll_chunks = audio.get_buffered_audio()
                            recorder.start(pre_roll_chunks=pre_roll_chunks)

                    else:
                        finished = recorder.process(chunk)

                        if finished:
                            utterance_audio = recorder.get_audio()
                            debugger.save_utterance(utterance_audio, "Eingabe")
                            audio.clear_buffer()
                            recorder.reset()
                            last_interaction_time = time.time()
                            state = State.TRANSCRIBING

        except KeyboardInterrupt:
            console.print("Beende Privo...")
        finally:
            audio.stop()


# --- DEL ---
# Statemachine
# Rich-text verarbeitung
# TODO:Silero-VAD überall benutzen!!! 
# TODO: LLM stream einbauen!!!