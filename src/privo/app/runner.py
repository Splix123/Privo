import time
import numpy as np
from enum import Enum, auto
from rich.console import Console
from .chat import Chat
from privo.app.module_builder import ModuleBuilder


class State(Enum):
    LISTENING = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    GENERATING = auto()
    SPEAKING = auto()
    FOLLOWUP = auto()


def run(debug: bool = False) -> None:
    console = Console()
    chat = Chat(console=console)
    console.print("\n\nStarte Privo...\n")

    builder = ModuleBuilder(debug=debug)
    config, debugger, audio, detector, recorder, stt, llm, tts = builder.build_all()

    state = State.LISTENING
    utterance_audio = None
    wwd_to_strip = config.get("wwd_to_strip", ["hey alexa", "alexa", "alex"])
    transcript = ""
    conversation_active = False
    last_interaction_time = 0.0
    silence_threshold = config.get("stt_silence_threshold", 500.0)
    conversation_timeout = config.get("llm_conversation_timeout", 8.0)
    system_prompt = config.get(
        "llm_system_prompt", "Du bist ein hilfreicher Sprachassistent"
    )
    answer = ""

    audio.start()
    with console.status("Höre auf Wakeword...", spinner="arc") as status:
        try:
            while True:
                chunk = audio.read_chunk()

                if state == State.LISTENING:
                    detected, wakeword, score = detector.process(chunk)

                    if detected:
                        debugger.save_text(
                            f"Wakeword erkannt: {wakeword} (Score: {score})\n",
                            "Wakeword",
                        )
                        if not conversation_active:
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
                        transcript = stt.transcribe_stream(utterance_audio)
                        chat.print_chat(
                            transcript,
                            name="Du",
                            align="left",
                        )
                        debugger.save_text(transcript, "Transkript")
                        cleaned = transcript.strip()
                        lower = cleaned.lower()

                        for wakeword_to_strip in wwd_to_strip:
                            if lower.startswith(wakeword_to_strip):
                                cleaned = cleaned[len(wakeword_to_strip) :].lstrip(
                                    " ,.!?:;"
                                )
                                break

                        if not cleaned:
                            console.print(
                                "\n[bold red]Nur Wakeword erkannt, keine eigentliche Eingabe.[/bold red]"
                            )
                            utterance_audio = None
                            conversation_active = True
                            last_interaction_time = time.time()
                            state = State.FOLLOWUP
                            continue
                        transcript = cleaned
                        debugger.save_text(transcript + "\n", "Bereinigtes Transkript")
                    else:
                        console.print(
                            "\n[bold red]Es konnte kein Audio transkribiert werden.[/bold red]"
                        )
                        state = State.LISTENING
                        continue
                    utterance_audio = None
                    last_interaction_time = time.time()
                    state = State.GENERATING

                elif state == State.GENERATING:
                    status.update("Generiere Antwort...")

                    if not transcript:
                        console.print(
                            "\n[bold red]Es wurde nur Stille aufgenommen – keine Antwort generiert.[/bold red]"
                        )
                        state = (
                            State.FOLLOWUP if conversation_active else State.LISTENING
                        )
                        continue

                    answer = llm.generate(
                        user_text=transcript,
                        system_prompt=system_prompt,
                    )
                    debugger.save_text(answer + "\n", "LLM-Antwort")
                    last_interaction_time = time.time()
                    state = State.SPEAKING

                elif state == State.SPEAKING:
                    chat.print_chat(
                        answer,
                        name="Privo",
                        align="right",
                    )
                    status.update("Sprechen...")
                    try:
                        audio.clear_buffer()
                        tts.stream_speak(answer)
                        audio.clear_buffer()
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
