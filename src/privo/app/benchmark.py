import os
import time
import psutil
import numpy as np
import soundfile as sf
from pathlib import Path
from rich.console import Console
from .chat import Chat
from privo.app.module_builder import ModuleBuilder


def get_resources(process: psutil.Process) -> dict:
    """Liest die aktuellen CPU- und RAM-Werte des laufenden Prozesses aus.

    Args:
        process (psutil.Process): Der zu überwachende Prozess.

    Returns:
        dict: Ein Dictionary mit den aktuellen CPU- und RAM-Werten.
    """

    return {
        "cpu_percent": process.cpu_percent(interval=None),
        "ram_mb": process.memory_info().rss / 1024 / 1024,
    }


def format_resources(before: dict, after: dict) -> str:
    """Formatiert zwei Ressourcen-Snapshots inklusive RAM-Differenz.

    Args:
        before (dict): Ressourcen-Snapshot vor der Ausführung.
        after (dict): Ressourcen-Snapshot nach der Ausführung.

    Returns:
        str: Formatierter String mit CPU- und RAM-Werten sowie RAM-Differenz.
    """

    ram_diff = after["ram_mb"] - before["ram_mb"]
    return (
        f" | CPU: {after['cpu_percent']:.1f}%"
        f" | RAM: {after['ram_mb']:.2f} MB"
        f" | RAM Δ: {ram_diff:+.2f} MB"
    )


def load_sample_chunks(
    sample_path: Path,
    sample_rate: int,
    block_ms: int,
) -> list[np.ndarray]:
    """Lädt ein WAV-Sample und zerlegt es in gepolsterte Int16-Blöcke.

    Args:
        sample_path (Path): Pfad zur WAV-Datei.
        sample_rate (int): Erwartete Abtastrate der WAV-Datei.
        block_ms (int): Größe der Blöcke in Millisekunden.

    Raises:
        ValueError: Wenn die Abtastrate der WAV-Datei nicht der erwarteten Abtastrate entspricht.

    Returns:
        list[np.ndarray]: Liste der gepolsterten Int16-Blöcke.
    """

    audio, file_sample_rate = sf.read(str(sample_path), dtype="int16", always_2d=False)

    if file_sample_rate != sample_rate:
        raise ValueError(
            f"Sample {sample_path.name} hat {file_sample_rate} Hz, "
            f"erwartet werden {sample_rate} Hz."
        )

    if audio.ndim > 1:
        audio = audio[:, 0]

    block_size = int(sample_rate * block_ms / 1000)

    chunks = []
    for start in range(0, len(audio), block_size):
        chunk = audio[start : start + block_size]

        if len(chunk) < block_size:
            chunk = np.pad(chunk, (0, block_size - len(chunk)))

        chunks.append(chunk.astype(np.int16))

    return chunks


def benchmark(debug: bool = True) -> None:
    """Führt den kompletten Offline-Benchmark für alle WAV-Samples aus.

    Args:
        debug (bool, optional): Aktiviert den Debug-Modus. Defaults to True.
    """

    console = Console()
    chat = Chat(console=console)
    console.print("\n\nStarte Privo Benchmark...\n")

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    builder = ModuleBuilder(console, debug=debug)
    config, debugger, detector, recorder, stt, llm, tts = builder.build_benchmark()

    samples_dir = Path(config.get("benchmark_samples_dir", "tests/audio_samples"))
    if not samples_dir.exists() or not samples_dir.is_dir():
        console.print(
            f"[bold red]Sample-Verzeichnis nicht gefunden:[/bold red] {samples_dir}"
        )
        debugger.save_text(
            f"Sample-Verzeichnis nicht gefunden: {samples_dir}\n", "Samples suchen"
        )
        return

    sample_files = sorted(samples_dir.glob("*.wav"))
    if sample_files:
        console.print(
            f"[bold green]Gefundene Sample-Dateien:[/bold green] {[f.name for f in sample_files]}"
        )
        debugger.save_text(
            f"Sample-Dateien: {[f.name for f in sample_files]}\n", "Samples suchen"
        )
    if not sample_files:
        console.print(
            f"[bold yellow]Keine .wav-Dateien in {samples_dir} gefunden[/bold yellow]"
        )
        debugger.save_text("Keine .wav-Dateien gefunden\n", "Samples suchen")
        return

    wakewords_to_strip = config.get("wwd_to_strip", ["hey alexa", "alexa", "alex"])

    with console.status("Bearbeite Samples...", spinner="arc") as status:
        try:
            for sample in sample_files:
                console.print(
                    f"\n\n[bold]Sample:[/bold] {sample.name}", justify="center"
                )
                debugger.save_text(f"Verarbeite Sample: {sample.name}", "Sample")

                sample_resources_start = get_resources(process)
                sample_start = time.perf_counter()

                status.update(f"Prüfe Wakeword in {sample.name}...")

                detector.reset()
                recorder.reset()

                sample_rate = config.get("au_sample_rate", 16000)
                block_ms = config.get("au_block_size", 80)
                ring_buffer_chunks = config.get("au_ring_buffer_chunks", 20)

                chunks = load_sample_chunks(
                    sample_path=sample,
                    sample_rate=sample_rate,
                    block_ms=block_ms,
                )

                ring_buffer: list[np.ndarray] = []
                wakeword_detected = False
                wakeword_name = None
                wakeword_score = None
                utterance_audio = None

                wwd_start = time.perf_counter()

                for chunk in chunks:
                    if not wakeword_detected:
                        ring_buffer.append(chunk.copy())

                        if len(ring_buffer) > ring_buffer_chunks:
                            ring_buffer.pop(0)

                        detected, wakeword, score = detector.process(chunk)

                        if detected:
                            wakeword_detected = True
                            wakeword_name = wakeword
                            wakeword_score = score

                            debugger.save_text(
                                f"Wakeword erkannt: {wakeword_name} "
                                f"(Score: {wakeword_score})\n",
                                "Wakeword Benchmark",
                            )

                            recorder.save_pre_roll(ring_buffer)
                            detector.reset()

                    else:
                        finished = recorder.process_chunk(chunk)

                        if finished:
                            utterance_audio = recorder.get_audio()
                            recorder.reset()
                            break

                wwd_time = time.perf_counter() - wwd_start

                if not wakeword_detected:
                    status.update(
                        f"[bold yellow]Kein Wakeword in {sample.name} erkannt.[/bold yellow]"
                    )
                    debugger.save_text(
                        f"Kein Wakeword erkannt in {sample.name}\n",
                        "Wakeword Benchmark",
                    )
                    time.sleep(1)
                    continue

                if utterance_audio is None:
                    utterance_audio = recorder.get_audio()
                    recorder.reset()

                if utterance_audio is None or len(utterance_audio) == 0:
                    status.update(
                        "[bold red]Wakeword erkannt, aber keine Äußerung aufgenommen.[/bold red]"
                    )
                    debugger.save_text(
                        "Wakeword erkannt, aber keine Äußerung aufgenommen.\n",
                        "Recording Benchmark",
                    )
                    time.sleep(1)
                    continue

                debugger.save_utterance(utterance_audio, "Benchmark Eingabe")

                status.update(f"Transkribiere {sample.name}...")
                stt_resources_start = get_resources(process)
                stt_start = time.perf_counter()
                transcript = stt.transcribe_stream(utterance_audio)
                stt_time = time.perf_counter() - stt_start
                stt_resources_end = get_resources(process)

                debugger.save_text(
                    f"Wakeword-Zeit: {wwd_time:.2f} Sekunden\n",
                    "Wakeword Benchmark",
                )

                if not transcript or not transcript.strip():
                    status.update(
                        "[bold red]Es konnte kein Audio transkribiert werden.[/bold red]"
                    )
                    debugger.save_text(
                        "Es konnte kein Audio transkribiert werden.\n", "Transkribieren"
                    )
                    time.sleep(1)
                    continue

                chat.print_chat(
                    transcript,
                    name=f"{sample.name}",
                    align="left",
                )

                debugger.save_text(
                    transcript,
                    "Transkript | in "
                    + stt_time.__format__(".2f")
                    + " Sekunden"
                    + format_resources(stt_resources_start, stt_resources_end),
                )

                cleaned = transcript.strip()
                lower = cleaned.lower()

                for wakeword in wakewords_to_strip:
                    if lower.startswith(wakeword.lower()):
                        cleaned = cleaned[len(wakeword) :].lstrip(" ,.!?:;")
                        break

                if not cleaned:
                    status.update(
                        "[bold yellow]Nur Wakeword erkannt, keine eigentliche Eingabe.[/bold yellow]"
                    )
                    debugger.save_text(
                        "Nur Wakeword erkannt, keine eigentliche Eingabe.\n",
                        "Transkript bereinigen",
                    )
                    time.sleep(1)
                    continue

                transcript = cleaned
                debugger.save_text(transcript, "Bereinigtes Transkript")

                status.update(f"Generiere Antwort für {sample.name}...")

                llm_resources_start = get_resources(process)
                llm_start = time.perf_counter()
                answer = llm.generate(user_text=transcript)
                llm_time = time.perf_counter() - llm_start
                llm_resources_end = get_resources(process)

                if not answer or not answer.strip():
                    status.update(
                        "[bold red]Es konnte keine Antwort generiert werden.[/bold red]"
                    )
                    debugger.save_text(
                        "Es konnte keine Antwort generiert werden.\n",
                        "Antwort generieren",
                    )
                    time.sleep(1)
                    continue
                chat.print_chat(
                    answer,
                    name="Privo",
                    align="right",
                )
                debugger.save_text(
                    answer,
                    "Antwort | in "
                    + llm_time.__format__(".2f")
                    + " Sekunden"
                    + format_resources(llm_resources_start, llm_resources_end),
                )
                status.update("Sprechen...")
                try:
                    tts_resources_start = get_resources(process)
                    tts_start = time.perf_counter()
                    tts.stream_speak(answer)
                    tts_time = time.perf_counter() - tts_start
                    tts_resources_end = get_resources(process)

                    debugger.save_text(
                        f"\n",
                        "Sprechen | in "
                        + tts_time.__format__(".2f")
                        + " Sekunden"
                        + format_resources(tts_resources_start, tts_resources_end),
                    )
                except Exception as e:
                    console.print(f"\n[bold red]TTS-Fehler:[/bold red] {e}")
                    debugger.save_text(f"TTS-Fehler: {e}\n", "TTS Fehler")

                sample_time = time.perf_counter() - sample_start
                sample_resources_end = get_resources(process)
                debugger.save_text(
                    (
                        f"Gesamtdauer: {sample_time:.2f} Sekunden\n"
                        f"CPU: {sample_resources_end['cpu_percent']:.1f}%\n"
                        f"RAM: {sample_resources_end['ram_mb']:.2f} MB\n"
                        f"RAM Δ: {sample_resources_end['ram_mb'] - sample_resources_start['ram_mb']:+.2f} MB\n"
                    ),
                    "Gesamtmessung Sample",
                )

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Benchmark abgebrochen.[/bold yellow]")

        console.print("Beende Privo...")
