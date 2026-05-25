import os
import time
import psutil
from pathlib import Path
from rich.console import Console
from .chat import Chat
from privo.app.module_builder import ModuleBuilder


def get_resources(process: psutil.Process) -> dict:
    return {
        "cpu_percent": process.cpu_percent(interval=None),
        "ram_mb": process.memory_info().rss / 1024 / 1024,
    }


def format_resources(before: dict, after: dict) -> str:
    ram_diff = after["ram_mb"] - before["ram_mb"]
    return (
        f" | CPU: {after['cpu_percent']:.1f}%"
        f" | RAM: {after['ram_mb']:.2f} MB"
        f" | RAM Δ: {ram_diff:+.2f} MB"
    )


def benchmark(debug: bool = True) -> None:
    console = Console()
    chat = Chat(console=console)
    console.print("\n\nStarte Privo Benchmark...\n")

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    builder = ModuleBuilder(console, debug=debug)
    config, debugger, stt, llm, tts = builder.build_benchmark()

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

                status.update(f"Transkribiere {sample.name}...")
                stt_resources_start = get_resources(process)
                stt_start = time.perf_counter()
                transcript = stt.transcribe_sample(sample)
                stt_time = time.perf_counter() - stt_start
                stt_resources_end = get_resources(process)

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
                answer = llm.generate(transcript)
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
