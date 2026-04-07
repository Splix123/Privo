import time
from pathlib import Path
from rich.console import Console
from privo.app.module_builder import ModuleBuilder


def benchmark(debug: bool = True) -> None:
    console = Console()
    console.print("\n\nStarte Privo Benchmark...\n")

    builder = ModuleBuilder(debug=debug)
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
                console.print(f"\n[bold]Sample:[/bold] {sample.name}")
                debugger.save_text(f"Verarbeite Sample: {sample.name}", "Sample")

                status.update(f"Transkribiere {sample.name}...")
                stt_start = time.perf_counter()
                transcript = stt.transcribe_sample(sample)
                stt_time = time.perf_counter() - stt_start

                if not transcript or not transcript.strip():
                    status.update(
                        "[bold red]Es konnte kein Audio transkribiert werden.[/bold red]"
                    )
                    debugger.save_text(
                        "Es konnte kein Audio transkribiert werden.\n", "Transkribieren"
                    )
                    time.sleep(1)
                    continue

                debugger.save_text(
                    transcript,
                    "Transkript | in " + stt_time.__format__(".2f") + " Sekunden",
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

                llm_start = time.perf_counter()
                answer = llm.generate(transcript)
                llm_time = time.perf_counter() - llm_start

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

                debugger.save_text(
                    answer, "Antwort | in " + llm_time.__format__(".2f") + " Sekunden"
                )
                status.update("[bold green]Antwort:[/bold green] " + answer)
                try:
                    tts_start = time.perf_counter()
                    tts.stream_speak(answer)
                    tts_time = time.perf_counter() - tts_start
                    debugger.save_text(
                        f"\n",
                        "Sprechen | in " + tts_time.__format__(".2f") + " Sekunden",
                    )
                except Exception as e:
                    console.print(f"\n[bold red]TTS-Fehler:[/bold red] {e}")
                    debugger.save_text(f"TTS-Fehler: {e}\n", "TTS Fehler")

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Benchmark abgebrochen.[/bold yellow]")

        console.print("Beende Privo...")
