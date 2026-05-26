import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RemoteEntryNotFoundError,
)
from huggingface_hub.utils import logging as hf_logging
from rich.console import Console
from rich.prompt import Confirm

hf_logging.set_verbosity_error()


@dataclass(frozen=True)
class ModelDownload:
    """Beschreibt ein Modell vom Hugging Face Hub."""

    name: str
    repo_id: str
    target_path: Path
    filename: str | None = None


MODEL_DIRS = (
    Path("models/wakeword"),
    Path("models/stt"),
    Path("models/llm"),
    Path("models/tts"),
)

DEFAULT_DOWNLOADS = (
    # ModelDownload(
    #     name="Wakeword-Modell",
    #     repo_id="davidscripka/openwakeword",
    #     filename="alexa_v0.1.onnx",
    #     target_path=Path("models/wakeword"),
    # ),
    ModelDownload(
        name="STT-Modell",
        repo_id="Systran/faster-whisper-small",
        target_path=Path("models/stt/faster-whisper-small"),
    ),
    ModelDownload(
        name="LLM-Modell",
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        target_path=Path("models/llm/qwen2.5-3b-instruct-q4_k_m.gguf"),
    ),
    # ModelDownload(
    #     name="TTS-Modell-EN",
    #     repo_id="rhasspy/piper-voices",
    #     filename="en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx",
    #     target_path=Path("models/tts/en_US-hfc_female-medium.onnx"),
    # ),
    # ModelDownload(
    #     name="TTS-Konfiguration-EN",
    #     repo_id="rhasspy/piper-voices",
    #     filename="en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json",
    #     target_path=Path("models/tts/en_US-hfc_female-medium.onnx.json"),
    # ),
    ModelDownload(
        name="TTS-Modell-DE",
        repo_id="Thorsten-Voice/Piper",
        filename="de_DE-thorsten-high.onnx",
        target_path=Path("models/tts/de_DE-thorsten-high.onnx"),
    ),
    ModelDownload(
        name="TTS-Konfiguration-DE",
        repo_id="Thorsten-Voice/Piper",
        filename="de_DE-thorsten-high.onnx.json",
        target_path=Path("models/tts/de_DE-thorsten-high.onnx.json"),
    ),
)


def install() -> None:
    """Installiert die benötigten Modelle für Privo."""
    project_dir = Path.cwd()
    console = Console()

    should_download = Confirm.ask(
        "\nSollen fehlende Modelle von Hugging Face Hub geladen werden? (Empfohlen)",
        default=True,
    )

    if should_download:
        download_all_models(console)
        return

    _print_manual_instructions(project_dir, console)


def download_model(model: ModelDownload, console: Console) -> bool:
    """Lädt ein Modell vom Hugging Face Hub herunter.

    Args:
        model (ModelDownload): Das zu ladende Modell.
        console (Console): Die Rich-Konsole für die Ausgabe.

    Returns:
        bool: True, wenn der Download erfolgreich war, False sonst.
    """

    target_dir = model.target_path.parent if model.filename else model.target_path
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with console.status(f"[bold blue]Lade {model.name} herunter...[/bold blue]"):
            if model.filename:
                hf_hub_download(
                    repo_id=model.repo_id,
                    filename=model.filename,
                    local_dir=target_dir,
                )
            else:
                snapshot_download(
                    repo_id=model.repo_id,
                    local_dir=target_dir,
                )

        console.print(f"[green]✓[/green] {model.name} heruntergeladen.")
        return True

    except RepositoryNotFoundError:
        console.print(
            f"[bold red]Repository nicht gefunden oder kein Zugriff: {model.repo_id}[/bold red]"
        )
        return False

    except LocalEntryNotFoundError:
        console.print(
            "[bold red]Datei nicht im Cache und keine Netzwerkverbindung verfügbar.[/bold red]"
        )
        return False

    except RemoteEntryNotFoundError:
        missing = model.filename or model.repo_id
        console.print(
            f"[bold red]Datei nicht im Repository gefunden: {missing}[/bold red]"
        )
        return False

    except Exception as e:
        console.print(
            f"[bold red]Fehler beim Download von {model.name}: {e}[/bold red]"
        )
        return False


def download_all_models(console: Console) -> None:
    """Lädt alle Modelle mittels download_model herunter.

    Args:
        console (Console): Die Rich-Konsole für die Ausgabe.
    """

    for model in DEFAULT_DOWNLOADS:
        if not download_model(model, console):
            console.print(
                f"[bold red]Fehler beim Herunterladen von {model.name}. Bitte versuche es später erneut.[/bold red]"
            )
            return

    console.print("\n[bold green]Installation abgeschlossen.[/bold green]")
    console.print("Du kannst Privo jetzt mit [cyan]privo run[/cyan] starten.")


def _print_manual_instructions(project_dir: Path, console: Console) -> None:
    """Gibt Anweisungen für die manuelle Installation der Modelle aus.

    Args:
        project_dir (Path): Das Projektverzeichnis.
        console (Console): Die Rich-Konsole für die Ausgabe.
    """

    console.print("\n[bold yellow]Manuelle Installation ausgewählt.[/bold yellow]")
    console.print("Bitte lege die Modelle exakt unter diesen Pfaden ab:")

    for download in DEFAULT_DOWNLOADS:
        console.print(f"  • [cyan]{project_dir / download.target_path}[/cyan]")

    console.print("\nDanach kannst du Privo mit [cyan]privo run[/cyan] starten.")
