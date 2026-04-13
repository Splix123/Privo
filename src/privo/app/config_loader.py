from __future__ import annotations

import shutil
import yaml
from importlib.resources import files, as_file
from pathlib import Path
from typing import Any, TypedDict, get_args, get_origin, get_type_hints
from rich import print


class Config(TypedDict, total=False):
    """Repräsentiert die Konfigurationswerte der Anwendung.

    Args:
        TypedDict (_type_): Basisklasse für typisierte Dictionaries.
        total (bool, optional): Gibt an, ob alle Schlüssel erforderlich sind. Defaults to False.
    """

    # Audio
    au_sample_rate: int
    au_block_size: int
    au_channels: int
    au_ring_buffer_chunks: int

    # Wakeword
    wwd_model_path: str
    wwd_threshold: float
    wwd_vad_threshold: float
    wwd_to_strip: list[str]

    # STT
    stt_silence_threshold: float
    stt_silence_blocks: int
    stt_model_path: str
    stt_device: str
    stt_compute_type: str
    stt_language: str
    stt_beam_size: int

    # LLM
    llm_model_path: str
    llm_n_ctx: int
    llm_n_gpu_layers: int
    llm_system_prompt: str
    llm_max_tokens: int
    llm_temperature: float
    llm_history_limit: int
    llm_conversation_timeout: float

    # TTS
    tts_model_path: str
    tts_config_path: str
    tts_length_scale: float
    tts_noise_scale: float
    tts_noise_w_scale: float
    tts_sentence_silence: float

    # Sonstiges
    debug_dir: str
    benchmark_samples_dir: str


class ConfigLoader:
    """Lädt und validiert die Konfigurationswerte der Anwendung basierend auf ihrem Typ."""

    EXPECTED_TYPES: dict[str, Any] = get_type_hints(Config)

    def __init__(self, filename: str = "config.yaml") -> None:
        """Setzt den Pfad zur Konfigurationsdatei. Standardmäßig wird "config.yaml" im aktuellen Verzeichnis verwendet.

        Args:
            filename (str, optional): Der Name der Konfigurationsdatei. Defaults to "config.yaml".
        """
        self.filename = filename

    def get_config_path(self) -> Path:
        """Gibt den Pfad zur Konfigurationsdatei zurück. Wenn die Datei nicht existiert, wird sie aus den Ressourcen kopiert.

        Returns:
            Path: Der Pfad zur Konfigurationsdatei.
        """
        path = Path.cwd() / self.filename

        if not path.exists():
            resource = files("privo").joinpath("config.yaml")
            with as_file(resource) as source_path:
                shutil.copyfile(source_path, path)

        return path

    def load(self) -> Config:
        """Lädt die Konfigurationsdatei.

        Returns:
            Config: Die geladene Konfiguration.
        """
        config_path = self.get_config_path()

        try:
            with config_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                print(f"[green]Config-Datei '{config_path}' geladen.[/green]\n")

        except FileNotFoundError:
            print(
                "[bold red]Config-Datei konnte nicht gefunden werden.[/bold red]\n"
                "[yellow]Standardwerte werden verwendet.[/yellow]\n"
            )
            return {}

        except yaml.YAMLError as e:
            print(
                f"[bold red]Fehler beim Parsen der YAML: {e}.[/bold red]\n"
                "[yellow]Standardwerte werden verwendet.[/yellow]\n"
            )
            return {}

        if not isinstance(raw, dict):
            print(
                "[bold red]Config-Datei hat kein gültiges Mapping-Format.[/bold red]\n"
                "[yellow]Standardwerte werden verwendet.[/yellow]\n"
            )
            return {}

        config, errors = self._validate(raw)

        if errors:
            print("[bold red]Ungültige Config-Werte gefunden:[/bold red]")
            for error in errors:
                print(f" - {error}")
            print("[yellow]Für diese Werte werden Standardwerte verwendet.[/yellow]\n")

        return config

    def _validate(self, raw: dict[str, Any]) -> tuple[Config, list[str]]:
        """Validiert die geladenen Konfigurationswerte nach dem erwarteten Typ.

        Args:
            raw (dict[str, Any]): Die rohen Konfigurationswerte.

        Returns:
            tuple[Config, list[str]]: Ein Tupel aus der validierten Konfiguration und einer Liste von Fehlern.
        """
        validated: Config = {}
        errors: list[str] = []

        for key, value in raw.items():
            expected_type = self.EXPECTED_TYPES.get(key)

            if expected_type is None:
                errors.append(
                    f"[red]'{key}'[/red] ist kein bekannter Konfigurationswert."
                )
                continue

            if not self._is_valid_type(value, expected_type):
                type_name = getattr(expected_type, "__name__", str(expected_type))
                errors.append(
                    f"[red]'{key}'[/red] hat ungültigen Typ: erwartet {type_name}, "
                    f"bekommen {type(value).__name__} ([red]{value!r}[/red])"
                )
                continue

            validated[key] = value

        return validated, errors

    def _is_valid_type(self, value: Any, expected_type: type) -> bool:
        """Überprüft, ob ein Wert dem erwarteten Typ entspricht.

        Args:
            value (Any): Der zu überprüfende Wert.
            expected_type (type): Der erwartete Typ.

        Returns:
            bool: True, wenn der Wert dem erwarteten Typ entspricht, False sonst.
        """
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        if expected_type is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        if expected_type is int:
            return isinstance(value, int) and not isinstance(value, bool)

        if expected_type is str:
            return isinstance(value, str)

        if origin is list:
            if not isinstance(value, list):
                return False

            if not args:
                return True

            item_type = args[0]
            return all(self._is_valid_type(item, item_type) for item in value)

        return isinstance(value, expected_type)
