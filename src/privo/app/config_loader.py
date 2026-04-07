import yaml
from typing import Any, TypedDict, get_args, get_origin
from rich import print


class Config(TypedDict, total=False):
    au_sample_rate: int
    au_block_size: int
    au_channels: int
    au_ring_buffer_chunks: int
    wwd_model_path: str
    wwd_threshold: float
    wwd_vad_threshold: float
    wwd_to_strip: list[str]
    stt_silence_threshold: float
    stt_silence_blocks: int
    stt_model_path: str
    stt_device: str
    stt_compute_type: str
    stt_language: str
    stt_beam_size: int
    llm_model_path: str
    llm_n_ctx: int
    llm_n_gpu_layers: int
    llm_system_prompt: str
    llm_max_tokens: int
    llm_temperature: float
    llm_history_limit: int
    llm_conversation_timeout: float
    tts_model_path: str
    tts_config_path: str
    tts_speaker: int
    tts_length_scale: float
    tts_noise_scale: float
    tts_noise_w_scale: float
    tts_sentence_silence: float
    debug_dir: str
    benchmark_samples_dir: str


class ConfigLoader:
    EXPECTED_TYPES: dict[str, type] = {
        "au_sample_rate": int,
        "au_block_size": int,
        "au_channels": int,
        "au_ring_buffer_chunks": int,
        "wwd_model_path": str,
        "wwd_threshold": float,
        "wwd_vad_threshold": float,
        "wwd_to_strip": list[str],
        "stt_silence_threshold": float,
        "stt_silence_blocks": int,
        "stt_model_path": str,
        "stt_device": str,
        "stt_compute_type": str,
        "stt_language": str,
        "stt_beam_size": int,
        "llm_model_path": str,
        "llm_n_ctx": int,
        "llm_n_gpu_layers": int,
        "llm_system_prompt": str,
        "llm_max_tokens": int,
        "llm_temperature": float,
        "llm_history_limit": int,
        "llm_conversation_timeout": float,
        "tts_model_path": str,
        "tts_config_path": str,
        "tts_speaker": int,
        "tts_length_scale": float,
        "tts_noise_scale": float,
        "tts_noise_w_scale": float,
        "tts_sentence_silence": float,
        "debug_dir": str,
        "benchmark_samples_dir": str
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

    def load(self) -> Config:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                print(f"[green]Config-Datei '{self.config_path}' geladen.[/green]\n")
        except FileNotFoundError:
            print("[bold red]Config-Datei konnte nicht gefunden werden.[/bold red]\n[yellow]Standardwerte werden verwendet.[/yellow]\n")
            return {}
        except yaml.YAMLError as e:
            print(f"[bold red]Fehler beim Parsen der YAML: {e}.[/bold red]\n[yellow]Standardwerte werden verwendet.[/yellow]\n")
            return {}

        if not isinstance(raw, dict):
            print("[bold red]Config-Datei hat kein gültiges Mapping-Format.[/bold red]\n[yellow]Standardwerte werden verwendet.[/yellow]\n")
            return {}

        config, errors = self._validate(raw)

        if errors:
            print("[bold red]Ungültige Config-Werte gefunden:[/bold red]")
            for error in errors:
                print(f" - {error}")
            print("[yellow]Für diese Werte werden Standardwerte verwendet.[/yellow]\n")
        return config

    def _validate(self, raw: dict[str, Any]) -> tuple[Config, list[str]]:
        validated: Config = {}
        errors: list[str] = []

        for key, value in raw.items():
            expected_type = self.EXPECTED_TYPES.get(key)

            if expected_type is None:
                errors.append(f"[red]'{key}'[/red] ist kein bekannter Konfigurationswert.")
                continue

            if self._is_valid_type(value, expected_type):
                validated[key] = value
            else:
                errors.append(
                    f"[red]'{key}'[/red] hat ungültigen Typ: erwartet {expected_type.__name__}, "
                    f"bekommen {type(value).__name__} ([red]{value!r}[/red])"
                )

        return validated, errors

    def _is_valid_type(self, value: Any, expected_type: type) -> bool:
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