import yaml
from privo.audio import AudioInput
from privo.wakeword import WakewordDetector

def run() -> None:
    print("Starte Privo...")

    # Config laden
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Config-Datei konnte nicht gefunden werden.")
        config = {}
    except yaml.YAMLError as e:
        print(f"Fehler beim Parsen der YAML: {e}")
        config = {}

    # Audio Initialisieren
    audio = AudioInput(
        sample_rate=config["sample_rate"],
        block_ms=config["block_size"],
        channels=config["channels"]
    )

    # Wakeword-Detector Initialisieren
    detector = WakewordDetector(
        model_path=config["model_path"],
        threshold=config["threshold"],
        vad_threshold=config["vad_threshold"]
    )

    audio.start()

    print("Höre auf Wakeword...")

    try:
        while True:
            chunk = audio.read_chunk()

            detected, wakeword, score = detector.process(chunk)

            if detected:
                print(f"Wakeword erkannt: {wakeword} ({score:.2f})")

                # 👉 HIER kannst du später weitermachen:
                # z.B. Aufnahme starten für Speech-to-Text

    except KeyboardInterrupt:
        print("Beende Privo...")
    finally:
        audio.stop()