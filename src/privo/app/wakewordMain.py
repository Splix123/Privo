import yaml
from privo.audio import AudioInput
from privo.wakeword import WakewordDetector

def main():
    try:
        with open("../config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Config-Datei konnte nicht gefunden werden.")
        config = {}
    except yaml.YAMLError as e:
        print(f"Fehler beim Parsen der YAML: {e}")
        config = {}

    audio = AudioInput(
        sample_rate=config["sample_rate"],
        block_ms=config["block_size"],
        channels=config["channels"]
    )
    detector = WakewordDetector(
        model_path=config["model_path"],
        threshold=config["threshold"],
        vad_threshold=config["vad_threshold"]
    )

    audio.start()

    print("Listening for wakeword...")

    try:
        while True:
            chunk = audio.read_chunk()

            detected, wakeword, score = detector.process(chunk)

            if detected:
                print(f"Wakeword erkannt: {wakeword} ({score:.2f})")

                # 👉 HIER kannst du später weitermachen:
                # z.B. Aufnahme starten für Speech-to-Text

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        audio.stop()


if __name__ == "__main__":
    main()