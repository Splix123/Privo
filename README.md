# 🧠 Offline-Sprachassistent – Prototyp

## 📌 Übersicht

Dieser Prototyp wurde im Rahmen der Bachelorarbeit zum Thema  
**„Privacy by Design: Entwicklung und Bewertung eines Offline-Sprachassistenten als Alternative zu Cloud-basierten Systemen“** entwickelt.

**Privo** ist ein modular aufgebauter Sprachassistent, der vollständig **lokal (offline)** arbeitet.
Alle Verarbeitungsschritte – von der Audioaufnahme bis zur Antwortausgabe – erfolgen direkt auf dem Endgerät, ohne externe Server oder Cloud-Dienste zu verwenden und ohne Informationen länger als ein Gespräch zu speichern.

👉 Ziel ist es, eine **datenschutzfreundliche** Alternative zu Systemen wie Amazon Alexa oder Google Assistant zu schaffen.

## 🎯 Zielsetzung

- Entwicklung eines vollständig offlinefähigen Sprachassistenten
- Umsetzung des Konzepts Privacy by Design
- Vermeidung jeglicher Cloud-Kommunikation
- Aufbau einer modularen, erweiterbaren Architektur
- Evaluation hinsichtlich:
  - Datenschutz
  - Performance (Latenz, Durchsatz)
  - Nutzererlebnis

## ⚙️ Funktionen

Der Prototyp unterstützt aktuell folgende Funktionen:

- 🗣️ **Wakeword Erkennung (OpenWakeWord)**  
  Zuhöeren ab einem bestimmten Wort

- 🎙️ **Spracherkennung (Whisper)**  
  Lokale Umwandlung von Sprache in Text

- 🧠 **LLM-Verarbeitung (llama.cpp)**  
  Antwortgenerierung basierend auf ausgewählten LLM-Modellen

- 🔊 **Sprachausgabe (Piper)**  
  Lokale Generierung von Antworten

## 🏗️ Architektur

Der Prototyp besteht aus mehreren Modulen:

1. **Audio-Input** – Aufnahme von Sprache über Mikrofon
2. **Wakeword Modul** - Wakeword Erkennung
3. **Utterance-Recorder** - Aufnehmen bis keine "Sprache" mehr erkannt wird
4. **Speech-to-Text Modul** – Lokale Transkription
5. **LLM Modul** – Antwortgenerierung
6. **Text-to-Speech Modul** – Sprechen der Antwort

Alle Komponenten arbeiten vollständig lokal auf dem Gerät.

---

## 💻 Installation

### Voraussetzungen

- Python 3.11
- Mikrofon
- Unterstütztes Betriebssystem (Windows, Linux, macOS)

### Setup

```bash
pip install git+https://github.com/USERNAME/privo.git
```

---

## 📊 Evaluation

Im Rahmen der Bachelorarbeit wird der Prototyp hinsichtlich folgender Kriterien bewertet:

- Datenschutzvorteile gegenüber Cloud-Lösungen
- Erkennungsgenauigkeit
- Reaktionszeit
- Ressourcenverbrauch
- Nutzerakzeptanz

---

## ⚠️ Einschränkungen

- Eingeschränkte Sprachmodelle im Vergleich zu Cloud-Systemen
- Hardwareabhängige Performance

---

## 👨‍💻 Autor

Prototyp zur Bachelorarbeit von:
Moritz Rühm
