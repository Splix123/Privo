# 🧠 Offline-Sprachassistent – Prototyp

## 📌 Übersicht

Dieser Prototyp wurde im Rahmen der Bachelorarbeit zum Thema  
**„Privacy by Design: Entwicklung und Bewertung eines Offline-Sprachassistenten als Alternative zu Cloud-basierten Systemen“** entwickelt.

Ziel ist es, einen Sprachassistenten zu realisieren, der vollständig **ohne Internetverbindung** funktioniert und somit die **Privatsphäre der Nutzer schützt**, indem keine Sprachdaten an externe Server übeen werden.

---

## 🎯 Zielsetzung

- Entwicklung eines funktionalen Offline-Sprachassistenten
- Vermeidung jeglicher Cloud-Abhängigkeiten
- Lokale Verarbeitung von Sprachbefehlen (Speech-to-Text, Intent-Erkennung, Text-to-Speech)
- Evaluierung hinsichtlich Datenschutz, Performance und Benutzerfreundlichkeit

---

## ⚙️ Funktionen

Der Prototyp unterstützt aktuell folgende Funktionen:

- 🎙️ **Spracherkennung (Speech-to-Text)**  
  Lokale Umwandlung von Sprache in Text

- 🧠 **Intent-Erkennung**  
  Verarbeitung und Interpretation von Nutzerbefehlen

- 🔊 **Sprachausgabe (Text-to-Speech)**  
  Lokale Generierung von Antworten

- 📅 **Beispielhafte Anwendungsfälle**
  - Uhrzeit abfragen
  - Einfache Wissensfragen
  - Systembefehle (z. B. Programme starten)

---

## 🏗️ Architektur

Der Prototyp besteht aus mehreren Modulen:

1. **Audio Input** – Aufnahme von Sprache über Mikrofon
2. **Speech-to-Text Engine** – Lokale Transkription
3. **NLU-Modul** – Intent-Erkennung und Verarbeitung
4. **Command Handler** – Ausführung von Aktionen
5. **Text-to-Speech Engine** – Generierung der Antwort

Alle Komponenten arbeiten vollständig lokal auf dem Gerät.

---

## 💻 Installation

### Voraussetzungen

- Python 3.x
- Mikrofon
- Unterstütztes Betriebssystem (Windows, Linux, macOS)

### Setup

```bash
git clone <>
cd privo
pip install -r requirements.txt
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
- Begrenzte Intent-Erkennung
- Hardwareabhängige Performance

---

## 👨‍💻 Autor

Prototyp zur Bachelorarbeit von:
Moritz Rühm
