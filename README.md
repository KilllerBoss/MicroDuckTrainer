# 🦆 MicroDuck Trainer v2.2

MuJoCo-Roboter-Training direkt auf dem Android-Handy – **komplett offline**. Enthält die **Original-MicroDuck-Ente** aus dem HF-Space `pollen-robotics/microduck-simulator` (MJCF, GLB, 9 ONNX-Policies) und den **Unitree G1** aus `google-deepmind/mujoco_menagerie`.

## Features (v2.2 – NEU)

| Feature | Beschreibung |
|---|---|
| 📂 **GLB-Upload gefixt** | Datei-Manager des Handys öffnet sich jetzt beim „GLB hochladen"-Button (WebView-`onShowFileChooser`) – war in v2.1 ohne Funktion |
| ☰ **Aufgeräumte Oberfläche** | Kompakte Topbar (passt auf jedes Handy) + **Vollbild-Menü** mit großen Touch-Kacheln für alle Panels statt 8 gequetschter Buttons |
| 👨‍💻 **Gemini Code-Experte** | Gemini **schreibt selbst echten JavaScript-Trainingscode** (Reward-Term) und entscheidet **Rundenlänge, Ziel-Generationen, Turbo, Reset** – den ganzen Trainingscode. Code-Vorschau, Gewichts-Slider, Anwenden, Kopieren, Download |
| ⏱ **Runden & Hyperparameter** | Rundenlänge (Policy-Schritte je Generation), **Ziel-Generationen mit Auto-Stopp**, Lernrate, Sigma – eigene Regler, per Modell gespeichert; Gemini darf sie mitentscheiden |
| 💾 **Export/Import** | Vollständiger Trainingsstand (Policy + Regeln + Runden-Einstellungen + Welt + Mapping) als `.json` in den **Downloads-Ordner** exportieren und über den Datei-Manager wieder importieren (APK-Bridge via MediaStore) |
| ✅ **Code-Sandbox** | KI-Code wird validiert (kein Netz/DOM/eval), kompiliert lokal, läuft in Main-Thread **und** Physik-Workern mit identischer Semantik; Laufzeitfehler bringen das Training nicht zu Fall |

## Features (v2.1)

| Feature | Beschreibung |
|---|---|
| 🌍 **Random-Welt-Generator** | Prozedurale Welten per Knopfdruck neu würfeln: **Treppen, Hügel, Löcher** (Fliesenboden), **Hindernisse**, **Balancierstange** – mit Schwierigkeit + Dichte-Reglern und Seed |
| 🎬 **GLB-Animations-Imitation** | Eigene `.glb`-Animationen hochladen (z. B. Mixamo) → Knochen werden automatisch auf die Roboter-Gelenke gemappt → die ES lernt sie nachzumachen (Phase sin/cos in der Observation, Test-Modus spielt endlos ohne Reset) |
| 🎯 **Punkt-Modus** | Der Joystick bewegt einen **3D-Punkt** in der Welt – die Trainingsregeln bestimmen, wie (und ob) der Roboter reagiert: „Zum Punkt laufen", „Vom Punkt weg" |
| 🧭 **Kamera-relativ** | Vorne ist immer dort, wohin du schaust: Joystick-oben = Blickrichtung der Kamera (nicht Welt-Norden) |
| 🎯↩ **Punkt-Modi** | **Aus / Frei / Umkreis / Pfad**: Punkt abschaltbar, frei in der Arena, auf einstellbaren Umkreis um den Roboter begrenzt, oder mit **physikalisch berechnetem Pfad inkl. Momentum** (Roboter geht der Schwung-Kurve nach) |
| 🏃 **Distanz → Tempo** | Einstellbar: Je weiter der Punkt, desto schneller läuft der Roboter (Max-Tempo + Volldistanz-Regler) |
| 🤖 **Gemini-Trainingsregeln** | Ziel in normalem Deutsch sagen („Ich will, dass der Roboter springen lernt") → **gemini-robotics-er-2-preview**, **gemini-3.5-flash-lite** oder **gemini-3.8-flash** passen Reward, Welt, Punkt-Modus und Turbo automatisch an – ganz ohne Programmieren (Standard-Key eingebaut, eigener Key optional) |

## Features (v2.0)

| Feature | Beschreibung |
|---|---|
| 🦆↔🧍 **Modell-Umschalter** | Zwischen Ente (MicroDuck, 14 Gelenke) und Mensch (Unitree G1, 29 Gelenke) wechseln |
| ⛶ **Vollbild** | Echter Vollbildmodus (Fullscreen-API + WebView-CustomView) |
| 🎮 **Touch-Gamepad** | Links Joystick, rechts Buttons **A–D** – Multi-Touch |
| 🗺 **Action-Mapping** | Joystick/A–D frei auf Gelenke, Policy-Commands, Policy-Wechsel oder Posen mappen (pro Roboter gespeichert) → Roboter ist teleoperierbar |
| ⚖ **Bewertungs-Editor** | Beliebig bestimmen, was gut/schlecht ist: Aufrecht, Höhe, Tempo, Energie-Strafe, Glattheit, Fall-Strafe … mit Gewichts-Slidern + Presets |
| 🧠 **KI-Bewertung (offline)** | Ziel-Chips antippen → KI-Regel-Templates bauen die Reward-Funktion + adaptive Trainings-Vorschläge |
| 🚀 **Turbo/Boost** | 1×/4×/16×/32×/**Hyper** (64×), headless-Rollouts, adaptives Sigma/Population, ONNX-Warm-Start, Early-Abort |
| ♾ **Kontinuierlicher Test** | Policy läuft endlos **ohne Reset/Teleport** – mit optionalem Auto-Recovery wie am echten Roboter |
| 💾 **Speichern/Laden** | Beste Policy lokal speichern, „Beste zeigen" |

## Installation

1. `MicroDuckTrainer-v2.2.apk` aus den [Releases](https://github.com/KilllerBoss/MicroDuckTrainer/releases/latest) herunterladen
2. Installieren („Unbekannte Quellen" erlauben)
3. App startet offline mit Tron-Arena + Ente

> **Update v2.1 → v2.2:** Gleiche Signatur – kann direkt **über** v2.1/v2.0 installiert werden.
> **Hinweis v1.0 → v2.x:** v1.0 hatte eine andere Signatur – v1.0 vorher deinstallieren.

### Gemini einrichten (optional)
1. Ein **Standard-Key ist eingebaut** – Gemini funktioniert sofort (sofern Land/Netz von Gemini unterstützt wird)
2. Eigener Key (optional): kostenlos erstellen: [aistudio.google.com](https://aistudio.google.com) → „Get API key"
3. In der App: **KI**-Panel → Key eintragen (wird nur lokal gespeichert) → Modell wählen → Ziel beschreiben → „Regeln automatisch anpassen"

## Nutzung

- **Manuell**: Gamepad steuert den Roboter direkt (Ente: Velocity-Commands an die ONNX-Policy, G1: direkte Gelenk-Targets)
- **Training**: ES (Evolution Strategy) trainiert ein MLP – Turbo-Stufe wählen, Fitness-Kurve live
- **Test**: Kontinuierlicher Dauerlauf ohne Episoden/Resets

## Technik

- MuJoCo-WASM (`@mujoco/mujoco`) + onnxruntime-web, three.js-Rendering (GLB-Rig für die Ente, STL-per-Body für den G1)
- Next.js 16 Static Export → WebView-APK mit virtuellem HTTPS-Origin (`appassets.local`) via `shouldInterceptRequest`
- APK-Build ohne Android Studio: `scripts/apk/build_apk.sh` (aapt2/D8/zipalign/apksigner)

## Build

```bash
bun install
NEXT_OUTPUT=export bunx next build   # → out/
bash scripts/apk/build_apk.sh        # → download/MicroDuckTrainer-v2.0.apk
```

Keystore: `microduck-trainer.keystore` (Pass: `microduck2026`, Alias: `microduck`) – liegt den Release-Assets bei.
