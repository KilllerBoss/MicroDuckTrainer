# 🦆 MicroDuck Trainer v2.0

MuJoCo-Roboter-Training direkt auf dem Android-Handy – **komplett offline**. Enthält die **Original-MicroDuck-Ente** aus dem HF-Space `pollen-robotics/microduck-simulator` (MJCF, GLB, 9 ONNX-Policies) und den **Unitree G1** aus `google-deepmind/mujoco_menagerie`.

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

1. `MicroDuckTrainer-v2.0.apk` aus den [Releases](https://github.com/KilllerBoss/MicroDuckTrainer/releases/latest) herunterladen
2. Installieren („Unbekannte Quellen" erlauben)
3. App startet offline mit Tron-Arena + Ente

> **Hinweis v1.0 → v2.0:** v2.0 ist neu signiert – vor der Installation die alte v1.0-App deinstallieren.

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
