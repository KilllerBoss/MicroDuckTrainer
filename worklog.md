---
Task ID: 7 (v2.3)
Agent: main (Super Z)
Task: Trainings-Kern sanieren (Ente läuft nicht / fällt, G1 zittert), Profi-Tricks, Original-Welten + Reset, GLB-Imitation-Fix, Gemini-Profi-Wissen, Joystick-Release + schwebender Pfad

Work Log:
- DIAGNOSE (Browser + Node-Sim + Python/ort):
  1) Original-ONNX-Policies sind 4-Layer-MLPs (512→256→128→14) MIT obs_normalizer
     (Sub/Div) und ELU-Aktivierung → alter 2-Layer-tanh-Extraktor schlug IMMER fehl
     → ES trainierte ab Zufalls-Init → „Ente bewegt sich nicht und fällt".
  2) Protobuf-Walker-Fix: (contentStart, contentLength)-Kontrakt wiederhergestellt
     (Regression aus v2.3-Entwurf: p2-len/p2 war falsch).
  3) Aktivierung wird jetzt aus dem Graph gelesen (NodeProto op_type; Elu) statt
     geraten; Folding W'=W/σ, b'=b−W·(μ/σ) numerisch gegen ort verifiziert (9,8e-8).
  4) ES-Explodierung bei 197k Gewichten: σ=0.08 ohne Skalierung → Verhaltens-Störung
     ~0.5 voraktiv pro Neuron → 100 % Sturz bei Schritt ~5 → FAN-IN-SKALIERUNG
     (eps_W = σ/√fan-in, eps_b = σ) eingeführt (muP-Stil).
  5) Summen-Fitness macht Fall-Strafe ×n binär → im Sum-Modus FLACHE Fall-Strafe
     (Ranking steigt stetig mit Überlebensdauer).
  6) theta-Drift bei Stagnation → Anker-Restart: theta ← bestTheta wenn
     sinceImprovement ≥ 3×Stagnation && Sturzrate > 0,5.
- PROFIS-TRICKS (TrainingPanel „Profi-Tricks", alle default AN): Befehle trainieren
  (Zufalls-cmd pro Paar + exp-Tracking-Reward MJX-Stil, cmdFwd-Deckel je Modell),
  Curriculum (Tempo-Treppe an Sturzrate), Aktions-Lowpass (EMA α), Zufalls-Stöße
  (alle 55 Steps ±0,35 m/s), Reset-Rauschen (Reference-State-Init), Summen-Fitness,
  Gewichtsbremse (Decay 0,02). EsStats.speedScale im Panel sichtbar.
- ENGINE: addResetNoise(), applyPush(), trackCmd-Flag; Vorwärts/Seitwärts-Term
  folgt cmd (exp-Kernel) wenn trackCmd; Worker (es-worker.js) spiegelt ALLES
  (N-Layer-MLP, Noise, Pushes, Lowpass, Tracking, Fitness-Modus).
- Imitation (Cross-Species): Clip-Ziele sind relativ → Zentrum-Pose (Ente:
  defaultPose, G1: standPose) + Aktuator-Clamp + Wurzel-Höhen-Skalierung
  (Roboterhöhe/Clip-Wurzelhöhe) → Ente kann aus HUMAN-Animationen lernen.
  Test-Modus wendet Center+Clamp+Scale ebenfalls an.
- UI: Welt-Panel „Original-Welt (flach, wie Simulator)"; Menü „Alles auf Original
  zurücksetzen" (löscht mdt_v2_*, Reload); Profi-Tricks-Sektion; Gemini-Prompts
  (Regeln + Code-Experte) mit Profi-Wissen + neue training-Knobs (cmdTrain,
  cmdFwd, curriculum, actionSmooth, pushes, noiseReset, fitnessMode, weightDecay)
  + Preview-Chips + applyGoal/applyCode durchreichen.
- Punkt-Modus: Joystick loslassen → Punkt gleitet (weiche Feder ~3,5/s, 0,15 s
  Delay) zum Roboter zurück (Umkreis + Pfad); PfadVisualisierung NEU: schwebende
  quadratische Bézier Roboter→Punkt mit wanderndem Leuchtpunkt (Game-Stil) statt
  Boden-Trail (rig.ts setTargetCurve, AdditiveBlending).
- Persistenz: theta/bestTheta als Base64 (197k-Gewichte sprengen sonst localStorage);
  TrainCfg-Migration (fehlende Profi-Felder → AN-Defaults), versionCode 5.
- QA (agent-browser, Mobile-Viewport): Boot clean; Training Gen 80+ stabil
  (BestEver 1034, Sturzrate schwankt 0,38–0,92, Anker-Restart holt immer zurück —
  VORHER: −93 Fitness und 100 % Sturz dauerhaft); Original-Welt-Button setzt
  Random-Welt aus; Punkt kehrt nach Loslassen heim (Telemetrie PUNKT:PFAD =
  Roboterposition); schwebende Kurve sichtbar (Screenshot); lint clean.
- Node-Sim (scripts/sim_probe.mjs, scripts/final_check.mjs): Ente steht 200 Steps,
  geht mit cmd ≥ 0,25 (x +0,39 m/4 s @0,25; +0,90 m @0,5), überlebt Noise+Pushes.

Stage Summary:
- Trainings-Kette end-to-end validiert (Extraktion → ES → Verhalten im Node-Sim
  und Browser). v2.3 behebt: Ente steht beim Training nur rum / fällt, G1-Zittern
  (Lowpass + Glättung + Decay), GLB-Imitations-Fehlpose (Center+Clamp+Scale).
- Bekannt: Sturzrate oszilliert während Exploration (Anker-Restart hält System
  in gesundem Bereich); Curriculum startet bei 50 % und passt Tempo selbst an.
- Build v2.3 (versionCode 5) folgt; Release v2.3.0 auf GitHub.

---
Task ID: 7a (v2.3 Release)
Agent: main (Super Z)
Task: APK v2.3 bauen + GitHub Release

Work Log:
- Android-SDK nach Workspace-Reset neu beschafft (build-tools r34 inkl.
  libc++.so-Extraktion aus dem Zip für zipalign, platform-34-ext7_r03, R8 8.5.35,
  Temurin JDK 21); platform-URL im Skript dauerhaft gefixt.
- Manifest: versionCode 5, versionName 2.3 (aapt2-Flags überschreiben manifest-
  Attribute NICHT → Manifest selbst geändert — Stolperstein dokumentiert).
- APK 39,7 MB signiert (v1+v2, gleiche Signatur wie v2.0–v2.2 → Update ohne
  Deinstallation); Badging + Signatur verifiziert.
- Release v2.3.0 erstellt, APK als Asset hochgeladen, Download verifiziert
  (HTTP 200, 39.689.391 Bytes, PK-Magic).

Stage Summary:
- Download: https://github.com/KilllerBoss/MicroDuckTrainer/releases/download/v2.3.0/MicroDuckTrainer-v2.3.apk
- Release-Seite: https://github.com/KilllerBoss/MicroDuckTrainer/releases/tag/v2.3.0

---
Task ID: 8 (v2.4 Live-Vorschau + Recovery-Fix)
Agent: main (Super Z)
Task: Nutzer-Meldung "beide Roboter bewegen sich nicht, lernt nichts" despite v2.3 → Root-Cause + Fix + QA + Release

Work Log:
- Verifiziert: v2.3-Trainingskern (Profi-Tricks, Cross-Species-Imitation, Worker-Spiegel) komplett im Repo; v2.3.0 war released.
- Root-Cause-Analyse der Nutzer-Wahrnehmung: (1) Training läuft headless in Workern → sichtbarer Roboter stand einfach rum (kein Feedback). (2) Test-Modus setzte cmd=(0,0,0) → cmd-trainierte Policy stand absichtlich. (3) Auto-Recovery hing ENDLOS in Bauchlage: upright-Bedingung verlangte projGravZ < -0.85, in Bauchlage ≈ 0 → nie erfüllbar.
- v2.4-Fixes (app-core.ts): Live-Trainings-Vorschau (startTraining → Test-Modus + source="es" + updateEsView; Warm-Start-Läufer sofort sichtbar), Auto-Geh-Befehl für Vorschau/„Beste zeigen" (Zufalls-cmd alle 2,5–5 s, 90 % vorwärts), Recovery-Timeout (150 Steps ≈ 3 s Sim → sanfter Keyframe-Reset) + erweiterte Upright-Bedingung (Höhe ≥ 80 % Ziel), Persistenz-Schalter mdt_v2_preview.
- v2.4-Fixes (es.ts): Curriculum hartnäckiger (Boden 0.35 statt 0.15, Aufstieg ab Sturzrate < 0.12 statt < 0.05, Abbau erst > 0.45) — Tempo klebte sonst am Minimum und die Policy lernte Stehen statt Laufen.
- UI: TrainingPanel Live-Vorschau-Switch (default AN) + TrainerApp-Verdrahtung (trainPreview in Telemetry/INITIAL_TEL).
- QA via agent-browser (headless, Timer gedrosselt auf ~3 Hz → Alles in Zeitlupe, echte Geräte 50 Hz): Training startet, Generatoren steigen (0→37), 732 Steps/s, 1 Worker, Best-Ever 1036. Ente läuft SICHTBAR in der Vorschau (gemessen 0.11 und 0.29 m/s), Sturz → Recovery → Timeout-Reset → weiterüben bestätigt. Screenshots: scripts/qa_duck_state.png, qa_duck_walking.png.
- Bekannt/erwartet: Sturzrate während Exploration 0.3-0.6 (Stöße+Noise+Sigma 0.08) — Anker-Restart + Curriculum halten das System gesund; Vorschau zeigt bestEver-Mitglied, wird mit Generationen stabiler.

Stage Summary:
- v2.4 (versionCode 6): Live-Vorschau + Auto-Gehen + Recovery-Timeout + Curriculum-Härtung.
- APK-Build + Release v2.4.0 folgen im nächsten Task.

---
Task ID: 9 (v2.5 Tempo-Sync + Obs-Timing-Fix + idle.glb)
Agent: main (Super Z)
Task: "Ente fällt beim Training auf die Fresse / bewegt sich nicht" — Ursache + Fix; Trainings-Tempo ↔ Physik synchronisieren; GLB idle.glb (Mixamo, 120 fps) fürs Lernen freischalten; Chat-Guide.

Work Log:
- ROOT-CAUSE (Endlich!): engine.stepWithAction baute die Observation NACH den 4
  Physik-Substeps. mj_step integriert sofort → sensordata (Gyro!) reflektierte
  den Zustand VOR der Integration, qpos/qvel den neuen → VERMISCHTER Obs-Zustand
  (Gyro t−1 + Gelenke t). Folge: Policy läuft ~1 s sauber an, bricht dann ein
  (Fell bei Step 20–45), Vorschau-Ente „fällt auf die Fresse". Der validierte
  Node-Sim (sim_probe) baut Obs VOR dem Step — deshalb lief dort alles.
  Beweiskette: kompilierte Modelle Browser↔Probe 0 Diffs; obs-Feld-Diff 0;
  Mini-Probe im Browser (manuelle obs) 120 Steps stabil; Engine-Pfad fiel bei 42.
- Fix (engine.ts): stepWithAction → buildObs() VOR den Substeps; stepPhysics
  behält buildObs (Recovery); resetToKeyframe baut Obs frisch (kein Stale-Start).
- QA nach Fix (Browser, deterministisch, Loop gestoppt): 300 Steps cmd 0.24 →
  dx 0→0.601 m (~0.24 m/s exakt Befehlstempo), z stabil 0.116–0.120, kein Fall,
  kein Zittern. Warm-Start-Läufer sichtbar in der Vorschau.
- Tempo-Sync (Nutzerwunsch): controlLoop multipliziert die Sim-Steps pro Frame
  mit dem Turbo-Faktor (cap 6×) während Training+Vorschau — adaptiver Abstieg
  bei schwachen Geräten (Iteration > 1,35× Budget → Stufe runter). Telemetrie
  + Panel-Badge „⏩ Welt: N× sync". Gemessen: Turbo 4 → previewSpeed 4.
- Vorschau-Härtung: previewCmd IMMER ≥ 0.24 m/s vorwärts (Ente fährt erst ab
  ~0.24 an — Node-Sim-Messung), sanftes Gieren ±0.2, kein Rückwärts-Zufall mehr;
  Recovery-Timeout setzt previewCmdT=0 (frischer sicherer Befehl).
- STEHEN-FALLE im Training behoben: Tracking-Reward gab Stehen bei Mini-cmd
  ~97 % Punktzahl (cmd 0.09, vx 0). Neu: meta.cmdFloor (Ente 0.26, G1 0.12),
  sampleCmd zieht Vorwärts-cmds NIE unter den Anfahr-Boden; Tracking-Kurve
  geschärft (σ² 0.25→0.09, gespiegelt in es-worker.js).
- Warm-Start-ANKER: bestEver startete bei −∞ → das erste ε-Mitglied setzte
  bestTheta (nicht der lauffähige Warm-Start!) → Vorschau zeigte Mediokrität.
  Jetzt: Warm-Start wird vor Gen 1 selbst evaluiert und als bestTheta verankert.
- Aktions-Glättung (EMA) jetzt auch im sichtbaren Pfad (identisch zum Training),
  Resets an allen Recovery-/Pose-/Keyframe-Stellen.
- Syntax-Bug startG1Blend behoben (korrupte Zeile aus unterbrochenem Edit).
- idle.glb (Mixamo-Humanoid, „KillerBossIdle_120fps", 9 s): activateAnimation →
  14/14 Enten-Gelenke gemappt (Hüfte×3/Knie/Knöchel je Seite + Hals), 271 Frames
  @30 Hz, baseY 0.936 → scaleY 0.128 (Cross-Species). Training läuft (Gen 170,
  Fitness 317, Ente hält Zielhöhe 0.12 — aufrecht, kein Sturz-Loop).
- APK v2.5 (versionCode 7), Release v2.5.0.

Stage Summary:
- v2.5 behebt DEN Kern: Obs-Timing-Bug im sichtbaren Pfad (Vorschau/ONNX/ES),
  Stehen-Falle im Reward, Warm-Start-Anker, Tempo-Sync (Turbo beschleunigt
  sichtbare Physik synchron, max 6×, adaptiv), idle.glb funktioniert end-to-end.
- Download: https://github.com/KilllerBoss/MicroDuckTrainer/releases/download/v2.5.0/MicroDuckTrainer-v2.5.apk
