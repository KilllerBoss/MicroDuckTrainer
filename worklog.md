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
