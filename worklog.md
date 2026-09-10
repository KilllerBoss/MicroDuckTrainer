# MicroDuck Trainer – Multi-Agent Worklog

---
Task ID: 1
Agent: main (Super Z)
Task: Umgebung-Rebuild nach Reset + Vorbereitung für APK v2.0

Work Log:
- Festgestellt: Workspace wurde komplett zurückgesetzt (nur leeres Scaffold + .git mit leerem Initial-Commit)
- GitHub-Token verifiziert: User = KilllerBoss (für Upload vorgesehen)
- Skill fullstack-dev geladen, Next.js 16 Scaffold via init-fullstack.sh neu aufgesetzt
- Original-Assets von HF-Space pollen-robotics/microduck-simulator heruntergeladen:
  * public/robot/mjlab/ (robot_allcollisions.xml, kinematics.json, microduck.glb, 43 STL-Meshes)
  * public/policies/ (9 ONNX: walking, stand, sitstand, roller, roller_crouch, ground_pick, kick_l/r, roulade)
- Unitree G1 aus google-deepmind/mujoco_menagerie (sparse clone) → public/robot/unitree_g1/
  * g1.xml = g1_29dof_rev_1_0: 29 POSITION-Aktuatoren (kp=500, dampratio=1), meshdir="assets", 51 STL, Keyframe "stand" (Höhe 0.79)
  * position-Aktuatoren: ctrl = Ziel-Gelenkwinkel (rad) → ideal für Gamepad-Mapping
- Original-Quellcode als Referenz nach reference/ geladen (game.js mit MuJoCo+ONNX-Bootmuster, duck.js Rig, controls/, constants.js)
  * MuJoCo: @mujoco/mujoco 3.11, loadMujocoFactory({locateFile}), MjVFS, from_xml_string, 50Hz Policy, dt=0.005, decimation 4, Obs 61D "new-cmd-obs"
  * ONNX: onnxruntime-web/wasm, ort.env.wasm.wasmPaths={wasm:url}, numThreads=1
- Android-SDK ohne Android Studio nach scripts/android-sdk/ geladen:
  * bt/aapt2 2.19, bt/zipalign, bt/apksigner(+jar), bt/d8.jar, plat/android.jar (API 34), r8.jar 8.5.35 (D8 läuft auf JDK 21)
  * JDK 21 mit javac im System vorhanden

Stage Summary:
- Alle Assets + Build-Tools wiederhergestellt; G1 nutzt NEUE Menagerie-Struktur (g1.xml + assets/, keine g1_29dof.xml/meshes mehr)
- Als nächstes: v2.0-App-Implementierung (Subagent), dann APK-Build (versionCode 2), dann GitHub-Upload
- HINWEIS: v1.0-Keystore ging beim Reset verloren → v2.0 bekommt NEUE Signatur → Nutzer muss v1.0 vor Installation deinstallieren
