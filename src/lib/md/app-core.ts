// ── MicroDuck Trainer v2.0 – App-Kern (Engine + Loop + Modi) ─────────────────
// Kapselt die komplette Laufzeit: MuJoCo-Engine, three.js-Welt, ONNX-Policies,
// ES-Training, Gamepad-Mappings, Modi (Manuell/Training/Test) und Auto-Recovery.
// Die React-UI erhält alles über Telemetrie-Callbacks (throttled, 4 Hz).

import { Engine } from "./engine";
import {
  createWorld, mountRig, syncDuck, syncG1, syncBall,
  type World,
} from "./rig";
import { OnnxPolicy, MlpPolicy, warmStartMlp, type MlpLayout } from "./policy";
import { getModel, type ModelId } from "./models";
import { EsTrainer, stepRewardValue, type EsStats, type TurboLevel, type ImitEvalData } from "./es";
import type { MappingEntry, SourceId } from "./mapping";
import { poseById, loadMapping, saveMapping } from "./mapping";
import { loadRewardConfig, saveRewardConfig, type RewardConfig } from "./rewards";
import { clearCustomFnCache } from "./customcode";
import {
  defaultWorldConfig, generateWorld, arenaHalf, randomSeed,
  buildWorldMeshes, type WorldConfig, type WorldBuild,
} from "./worldgen";
import {
  buildImitClip, targetAt, rootDeltaAt, saveImitPrefs,
  type ImitClip,
} from "./imitation";
import { defaultRunCfg, type RunCfg } from "./es";


export type Mode = "manuell" | "training" | "test";
export type PolicySource = "onnx" | "es";

// ── v2.2: Punkt-Modus-Konfiguration ──
export type PointModeState = "aus" | "frei" | "umkreis" | "pfad";

export interface PointCfg {
  mode: PointModeState;
  /** Max. Abstand Punkt ↔ Roboter in m (umkreis & pfad). */
  radius: number;
  /** Je weiter der Punkt, desto schneller läuft der Roboter. */
  speedByDist: boolean;
  /** Lauf-Tempo (m/s) bei voller Distanz. */
  maxSpeed: number;
  /** Distanz (m), ab der maxSpeed erreicht ist. */
  fullDist: number;
}

export function defaultPointCfg(modelId: ModelId | null): PointCfg {
  return {
    mode: "aus",
    radius: 2,
    speedByDist: true,
    maxSpeed: modelId === "unitree_g1" ? 0.6 : 0.25,
    fullDist: 1.5,
  };
}

// ── v2.2: Trainings-Konfiguration („Runden“ + Hyperparameter) ──
// v2.3: + Profi-Tricks (Befehle, Curriculum, Glättung, Stöße, Fitness-Modus)
export interface TrainCfg {
  /** Rollout-Länge in Policy-Steps (Rundenlänge). */
  rolloutSteps: number;
  /** Ziel-Generationen; 0 = unbegrenzt (läuft bis Stop). */
  maxGenerations: number;
  /** ES-Lernrate. */
  lr: number;
  /** Start-/Basis-Sigma (Rauschen). */
  sigma: number;
  /** Profi-Trick: Befehle pro Runde zufällig ziehen + Belohnung fürs Folgen
   *  (löst das „Ente steht rum“-Problem: die Policy KENNT ein Ziel-Tempo). */
  cmdTrain: boolean;
  /** Max-Tempo der Trainings-Befehle (m/s) – Deckel: Modell-Geschwindigkeitslimit. */
  cmdFwd: number;
  /** Curriculum: Tempo automatisch an Sturzrate anpassen (Profi-Trick). */
  curriculum: boolean;
  /** Action-Lowpass (1 = aus, 0.3 = stark geglättet) – killt Zittern. */
  actionSmooth: number;
  /** Zufalls-Stöße während der Runden (Domain-Randomization). */
  pushes: boolean;
  /** Reset mit Zustands-Rauschen (Reference-State-Init). */
  noiseReset: boolean;
  /** Fitness = Summe (Überleben zählt) oder Mittelwert (klassisch). */
  fitnessMode: "sum" | "mean";
  /** Gewichtsbremse 0–0.05 (Decay gegen saturierte tanh-Ausgänge). */
  weightDecay: number;
}

export function defaultTrainCfg(): TrainCfg {
  return {
    rolloutSteps: 200, maxGenerations: 0, lr: 0.03, sigma: 0.08,
    cmdTrain: true, cmdFwd: 0.25, curriculum: true, actionSmooth: 0.6,
    pushes: true, noiseReset: true, fitnessMode: "sum", weightDecay: 0.02,
  };
}

export interface Telemetry {
  booted: boolean;
  loading: boolean;
  loadingText: string;
  error: string | null;
  modelId: ModelId | null;
  mode: Mode;
  source: PolicySource;
  onnxId: string;
  overridePolicy: string | null;
  esReady: boolean; // ES-Policy vorhanden (trainiert/geladen)?
  speed: number;
  height: number;
  ctrlHz: number;
  recovering: boolean;
  fallen: boolean;
  testUptime: number;
  testReward: number;
  training: boolean;
  es: EsStats | null;
  mappings: MappingEntry[];
  reward: RewardConfig;
  // ── v2.1/2.2 ──
  trainCfg: TrainCfg;
  trainPreview: boolean; // v2.4: Live-Vorschau an/aus
  previewSpeed: number; // v2.5: effektives Vorschau-Tempo (Turbo-Sync)
  customCode: { name: string; enabled: boolean } | null;
  world: WorldConfig;
  pointMode: boolean; // abgeleitet: mode !== "aus"
  pointCfg: PointCfg;
  point: [number, number];
  imitation: {
    name: string; duration: number; playing: boolean; time: number; mapped: number; manual: boolean;
  } | null;
}

// Enten-Tempos wie reference/constants.js
const VEL_FWD = 0.25, VEL_BACK = -0.2, VEL_ANG = 1.0, VEL_LAT = 0.4;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export class TrainerCore {
  // ── Callbacks (React-Seite) ──
  onTelemetry: (t: Telemetry) => void = () => {};
  /** v2.2: Hinweise (Auto-Stop, Import-Ergebnis …) als Toast. */
  onNotice: (title: string, message: string) => void = () => {};

  // ── Laufzeit-Zustand ──
  private world: World | null = null;
  private engine = new Engine();
  private modelId: ModelId | null = null;

  private sessions = new Map<string, Promise<OnnxPolicy>>();
  private esView: MlpPolicy | null = null;
  private trainer: EsTrainer | null = null;

  // UI-Modi
  private mode: Mode = "manuell";
  private source: PolicySource = "onnx";
  private selectedOnnx = "";
  private overridePolicy: string | null = null; // Gamepad-Button hat umgeschaltet
  private mappings: MappingEntry[] = [];
  private rewardCfg: RewardConfig | null = null;
  private trainCfg: TrainCfg = defaultTrainCfg();
  private autoRecovery = true; // Test-Modus: sanftes Aufrichten nach Sturz

  // Gamepad-Eingaben
  private joy = { x: 0, y: 0 };
  private pendingPolicyId: string | null = null;
  private pendingPoseId: string | null = null;

  // ── v2.1: Welt, Imitation · v2.2: Punkt-Modus ──
  private worldCfg: WorldConfig = defaultWorldConfig();
  private worldBuild: WorldBuild | null = null;
  private pointCfg: PointCfg = defaultPointCfg(null);
  private point: [number, number] = [0.8, 0];
  /** Momentum-Führpunkt (Modus "pfad"): folgt dem Punkt mit Feder-Dämpfer. */
  private leader: { x: number; y: number; vx: number; vy: number } | null = null;
  private markerShown = false;
  private imitClip: ImitClip | null = null;
  private imitPlaying = false;
  private imitManual = false;
  private imitTime = 0;
  private imitTargetBuf: Float32Array | null = null;
  /** v2.3: Zentrum-Pose für Imitations-Ziele (Ente: defaultPose, G1: Standpose). */
  private imitCenter: Float32Array | null = null;
  /** v2.3: Höhen-Skalierung der Clip-Wurzel (Human → Roboter). */
  private imitScaleY = 1;
  // v2.4: Live-Trainings-Vorschau – der sichtbare Roboter übt sichtbar mit
  // (Headless-Rollouts allein sahen wie „bewegt sich nie“ aus!)
  private trainPreview = true;
  private previewCmd: [number, number, number] = [0.15, 0, 0];
  private previewCmdT = 0;
  // v2.5: Trainings-Tempo ↔ Physik-SYNC. Der Turbo-Faktor beschleunigt nicht
  // nur die (headlessen) Rollouts, sondern auch die SICHTBARE Simulation:
  // Mehr Sim-Steps pro Frame bei gleichem dt → Roboter+Welt laufen sichtbar
  // schneller mit, genau so schnell wie das Training selbst.
  private turbo: TurboLevel = 1;
  private previewSpeed = 1;   // effektiv aktive Vorschau-Tempo-Stufe (adaptiv)
  private lastIterMs = 20;    // Dauer der letzten Loop-Iteration (Adaption)
  private esActSm: Float32Array | null = null; // Aktions-Glättung wie im Training
  /** v2.3: Wie lange der Joystick schon losgelassen ist (s) – Punkt fährt heim. */
  private joyIdle = 0;
  private esObsExtra = 0; // Obs-Extra der aktuellen esView

  // Posen-Blending
  private g1Base: Float32Array | null = null;
  private g1Blend: { from: Float32Array; to: Float32Array; t0: number; dur: number } | null = null;
  private duckPoseHold: { from: Float32Array; to: Float32Array; t0: number; blendMs: number; holdMs: number } | null = null;

  // Auto-Recovery (Test-Modus)
  private recovering = false;
  private recoverySteps = 0;
  private recoveryUpright = 0;
  private fallDebounce = 0;

  // Test-Statistik
  private testStart = 0;
  private testUptime = 0;
  private testReward = 0;
  private testStepN = 0;
  /** v2.6: Sturz-Strafe im Test-Modus nur EINMAL pro Fall abziehen. */
  private testFallPenalized = false;
  private prevAct: Float32Array = new Float32Array(0);
  private actBuf: Float32Array = new Float32Array(0);

  // Loops
  private running = false;
  private trainingActive = false;
  private ctrlHz = 0;
  private lastFrameT = 0;
  private lastEmit = 0;

  // Boot/Load-Status
  private booted = false;
  private loading = false;
  private loadingText = "";
  private error: string | null = null;

  // ── Boot ────────────────────────────────────────────────────────────────────

  async boot(container: HTMLElement): Promise<void> {
    try {
      this.world = await createWorld(container);
      await this.loadModel("microduck");
      this.booted = true;
      this.running = true;
      if (typeof window !== "undefined") (window as any).__mdt = this; // QA-Hook
      this.lastFrameT = performance.now();
      void this.controlLoop();
      requestAnimationFrame(this.renderLoop);
      this.emit();
    } catch (err: any) {
      this.error = err?.message || String(err);
      console.error("[core] Boot fehlgeschlagen:", err);
      this.emit();
    }
  }

  /** Modell wechseln (oder gleiches neu starten): disposed alles Alte. */
  async loadModel(id: ModelId): Promise<void> {
    if (this.loading) return;
    this.loading = true;
    this.loadingText = id === "microduck" ? "Lade Ente…" : "Lade G1 (29 Gelenke)…";
    this.emit();
    try {
      // Training anhalten + Trainer wegwerfen (Obs-Dimensionen ändern sich!)
      this.trainingActive = false;
      this.trainer?.dispose();
      this.trainer = null;
      this.esView = null;
      this.resetTransientState();
      this.mode = "manuell";

      const meta = getModel(id);
      const needRig = this.modelId !== id ||
        !(this.world && (id === "microduck" ? this.world.duckRig : this.world.g1Rig));

      // Welt-Konfiguration (global persistiert) auf dieses Modell anwenden
      this.worldCfg = loadWorldPrefs();
      this.worldBuild = this.worldCfg.enabled ? generateWorld(this.worldCfg, id) : null;
      this.applyWorldVisuals();
      this.pointCfg = loadPointPrefs(id);
      this.point = [0.8, 0];
      this.leader = null;

      const tasks: Promise<any>[] = [this.engine.load(id, this.worldBuild)];
      if (needRig && this.world) tasks.push(mountRig(this.world, id));
      if (id === "microduck") tasks.push(this.getOnnxSession(meta.defaultOnnx).catch(() => null));
      await Promise.all(tasks);

      this.modelId = id;
      this.selectedOnnx = meta.defaultOnnx;
      this.source = "onnx";
      this.g1Base = new Float32Array(this.engine.standPose);
      this.esObsExtra = 0;
      this.engine.setImitObs(false);
      this.engine.targetPoint = this.pointCfg.mode !== "aus" ? [...this.point] as [number, number] : null;
      this.prevAct = new Float32Array(meta.actionDim);
      this.actBuf = new Float32Array(meta.actionDim);
      // Mappings + Bewertung + TrainCfg pro Modell laden (Persistenz)
      this.mappings = loadMapping(id);
      this.rewardCfg = loadRewardConfig(id);
      this.trainCfg = loadTrainPrefs(id);
      try {
        const pv = localStorage.getItem("mdt_v2_preview");
        if (pv !== null) this.trainPreview = pv === "1";
      } catch { /* ignore */ }
      this.testUptime = 0;
      this.testReward = 0;
      this.testStepN = 0;
      this.error = null;
    } catch (err: any) {
      this.error = err?.message || String(err);
      console.error("[core] Modell-Load fehlgeschlagen:", err);
    } finally {
      this.loading = false;
      this.emit();
    }
  }

  private resetTransientState() {
    this.overridePolicy = null;
    this.pendingPolicyId = null;
    this.pendingPoseId = null;
    this.duckPoseHold = null;
    this.g1Blend = null;
    this.recovering = false;
    this.fallDebounce = 0;
    this.joy.x = 0;
    this.joy.y = 0;
    this.imitTime = 0;
    this.leader = null;
    this.joyIdle = 0;
  }

  // ── Setter (von der UI) ─────────────────────────────────────────────────────

  setMode(mode: Mode) {
    if (this.mode === mode) return;
    this.mode = mode;
    this.overridePolicy = null;
    this.recovering = false;
    this.fallDebounce = 0;
    if (mode === "test") {
      this.testStart = performance.now();
      this.testUptime = 0;
      this.testReward = 0;
      this.testStepN = 0;
      this.prevAct.fill(0);
    }
    this.emit();
  }

  setSource(source: PolicySource) {
    this.source = source;
    this.overridePolicy = null;
    this.emit();
  }

  setOnnx(id: string) {
    this.selectedOnnx = id;
    this.overridePolicy = null;
    void this.getOnnxSession(id);
    this.emit();
  }

  setMappings(mappings: MappingEntry[], persist = true) {
    this.mappings = mappings;
    if (persist && this.modelId) saveMapping(this.modelId, mappings);
    this.emit();
  }

  setReward(cfg: RewardConfig, persist = true) {
    this.rewardCfg = cfg;
    this.trainer?.setReward(cfg);
    if (persist && this.modelId) saveRewardConfig(this.modelId, cfg);
    this.emit();
  }

  /** v2.2: KI-Code-Term setzen (null = entfernen). */
  setCustomTerm(term: RewardConfig["custom"] | null): void {
    if (!this.rewardCfg) return;
    const next: RewardConfig = { ...this.rewardCfg };
    if (term && term.code && typeof term.weight === "number") {
      next.custom = {
        enabled: term.enabled !== false,
        weight: Math.min(20, Math.max(-20, term.weight)),
        name: (term.name || "KI-Code").slice(0, 40),
        code: term.code,
      };
    } else {
      delete next.custom;
      clearCustomFnCache();
    }
    this.setReward(next);
  }

  /** v2.2: Rundenlänge / Ziel-Generationen / lr / sigma. v2.3: + Profi-Tricks. */
  setTrainCfg(patch: Partial<TrainCfg>): void {
    const next: TrainCfg = { ...this.trainCfg };
    const meta = this.modelId ? getModel(this.modelId) : null;
    if (Number.isFinite(patch.rolloutSteps)) next.rolloutSteps = clamp(Math.round(patch.rolloutSteps!), 30, 1000);
    if (Number.isFinite(patch.maxGenerations)) next.maxGenerations = clamp(Math.round(patch.maxGenerations!), 0, 100000);
    if (Number.isFinite(patch.lr)) next.lr = clamp(patch.lr!, 0.002, 0.2);
    if (Number.isFinite(patch.sigma)) next.sigma = clamp(patch.sigma!, 0.005, 0.3);
    if (typeof patch.cmdTrain === "boolean") next.cmdTrain = patch.cmdTrain;
    if (Number.isFinite(patch.cmdFwd)) {
      const cap = meta ? Math.max(0.05, meta.velocityLimit.fwd) : 0.6;
      next.cmdFwd = clamp(patch.cmdFwd!, 0.05, cap);
    }
    if (typeof patch.curriculum === "boolean") next.curriculum = patch.curriculum;
    if (Number.isFinite(patch.actionSmooth)) next.actionSmooth = clamp(patch.actionSmooth!, 0.3, 1);
    if (typeof patch.pushes === "boolean") next.pushes = patch.pushes;
    if (typeof patch.noiseReset === "boolean") next.noiseReset = patch.noiseReset;
    if (patch.fitnessMode === "sum" || patch.fitnessMode === "mean") next.fitnessMode = patch.fitnessMode;
    if (Number.isFinite(patch.weightDecay)) next.weightDecay = clamp(patch.weightDecay!, 0, 0.05);
    this.trainCfg = next;
    if (this.modelId) saveTrainPrefs(this.modelId, next);
    this.applyTrainCfg();
    this.emit();
  }

  private applyTrainCfg(): void {
    const t = this.trainer;
    if (!t) return;
    t.rolloutSteps = this.trainCfg.rolloutSteps;
    t.lr = this.trainCfg.lr;
    t.sigma = this.trainCfg.sigma;
    t.baseSigma = this.trainCfg.sigma;
    t.maxGenerations = this.trainCfg.maxGenerations;
    // v2.3: Profi-Trainings-Konfiguration an den Trainer durchreichen
    t.setRunCfg({
      cmdTrain: this.trainCfg.cmdTrain,
      cmdFwd: this.trainCfg.cmdFwd,
      cmdLat: 0.15,
      cmdAng: 0.8,
      curriculum: this.trainCfg.curriculum,
      actionSmooth: this.trainCfg.actionSmooth,
      pushes: this.trainCfg.pushes,
      noiseReset: this.trainCfg.noiseReset,
      fitnessMode: this.trainCfg.fitnessMode,
      weightDecay: this.trainCfg.weightDecay,
    });
  }

  setJoystick(x: number, y: number) {
    this.joy.x = clamp(x, -1, 1);
    this.joy.y = clamp(y, -1, 1);
  }

  setAutoRecovery(on: boolean) {
    this.autoRecovery = on;
    if (!on) this.recovering = false;
    this.emit();
  }

  // ── v2.1: Welt ─────────────────────────────────────────────────────────────

  async setWorld(cfg: WorldConfig): Promise<void> {
    if (this.loading) return;
    this.loading = true;
    this.loadingText = "Baue Welt…";
    this.emit();
    try {
      this.stopTraining();
      this.trainer?.dispose();
      this.trainer = null;
      this.esView = null;
      this.esObsExtra = 0;
      this.worldCfg = { ...cfg, features: { ...cfg.features } };
      saveWorldPrefs(this.worldCfg);
      this.worldBuild = this.worldCfg.enabled
        ? generateWorld(this.worldCfg, this.modelId ?? "microduck")
        : null;
      this.applyWorldVisuals();
      await this.engine.load(this.modelId ?? "microduck", this.worldBuild);
      this.engine.targetPoint = this.pointCfg.mode !== "aus" ? ([...this.point] as [number, number]) : null;
      this.error = null;
    } catch (err: any) {
      this.error = err?.message || String(err);
      console.error("[core] Welt-Build fehlgeschlagen:", err);
    } finally {
      this.loading = false;
      this.emit();
    }
  }

  rerollWorld(): void {
    void this.setWorld({ ...this.worldCfg, seed: randomSeed() });
  }

  get worldConfig(): WorldConfig {
    return this.worldCfg;
  }

  private applyWorldVisuals(): void {
    if (!this.world) return;
    this.world.setWorldProps(this.worldBuild ? buildWorldMeshes(this.worldBuild) : null);
  }

  // ── v2.1/2.2: Punkt-Modus ────────────────────────────────────────────────

  /** Legacy-Schalter (bool): true = frei, false = aus. */
  setPointMode(on: boolean): void {
    const mode: PointModeState = on
      ? (this.pointCfg.mode === "aus" ? "frei" : this.pointCfg.mode)
      : "aus";
    this.setPointCfg({ mode });
  }

  setPointCfg(patch: Partial<PointCfg>): void {
    const prevMode = this.pointCfg.mode;
    const next: PointCfg = { ...this.pointCfg, ...patch };
    // Sanfte Validierung
    next.mode = ("aus|frei|umkreis|pfad".split("|") as PointModeState[]).includes(next.mode)
      ? next.mode : "aus";
    next.radius = clamp(next.radius, 0.3, 6);
    next.maxSpeed = clamp(next.maxSpeed, 0.05, 1.2);
    next.fullDist = clamp(next.fullDist, 0.3, 6);
    this.pointCfg = next;
    if (this.modelId) savePointPrefs(this.modelId, next);
    if (next.mode !== prevMode) {
      this.leader = null; // Momentum neu starten
      if (next.mode === "aus") this.world?.setTargetCurve(null, null);
    }
    const active = next.mode !== "aus";
    if (!active) {
      this.engine.targetPoint = null;
      this.trainer?.setTargetPoint(null);
      this.world?.setTargetPoint(0, 0, false);
      this.world?.setTargetCurve(null, null);
      this.markerShown = false;
    } else if (!this.engine.targetPoint) {
      this.engine.targetPoint = [...this.point] as [number, number];
    }
    this.emit();
  }

  get isPointMode(): boolean {
    return this.pointCfg.mode !== "aus";
  }

  get pointConfig(): PointCfg {
    return { ...this.pointCfg };
  }

  // ── v2.1: Imitation ────────────────────────────────────────────────────────

  /** GLB-Clip bauen und aktivieren; Reward-Terme automatisch scharf schalten. */
  async activateAnimation(
    buffer: ArrayBuffer, clipIndex: number, mirror: boolean, modelId: ModelId,
  ): Promise<{ ok: boolean; message: string }> {
    try {
      if (this.trainingActive) this.stopTraining();
      this.trainer?.dispose();
      this.trainer = null;
      this.esView = null;
      this.esObsExtra = 0;
      const meta = getModel(modelId);
      // v2.3: Center = aktuelle Ruhelage des Roboters (G1: Stand-Pose mit
      // entspannten Armen) + Zielhöhe für Cross-Species-Skalierung.
      const clip = await buildImitClip(
        buffer, clipIndex, meta, mirror,
        this.engine.standPose.length === meta.actionDim
          ? this.engine.standPose : null,
        meta.targetHeight,
      );
      if (clip.mapped === 0) {
        return { ok: false, message: "Keine Gelenke gemappt – Skeleton-Namen nicht erkannt." };
      }
      this.imitClip = clip;
      this.imitPlaying = true;
      this.imitTime = 0;
      this.imitTargetBuf = new Float32Array(clip.dim);
      // v2.3: Cross-Species-Retargeting – Clip-Ziele sind RELATIV zur Ruhelage.
      // Zentrum (Ente: defaultPose, G1: Keyframe-Standpose) addieren, sonst
      // vergleicht der Imitations-Reward absolute Gelenkwinkel mit Offset~0
      // und zieht den Roboter in eine nicht-standfähige Pose (→ Sturz).
      const meta2 = getModel(modelId);
      const centerSrc = meta2.id === "microduck"
        ? meta2.defaultPose
        : (this.engine.standPose.length === meta2.actionDim ? this.engine.standPose : meta2.defaultPose);
      this.imitCenter = Float32Array.from(centerSrc);
      // Höhen-Skalierung: Human-Wurzel (~1 m) → Roboterhöhe (Ente 0.12 m)
      this.imitScaleY = clamp(meta2.targetHeight / Math.max(0.25, clip.baseY), 0.05, 1.2);
      this.engine.imitTarget = this.imitTargetBuf;
      this.engine.imitRootDelta = null;
      this.engine.setImitObs(true);
      saveImitPrefs(modelId, clip.name, mirror);
      // Terme automatisch aktivieren, falls beide aus
      if (this.rewardCfg) {
        const im = this.rewardCfg.terms.imitate;
        const ih = this.rewardCfg.terms.imitHeight;
        let changed = false;
        if (im && !im.enabled) { im.enabled = true; im.weight = Math.max(1.5, Math.abs(im.weight)); changed = true; }
        if (ih && !ih.enabled) { ih.enabled = true; ih.weight = Math.max(1.0, Math.abs(ih.weight)); changed = true; }
        if (changed && this.modelId) saveRewardConfig(this.modelId, this.rewardCfg);
      }
      this.emit();
      return {
        ok: true,
        message: `${clip.mapped} Gelenke gemappt · ${clip.duration.toFixed(1)} s · ${clip.name}`,
      };
    } catch (err: any) {
      console.error("[core] Animations-Load fehlgeschlagen:", err);
      return { ok: false, message: err?.message || String(err) };
    }
  }

  clearAnimation(): void {
    this.imitClip = null;
    this.imitPlaying = false;
    this.imitManual = false;
    this.imitTargetBuf = null;
    this.imitCenter = null;
    this.imitScaleY = 1;
    this.engine.imitTarget = null;
    this.engine.imitRootDelta = null;
    this.engine.imitPhase = null;
    this.engine.setImitObs(false);
    this.emit();
  }

  setImitPlaying(on: boolean): void {
    if (!this.imitClip) return;
    this.imitPlaying = on;
    if (!on) {
      this.engine.imitTarget = null;
      this.engine.imitRootDelta = null;
      this.engine.imitPhase = null;
    } else {
      this.engine.imitTarget = this.imitTargetBuf;
    }
    this.emit();
  }

  get imitState(): ImitClip | null {
    return this.imitClip;
  }

  get isImitPlaying(): boolean {
    return this.imitPlaying;
  }

  get isImitManual(): boolean {
    return this.imitManual;
  }

  setImitManual(on: boolean): void {
    this.imitManual = on;
    this.emit();
  }

  pressButton(source: SourceId) {
    const m = this.mappings.find((e) => e.source === source);
    if (!m) return;
    if (m.targetType === "policy") this.pendingPolicyId = m.target;
    else if (m.targetType === "pose") this.pendingPoseId = m.target;
  }

  resetSim() {
    this.resetTransientState();
    this.engine.resetToKeyframe();
    if (this.modelId === "unitree_g1") this.g1Base = new Float32Array(this.engine.standPose);
    this.prevAct.fill(0);
    this.esActSm?.fill(0); // v2.5
    this.emit();
  }

  // ── ONNX-Sessions (lazy, gecacht) ───────────────────────────────────────────

  private getOnnxSession(id: string): Promise<OnnxPolicy | null> {
    const meta = this.modelId ? getModel(this.modelId) : null;
    const def = meta?.policies.find((p) => p.id === id);
    if (!def) return Promise.resolve(null);
    let p = this.sessions.get(def.url);
    if (!p) {
      p = OnnxPolicy.create(def.url).catch((err) => {
        console.warn("[core] ONNX-Load fehlgeschlagen:", def.url, err);
        this.sessions.delete(def.url);
        throw err;
      });
      this.sessions.set(def.url, p);
    }
    return p;
  }

  // ── ES-Training ─────────────────────────────────────────────────────────────

  async startTraining(): Promise<void> {
    if (this.trainingActive || !this.modelId) return;
    try {
      const imitActive = !!(this.imitClip && this.imitPlaying);
      const obsExtra = imitActive ? 2 : 0;
      const imitData: ImitEvalData | null = imitActive && this.imitClip
        ? {
            dim: this.imitClip.dim, frames: this.imitClip.frames, fps: this.imitClip.fps,
            duration: this.imitClip.duration, targets: this.imitClip.targets,
            rootY: this.imitClip.rootY, baseY: this.imitClip.baseY,
            // v2.3: Cross-Species-Retargeting – Zentrum-Pose + Höhen-Skalierung
            center: this.imitCenter ? new Float32Array(this.imitCenter) : undefined,
            scaleY: this.imitScaleY,
          }
        : null;
      const imitSig = this.imitClip
        ? `${this.imitClip.name}|${this.imitClip.frames}|${this.imitClip.mapped}`
        : "";
      // Trainer wegwerfen, wenn Welt/Imitation/Obs-Layout sich geändert haben
      if (this.trainer) {
        const stale =
          this.trainer.obsExtra !== obsExtra ||
          this.trainer.world !== this.worldBuild ||
          this.trainer.imitSig !== imitSig;
        if (stale) {
          this.trainer.dispose();
          this.trainer = null;
          this.esView = null;
          this.esObsExtra = 0;
        }
      }
      if (!this.trainer) {
        const warm = await warmStartMlp(getModel(this.modelId), obsExtra);
        this.trainer = new EsTrainer(this.modelId, warm.theta, warm.layout, this.rewardCfg!, {
          obsExtra, world: this.worldBuild, imit: imitData, imitSig,
        });
        await this.trainer.init();
      }
      this.trainer.setReward(this.rewardCfg!);
      this.trainer.setTargetPoint(this.pointCfg.mode !== "aus" ? ([...this.point] as [number, number]) : null);
      this.applyTrainCfg();
      // v2.4: Live-Vorschau – Test-Modus + beste Policy sofort sichtbar (vor
      // Gen 1: die Warm-Start-Policy, die schon laufen kann!). Sonst steht der
      // sichtbare Roboter nur rum und Training wirkt wie „passiert nichts“.
      // Quelle "es" (winziges MLP statt ONNX-WASM): kein CPU-Wettkampf mit
      // den Eval-Workern → Kontroll-Loop bleibt bei 50 Hz.
      if (this.mode !== "test") this.setMode("test");
      this.updateEsView();
      this.source = "es";
      this.trainingActive = true;
      this.emit();
      await this.trainer.tryStartWorkers();
      if (!this.trainingActive) return; // inzwischen gestoppt
      void this.trainingLoop();
    } catch (err: any) {
      console.error("[core] Training-Start fehlgeschlagen:", err);
      this.trainingActive = false;
      this.emit();
    }
  }

  stopTraining() {
    this.trainingActive = false;
    this.emit();
  }

  setTurbo(t: TurboLevel) {
    this.turbo = t; // v2.5: Turbo skaliert auch die sichtbare Physik (Sync)
    this.trainer?.setTurbo(t);
    this.emit();
  }

  private async trainingLoop() {
    while (this.trainingActive && this.trainer) {
      await this.trainer.runGeneration();
      this.updateEsView();
      this.emit();
      // v2.2: Ziel-Generationen erreicht → automatisch stoppen
      const maxGen = this.trainer.maxGenerations;
      if (maxGen > 0 && this.trainer.generation >= maxGen) {
        this.trainingActive = false;
        this.onNotice(
          "Ziel erreicht",
          `${maxGen} Generationen fertig (Best: ${Number.isFinite(this.trainer.bestEver) ? this.trainer.bestEver.toFixed(2) : "–"}). Training gestoppt.`,
        );
        this.emit();
        break;
      }
      await new Promise((r) => setTimeout(r, 30));
    }
  }

  private updateEsView() {
    if (!this.trainer) {
      this.esView = null;
      return;
    }
    try {
      this.esView = new MlpPolicy(this.trainer.layout, this.trainer.bestTheta);
      this.esObsExtra = this.trainer.layout.obsDim - this.trainer.meta.obsDim;
    } catch {
      this.esView = null;
    }
  }

  /** "Beste zeigen": Quelle auf ES-Policy (bestTheta) umschalten. */
  showBest() {
    this.updateEsView();
    this.source = "es";
    // v2.4: Test-Modus + Auto-Gehen, sonst steht die cmd-trainierte Policy
    // bei Befehl (0,0,0) nur rum und der Nutzer denkt, sie kann nichts.
    if (this.mode !== "test") this.setMode("test");
    this.emit();
  }

  /** v2.4: Live-Trainings-Vorschau ein/aus (persistiert global). */
  setTrainPreview(on: boolean): void {
    this.trainPreview = on;
    try { localStorage.setItem("mdt_v2_preview", on ? "1" : "0"); } catch { /* ignore */ }
    this.emit();
  }

  // ── Theta Persistenz ────────────────────────────────────────────────────────

  private thetaKey(): string {
    return `mdt_v2_theta_${this.modelId}`;
  }

  async saveTheta(): Promise<boolean> {
    if (!this.trainer || !this.modelId) return false;
    try {
      const t = this.trainer;
      localStorage.setItem(this.thetaKey(), JSON.stringify({
        layout: t.layout,
        thetaB64: f32ToB64(t.theta),
        bestThetaB64: f32ToB64(t.bestTheta),
        generation: t.generation,
        bestEver: t.bestEver,
        history: t.history.slice(-200),
      }));
      return true;
    } catch {
      return false;
    }
  }

  hasSavedTheta(): boolean {
    try {
      return !!localStorage.getItem(this.thetaKey());
    } catch {
      return false;
    }
  }

  async loadTheta(): Promise<boolean> {
    if (!this.modelId) return false;
    try {
      const raw = localStorage.getItem(this.thetaKey());
      if (!raw) return false;
      const d = JSON.parse(raw) as {
        layout: MlpLayout; thetaB64?: string; theta?: number[];
        bestThetaB64?: string; bestTheta?: number[];
        generation: number; bestEver: number; history: number[];
      };
      return await this.restoreTheta(d);
    } catch (err) {
      console.warn("[core] Theta-Load fehlgeschlagen:", err);
      return false;
    }
  }

  /** v2.2: Theta-Daten in einen neuen Trainer übernehmen (Load + Import). */
  private async restoreTheta(d: {
    layout: MlpLayout; thetaB64?: string; theta?: number[];
    bestThetaB64?: string; bestTheta?: number[];
    generation?: number; bestEver?: number; history?: number[];
  }): Promise<boolean> {
    try {
      const theta = d?.thetaB64 ? b64ToF32(d.thetaB64) : (d?.theta ? new Float32Array(d.theta) : null);
      if (!theta?.length || !d?.layout) return false;
      this.trainingActive = false;
      this.trainer?.dispose();
      this.trainer = new EsTrainer(this.modelId!, theta, d.layout, this.rewardCfg!);
      await this.trainer.init();
      const best = d.bestThetaB64 ? b64ToF32(d.bestThetaB64) : (d.bestTheta ? new Float32Array(d.bestTheta) : null);
      this.trainer.bestTheta = best ?? theta.slice();
      this.trainer.generation = d.generation ?? 0;
      this.trainer.bestEver = d.bestEver ?? -Infinity;
      this.trainer.history = Array.isArray(d.history) ? d.history : [];
      this.applyTrainCfg();
      this.updateEsView();
      this.emit();
      return true;
    } catch (err) {
      console.warn("[core] Theta-Restore fehlgeschlagen:", err);
      return false;
    }
  }

  /** v2.2: Alles (Policy + Regeln + TrainCfg + Mapping) als JSON-String exportieren. */
  exportAll(): string | null {
    if (!this.modelId) return null;
    const t = this.trainer;
    return JSON.stringify({
      app: "microduck-trainer",
      exportVersion: 2,
      exportedAt: new Date().toISOString(),
      modelId: this.modelId,
      thetaB64: t ? f32ToB64(t.theta) : null,
      bestThetaB64: t ? f32ToB64(t.bestTheta) : null,
      layout: t ? t.layout : null,
      generation: t ? t.generation : 0,
      bestEver: t ? t.bestEver : null,
      history: t ? t.history.slice(-200) : [],
      reward: this.rewardCfg,
      trainCfg: this.trainCfg,
      pointCfg: this.pointCfg,
      worldCfg: this.worldCfg,
      mappings: this.mappings,
    });
  }

  /** v2.2: Export-JSON wiederherstellen (gleiches Modell oder Modellwechsel). */
  async importAll(json: string): Promise<{ ok: boolean; message: string }> {
    try {
      const d = JSON.parse(json) as any;
      if (d?.app !== "microduck-trainer" || !d.modelId) {
        return { ok: false, message: "Keine MicroDuck-Trainer-Exportdatei." };
      }
      if (d.modelId !== this.modelId) {
        if (d.modelId !== "microduck" && d.modelId !== "unitree_g1") {
          return { ok: false, message: `Unbekanntes Modell: ${d.modelId}` };
        }
        await this.loadModel(d.modelId);
      }
      const parts: string[] = [];
      if (d.reward?.terms) { this.setReward(d.reward); parts.push("Bewertung"); }
      if (d.trainCfg) { this.setTrainCfg(d.trainCfg); parts.push("Runden-Einstellungen"); }
      if (Array.isArray(d.mappings)) { this.setMappings(d.mappings); parts.push("Mapping"); }
      if (d.pointCfg?.mode) { this.setPointCfg(d.pointCfg); parts.push("Punkt-Modus"); }
      if (d.worldCfg && typeof d.worldCfg.seed === "number") {
        await this.setWorld(d.worldCfg);
        parts.push("Welt");
      }
      let policyOk = false;
      if ((d.thetaB64 || d.theta?.length) && d.layout) {
        policyOk = await this.restoreTheta(d);
        if (policyOk) parts.push(`Policy (Gen ${d.generation ?? 0})`);
      }
      if (parts.length === 0) {
        return { ok: false, message: "Datei enthält keine verwertbaren Daten." };
      }
      const msg = `Übernommen: ${parts.join(", ")}`;
      this.onNotice("Import abgeschlossen", msg);
      return { ok: true, message: msg };
    } catch (err: any) {
      return { ok: false, message: err?.message || String(err) };
    }
  }

  // ── Kontroll-Loop (50 Hz, async wegen ONNX) ─────────────────────────────────

  private stepMs(): number {
    const m = this.engine.meta;
    return m ? m.timestep * m.decimation * 1000 : 20;
  }

  private async controlLoop() {
    let next = performance.now();
    let count = 0;
    let hzT0 = next;
    while (this.running) {
      const t0 = performance.now();
      // v2.5: Tempo-Sync — während des Trainings läuft die sichtbare Welt mit
      // dem Turbo-Faktor (cap 6×, damit die Darstellung lesbar bleibt). Adaptiv:
      // Schafft das Gerät die Stufe nicht (Iteration > 1,35× Budget), wird eine
      // Stufe abgebaut; Trainingsstopp/aus = sofort 1×.
      const target = this.trainPreview && this.trainingActive && this.mode === "test"
        ? Math.min(6, this.turbo)
        : 1;
      const stepBudget = this.stepMs();
      if (this.previewSpeed > target) {
        this.previewSpeed = target; // Training gestoppt/aus → sofort 1×
      } else if (this.lastIterMs > stepBudget * 1.35 && this.previewSpeed > 1) {
        this.previewSpeed--; // zu langsam für diese Stufe → abbauen
      } else if (this.lastIterMs <= stepBudget * 0.5 && this.previewSpeed < target) {
        this.previewSpeed++; // Luft da → synchron hochdrehen
      }
      const n = Math.max(1, this.previewSpeed);
      for (let k = 0; k < n && this.running; k++) {
        try {
          await this.controlStep();
        } catch (err) {
          console.warn("[core] controlStep:", err);
        }
      }
      count += n; // ctrlHz = Sim-Steps je Wandsekunde (zeigt den Sync-Faktor)
      this.lastIterMs = performance.now() - t0;
      const now = performance.now();
      if (now - hzT0 > 500) {
        this.ctrlHz = (count * 1000) / (now - hzT0);
        count = 0;
        hzT0 = now;
      }
      next += stepBudget;
      const wait = next - performance.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      else next = performance.now(); // zurückgefallen: nicht spiralen
    }
  }

  private async controlStep(): Promise<void> {
    const engine = this.engine;
    if (!engine.data || !engine.meta || !this.modelId) return;

    // ── v2.2: Punkt-Modus – Joystick bewegt den 3D-Punkt (KAMERA-RELATIV) ──
    // Vorne ist immer die Blickrichtung der Kamera (Touch-Drehsensor der Chase-Cam):
    // Joystick oben = weg von der Kamera, rechts = rechts von der Blickrichtung.
    const dtStep = this.stepMs() / 1000;
    // ── v2.4: Live-Vorschau – Befehls-Zufall wie im Training (2,5–5 s Takt) ──
    if (this.mode === "test" && this.trainPreview) {
      this.previewCmdT -= dtStep;
      if (this.previewCmdT <= 0) {
        this.previewCmdT = 2.5 + Math.random() * 2.5;
        const cap = Math.max(0.1, this.trainCfg.cmdFwd);
        // v2.5: Vorschau IMMER vorwärts ÜBER der Anfahr-Schwelle (Ente läuft
        // erst ab ~0.24 m/s, darunter steht sie – Node-Sim gemessen). Der
        // frühere Rückwärts-/Mini-Tempo-Zufall war der „fällt auf die Fresse /
        // bewegt sich nicht"-Effekt. Stürze der Exploration gehören ins
        // headless-Training – nicht in die sichtbare Show.
        this.previewCmd = [
          Math.max(0.24, cap * (0.55 + Math.random() * 0.35)), 0,
          (Math.random() * 2 - 1) * 0.2,
        ];
      }
    }
    const pm = this.pointCfg.mode;
    if (pm !== "aus") {
      const cy = this.world?.getCamYaw() ?? 0;
      // Blickrichtung (MuJoCo-Frame, z-up): three-Cam-Yaw → Kamera sitzt bei
      // (cos cy, -sin cy) relativ zum Roboter → vorne = (-cos cy, +sin cy).
      const fx = -Math.cos(cy), fy = Math.sin(cy);
      const rx = fy, ry = -fx; // rechts = (f_y, -f_x)
      const sp = 1.4 * dtStep;
      const lim = arenaHalf(this.modelId) - 0.15;
      // v2.3: Joystick losgelassen? (Umkreis/Pfad → Punkt fährt heim)
      const joyMag = Math.hypot(this.joy.x, this.joy.y);
      if (joyMag > 0.02) this.joyIdle = 0; else this.joyIdle += dtStep;
      this.point[0] = clamp(this.point[0] + (this.joy.y * fx + this.joy.x * rx) * sp, -lim, lim);
      this.point[1] = clamp(this.point[1] + (this.joy.y * fy + this.joy.x * ry) * sp, -lim, lim);

      // Umkreis/Pfad: Punkt darf max. `radius` vom Roboter entfernt bleiben
      const p0 = engine.torsoPos();
      if (pm === "umkreis" || pm === "pfad") {
        // v2.3: Loslassen → Punkt schwebt sanft zum Roboter zurück (weiche
        // Feder ~3.5/s nach kurzem Delay, damit kurze Stöße nicht zappeln)
        if (joyMag <= 0.02 && this.joyIdle > 0.15) {
          const k = 1 - Math.exp(-dtStep * 3.5);
          this.point[0] += (p0[0] - this.point[0]) * k;
          this.point[1] += (p0[1] - this.point[1]) * k;
        }
        const dxp = this.point[0] - p0[0], dyp = this.point[1] - p0[1];
        const dp = Math.hypot(dxp, dyp);
        if (dp > this.pointCfg.radius) {
          const k = this.pointCfg.radius / dp;
          this.point[0] = p0[0] + dxp * k;
          this.point[1] = p0[1] + dyp * k;
        }
      }

      // Ziel für Roboter/Reward: bei "pfad" der Momentum-Führpunkt (Feder-Dämpfer)
      let target: [number, number] = [this.point[0], this.point[1]];
      if (pm === "pfad") {
        if (!this.leader) this.leader = { x: p0[0], y: p0[1], vx: 0, vy: 0 };
        const L = this.leader;
        const kSpring = 16, zeta = 0.8; // Federrate + Dämpfungsgrad
        const cDamp = 2 * Math.sqrt(kSpring) * zeta;
        L.vx += (kSpring * (this.point[0] - L.x) - cDamp * L.vx) * dtStep;
        L.vy += (kSpring * (this.point[1] - L.y) - cDamp * L.vy) * dtStep;
        const vAbs = Math.hypot(L.vx, L.vy);
        const vMax = Math.max(0.3, this.pointCfg.maxSpeed * 2.5);
        if (vAbs > vMax) { L.vx *= vMax / vAbs; L.vy *= vMax / vAbs; }
        L.x = clamp(L.x + L.vx * dtStep, -lim, lim);
        L.y = clamp(L.y + L.vy * dtStep, -lim, lim);
        target = [L.x, L.y];
      }

      engine.targetPoint = target;
      this.trainer?.setTargetPoint(target);
      this.world?.setTargetPoint(this.point[0], this.point[1], true);
      this.markerShown = true;
      // v2.3: Pfad als schwebende, flüssige Kurve Roboter → Punkt (Game-Stil,
      // wie Waypoint-Anzeige) – kein Boden-Gemälde mehr.
      if (pm === "pfad") {
        this.world?.setTargetCurve(p0, target);
      } else {
        this.world?.setTargetCurve(null, null);
      }
    } else if (engine.targetPoint || this.markerShown) {
      engine.targetPoint = null;
      this.trainer?.setTargetPoint(null);
      this.world?.setTargetPoint(0, 0, false);
      this.world?.setTargetCurve(null, null);
      this.markerShown = false;
    }

    // ── v2.1: Imitation – Zielpose je Policy-Step (v2.3: mit Center+Skalierung) ──
    if (this.imitClip && this.imitPlaying && this.imitTargetBuf) {
      this.imitTime += dtStep;
      targetAt(this.imitClip, this.imitTime, this.imitTargetBuf);
      if (this.imitCenter) {
        for (let j = 0; j < this.imitTargetBuf.length; j++) this.imitTargetBuf[j] += this.imitCenter[j];
      }
      // Auf Aktuator-Range klemmen (Human-Amplituden > Gelenklimits)
      const lims = engine.ctrlLimits;
      if (lims) {
        for (let j = 0; j < this.imitTargetBuf.length; j++) {
          const lo = lims[j * 2], hi = lims[j * 2 + 1];
          if (hi > lo) this.imitTargetBuf[j] = Math.min(hi, Math.max(lo, this.imitTargetBuf[j]));
        }
      }
      engine.imitTarget = this.imitTargetBuf;
      engine.imitRootDelta = rootDeltaAt(this.imitClip, this.imitTime) * this.imitScaleY;
      const ph = this.imitClip.duration > 0
        ? ((this.imitTime % this.imitClip.duration) / this.imitClip.duration) * 2 * Math.PI
        : 0;
      engine.imitPhase = [Math.sin(ph), Math.cos(ph)];
    }

    if (this.recovering) {
      this.stepRecovery();
    } else if (this.modelId === "microduck") {
      await this.stepDuck();
    } else {
      this.stepG1();
    }
    if (!engine.data) return; // Modellwechsel während des Steps (dispose)

    // Auto-Recovery nur im Test-Modus (Zustand läuft physisch weiter)
    if (this.mode === "test" && this.autoRecovery && !this.recovering && engine.isFallen()) {
      if (++this.fallDebounce >= 10) this.startRecovery();
    } else if (!this.recovering) {
      this.fallDebounce = 0;
    }

    if (this.mode === "test") {
      // v2.3: Sim-Zeit (Steps × dt) statt Wanduhr – korrekt auch bei Tempo 4×
      this.testUptime = this.testStepN * (this.stepMs() / 1000);
    }
  }

  // ── Enten-Step (cmd + ONNX/ES-Policy) ───────────────────────────────────────

  private async stepDuck(): Promise<void> {
    const engine = this.engine;
    const meta = engine.meta;

    // Command-Slots: Punkt-Modus → Richtung zum Punkt; sonst Mappings/Test
    if (this.pointCfg.mode !== "aus" && engine.targetPoint) {
      const [tx, ty] = engine.targetPoint;
      const p = engine.torsoPos();
      const dx = tx - p[0], dy = ty - p[1];
      const dist = Math.hypot(dx, dy);
      const yaw = engine.yaw();
      const fwd = Math.cos(yaw) * dx + Math.sin(yaw) * dy;
      const lat = -Math.sin(yaw) * dx + Math.cos(yaw) * dy;
      const desired = Math.atan2(dy, dx);
      let yerr = desired - yaw;
      while (yerr > Math.PI) yerr -= 2 * Math.PI;
      while (yerr < -Math.PI) yerr += 2 * Math.PI;
      if (dist > 0.12) {
        // v2.2: Tempo optional mit Distanz skalieren (weiter = schneller)
        const scale = this.pointCfg.speedByDist
          ? clamp(dist / Math.max(0.2, this.pointCfg.fullDist), 0.15, 1)
          : 1;
        engine.cmd[0] = clamp(this.pointCfg.maxSpeed * scale * Math.max(0, Math.cos(yerr)), VEL_BACK, VEL_FWD);
        engine.cmd[1] = clamp(lat * 1.2, -VEL_LAT, VEL_LAT);
        engine.cmd[2] = clamp(1.4 * yerr, -VEL_ANG, VEL_ANG);
      } else {
        engine.cmd[0] = 0; engine.cmd[1] = 0; engine.cmd[2] = 0;
      }
    } else {
      let vx = 0, vy = 0, wz = 0;
      // v2.4: Live-Vorschau/Auto-Gehen – die ES-Policy bekommt einen sichtbaren
      // Geh-Befehl (Training läuft headless; ohne das wirkt es wie „bewegt sich
      // nie“. Auch nach "Beste zeigen": cmd 0 = Policy steht absichtlich!)
      const previewOn = this.mode === "test" && this.trainPreview
        && (this.trainingActive || this.source === "es") && !this.recovering;
      if (previewOn) {
        vx = this.previewCmd[0]; vy = this.previewCmd[1]; wz = this.previewCmd[2];
      } else if (this.mode !== "test") {
        for (const m of this.mappings) {
          if (m.targetType !== "cmd") continue;
          const v = (m.source === "joyX" ? this.joy.x : this.joy.y) * m.gain;
          if (m.target === "cmd_x") vx += v;
          else if (m.target === "cmd_y") vy += v;
          else if (m.target === "cmd_yaw") wz -= v; // Stick rechts = Rechtsdrehung
        }
      }
      engine.cmd[0] = clamp(vx, VEL_BACK, VEL_FWD);
      engine.cmd[1] = clamp(vy, -VEL_LAT, VEL_LAT);
      engine.cmd[2] = clamp(wz, -VEL_ANG, VEL_ANG);
    }

    // Button-Trigger (Edge)
    if (this.pendingPolicyId) {
      this.overridePolicy = this.pendingPolicyId;
      this.pendingPolicyId = null;
    }
    if (this.pendingPoseId) {
      const pose = poseById("microduck", this.pendingPoseId);
      this.pendingPoseId = null;
      if (pose) this.startDuckPose(pose.targets);
    }

    // v2.1: Imitation manuell anwenden (Zielpose direkt als ctrl)
    if (this.mode === "manuell" && this.imitManual && this.imitClip && this.imitPlaying && this.imitTargetBuf) {
      engine.applyCtrlFromPose(this.imitTargetBuf);
      engine.stepPhysics();
      return;
    }

    // Pose-Hold (kurze Overlay-Phase, dann zurück zur Policy)
    if (this.duckPoseHold) {
      const hold = this.duckPoseHold;
      const now = performance.now();
      const u = clamp((now - hold.t0) / hold.blendMs, 0, 1);
      const ctrl = engine.data.ctrl as Float32Array;
      for (let j = 0; j < meta.actionDim; j++) {
        ctrl[j] = hold.from[j] + (hold.to[j] - hold.from[j]) * u;
      }
      engine.stepPhysics();
      if (now - hold.t0 > hold.blendMs + hold.holdMs) {
        this.duckPoseHold = null;
        engine.lastAction.fill(0);
        this.esActSm?.fill(0); // v2.5
      }
      return;
    }

    // Policy wählen
    let policy: OnnxPolicy | MlpPolicy | null = null;
    if (this.source === "es" && this.esView) {
      policy = this.esView;
    } else {
      const id = this.overridePolicy ?? this.selectedOnnx;
      policy = (await this.getOnnxSession(id)) ?? (await this.getOnnxSession(meta.defaultOnnx));
      if (!engine.data) return; // Modellwechsel während des Session-Ladens
    }
    if (!policy) {
      engine.stepPhysics();
      return;
    }

    // ONNX braucht exakte Basis-Dimension; ES-MLP ggf. +2 (Imitations-Phase)
    const obs = policy instanceof MlpPolicy
      ? engine.obsFor(this.esObsExtra >= 2)
      : engine.obsFor(false);
    const act = policy instanceof MlpPolicy
      ? policy.forward(obs, this.actBuf)
      : await policy.forward(obs);
    this.actBuf.fill(0);
    this.actBuf.set(act.subarray(0, Math.min(act.length, this.actBuf.length)));
    // v2.5: Aktions-Glättung im sichtbaren Pfad EXAKT wie im Training (ctxStep):
    // gefilterte Aktion steuert Physik UND Obs → Vorschau/Test zeigen das
    // trainierte Verhalten (kein Zittern, kein Sturz nur wegen Filter-Delta).
    let actUsed = this.actBuf;
    if (policy instanceof MlpPolicy && this.trainCfg.actionSmooth < 0.999) {
      const a = this.trainCfg.actionSmooth;
      if (!this.esActSm || this.esActSm.length !== this.actBuf.length) {
        this.esActSm = new Float32Array(this.actBuf.length);
      }
      const sm = this.esActSm;
      for (let j = 0; j < sm.length; j++) sm[j] = a * this.actBuf[j] + (1 - a) * sm[j];
      actUsed = sm;
    }
    engine.lastAction.set(actUsed);
    engine.stepWithAction(actUsed);

    if (this.mode === "test") {
      this.testStepN++;
      // v2.6: Ehrlicher Reward — Bodenzeit bringt NICHTS. Am Boden liegend
      // oder während der Aufrichtung gibt es keine Punkte mehr.
      if (!engine.isFallen() && !this.recovering) {
        const r = stepRewardValue(
          engine, this.rewardCfg!, actUsed, this.prevAct,
          { t: this.testUptime, step: this.testStepN, dt: this.stepMs() / 1000 },
        );
        if (Number.isFinite(r)) this.testReward += r;
      }
      // v2.6: Sturz kostet (einmalig pro Fall) — Fälle dürfen sich NIEMALS
      // auszahlen, sonst lernt die Evolution falsches Verhalten.
      if (engine.isFallen()) {
        if (!this.testFallPenalized) {
          this.testFallPenalized = true;
          const fw = this.rewardCfg?.terms.fall;
          this.testReward -= Number.isFinite(fw?.weight) && fw!.weight > 0 ? fw!.weight : 10;
        }
      } else {
        this.testFallPenalized = false;
      }
      this.prevAct.set(actUsed);
    }
  }

  private startDuckPose(targets: Record<string, number>) {
    const meta = this.engine.meta;
    const from = new Float32Array(this.engine.data.ctrl as Float32Array);
    const to = new Float32Array(meta.actionDim);
    for (let j = 0; j < meta.actionDim; j++) {
      const name = meta.jointNames[j];
      to[j] = targets[name] ?? meta.defaultPose[j];
    }
    this.duckPoseHold = { from, to, t0: performance.now(), blendMs: 400, holdMs: 1600 };
  }

  // ── G1-Step (direkte Gelenk-Targets / ES-Policy) ────────────────────────────

  private stepG1(): void {
    const engine = this.engine;
    const meta = engine.meta;
    const dim = meta.actionDim;

    // Pose-Trigger (Edge)
    if (this.pendingPoseId) {
      const pose = poseById("unitree_g1", this.pendingPoseId);
      this.pendingPoseId = null;
      if (pose) this.startG1Blend(pose.targets);
    }

    // Basis-Blending (0.4 s)
    if (this.g1Blend && this.g1Base) {
      const b = this.g1Blend;
      const u = clamp((performance.now() - b.t0) / b.dur, 0, 1);
      const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      for (let j = 0; j < dim; j++) this.g1Base[j] = b.from[j] + (b.to[j] - b.from[j]) * e;
      if (u >= 1) this.g1Blend = null;
    }

    if (this.mode === "manuell") {
      // Joystick-Offsets auf gemappten Gelenken
      const targets = new Float32Array(dim);
      targets.set(this.g1Base ?? engine.standPose);
      for (const m of this.mappings) {
        if (m.targetType !== "gelenk") continue;
        const v = (m.source === "joyX" ? this.joy.x : this.joy.y) * m.gain;
        for (const jn of m.target.split(",").map((s) => s.trim())) {
          const idx = meta.jointNames.indexOf(jn);
          if (idx >= 0) targets[idx] += v;
        }
      }
      engine.applyCtrlFromPose(targets);
      engine.stepPhysics();
      return;
    }

    // v2.2: Punkt-Modus – cmd Richtung Punkt (für ES-Obs), Tempo mit Distanz
    if (this.pointCfg.mode !== "aus" && engine.targetPoint) {
      const [tx, ty] = engine.targetPoint;
      const p = engine.torsoPos();
      const dx = tx - p[0], dy = ty - p[1];
      const dist = Math.hypot(dx, dy);
      const yaw = engine.yaw();
      const desired = Math.atan2(dy, dx);
      let yerr = desired - yaw;
      while (yerr > Math.PI) yerr -= 2 * Math.PI;
      while (yerr < -Math.PI) yerr += 2 * Math.PI;
      const scale = this.pointCfg.speedByDist
        ? clamp(dist / Math.max(0.2, this.pointCfg.fullDist), 0.15, 1)
        : 1;
      engine.cmd[0] = dist > 0.15 ? this.pointCfg.maxSpeed * scale * Math.max(0, Math.cos(yerr)) : 0;
      engine.cmd[2] = clamp(1.2 * yerr, -1, 1);
    } else if (this.mode === "test" && this.trainPreview
      && (this.trainingActive || this.source === "es") && !this.recovering) {
      // v2.4: Live-Vorschau – Geh-Befehl, damit der G1 sichtbar übt/zeigt
      engine.cmd[0] = this.previewCmd[0];
      engine.cmd[2] = this.previewCmd[2];
    }

    // Training/Test: ES-Policy (G1 hat keine ONNX-Policies)
    const mlp = this.esView;
    if (!mlp || !this.g1Base) {
      // noch nichts trainiert: Stand-Pose halten, Physik läuft weiter
      engine.applyCtrlFromPose(engine.standPose);
      engine.stepPhysics();
      return;
    }
    const obs = engine.obsFor(this.esObsExtra >= 2);
    const act = mlp.forward(obs, this.actBuf);
    // v2.5: Glättung wie im Training (sonst zittert G1 in der Vorschau)
    let actUsed: Float32Array = act;
    if (this.trainCfg.actionSmooth < 0.999) {
      const a = this.trainCfg.actionSmooth;
      if (!this.esActSm || this.esActSm.length !== act.length) {
        this.esActSm = new Float32Array(act.length);
      }
      const sm = this.esActSm;
      for (let j = 0; j < sm.length; j++) sm[j] = a * act[j] + (1 - a) * sm[j];
      actUsed = sm;
    }
    engine.lastAction.set(actUsed);
    engine.stepWithAction(actUsed);
    if (this.mode === "test") {
      this.testStepN++;
      // v2.6: Ehrlicher Reward (identisch zum Enten-Pfad) — Bodenzeit bringt nichts.
      if (!engine.isFallen() && !this.recovering) {
        const r = stepRewardValue(
          engine, this.rewardCfg!, actUsed, this.prevAct,
          { t: this.testUptime, step: this.testStepN, dt: this.stepMs() / 1000 },
        );
        if (Number.isFinite(r)) this.testReward += r;
      }
      if (engine.isFallen()) {
        if (!this.testFallPenalized) {
          this.testFallPenalized = true;
          const fw = this.rewardCfg?.terms.fall;
          this.testReward -= Number.isFinite(fw?.weight) && fw!.weight > 0 ? fw!.weight : 10;
        }
      } else {
        this.testFallPenalized = false;
      }
      this.prevAct.set(actUsed);
    }
  }

  private startG1Blend(targets: Record<string, number>) {
    const meta = this.engine.meta;
    const to = new Float32Array(meta.actionDim);
    for (let j = 0; j < meta.actionDim; j++) {
      to[j] = targets[meta.jointNames[j]] ?? this.engine.standPose[j];
    }
    this.g1Blend = { from: (this.g1Base ?? this.engine.standPose).slice(), to, t0: performance.now(), dur: 400 };
  }

  // ── Auto-Recovery (sanftes Blenden zur Stand-Pose, kein Teleport) ───────────

  private startRecovery() {
    this.recovering = true;
    this.recoverySteps = 0;
    this.recoveryUpright = 0;
    this.fallDebounce = 0;
    this.engine.lastAction.fill(0);
    this.esActSm?.fill(0); // v2.5: Filter-State auch zuruecksetzen
    this.emit();
  }

  private stepRecovery() {
    const engine = this.engine;
    const meta = engine.meta;
    // v2.4: Notbremse – schafft die Policy die Standpose nicht (z. B. Ente in
    // Bauchlage: projGravZ ≈ 0 → "aufrecht" nie erfüllbar → endlose Liegezeit),
    // nach ~3 s Sim sanft aufs Keyframe zurücksetzen. Letzte Stufe der Kette
    // (Blenden → Keyframe), damit die Vorschau immer weiterläuft.
    if (this.recoverySteps >= 150) {
      this.recovering = false;
      this.recoverySteps = 0;
      this.recoveryUpright = 0;
      this.fallDebounce = 0;
      engine.resetToKeyframe();
      engine.lastAction.fill(0);
      this.esActSm?.fill(0); // v2.5
      this.prevAct.fill(0);
      this.previewCmdT = 0; // v2.5: sofort frischen (sicheren) Vorschau-Befehl
      this.emit();
      return;
    }
    const stand = meta.id === "microduck" ? Float32Array.from(meta.defaultPose) : engine.standPose;
    const ctrl = engine.data.ctrl as Float32Array;
    for (let j = 0; j < meta.actionDim; j++) {
      ctrl[j] += (stand[j] - ctrl[j]) * 0.06; // sanft, pro Policy-Step
    }
    engine.stepPhysics();
    this.recoverySteps++;
    // v2.4: Ente in Bauchlage hat projGravZ ≈ 0 → zusätzlich Höhe akzeptieren:
    // aufrecht = entweder sauber aufrecht ODER Höhe wieder im Stand-Bereich.
    const gz = engine.projGravZ();
    const upright = (gz < -0.85 || engine.height() >= meta.targetHeight * 0.8)
      && !engine.isFallen();
    this.recoveryUpright = upright ? this.recoveryUpright + 1 : 0;
    if (this.recoveryUpright >= 50) {
      // 1 s aufrecht → Policy übernimmt wieder (Zustand läuft weiter)
      this.recovering = false;
      this.recoveryUpright = 0;
      engine.lastAction.fill(0);
      this.esActSm?.fill(0); // v2.5
      this.prevAct.fill(0);
      this.emit();
    }
  }

  // ── Render-Loop (rAF) ───────────────────────────────────────────────────────

  private renderLoop = () => {
    if (!this.running || !this.world) return;
    const now = performance.now();
    const dt = Math.min(0.05, (now - this.lastFrameT) / 1000 || 0.016);
    this.lastFrameT = now;
    const engine = this.engine;
    if (engine.data && engine.meta) {
      const [x, y, z] = engine.torsoPos();
      this.world.setCameraTarget(x, z, -y);
      if (engine.meta.id === "microduck") {
        syncDuck(this.world, engine);
      } else {
        syncG1(this.world, engine, engine.mujoco);
      }
      syncBall(this.world, engine);
      this.world.updateCamera(dt);
      this.world.renderer.render(this.world.scene, this.world.camera);
    }
    if (now - this.lastEmit > 250) {
      this.lastEmit = now;
      this.emit();
    }
    requestAnimationFrame(this.renderLoop);
  };

  // ── Telemetrie ──────────────────────────────────────────────────────────────

  private emit() {
    const engine = this.engine;
    const qvel = engine.data ? (engine.data.qvel as Float32Array) : null;
    const t: Telemetry = {
      booted: this.booted,
      loading: this.loading,
      loadingText: this.loadingText,
      error: this.error,
      modelId: this.modelId,
      mode: this.mode,
      source: this.source,
      onnxId: this.selectedOnnx,
      overridePolicy: this.overridePolicy,
      esReady: this.esView !== null,
      speed: qvel ? Math.hypot(qvel[0], qvel[1]) : 0,
      height: engine.data ? engine.height() : 0,
      ctrlHz: Math.round(this.ctrlHz),
      recovering: this.recovering,
      fallen: engine.data ? engine.isFallen() : false,
      testUptime: this.testUptime,
      testReward: this.testReward,
      training: this.trainingActive,
      es: this.trainer ? this.trainer.stats() : null,
      mappings: this.mappings,
      reward: this.rewardCfg ?? (this.modelId ? loadRewardConfig(this.modelId) : ({} as RewardConfig)),
      trainCfg: { ...this.trainCfg },
      trainPreview: this.trainPreview,
      previewSpeed: this.previewSpeed, // v2.5: Sync-Faktor der sichtbaren Physik
      customCode: this.rewardCfg?.custom
        ? { name: this.rewardCfg.custom.name, enabled: this.rewardCfg.custom.enabled }
        : null,
      world: this.worldCfg,
      pointMode: this.pointCfg.mode !== "aus",
      pointCfg: { ...this.pointCfg },
      point: [this.point[0], this.point[1]],
      imitation: this.imitClip
        ? {
            name: this.imitClip.name, duration: this.imitClip.duration,
            playing: this.imitPlaying,
            // v2.3: Anzeigezeit wrappen (49.2/9.0 s → 4.2/9.0 s)
            time: this.imitClip.duration > 0
              ? this.imitTime % this.imitClip.duration
              : this.imitTime,
            mapped: this.imitClip.mapped, manual: this.imitManual,
          }
        : null,
    };
    try {
      this.onTelemetry(t);
    } catch {
      // UI-Fehler dürfen die Sim nicht abwürgen
    }
  }

  get isTraining(): boolean {
    return this.trainingActive;
  }

  // ── Vollbild ────────────────────────────────────────────────────────────────

  static async toggleFullscreen(el: HTMLElement): Promise<void> {
    const doc = document as Document & {
      webkitFullscreenElement?: Element;
      webkitExitFullscreen?: () => Promise<void>;
    };
    const full = el as HTMLElement & { webkitRequestFullscreen?: () => Promise<void> };
    const active = document.fullscreenElement ?? doc.webkitFullscreenElement;
    try {
      if (active) {
        if (document.exitFullscreen) await document.exitFullscreen();
        else await doc.webkitExitFullscreen?.();
        return;
      }
      if (el.requestFullscreen) await el.requestFullscreen();
      else if (full.webkitRequestFullscreen) await full.webkitRequestFullscreen();
      const orientation = screen.orientation as ScreenOrientation & {
        lock?: (o: string) => Promise<void>;
      };
      await orientation?.lock?.("landscape")?.catch?.(() => {});
    } catch {
      // Vollbild/Rotation verweigert (z. B. iOS Safari) – ignorieren
    }
  }

  // ── Aufräumen ───────────────────────────────────────────────────────────────

  dispose() {
    this.running = false;
    this.trainingActive = false;
    this.trainer?.dispose();
    this.trainer = null;
    this.engine.dispose();
    this.world?.dispose();
    this.world = null;
    this.booted = false;
  }
}

// ── v2.1: Persistenz (Welt + Punkt-Modus) ────────────────────────────────────

function loadWorldPrefs(): WorldConfig {
  try {
    const raw = localStorage.getItem("mdt_v2_world");
    if (raw) {
      const p = JSON.parse(raw) as WorldConfig;
      if (p && typeof p.seed === "number") {
        return { ...defaultWorldConfig(), ...p, features: { ...defaultWorldConfig().features, ...p.features } };
      }
    }
  } catch { /* ignore */ }
  return defaultWorldConfig();
}

function saveWorldPrefs(cfg: WorldConfig): void {
  try { localStorage.setItem("mdt_v2_world", JSON.stringify(cfg)); } catch { /* ignore */ }
}

function loadPointPrefs(modelId: ModelId): PointCfg {
  const def = defaultPointCfg(modelId);
  try {
    const raw = localStorage.getItem(`mdt_v2_pointcfg_${modelId}`);
    if (raw) {
      const p = JSON.parse(raw) as Partial<PointCfg>;
      if (p && typeof p === "object") {
        return {
          ...def,
          ...p,
          mode: ("aus|frei|umkreis|pfad".split("|") as PointModeState[]).includes(p.mode as PointModeState)
            ? (p.mode as PointModeState) : def.mode,
        };
      }
    }
    // Migration vom alten Boolean-Pref
    if (localStorage.getItem(`mdt_v2_point_${modelId}`) === "1") {
      return { ...def, mode: "frei" };
    }
  } catch { /* ignore */ }
  return def;
}

function savePointPrefs(modelId: ModelId, cfg: PointCfg): void {
  try { localStorage.setItem(`mdt_v2_pointcfg_${modelId}`, JSON.stringify(cfg)); } catch { /* ignore */ }
}

function loadTrainPrefs(modelId: ModelId): TrainCfg {
  const def = defaultTrainCfg();
  // v2.3: Modell-abhängiges Befehls-Tempo (Ente 0.25, G1 0.5)
  const fwdCap = getModel(modelId).velocityLimit.fwd;
  def.cmdFwd = Math.max(0.1, fwdCap * 0.8);
  try {
    const raw = localStorage.getItem(`mdt_v2_traincfg_${modelId}`);
    if (raw) {
      const p = JSON.parse(raw) as Partial<TrainCfg>;
      if (p && typeof p === "object") {
        return {
          rolloutSteps: Number.isFinite(p.rolloutSteps) ? clamp(Math.round(p.rolloutSteps!), 30, 1000) : def.rolloutSteps,
          maxGenerations: Number.isFinite(p.maxGenerations) ? clamp(Math.round(p.maxGenerations!), 0, 100000) : def.maxGenerations,
          lr: Number.isFinite(p.lr) ? clamp(p.lr!, 0.002, 0.2) : def.lr,
          sigma: Number.isFinite(p.sigma) ? clamp(p.sigma!, 0.005, 0.3) : def.sigma,
          // v2.3: Migration – fehlende Profi-Felder erhalten die AN-Defaults
          cmdTrain: typeof p.cmdTrain === "boolean" ? p.cmdTrain : def.cmdTrain,
          cmdFwd: Number.isFinite(p.cmdFwd) ? clamp(p.cmdFwd!, 0.05, Math.max(0.05, fwdCap)) : def.cmdFwd,
          curriculum: typeof p.curriculum === "boolean" ? p.curriculum : def.curriculum,
          actionSmooth: Number.isFinite(p.actionSmooth) ? clamp(p.actionSmooth!, 0.3, 1) : def.actionSmooth,
          pushes: typeof p.pushes === "boolean" ? p.pushes : def.pushes,
          noiseReset: typeof p.noiseReset === "boolean" ? p.noiseReset : def.noiseReset,
          fitnessMode: p.fitnessMode === "mean" ? "mean" : def.fitnessMode,
          weightDecay: Number.isFinite(p.weightDecay) ? clamp(p.weightDecay!, 0, 0.05) : def.weightDecay,
        };
      }
    }
  } catch { /* ignore */ }
  return def;
}

function saveTrainPrefs(modelId: ModelId, cfg: TrainCfg): void {
  try { localStorage.setItem(`mdt_v2_traincfg_${modelId}`, JSON.stringify(cfg)); } catch { /* ignore */ }
}

// ── v2.3: Float32 ⇄ Base64 (kompakte Gewichte-Speicherung) ─────────────────

function f32ToB64(f: Float32Array): string {
  const bytes = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
  let out = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    out += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunk)));
  }
  return btoa(out);
}

function b64ToF32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export default TrainerCore;
