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
import { EsTrainer, stepRewardValue, type EsStats, type TurboLevel } from "./es";
import type { MappingEntry, SourceId } from "./mapping";
import { poseById, loadMapping, saveMapping } from "./mapping";
import { loadRewardConfig, saveRewardConfig, type RewardConfig } from "./rewards";


export type Mode = "manuell" | "training" | "test";
export type PolicySource = "onnx" | "es";

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
}

// Enten-Tempos wie reference/constants.js
const VEL_FWD = 0.25, VEL_BACK = -0.2, VEL_ANG = 1.0, VEL_LAT = 0.4;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export class TrainerCore {
  // ── Callbacks (React-Seite) ──
  onTelemetry: (t: Telemetry) => void = () => {};

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
  private autoRecovery = true; // Test-Modus: sanftes Aufrichten nach Sturz

  // Gamepad-Eingaben
  private joy = { x: 0, y: 0 };
  private pendingPolicyId: string | null = null;
  private pendingPoseId: string | null = null;

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

      const tasks: Promise<any>[] = [this.engine.load(id)];
      if (needRig && this.world) tasks.push(mountRig(this.world, id));
      if (id === "microduck") tasks.push(this.getOnnxSession(meta.defaultOnnx).catch(() => null));
      await Promise.all(tasks);

      this.modelId = id;
      this.selectedOnnx = meta.defaultOnnx;
      this.source = "onnx";
      this.g1Base = new Float32Array(this.engine.standPose);
      this.prevAct = new Float32Array(meta.actionDim);
      this.actBuf = new Float32Array(meta.actionDim);
      // Mappings + Bewertung pro Modell laden (Persistenz)
      this.mappings = loadMapping(id);
      this.rewardCfg = loadRewardConfig(id);
      this.testUptime = 0;
      this.testReward = 0;
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

  setJoystick(x: number, y: number) {
    this.joy.x = clamp(x, -1, 1);
    this.joy.y = clamp(y, -1, 1);
  }

  setAutoRecovery(on: boolean) {
    this.autoRecovery = on;
    if (!on) this.recovering = false;
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
      if (!this.trainer) {
        const warm = await warmStartMlp(getModel(this.modelId));
        this.trainer = new EsTrainer(this.modelId, warm.theta, warm.layout, this.rewardCfg!);
        await this.trainer.init();
      }
      this.trainer.setReward(this.rewardCfg!);
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
    this.trainer?.setTurbo(t);
    this.emit();
  }

  private async trainingLoop() {
    while (this.trainingActive && this.trainer) {
      await this.trainer.runGeneration();
      this.updateEsView();
      this.emit();
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
    } catch {
      this.esView = null;
    }
  }

  /** "Beste zeigen": Quelle auf ES-Policy (bestTheta) umschalten. */
  showBest() {
    this.updateEsView();
    this.source = "es";
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
        theta: Array.from(t.theta),
        bestTheta: Array.from(t.bestTheta),
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
        layout: MlpLayout; theta: number[]; bestTheta: number[];
        generation: number; bestEver: number; history: number[];
      };
      if (!d?.theta?.length || !d?.layout) return false;
      this.trainingActive = false;
      this.trainer?.dispose();
      this.trainer = new EsTrainer(this.modelId, new Float32Array(d.theta), d.layout, this.rewardCfg!);
      await this.trainer.init();
      this.trainer.bestTheta = new Float32Array(d.bestTheta ?? d.theta);
      this.trainer.generation = d.generation ?? 0;
      this.trainer.bestEver = d.bestEver ?? -Infinity;
      this.trainer.history = Array.isArray(d.history) ? d.history : [];
      this.updateEsView();
      this.emit();
      return true;
    } catch (err) {
      console.warn("[core] Theta-Load fehlgeschlagen:", err);
      return false;
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
      try {
        await this.controlStep();
      } catch (err) {
        console.warn("[core] controlStep:", err);
      }
      count++;
      const now = performance.now();
      if (now - hzT0 > 500) {
        this.ctrlHz = (count * 1000) / (now - hzT0);
        count = 0;
        hzT0 = now;
      }
      next += this.stepMs();
      const wait = next - performance.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      else next = performance.now(); // zurückgefallen: nicht spiralen
    }
  }

  private async controlStep(): Promise<void> {
    const engine = this.engine;
    if (!engine.data || !engine.meta || !this.modelId) return;

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
      this.testUptime = (performance.now() - this.testStart) / 1000;
    }
  }

  // ── Enten-Step (cmd + ONNX/ES-Policy) ───────────────────────────────────────

  private async stepDuck(): Promise<void> {
    const engine = this.engine;
    const meta = engine.meta;

    // Command-Slots (Manuell/Training: Joystick; Test: Null-Kommando)
    let vx = 0, vy = 0, wz = 0;
    if (this.mode !== "test") {
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

    const obs = engine.buildObs();
    const act = policy instanceof MlpPolicy
      ? policy.forward(obs, this.actBuf)
      : await policy.forward(obs);
    this.actBuf.fill(0);
    this.actBuf.set(act.subarray(0, Math.min(act.length, this.actBuf.length)));
    engine.lastAction.set(this.actBuf);
    engine.stepWithAction(this.actBuf);

    if (this.mode === "test") {
      this.testReward += stepRewardValue(engine, this.rewardCfg!, this.actBuf, this.prevAct);
      this.prevAct.set(this.actBuf);
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

    // Training/Test: ES-Policy (G1 hat keine ONNX-Policies)
    const mlp = this.esView;
    if (!mlp || !this.g1Base) {
      // noch nichts trainiert: Stand-Pose halten, Physik läuft weiter
      engine.applyCtrlFromPose(engine.standPose);
      engine.stepPhysics();
      return;
    }
    const obs = engine.buildObs();
    const act = mlp.forward(obs, this.actBuf);
    engine.lastAction.set(act);
    engine.stepWithAction(act);
    if (this.mode === "test") {
      this.testReward += stepRewardValue(engine, this.rewardCfg!, act, this.prevAct);
      this.prevAct.set(act);
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
    this.emit();
  }

  private stepRecovery() {
    const engine = this.engine;
    const meta = engine.meta;
    const stand = meta.id === "microduck" ? Float32Array.from(meta.defaultPose) : engine.standPose;
    const ctrl = engine.data.ctrl as Float32Array;
    for (let j = 0; j < meta.actionDim; j++) {
      ctrl[j] += (stand[j] - ctrl[j]) * 0.06; // sanft, pro Policy-Step
    }
    engine.stepPhysics();
    this.recoverySteps++;
    const upright = engine.projGravZ() < -0.85 && !engine.isFallen();
    this.recoveryUpright = upright ? this.recoveryUpright + 1 : 0;
    if (this.recoveryUpright >= 50) {
      // 1 s aufrecht → Policy übernimmt wieder (Zustand läuft weiter)
      this.recovering = false;
      this.recoveryUpright = 0;
      engine.lastAction.fill(0);
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

export default TrainerCore;
