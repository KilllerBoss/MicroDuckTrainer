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
import {
  defaultWorldConfig, generateWorld, arenaHalf, randomSeed,
  buildWorldMeshes, type WorldConfig, type WorldBuild,
} from "./worldgen";
import {
  buildImitClip, targetAt, rootDeltaAt, saveImitPrefs,
  type ImitClip,
} from "./imitation";


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
  // ── v2.1 ──
  world: WorldConfig;
  pointMode: boolean;
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

  // ── v2.1: Welt, Punkt-Modus, Imitation ──
  private worldCfg: WorldConfig = defaultWorldConfig();
  private worldBuild: WorldBuild | null = null;
  private pointMode = false;
  private point: [number, number] = [0.8, 0];
  private imitClip: ImitClip | null = null;
  private imitPlaying = false;
  private imitManual = false;
  private imitTime = 0;
  private imitTargetBuf: Float32Array | null = null;
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
      this.pointMode = loadPointPrefs(id);
      this.point = [0.8, 0];

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
      this.engine.targetPoint = this.pointMode ? [...this.point] as [number, number] : null;
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
    this.imitTime = 0;
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
      this.engine.targetPoint = this.pointMode ? ([...this.point] as [number, number]) : null;
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

  // ── v2.1: Punkt-Modus ──────────────────────────────────────────────────────

  setPointMode(on: boolean): void {
    this.pointMode = on;
    if (this.modelId) savePointPrefs(this.modelId, on);
    this.engine.targetPoint = on ? ([...this.point] as [number, number]) : null;
    this.trainer?.setTargetPoint(on ? ([...this.point] as [number, number]) : null);
    this.emit();
  }

  get isPointMode(): boolean {
    return this.pointMode;
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
      const clip = await buildImitClip(buffer, clipIndex, meta, mirror);
      if (clip.mapped === 0) {
        return { ok: false, message: "Keine Gelenke gemappt – Skeleton-Namen nicht erkannt." };
      }
      this.imitClip = clip;
      this.imitPlaying = true;
      this.imitTime = 0;
      this.imitTargetBuf = new Float32Array(clip.dim);
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
      this.trainer.setTargetPoint(this.pointMode ? ([...this.point] as [number, number]) : null);
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
      this.esObsExtra = this.trainer.layout.obsDim - this.trainer.meta.obsDim;
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

    // ── v2.1: Punkt-Modus – Joystick bewegt den 3D-Punkt ──
    const dtStep = this.stepMs() / 1000;
    if (this.pointMode) {
      const sp = 1.4 * dtStep;
      const lim = arenaHalf(this.modelId) - 0.15;
      this.point[0] = Math.min(lim, Math.max(-lim, this.point[0] + this.joy.x * sp));
      this.point[1] = Math.min(lim, Math.max(-lim, this.point[1] + this.joy.y * sp));
      engine.targetPoint = [this.point[0], this.point[1]];
      this.trainer?.setTargetPoint([this.point[0], this.point[1]]);
      this.world?.setTargetPoint(this.point[0], this.point[1], true);
    } else if (engine.targetPoint) {
      engine.targetPoint = null;
      this.world?.setTargetPoint(0, 0, false);
    }

    // ── v2.1: Imitation – Zielpose je Policy-Step ──
    if (this.imitClip && this.imitPlaying && this.imitTargetBuf) {
      this.imitTime += dtStep;
      targetAt(this.imitClip, this.imitTime, this.imitTargetBuf);
      engine.imitTarget = this.imitTargetBuf;
      engine.imitRootDelta = rootDeltaAt(this.imitClip, this.imitTime);
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
      this.testUptime = (performance.now() - this.testStart) / 1000;
    }
  }

  // ── Enten-Step (cmd + ONNX/ES-Policy) ───────────────────────────────────────

  private async stepDuck(): Promise<void> {
    const engine = this.engine;
    const meta = engine.meta;

    // Command-Slots: Punkt-Modus → Richtung zum Punkt; sonst Mappings/Test
    if (this.pointMode && engine.targetPoint) {
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
        engine.cmd[0] = clamp(VEL_FWD * 0.9 * Math.max(0, Math.cos(yerr)), VEL_BACK, VEL_FWD);
        engine.cmd[1] = clamp(lat * 1.2, -VEL_LAT, VEL_LAT);
        engine.cmd[2] = clamp(1.4 * yerr, -VEL_ANG, VEL_ANG);
      } else {
        engine.cmd[0] = 0; engine.cmd[1] = 0; engine.cmd[2] = 0;
      }
    } else {
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

    // v2.1: Punkt-Modus – cmd Richtung Punkt (für ES-Obs)
    if (this.pointMode && engine.targetPoint) {
      const [tx, ty] = engine.targetPoint;
      const p = engine.torsoPos();
      const dx = tx - p[0], dy = ty - p[1];
      const dist = Math.hypot(dx, dy);
      const yaw = engine.yaw();
      const desired = Math.atan2(dy, dx);
      let yerr = desired - yaw;
      while (yerr > Math.PI) yerr -= 2 * Math.PI;
      while (yerr < -Math.PI) yerr += 2 * Math.PI;
      engine.cmd[0] = dist > 0.15 ? 0.35 * Math.max(0, Math.cos(yerr)) : 0;
      engine.cmd[2] = clamp(1.2 * yerr, -1, 1);
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
      world: this.worldCfg,
      pointMode: this.pointMode,
      point: [this.point[0], this.point[1]],
      imitation: this.imitClip
        ? {
            name: this.imitClip.name, duration: this.imitClip.duration,
            playing: this.imitPlaying, time: this.imitTime,
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

function loadPointPrefs(modelId: ModelId): boolean {
  try { return localStorage.getItem(`mdt_v2_point_${modelId}`) === "1"; } catch { return false; }
}

function savePointPrefs(modelId: ModelId, on: boolean): void {
  try { localStorage.setItem(`mdt_v2_point_${modelId}`, on ? "1" : "0"); } catch { /* ignore */ }
}

export default TrainerCore;
