// ── MicroDuck Trainer v2.0 – Evolution Strategy (OpenAI-ES) ──────────────────
// Antithetic Sampling, Rank-Normalisierung, adaptives Sigma. Evaluierung läuft
// bevorzugt in Web-Workern (headless, reine Physik) – Fallback: Main-Thread-
// Interleaving (mehrere MjData-Instanzen rundenweise, N = Turbo-Stufe).
// Die laufende Entscheidung steht im Worklog (public/workers/es-worker.js).

import type { ModelMeta, ModelId } from "./models";
import { getModel } from "./models";
import { Engine } from "./engine";
import { MlpPolicy, hiddenList, type MlpLayout } from "./policy";
import type { RewardConfig } from "./rewards";
import type { WorldBuild } from "./worldgen";
import { getCustomFn, evalCustomFn, type CustomRewardFn } from "./customcode";

/** Imitations-Daten für Rollouts (Main-Thread + Worker, identisch). */
export interface ImitEvalData {
  dim: number;
  frames: number;
  fps: number;
  duration: number;
  targets: Float32Array; // frames × dim (RELATIV zur Ruhelage, v2.3)
  rootY: Float32Array; // frames (GLB y-up)
  baseY: number;
  // v2.3: Cross-Species-Retargeting (Ente lernt Human-Clips)
  /** Zentrum-Pose (Ente: defaultPose, G1: standPose) – wird zu den relativen
   *  Clip-Zielen addiert, damit die Zielpose im Stehen exakt die Standpose ist. */
  center?: Float32Array;
  /** Wurzel-Höhen-Skalierung (Roboterhöhe / Clip-Wurzelhöhe), default 1. */
  scaleY?: number;
}

/** Zielpose + Wurzel-Delta zum Zeitpunkt t (loop). Schreibt in out (ABSOLUT). */
export function imitSampleAt(d: ImitEvalData, t: number, out: Float32Array): number {
  const tt = d.duration > 0 ? ((t % d.duration) + d.duration) % d.duration : 0;
  const x = tt * d.fps;
  const f0 = Math.min(d.frames - 1, Math.floor(x));
  const f1 = Math.min(d.frames - 1, f0 + 1);
  const u = x - f0;
  const c = d.center;
  for (let j = 0; j < d.dim; j++) {
    const a = d.targets[f0 * d.dim + j];
    const b = d.targets[f1 * d.dim + j];
    let v = (a + (b - a) * u) + (c ? c[j] : 0);
    // v2.6 NaN-Schutz: kaputte Clip-Werte → Ruhelage (statt REWARD NaN)
    out[j] = Number.isFinite(v) ? v : (c ? c[j] : 0);
  }
  const raw = (d.rootY[f0] + (d.rootY[f1] - d.rootY[f0]) * u) - d.baseY;
  const s = d.scaleY ?? 1;
  return Number.isFinite(raw) ? raw * s : 0;
}

/**
 * v2.3: Profi-Trainings-Konfiguration (Professional-Tricks, an/aus + Werte).
 * Identisch an Main-Thread-Rollouts und ES-Worker übergeben.
 */
export interface RunCfg {
  /** Befehle (cmd) pro Rollout zufällig ziehen + Tracking-Reward → Joystick-fähige Policies. */
  cmdTrain: boolean;
  cmdFwd: number; // max Vorwärtstempo der Befehle (m/s)
  cmdLat: number; // max Seitwärtstempo (m/s)
  cmdAng: number; // max Gierrate (rad/s)
  /** Curriculum: Befehls-Tempo automatisch an die Sturzrate anpassen. */
  curriculum: boolean;
  /** Action-Lowpass: 1 = aus, sonst EMA-Alpha (0.3–0.9). Gegen Zittern. */
  actionSmooth: number;
  /** Zufalls-Stöße auf den Basis-Körper (Domain-Randomization). */
  pushes: boolean;
  /** Reset mit kleinem Zustands-Rauschen (Reference-State-Init). */
  noiseReset: boolean;
  /** Fitness = Summe (Überleben zählt) oder Mittelwert (klassisch). */
  fitnessMode: "sum" | "mean";
  /** Gewichtsbremse (theta-Decay gegen saturierte tanh-Ausgänge), 0 = aus. */
  weightDecay: number;
}

export function defaultRunCfg(): RunCfg {
  return {
    cmdTrain: true, cmdFwd: 0.25, cmdLat: 0.15, cmdAng: 0.8,
    curriculum: true, actionSmooth: 0.6, pushes: true, noiseReset: true,
    fitnessMode: "sum", weightDecay: 0.02,
  };
}

export type TurboLevel = 1 | 4 | 16 | 32 | 64;

export const TURBO_LEVELS: TurboLevel[] = [1, 4, 16, 32, 64];

export function popSizeFor(turbo: TurboLevel): number {
  switch (turbo) {
    case 1: return 24;
    case 4: return 32;
    case 16: return 48;
    case 32: return 64;
    case 64: return 96;
  }
}

export interface EsStats {
  generation: number;
  bestFitness: number;
  bestEver: number;
  sigma: number;
  stepsPerSec: number;
  history: number[]; // beste Fitness je Generation
  meanFitness: number;
  fellRate: number;
  evaluator: "worker" | "main";
  workersReady: number;
  speedScale: number; // v2.3: Curriculum-Tempo-Stufe
}

const BASE_SIGMA = 0.08;
const LR = 0.03;
const STAGNATION_GENS = 8;
/** v2.3: Curriculum-Schwellen (Sturzrate → Tempo-Skalierung). */
const CURRICULUM_UP = 0.08; // unter 8 % Stürze → Tempo hoch
const CURRICULUM_DOWN = 0.45; // über 45 % Stürze → Tempo runter

/** v2.3: Layer-Dimensionen eines Layouts ([obs, ...hidden, action]). */
export function layerDims(layout: MlpLayout): number[] {
  const hs = hiddenList(layout.hidden);
  return [layout.obsDim, ...hs, layout.actionDim];
}

// ── Rollout-Kontext (kooperativ schrittweise, für Interleaving) ──────────────

interface RolloutCtx {
  data: any; // eigenes MjData
  mlp: MlpPolicy;
  steps: number;
  n: number;
  t: number; // Imitations-Zeit (s)
  sum: number;
  fell: boolean;
  done: boolean;
  prevAct: Float32Array;
  act: Float32Array;
  actF: Float32Array; // v2.3: tiefpassgefilterte Aktion (Action-Lowpass)
  pushTick: number; // v2.3: Schritt-Zähler für Zufalls-Stöße
}

function resetCtx(
  engine: Engine, ctx: RolloutCtx, run: RunCfg,
  cmd: Float32Array | null,
) {
  engine.setData(ctx.data);
  engine.resetToKeyframe();
  engine.trackCmd = false;
  if (run.noiseReset) engine.addResetNoise();
  engine.cmd.fill(0);
  if (cmd) engine.cmd.set(cmd); // v2.3: Befehl des Paares (Domain-Randomization)
  engine.trackCmd = run.cmdTrain && !!cmd;
  ctx.n = 0;
  ctx.t = 0;
  ctx.sum = 0;
  ctx.fell = false;
  ctx.done = false;
  ctx.prevAct.fill(0);
  ctx.actF.fill(0);
  ctx.pushTick = 0;
}

/** Zusatz-Info für den Custom-Code-Term (Zeit/Schritt). */
export interface StepCtxInfo { t: number; step: number; dt: number }

/**
 * Reward-Terme für EINEN Policy-Step (Zustand NACH dem Physik-Step).
 * Geteilt zwischen Rollouts (ES) und Test-Modus (kumulierter Reward).
 */
export function stepRewardValue(
  engine: Engine, cfg: RewardConfig, act: Float32Array, prevAct: Float32Array,
  ctxInfo?: StepCtxInfo,
): number {
  const T = cfg.terms;
  const gz = engine.projGravZ();
  const upZ = Math.max(0, Math.min(1, -gz));
  const h = engine.height();
  const yaw = engine.yaw();
  const qvel = engine.data.qvel as Float32Array;
  const vx = Math.cos(yaw) * qvel[0] + Math.sin(yaw) * qvel[1];
  const vy = -Math.sin(yaw) * qvel[0] + Math.cos(yaw) * qvel[1];
  const omega = qvel[5];
  // v2.3: Befehls-Tracking (Profi-Stil MJX-Playground): Vorwärts/Seitwärts-Term
  // folgt dem aktiven Befehl (Training: zufällig gezogen, Test: Joystick) mit
  // exp(-Fehler²/σ²) statt fester Linear-Skala.
  const tracking = engine.trackCmd === true;
  const cmd0 = engine.cmd[0] ?? 0;
  const cmd1 = engine.cmd[1] ?? 0;
  let val = 0;
  if (T.upright?.enabled) val += T.upright.weight * upZ;
  if (T.height?.enabled) {
    val += T.height.weight * (1 - Math.abs(h - T.height.param) / Math.max(0.05, T.height.param));
  }
  if (T.forward?.enabled) {
    if (tracking) {
      const dvx = vx - cmd0, dvy = vy - cmd1;
      // v2.5: schärfere Kurve (σ² 0.25 → 0.09): Stehen bei cmd 0.26 gibt nur
      // noch ~47 % statt 76 % — Stehen lohnt sich nicht mehr, LAUFEN schon.
      val += T.forward.weight * Math.exp(-(dvx * dvx + dvy * dvy) / 0.09);
    } else {
      val += T.forward.weight * Math.max(0, 1 - Math.abs(vx - T.forward.param) / 0.5);
    }
  }
  if (T.lateral?.enabled) {
    if (tracking) {
      const dvy = vy - cmd1;
      val += T.lateral.weight * Math.exp(-(dvy * dvy) / 0.25);
    } else {
      val += T.lateral.weight * Math.max(-1, Math.min(1, vy));
    }
  }
  if (T.yaw?.enabled) val += T.yaw.weight * (1 - Math.min(1, Math.abs(omega - (engine.cmd[2] ?? 0))));
  if (T.alive?.enabled) val += T.alive.weight;
  if (T.energy?.enabled) {
    let e = 0;
    for (let j = 0; j < act.length; j++) e += act[j] * act[j];
    val -= T.energy.weight * (e / act.length);
  }
  if (T.smoothness?.enabled) {
    let d = 0;
    for (let j = 0; j < act.length; j++) {
      const diff = act[j] - prevAct[j];
      d += diff * diff;
    }
    val -= T.smoothness.weight * (d / act.length);
  }
  if (T.jointAccel?.enabled) {
    const qacc = engine.jointAccel();
    let s = 0;
    for (const a of qacc) s += Math.abs(a);
    val -= T.jointAccel.weight * (s / qacc.length / 1000);
  }
  if (T.contact?.enabled) {
    const qacc = engine.data.qacc as Float32Array;
    let s = 0;
    for (let j = 0; j < 6; j++) s += Math.abs(qacc[j]);
    val -= T.contact.weight * (s / 6 / 1000);
  }
  if (T.jointLimit?.enabled) {
    const angles = engine.jointAngles();
    const ranges = engine.jntRanges;
    let pen = 0, cnt = 0;
    for (let j = 0; j < angles.length; j++) {
      const [lo, hi] = ranges[j] ?? [0, 0];
      const lim = Math.max(Math.abs(lo), Math.abs(hi));
      if (lim < 1e-6) continue;
      cnt++;
      const over = Math.max(0, (Math.abs(angles[j]) - 0.9 * lim) / (0.1 * lim));
      pen += Math.min(1, over);
    }
    if (cnt > 0) val -= T.jointLimit.weight * (pen / cnt);
  }
  // ── v2.1: Imitation ──
  if (T.imitate?.enabled && engine.imitTarget) {
    const angles = engine.jointAngles();
    const tgt = engine.imitTarget;
    let err = 0;
    for (let j = 0; j < angles.length; j++) err += Math.abs(angles[j] - tgt[j]);
    err /= Math.max(1, angles.length);
    val += T.imitate.weight * Math.max(0, 1 - err / Math.max(0.05, T.imitate.param));
  }
  if (T.imitHeight?.enabled && engine.imitRootDelta !== null) {
    const zt = engine.meta.targetHeight + engine.imitRootDelta;
    val -= T.imitHeight.weight
      * Math.min(1, Math.abs(engine.height() - zt) / Math.max(0.05, T.imitHeight.param));
  }
  // ── v2.1: Punkt-Modus ──
  if ((T.pointChase?.enabled || T.pointAvoid?.enabled) && engine.targetPoint) {
    const [tx, ty] = engine.targetPoint;
    const p = engine.torsoPos();
    const d = Math.hypot(p[0] - tx, p[1] - ty);
    if (T.pointChase?.enabled) {
      val += T.pointChase.weight * Math.max(0, 1 - d / Math.max(0.05, T.pointChase.param));
    }
    if (T.pointAvoid?.enabled) {
      val += T.pointAvoid.weight * Math.min(1, d / Math.max(0.05, T.pointAvoid.param));
    }
  }
  // ── v2.2: KI-Code-Term (Gemini-geschriebener Funktionskörper) ──
  const C = cfg.custom;
  if (C?.enabled && C.code) {
    const fn: CustomRewardFn | null = getCustomFn(C.code);
    if (fn) {
      const qvel2 = engine.data.qvel as Float32Array;
      const meta2 = engine.meta;
      val += C.weight * evalCustomFn(fn, {
        h: h, height: h,
        upZ, gz,
        vx, vy, vz: qvel2[2] ?? 0, omega,
        angles: engine.jointAngles(),
        act, prevAct,
        qpos: engine.data.qpos as Float32Array,
        qvel: qvel2,
        qacc: engine.data.qacc as Float32Array,
        torso: engine.torsoPos(),
        target: engine.targetPoint ? [engine.targetPoint[0], engine.targetPoint[1]] : null,
        cmd: engine.cmd,
        imitDelta: engine.imitRootDelta,
        imitTarget: engine.imitTarget,
        dt: ctxInfo?.dt ?? (meta2 ? meta2.timestep * meta2.decimation : 0.02),
        t: ctxInfo?.t ?? 0,
        step: ctxInfo?.step ?? 0,
      });
    }
  }
  return val;
}

/** Genau EINEN Policy-Step des Rollouts (Rundeneinteilung für Interleaving). */
function ctxStep(
  engine: Engine, ctx: RolloutCtx, cfg: RewardConfig,
  imit: ImitEvalData | null, imitBuf: Float32Array | null, policyDt: number,
  run: RunCfg,
): void {
  const T = cfg.terms;
  engine.setData(ctx.data);
  if (imit && imitBuf) {
    const delta = imitSampleAt(imit, ctx.t, imitBuf);
    // v2.3: Imitations-Ziele auf Aktuator-Range klemmen (Human-Clips haben
    // größere Amplituden als die Gelenke erlauben)
    const lims = engine.ctrlLimits;
    if (lims) {
      for (let j = 0; j < imitBuf.length; j++) {
        const lo = lims[j * 2], hi = lims[j * 2 + 1];
        if (hi > lo) imitBuf[j] = Math.min(hi, Math.max(lo, imitBuf[j]));
      }
    }
    engine.imitTarget = imitBuf;
    engine.imitRootDelta = delta;
    const ph = (imit.duration > 0 ? (ctx.t % imit.duration) / imit.duration : 0) * 2 * Math.PI;
    engine.imitPhase = [Math.sin(ph), Math.cos(ph)];
  }
  const obs = engine.obsFor(!!imit);
  ctx.mlp.forward(obs, ctx.act);
  // v2.3: Action-Lowpass (Profi-Trick gegen Zittern): gefilterte Aktion
  // steuert Physik UND Obs; Smoothness vergleicht die gefilterten Werte.
  const a = run.actionSmooth;
  if (a < 0.999) {
    for (let j = 0; j < ctx.act.length; j++) {
      ctx.actF[j] = a * ctx.act[j] + (1 - a) * ctx.actF[j];
    }
  } else {
    ctx.actF.set(ctx.act);
  }
  engine.lastAction.set(ctx.actF);
  engine.stepWithAction(ctx.actF);
  ctx.n++;
  ctx.t += policyDt;

  // v2.3: Zufalls-Stöße (Domain-Randomization, Profi-Trick für Robustheit)
  if (run.pushes && ++ctx.pushTick >= 55) {
    ctx.pushTick = 0;
    engine.applyPush();
  }

  // v2.6: ERST Sturz prüfen, DANN Reward — und NIE NaN in sum_addieren.
  // Genau das verursachte "REWARD NaN": divergierte Physik (qpos=NaN) lieferte
  // NaN-Rewards, vergiftete sum und damit die komplette Generation.
  if (engine.isFallen()) {
    ctx.fell = true;
    ctx.done = true;
    if (T.fall?.enabled) {
      // v2.3: Summen-Modus → FLACHE Strafe (Ranking steigt stetig mit der
      // Überlebensdauer); Mittel-Modus → ×n (pro-step-Beitrag wie bisher).
      ctx.sum -= run.fitnessMode === "sum"
        ? T.fall.weight
        : T.fall.weight * ctx.n;
    }
    return;
  }
  const val = stepRewardValue(
    engine, cfg, ctx.actF, ctx.prevAct, { t: ctx.t, step: ctx.n, dt: policyDt },
  );
  if (!Number.isFinite(val)) {
    // Physik divergiert (NaN/Inf) → Runde sofort als gescheitert werten
    ctx.fell = true;
    ctx.done = true;
    ctx.sum -= T.fall?.enabled ? T.fall.weight : 5;
    return;
  }
  ctx.sum += val;
  ctx.prevAct.set(ctx.actF);

  if (ctx.n >= ctx.steps) {
    ctx.done = true;
  }
}

function finishCtx(ctx: RolloutCtx, mode: "sum" | "mean"): number {
  // v2.3: Summen-Modus (Profi-Trick): länger überleben = mehr Reward-Akkumulation
  // → natürlicher Überlebensdruck; Mittelwert macht Sturzzeitpunkt egal.
  // v2.6: NaN-Schutz — vergiftete Runden zählen als Totalausfall.
  const f = mode === "sum" ? ctx.sum : (ctx.n > 0 ? ctx.sum / ctx.n : 0);
  return Number.isFinite(f) ? f : -100;
}

// ── EsTrainer ────────────────────────────────────────────────────────────────

export class EsTrainer {
  meta: ModelMeta;
  modelId: ModelId;
  engine: Engine; // eigener headless-Kontext (kein Rendering)

  theta: Float32Array;
  layout: MlpLayout;
  sigma = BASE_SIGMA;
  baseSigma = BASE_SIGMA;
  lr = LR;
  rolloutSteps = 200;
  /** v2.2: Training stoppt automatisch nach N Generationen (0 = unbegrenzt). */
  maxGenerations = 0;
  turbo: TurboLevel = 1;
  reward: RewardConfig;

  generation = 0;
  bestEver = -Infinity;
  bestTheta: Float32Array;
  history: number[] = [];
  lastGenBest = 0;
  lastGenMean = 0;
  lastFellRate = 0;
  stepsPerSec = 0;
  sinceImprovement = 0;
  evaluator: "worker" | "main" = "main";
  workersReady = 0;
  // v2.1: Welt + Imitation für Rollouts
  obsExtra = 0;
  world: WorldBuild | null = null;
  imit: ImitEvalData | null = null;
  imitSig = "";
  obsExtraActive = false; // Phase aktuell in der Obs?
  // v2.3: Profi-Trainings-Konfiguration + Curriculum-Phase
  runCfg: RunCfg = defaultRunCfg();
  speedScale = 0.5; // Curriculum: Befehls-Tempo-Stufe (0.15..1)

  private workers: Worker[] = [];
  private pending = new Map<number, (r: { fitness: number; fell: boolean; steps: number }) => void>();
  private jobId = 0;
  private workerBooted = false;
  private stepsCounter = 0;
  // MjData-Pool: Rollout-Instanzen werden wiederverwendet (kein Leak je Generation)
  private dataPool: any[] = [];
  private imitBuf: Float32Array | null = null;
  private policyDt = 0.02;

  constructor(
    modelId: ModelId, theta: Float32Array, layout: MlpLayout, reward: RewardConfig,
    opts?: { obsExtra?: number; world?: WorldBuild | null; imit?: ImitEvalData | null; imitSig?: string },
  ) {
    this.modelId = modelId;
    this.meta = getModel(modelId);
    this.theta = theta.slice();
    this.layout = layout;
    this.reward = reward;
    this.bestTheta = theta.slice();
    this.obsExtra = opts?.obsExtra ?? 0;
    this.world = opts?.world ?? null;
    this.imit = opts?.imit ?? null;
    this.imitSig = opts?.imitSig ?? "";
    this.engine = new Engine();
  }

  async init(): Promise<void> {
    this.policyDt = this.meta.timestep * this.meta.decimation;
    await this.engine.load(this.modelId, this.world ?? undefined);
    this.imitBuf = this.imit ? new Float32Array(this.imit.dim) : null;
    this.engine.setImitObs(this.obsExtra > 0);
    this.obsExtraActive = this.obsExtra > 0;
  }

  /** Versucht, Eval-Worker zu starten (Module-Worker mit import('/wasm/mujoco.js')). */
  async tryStartWorkers(): Promise<boolean> {
    if (this.workerBooted) return this.workersReady > 0;
    try {
      const count = Math.max(1, Math.min(3, Math.floor((navigator.hardwareConcurrency || 4) / 2)));
      const payload = await this.buildWorkerPayload();
      const ready = await Promise.all(
        Array.from({ length: count }, (_, i) => this.bootWorker(i, payload).catch(() => false)),
      );
      this.workersReady = ready.filter(Boolean).length;
    } catch (err) {
      console.warn("[es] Worker-Start fehlgeschlagen:", err);
      this.workersReady = 0;
    }
    this.workerBooted = true;
    this.evaluator = this.workersReady > 0 ? "worker" : "main";
    return this.workersReady > 0;
  }

  /** Dieselbe XML-Aufbereitung wie der Engine-Boot (ausgelagert nach xml.ts). */
  private async buildWorkerPayload(): Promise<WorkerInitPayload> {
    const src = await (await fetch(this.meta.mjcf)).text();
    const { buildPhysicsXmlForWorker } = await import("./xml");
    const base = buildPhysicsXmlForWorker(this.modelId, src, this.meta, this.world ?? null);
    return {
      ...base,
      cmdSize: this.meta.cmdSize,
      obsType: this.meta.obsType,
      decimation: this.meta.decimation,
      actionScale: this.meta.actionScale,
      defaultPose: this.meta.defaultPose,
      jointNames: this.meta.jointNames,
      torsoBody: this.meta.torsoBody,
      gyroSensor: this.meta.gyroSensor,
      keyframe: this.meta.keyframe,
      obsExtra: this.obsExtra,
      targetHeight: this.meta.targetHeight,
      imitation: this.imit
        ? {
            dim: this.imit.dim, frames: this.imit.frames, fps: this.imit.fps,
            duration: this.imit.duration, targets: this.imit.targets.buffer,
            rootY: this.imit.rootY.buffer, baseY: this.imit.baseY,
            // v2.3: Cross-Species-Retargeting (Center-Pose + Höhen-Skalierung)
            center: this.imit.center ? this.imit.center.buffer : null,
            scaleY: this.imit.scaleY ?? 1,
          }
        : null,
    };
  }

  private async bootWorker(
    idx: number,
    payload: WorkerInitPayload,
  ): Promise<boolean> {
    const w = new Worker("/workers/es-worker.js", { type: "module" });
    this.workers.push(w);
    const ready = await new Promise<boolean>((resolve) => {
      const to = setTimeout(() => resolve(false), 30000);
      w.onmessage = (e: MessageEvent) => {
        const msg = e.data as { type: string; message?: string };
        if (msg.type === "ready") {
          clearTimeout(to);
          resolve(true);
        } else if (msg.type === "bootError") {
          console.warn("[es] Worker-Boot fehlgeschlagen:", msg.message);
          clearTimeout(to);
          resolve(false);
        } else if (msg.type === "result") {
          this.onWorkerResult(e.data as WorkerResultMsg);
        }
      };
      w.onerror = () => {
        clearTimeout(to);
        resolve(false);
      };
      w.postMessage({ type: "init", workerId: idx, ...payload });
    });
    return ready;
  }

  private onWorkerResult(msg: WorkerResultMsg) {
    const resolve = this.pending.get(msg.jobId);
    if (!resolve) return;
    this.pending.delete(msg.jobId);
    this.stepsCounter += msg.steps;
    resolve({ fitness: msg.fitness, fell: msg.fell, steps: msg.steps });
  }

  private evalViaWorker(
    theta: Float32Array, cmd: Float32Array | null,
  ): Promise<{ fitness: number; fell: boolean; steps: number }> {
    const jobId = ++this.jobId;
    const w = this.workers[jobId % this.workers.length];
    const p = new Promise<{ fitness: number; fell: boolean; steps: number }>((resolve) => {
      this.pending.set(jobId, resolve);
    });
    const copy = theta.slice();
    const cmdCopy = cmd ? cmd.slice() : null;
    w.postMessage(
      {
        type: "eval", jobId,
        theta: copy.buffer, layout: this.layout, reward: this.reward,
        rolloutSteps: this.rolloutSteps,
        targetPoint: this.pointSnapshot ? [...this.pointSnapshot] : null,
        // v2.3: Profi-Rollout-Konfiguration + Befehl des Paares
        run: { ...this.runCfg },
        cmd: cmdCopy ? Array.from(cmdCopy) : null,
      },
      [copy.buffer],
    );
    return p;
  }

  /** Aktueller Punkt-Snapshot für Worker-Rewards (Punkt-Modus). */
  pointSnapshot: [number, number] | null = null;
  setTargetPoint(p: [number, number] | null): void {
    this.pointSnapshot = p;
  }

  /** Evaluierung der Population: Worker-Pool oder Main-Thread-Interleaving. */
  private async evalBatch(
    members: Float32Array[], cmds: (Float32Array | null)[],
  ): Promise<{ fitness: number; fell: boolean }[]> {
    const results: { fitness: number; fell: boolean }[] = new Array(members.length);
    this.stepsCounter = 0;
    if (this.evaluator === "worker" && this.workers.length > 0) {
      await Promise.all(
        members.map(async (m, i) => {
          const r = await this.evalViaWorker(m, cmds[i]);
          // v2.6: NaN-Schutz — Worker-Ergebnisse sind immer endlich.
          results[i] = {
            ...r,
            fitness: Number.isFinite(r.fitness) ? r.fitness : -100,
          };
        }),
      );
    } else {
      // Main-Thread-Interleaving: bis zu 6 gepoolte MjData-Instanzen rundenweise
      // je 1 Policy-Step, nach jedem Rundendurchlauf Yield an den Event-Loop.
      const slots = Math.min(6, members.length);
      for (let batch = 0; batch < members.length; batch += slots) {
        const batchMembers = members.slice(batch, batch + slots);
        const ctxs: RolloutCtx[] = batchMembers.map((m, k) => {
          const ctx: RolloutCtx = {
            data: this.dataPool.pop() ?? this.engine.newData(),
            mlp: new MlpPolicy(this.layout, m),
            steps: this.rolloutSteps,
            n: 0, t: 0, sum: 0, fell: false, done: false,
            prevAct: new Float32Array(this.layout.actionDim),
            act: new Float32Array(this.layout.actionDim),
            actF: new Float32Array(this.layout.actionDim),
            pushTick: 0,
          };
          resetCtx(this.engine, ctx, this.runCfg, cmds[batch + k] ?? null);
          return ctx;
        });
        while (ctxs.some((c) => !c.done)) {
          for (const ctx of ctxs) {
            if (!ctx.done) {
              ctxStep(this.engine, ctx, this.reward, this.imit, this.imitBuf, this.policyDt, this.runCfg);
            }
          }
          await new Promise<void>((r) => setTimeout(r, 0)); // UI-Yield
        }
        for (let i = 0; i < ctxs.length; i++) {
          results[batch + i] = { fitness: finishCtx(ctxs[i], this.runCfg.fitnessMode), fell: ctxs[i].fell };
          this.stepsCounter += ctxs[i].n;
          this.dataPool.push(ctxs[i].data); // zurück in den Pool statt GC/Leak
        }
      }
    }
    return results;
  }

  setTurbo(turbo: TurboLevel): void {
    this.turbo = turbo;
  }

  setReward(cfg: RewardConfig): void {
    this.reward = cfg;
  }

  /** v2.3: Profi-Trainings-Konfiguration übernehmen. */
  setRunCfg(cfg: RunCfg): void {
    this.runCfg = cfg;
    if (!cfg.curriculum) this.speedScale = 1;
  }

  /** v2.3: Befehl für ein antithetisches Paar ziehen (beide Mitglieder gleich,
   *  sonst vergleicht der Gradient Äpfel mit Birnen).
   *  v2.5: Anfahr-Boden gegen die STEHEN-FALLE: Der Tracking-Reward gab
   *  Stehen bei Mini-cmd fast die volle Punktzahl (cmd 0.09, vx 0 → 97 %).
   *  Vorwärts-Befehle liegen deshalb nie unter meta.cmdFloor (Ente 0.26:
   *  darunter fährt die Warm-Start-Policy real nicht an). Curriculum wirkt
   *  jetzt auf dem Bereich ÜBER dem Boden (Tempo-Vielfalt), nicht darunter. */
  private sampleCmd(): Float32Array | null {
    if (!this.runCfg.cmdTrain) return null;
    const s = this.speedScale;
    const cmd = new Float32Array(this.meta.cmdSize);
    // ~75 % vorwärts, sonst leicht rückwärts/stand – repräsentative Mischung
    const draw = Math.random();
    const hi = this.runCfg.cmdFwd * s;
    if (draw < 0.75) {
      // vorwärts: [0,1] → [max(cmdFloor, hi), hi], immer ≥ Anfahr-Boden
      const lo = Math.min(this.meta.cmdFloor, this.runCfg.cmdFwd);
      cmd[0] = Math.max(lo, hi);
    } else {
      cmd[0] = hi * (Math.random() * 0.35 - 0.35);
    }
    if (cmd.length > 1) cmd[1] = this.runCfg.cmdLat * (Math.random() * 2 - 1) * 0.6 * s;
    if (cmd.length > 2) cmd[2] = this.runCfg.cmdAng * (Math.random() * 2 - 1) * 0.7;
    return cmd;
  }

  /** v2.3: Curriculum-Schritt nach jeder Generation (Tempo-Treppe).
   *  v2.4: Hartnäckiger ausgelegt – mit der alten Schwelle (Erhöhung erst ab
   *  Sturzrate < 0.05, Boden 0.15) klebte das Tempo am Minimum, sobald Stöße/
   *  Reset-Rauschen die Sturzrate bei ~0.3 hielten: Die Policy lernte Stehen
   *  mit Mini-Schritten statt sichtbares Laufen. Jetzt: Boden 0.35, Aufstieg
   *  ab Sturzrate < 0.12, Abbau erst ab > 0.45. */
  private curriculumStep(): void {
    if (!this.runCfg.curriculum) return;
    if (this.lastFellRate > 0.45) {
      this.speedScale = Math.max(0.35, this.speedScale * 0.85);
    } else if (this.lastFellRate < 0.12) {
      this.speedScale = Math.min(1, this.speedScale * 1.1 + 0.03);
    }
  }

  /** Eine ES-Generation: antithetisches Sampling → Evaluierung → Update. */
  async runGeneration(): Promise<void> {
    const pop = popSizeFor(this.turbo);
    const pairs = Math.ceil(pop / 2);
    const members: Float32Array[] = [];
    const epsilons: Float32Array[] = [];
    const cmds: (Float32Array | null)[] = [];
    // v2.3: FAN-IN-SKALIERUNG (Profi-Trick für große Netze): Störung je Gewicht
    // wird durch √(fan-in) der Schicht geteilt, sonst ist die Verhaltens-Störung
    // bei 197k-Gewicht-Netzen (512→256→128) so groß, dass JEDES Mitglied
    // sofort stürzt (beobachtet: Sturzrate 100 % bei Schritt ~5).
    const dims = layerDims(this.layout);
    for (let i = 0; i < pairs; i++) {
      const eps = new Float32Array(this.theta.length);
      let off = 0;
      for (let l = 0; l < dims.length - 1; l++) {
        const inDim = dims[l], outDim = dims[l + 1];
        const wScale = this.sigma / Math.sqrt(inDim);
        const nW = outDim * inDim;
        for (let j = 0; j < nW; j++) eps[off + j] = gauss() * wScale;
        off += nW;
        for (let j = 0; j < outDim; j++) eps[off + j] = gauss() * this.sigma;
        off += outDim;
      }
      epsilons.push(eps);
      members.push(Float32Array.from(this.theta.map((v, j) => v + eps[j])));
      members.push(Float32Array.from(this.theta.map((v, j) => v - eps[j])));
      // v2.3: EIN Befehl pro Paar (fairer Gradientenvergleich)
      const c = this.sampleCmd();
      cmds.push(c, c);
    }
    const t0 = performance.now();
    // v2.5: WARM-START-ANKER: Vor der ersten Generation den Ausgangs-Theta
    // (Warm-Start-Läufer) selbst evaluieren und als bestEver/bestTheta
    // verankern. Sonst setzte das erste epsilon-Mitglied bestEver/bestTheta
    // (bestEver startete bei -Infinity!) und die Live-Vorschau zeigte ein
    // verrauschtes Mediokritäts-Glied statt des lauffähigen Warm-Starts —
    // der sichtbare „fällt auf die Fresse"-Effekt. bestTheta ist jetzt nie
    // schlechter als der Start-Punkt.
    if (this.bestEver === -Infinity) {
      const [warmRes] = await this.evalBatch([this.theta.slice()], [this.sampleCmd()]);
      this.bestEver = warmRes.fitness;
      this.bestTheta = this.theta.slice();
    }
    const results = await this.evalBatch(members.slice(0, pop), cmds.slice(0, pop));
    const dt = Math.max(1, performance.now() - t0);
    this.stepsPerSec = (this.stepsCounter / dt) * 1000;

    // Rank-Normalisierung (zentriert auf [-0.5, 0.5], Fitness-Shaping)
    const n = results.length;
    const order = results.map((_, i) => i).sort((a, b) => results[a].fitness - results[b].fitness);
    const shaped = new Float32Array(n);
    order.forEach((memberIdx, rank) => {
      shaped[memberIdx] = rank / (n - 1) - 0.5;
    });

    // Gradienten-Update (antithetisch): theta += lr/sigma * Σ_i (s⁺_i − s⁻_i) · eps_i
    // members[2i] = theta+eps_i, members[2i+1] = theta−eps_i → epsilons hat
    // nur `pairs` Einträge (v2.0-Bug: Loop lief über n=pop → undefined-Crash).
    const grad = new Float32Array(this.theta.length);
    for (let i = 0; i < pairs; i++) {
      const eps = epsilons[i];
      const sPlus = shaped[2 * i] ?? 0;
      const sMinus = shaped[2 * i + 1] ?? 0;
      const s = sPlus - sMinus;
      if (s === 0) continue;
      for (let j = 0; j < grad.length; j++) grad[j] += s * eps[j];
    }
    const norm = this.lr / this.sigma;
    const wd = this.runCfg.weightDecay;
    for (let j = 0; j < this.theta.length; j++) {
      // v2.3: Gewichtsbremse (Decay gegen saturierte tanh → weniger Zittern)
      this.theta[j] = this.theta[j] * (1 - this.lr * wd) + grad[j] * norm;
    }

    // v2.3: Weight-Decay – hält Theta klein (tanh bleibt sensitiv, kein Drift)
    if (this.runCfg.weightDecay > 0) {
      const wd = this.runCfg.weightDecay;
      for (let j = 0; j < this.theta.length; j++) this.theta[j] *= 1 - wd;
    }

    const bestIdx = order[n - 1];
    const bestFit = results[bestIdx].fitness;
    let meanFit = 0;
    for (const r of results) meanFit += Number.isFinite(r.fitness) ? r.fitness : 0;
    meanFit /= n;
    this.generation++;
    this.lastGenBest = bestFit;
    this.lastGenMean = meanFit;
    this.lastFellRate = results.filter((r) => r.fell).length / n;
    this.history.push(bestFit);
    if (this.history.length > 200) this.history.shift();

    if (bestFit > this.bestEver) {
      this.bestEver = bestFit;
      this.bestTheta = members[bestIdx].slice();
      this.sinceImprovement = 0;
      this.sigma += (this.baseSigma - this.sigma) * 0.2; // Richtung Basis zurück
    } else {
      this.sinceImprovement++;
      if (this.sinceImprovement >= STAGNATION_GENS && this.sigma > 0.01) {
        this.sigma = Math.max(0.005, this.sigma * 0.85); // adaptiv schrumpfen
      }
    }
    // v2.3: Anker-Restart (Profi-Trick): driftet theta während der Exploration
    // in eine schlechte Region (hohe Sturzrate trotz geschrumpftem Sigma),
    // kehrt es zum besten je gefundenen Punkt zurück und versucht neu.
    if (this.sinceImprovement >= STAGNATION_GENS * 3 && this.lastFellRate > 0.5) {
      this.theta = this.bestTheta.slice();
      this.sinceImprovement = 0;
      this.sigma = Math.min(this.baseSigma, this.sigma * 1.5 + 0.005);
    }
    this.curriculumStep(); // v2.3: Tempo-Treppe nach jeder Generation
  }

  stats(): EsStats {
    return {
      generation: this.generation,
      bestFitness: this.lastGenBest,
      bestEver: this.bestEver,
      sigma: this.sigma,
      stepsPerSec: this.stepsPerSec,
      history: this.history,
      meanFitness: this.lastGenMean,
      fellRate: this.lastFellRate,
      evaluator: this.evaluator,
      workersReady: this.workersReady,
      speedScale: this.speedScale,
    };
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    for (const d of this.dataPool) {
      try { d?.delete?.(); } catch { /* Emscripten-Handle */ }
    }
    this.dataPool = [];
    this.engine.dispose();
  }
}

export interface WorkerInitPayload {
  xml: string; meshBase: string; vfsPrefix: string; meshFiles: string[];
  modelId: ModelId; obsDim: number; actionDim: number;
  cmdSize: number; obsType: string; decimation: number; actionScale: number;
  defaultPose: number[]; jointNames: string[];
  torsoBody: string; gyroSensor: string | null; keyframe: string;
  // v2.1
  obsExtra: number;
  targetHeight: number;
  imitation: {
    dim: number; frames: number; fps: number; duration: number;
    targets: ArrayBufferLike; rootY: ArrayBufferLike; baseY: number;
    center: ArrayBufferLike | null; // v2.3: Zentrum-Pose (absolut)
    scaleY: number; // v2.3: Wurzel-Höhen-Skalierung
  } | null;
}

interface WorkerResultMsg {
  type: "result";
  jobId: number;
  fitness: number;
  fell: boolean;
  steps: number;
}

// Box–Muller für normalverteiltes Rauschen
function gauss(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
