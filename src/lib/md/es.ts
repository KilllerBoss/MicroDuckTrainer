// ── MicroDuck Trainer v2.0 – Evolution Strategy (OpenAI-ES) ──────────────────
// Antithetic Sampling, Rank-Normalisierung, adaptives Sigma. Evaluierung läuft
// bevorzugt in Web-Workern (headless, reine Physik) – Fallback: Main-Thread-
// Interleaving (mehrere MjData-Instanzen rundenweise, N = Turbo-Stufe).
// Die laufende Entscheidung steht im Worklog (public/workers/es-worker.js).

import type { ModelMeta, ModelId } from "./models";
import { getModel } from "./models";
import { Engine } from "./engine";
import { MlpPolicy, type MlpLayout } from "./policy";
import type { RewardConfig } from "./rewards";
import type { WorldBuild } from "./worldgen";
import { getCustomFn, evalCustomFn, type CustomRewardFn } from "./customcode";

/** Imitations-Daten für Rollouts (Main-Thread + Worker, identisch). */
export interface ImitEvalData {
  dim: number;
  frames: number;
  fps: number;
  duration: number;
  targets: Float32Array; // frames × dim – v2.3: RELATIV zur Ruhelage
  rootY: Float32Array; // frames (GLB y-up)
  baseY: number;
  /** v2.3: Ruhelage des Roboters (Gelenkwinkel) – Ziele absolut = rel + center. */
  center?: Float32Array | null;
  /** v2.3: Roboterhöhe / Clip-Wurzelhöhe (Cross-Species-Skalierung). */
  scaleY?: number;
}

/** Zielpose (ABSOLUT) + Wurzel-Delta zum Zeitpunkt t (loop). Schreibt in out. */
export function imitSampleAt(d: ImitEvalData, t: number, out: Float32Array): number {
  const tt = d.duration > 0 ? ((t % d.duration) + d.duration) % d.duration : 0;
  const x = tt * d.fps;
  const f0 = Math.min(d.frames - 1, Math.floor(x));
  const f1 = Math.min(d.frames - 1, f0 + 1);
  const u = x - f0;
  const center = d.center;
  for (let j = 0; j < d.dim; j++) {
    const a = d.targets[f0 * d.dim + j];
    const b = d.targets[f1 * d.dim + j];
    let v = a + (b - a) * u;
    if (center && j < center.length) v += center[j];
    // v2.3 NaN-Schutz: kaputte Clip-Werte → Ruhelage
    out[j] = Number.isFinite(v) ? v : (center ? center[j] : 0);
  }
  const raw = (d.rootY[f0] + (d.rootY[f1] - d.rootY[f0]) * u) - d.baseY;
  const s = d.scaleY ?? 1;
  return Number.isFinite(raw) ? raw * s : 0;
}

// ── v2.3: Profi-Tricks (Lauftraining nach Stand der Technik) ───────────────────

/** Konfiguration der Profi-Tricks (TrainingPanel + Gemini). */
export interface RunCfg {
  /** Zufällige Geschwindigkeitsbefehle pro Runde (Joystick-Fähigkeit lernen). */
  cmdTrain: boolean;
  cmdFwd: number; // max. Vorwärtstempo (m/s)
  cmdLat: number; // max. Seitwärtstempo (m/s)
  cmdAng: number; // max. Gierrate (rad/s)
  /** Curriculum: Tempo bei vielen Stürzen automatisch reduzieren. */
  curriculum: boolean;
  /** Aktions-Glättung (EMA): 0.3 = sehr glatt, 1 = aus. Gegen Zittern. */
  actionSmooth: number;
  /** Zufalls-Stöße auf den Körper (Domain Randomization → robustes Gehen). */
  pushes: boolean;
  /** Reset-Rauschen (jeder Start leicht anders → keine Memory-Löcher). */
  noiseReset: boolean;
  /** Fitness = Summe (Überleben zählt) statt Mittelwert. */
  fitnessMode: "sum" | "mean";
  /** Weight-Decay gegen Theta-Drift/tanh-Sättigung (0 = aus). */
  weightDecay: number;
}

export function defaultRunCfg(): RunCfg {
  return {
    cmdTrain: true,
    cmdFwd: 0.3, cmdLat: 0.15, cmdAng: 0.8,
    curriculum: true,
    actionSmooth: 0.6,
    pushes: true,
    noiseReset: true,
    fitnessMode: "sum",
    weightDecay: 0.005,
  };
}

function sanitizeRunCfg(p: unknown): RunCfg {
  const d = defaultRunCfg();
  if (!p || typeof p !== "object") return d;
  const o = p as Partial<RunCfg>;
  const num = (v: unknown, def: number, lo: number, hi: number) =>
    typeof v === "number" && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : def;
  return {
    cmdTrain: typeof o.cmdTrain === "boolean" ? o.cmdTrain : d.cmdTrain,
    cmdFwd: num(o.cmdFwd, d.cmdFwd, 0, 1.2),
    cmdLat: num(o.cmdLat, d.cmdLat, 0, 0.6),
    cmdAng: num(o.cmdAng, d.cmdAng, 0, 2),
    curriculum: typeof o.curriculum === "boolean" ? o.curriculum : d.curriculum,
    actionSmooth: num(o.actionSmooth, d.actionSmooth, 0.2, 1),
    pushes: typeof o.pushes === "boolean" ? o.pushes : d.pushes,
    noiseReset: typeof o.noiseReset === "boolean" ? o.noiseReset : d.noiseReset,
    fitnessMode: o.fitnessMode === "mean" ? "mean" : "sum",
    weightDecay: num(o.weightDecay, d.weightDecay, 0, 0.05),
  };
}

/** Zufalls-Bewegungsbefehl für eine Runde (mit Curriculum-Skalierung). */
function drawCmd(cfg: RunCfg, cmdScale: number, out: Float32Array): void {
  if (cfg.cmdTrain) {
    out[0] = Math.random() * cfg.cmdFwd * cmdScale;
    if (out.length > 1) out[1] = (Math.random() * 2 - 1) * cfg.cmdLat * cmdScale;
    if (out.length > 2) out[2] = (Math.random() * 2 - 1) * cfg.cmdAng * cmdScale;
  } else {
    out.fill(0);
    if (out.length > 0) out[0] = 0.25 * cmdScale;
  }
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
}

const BASE_SIGMA = 0.08;
const LR = 0.03;
const STAGNATION_GENS = 8;
/** v2.3: Curriculum-Schwellen (Sturzrate → Tempo-Skalierung). */
const CURRICULUM_UP = 0.08; // unter 8 % Stürze → Tempo hoch
const CURRICULUM_DOWN = 0.45; // über 45 % Stürze → Tempo runter

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
  cmd: Float32Array; // v2.3: Bewegungsbefehl dieser Runde
  pushTimer: number; // v2.3: Steps bis zum nächsten Zufalls-Stoß
  poisoned: boolean; // v2.3: Physik divergiert (NaN) → Runde abbrechen
}

function resetCtx(engine: Engine, ctx: RolloutCtx, cfg: RunCfg, cmdScale: number) {
  engine.setData(ctx.data);
  engine.resetToKeyframe();
  // v2.3: Reset-Rauschen – jeder Start leicht anders (Reference-State-Init)
  if (cfg.noiseReset) {
    const qpos = engine.data.qpos as Float32Array;
    const qvel = engine.data.qvel as Float32Array;
    for (let j = 0; j < engine.meta.actionDim; j++) {
      qpos[engine.qposAdrOf(j)] += (Math.random() * 2 - 1) * 0.04;
    }
    for (let j = 0; j < Math.min(6, qvel.length); j++) {
      qvel[j] += (Math.random() * 2 - 1) * 0.08;
    }
    engine.mujoco.mj_forward(engine.model, engine.data);
  }
  drawCmd(cfg, cmdScale, ctx.cmd);
  engine.cmd.set(ctx.cmd);
  ctx.n = 0;
  ctx.t = 0;
  ctx.sum = 0;
  ctx.fell = false;
  ctx.done = false;
  ctx.poisoned = false;
  ctx.pushTimer = 60 + Math.floor(Math.random() * 60);
  ctx.prevAct.fill(0);
  ctx.act.fill(0);
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
  let val = 0;
  if (T.upright?.enabled) val += T.upright.weight * upZ;
  if (T.height?.enabled) {
    val += T.height.weight * (1 - Math.abs(h - T.height.param) / Math.max(0.05, T.height.param));
  }
  if (T.forward?.enabled) {
    val += T.forward.weight * Math.max(0, 1 - Math.abs(vx - T.forward.param) / 0.5);
  }
  if (T.lateral?.enabled) val += T.lateral.weight * Math.max(-1, Math.min(1, vy));
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
  engine: Engine, ctx: RolloutCtx, cfg: RewardConfig, runCfg: RunCfg,
  imit: ImitEvalData | null, imitBuf: Float32Array | null, policyDt: number,
): void {
  const T = cfg.terms;
  engine.setData(ctx.data);
  if (imit && imitBuf) {
    const delta = imitSampleAt(imit, ctx.t, imitBuf);
    engine.imitTarget = imitBuf;
    engine.imitRootDelta = delta;
    const ph = (imit.duration > 0 ? (ctx.t % imit.duration) / imit.duration : 0) * 2 * Math.PI;
    engine.imitPhase = [Math.sin(ph), Math.cos(ph)];
  }
  const obs = engine.obsFor(!!imit);
  ctx.mlp.forward(obs, ctx.act);
  // v2.3: Aktions-EMA-Glättung – killt Zittern (Profi-Trick aus RL-Laufwerken)
  const a = runCfg.actionSmooth;
  if (a < 1) {
    for (let j = 0; j < ctx.act.length; j++) {
      ctx.act[j] = a * ctx.act[j] + (1 - a) * ctx.prevAct[j];
    }
  }
  engine.lastAction.set(ctx.act);
  engine.stepWithAction(ctx.act);
  ctx.n++;
  ctx.t += policyDt;

  // v2.3: Zufalls-Stöße (Domain Randomization) nach kurzer Schonfrist
  if (runCfg.pushes && ctx.n > 40 && !ctx.poisoned) {
    if (--ctx.pushTimer <= 0) {
      ctx.pushTimer = 80 + Math.floor(Math.random() * 120);
      const qvel = engine.data.qvel as Float32Array;
      const s = 0.35;
      if (qvel.length >= 6) {
        qvel[0] += (Math.random() * 2 - 1) * s;
        qvel[1] += (Math.random() * 2 - 1) * s;
        qvel[5] += (Math.random() * 2 - 1) * s * 0.8;
      }
    }
  }

  // v2.3: ERST Sturz/NaN prüfen, DANN Reward (NaN darf sum nie vergiften)
  if (engine.isFallen()) {
    ctx.fell = true;
    ctx.done = true;
    if (T.fall?.enabled) ctx.sum -= T.fall.weight; // einmalige Strafe
    return;
  }
  const val = stepRewardValue(
    engine, cfg, ctx.act, ctx.prevAct, { t: ctx.t, step: ctx.n, dt: policyDt },
  );
  if (!Number.isFinite(val)) {
    // Physik divergiert → Runde als gescheitert werten (kein NaN in sum!)
    ctx.poisoned = true;
    ctx.fell = true;
    ctx.done = true;
    ctx.sum -= T.fall?.enabled ? T.fall.weight : 5;
    return;
  }
  ctx.sum += val;
  ctx.prevAct.set(ctx.act);

  if (ctx.n >= ctx.steps) {
    ctx.done = true;
  }
}

function finishCtx(ctx: RolloutCtx, runCfg: RunCfg): number {
  if (ctx.n <= 0) return 0;
  let f: number;
  if (runCfg.fitnessMode === "sum") {
    // Sum-Modus, über Rundenlängen vergleichbar: Reward-Dichte + Überlebensbonus
    f = (ctx.sum / Math.max(1, ctx.steps)) * 100 + (ctx.n / Math.max(1, ctx.steps)) * 20;
  } else {
    f = ctx.sum / ctx.n;
  }
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
  // v2.3: Profi-Tricks
  runCfg: RunCfg = defaultRunCfg();
  cmdScale = 0.4; // Curriculum: startet konservativ, wächst bei wenig Stürzen
  // v2.1: Welt + Imitation für Rollouts
  obsExtra = 0;
  world: WorldBuild | null = null;
  imit: ImitEvalData | null = null;
  imitSig = "";
  obsExtraActive = false; // Phase aktuell in der Obs?

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

  private evalViaWorker(theta: Float32Array): Promise<{ fitness: number; fell: boolean; steps: number }> {
    const jobId = ++this.jobId;
    const w = this.workers[jobId % this.workers.length];
    const p = new Promise<{ fitness: number; fell: boolean; steps: number }>((resolve) => {
      this.pending.set(jobId, resolve);
    });
    const copy = theta.slice();
    w.postMessage(
      {
        type: "eval", jobId,
        theta: copy.buffer, layout: this.layout, reward: this.reward,
        rolloutSteps: this.rolloutSteps,
        runCfg: this.runCfg,
        cmdScale: this.cmdScale,
        targetPoint: this.pointSnapshot ? [...this.pointSnapshot] : null,
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
  private async evalBatch(members: Float32Array[]): Promise<{ fitness: number; fell: boolean }[]> {
    const results: { fitness: number; fell: boolean }[] = new Array(members.length);
    this.stepsCounter = 0;
    if (this.evaluator === "worker" && this.workers.length > 0) {
      await Promise.all(
        members.map(async (m, i) => {
          const r = await this.evalViaWorker(m);
          results[i] = {
            fitness: Number.isFinite(r.fitness) ? r.fitness : -100,
            fell: r.fell,
          };
        }),
      );
    } else {
      // Main-Thread-Interleaving: bis zu 6 gepoolte MjData-Instanzen rundenweise
      // je 1 Policy-Step, nach jedem Rundendurchlauf Yield an den Event-Loop.
      const slots = Math.min(6, members.length);
      for (let batch = 0; batch < members.length; batch += slots) {
        const batchMembers = members.slice(batch, batch + slots);
        const ctxs: RolloutCtx[] = batchMembers.map((m) => {
          const ctx: RolloutCtx = {
            data: this.dataPool.pop() ?? this.engine.newData(),
            mlp: new MlpPolicy(this.layout, m),
            steps: this.rolloutSteps,
            n: 0, t: 0, sum: 0, fell: false, done: false,
            prevAct: new Float32Array(this.layout.actionDim),
            act: new Float32Array(this.layout.actionDim),
            cmd: new Float32Array(Math.max(3, this.meta.cmdSize)),
            pushTimer: 100,
            poisoned: false,
          };
          resetCtx(this.engine, ctx, this.runCfg, this.cmdScale);
          return ctx;
        });
        while (ctxs.some((c) => !c.done)) {
          for (const ctx of ctxs) {
            if (!ctx.done) {
              ctxStep(this.engine, ctx, this.reward, this.runCfg, this.imit, this.imitBuf, this.policyDt);
            }
          }
          await new Promise<void>((r) => setTimeout(r, 0)); // UI-Yield
        }
        for (let i = 0; i < ctxs.length; i++) {
          results[batch + i] = { fitness: finishCtx(ctxs[i], this.runCfg), fell: ctxs[i].fell };
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

  /** v2.3: Profi-Tricks setzen (immer sanitiert). */
  setRunCfg(cfg: Partial<RunCfg> | null | undefined): void {
    this.runCfg = sanitizeRunCfg(cfg);
  }

  getRunCfg(): RunCfg {
    return { ...this.runCfg };
  }

  /** Eine ES-Generation: antithetisches Sampling → Evaluierung → Update. */
  async runGeneration(): Promise<void> {
    const pop = popSizeFor(this.turbo);
    const pairs = Math.ceil(pop / 2);
    const members: Float32Array[] = [];
    const epsilons: Float32Array[] = [];
    for (let i = 0; i < pairs; i++) {
      const eps = new Float32Array(this.theta.length);
      for (let j = 0; j < eps.length; j++) eps[j] = gauss() * this.sigma;
      epsilons.push(eps);
      members.push(Float32Array.from(this.theta.map((v, j) => v + eps[j])));
      members.push(Float32Array.from(this.theta.map((v, j) => v - eps[j])));
    }
    const t0 = performance.now();
    const results = await this.evalBatch(members.slice(0, pop));
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
    for (let j = 0; j < this.theta.length; j++) {
      this.theta[j] += grad[j] * norm;
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

    // v2.3: Curriculum – Tempo hoch bei wenig Stürzen, runter bei vielen
    if (this.runCfg.curriculum) {
      if (this.lastFellRate < CURRICULUM_UP && this.cmdScale < 1) {
        this.cmdScale = Math.min(1, this.cmdScale + 0.15);
      } else if (this.lastFellRate > CURRICULUM_DOWN && this.cmdScale > 0.15) {
        this.cmdScale = Math.max(0.15, this.cmdScale - 0.1);
      }
    } else {
      this.cmdScale = 1;
    }

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
    center: ArrayBufferLike | null; // v2.3: Ruhelage (Gelenkwinkel)
    scaleY: number; // v2.3: Höhen-Skalierung (Cross-Species)
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
