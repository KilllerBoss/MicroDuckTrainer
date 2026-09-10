// ── MicroDuck Trainer v2.0 – ES-Worker (headless Rollouts) ───────────────────
// Module-Worker: lädt MuJoCo nativ via import('/wasm/mujoco.js') (die Datei ist
// ein echtes ES-Modul – Entscheidung im Worklog dokumentiert). Kein Rendering:
// reine Physik + MLP-Policy + Reward-Berechnung, identisch zu src/lib/md/es.ts.

let mujoco = null;
let model = null;
let keyId = -1;
let ctx = null; // { modelId, obsDim, actionDim, cmdSize, obsType, decimation, actionScale, defaultPose, obsExtra, targetHeight, imitation }
let addrs = null; // { qposAdr, dofAdr, gyroAdr, torsoId, ballQposAdr, ballDofAdr, ctrlRange, jntRange }
let standPose = null;
let data = null;
let imit = null; // { dim, frames, fps, duration, targets, rootY, baseY, center, scaleY }
let imitBuf = null;
let policyDt = 0.02;
let pointSnapshot = null; // [x, y] | null (Punkt-Modus, Snapshot je Eval)

// ── v2.3: Profi-Tricks (identisch zu src/lib/md/es.ts) ──
function defaultRunCfg() {
  return {
    cmdTrain: true, cmdFwd: 0.3, cmdLat: 0.15, cmdAng: 0.8,
    curriculum: true, actionSmooth: 0.6, pushes: true, noiseReset: true,
    fitnessMode: "sum", weightDecay: 0.005,
  };
}
function sanitizeRunCfg(p) {
  const d = defaultRunCfg();
  if (!p || typeof p !== "object") return d;
  const num = (v, def, lo, hi) =>
    typeof v === "number" && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : def;
  return {
    cmdTrain: typeof p.cmdTrain === "boolean" ? p.cmdTrain : d.cmdTrain,
    cmdFwd: num(p.cmdFwd, d.cmdFwd, 0, 1.2),
    cmdLat: num(p.cmdLat, d.cmdLat, 0, 0.6),
    cmdAng: num(p.cmdAng, d.cmdAng, 0, 2),
    curriculum: typeof p.curriculum === "boolean" ? p.curriculum : d.curriculum,
    actionSmooth: num(p.actionSmooth, d.actionSmooth, 0.2, 1),
    pushes: typeof p.pushes === "boolean" ? p.pushes : d.pushes,
    noiseReset: typeof p.noiseReset === "boolean" ? p.noiseReset : d.noiseReset,
    fitnessMode: p.fitnessMode === "mean" ? "mean" : "sum",
    weightDecay: num(p.weightDecay, d.weightDecay, 0, 0.05),
  };
}
function drawCmd(cfg, cmdScale, out) {
  if (cfg.cmdTrain) {
    out[0] = Math.random() * cfg.cmdFwd * cmdScale;
    if (out.length > 1) out[1] = (Math.random() * 2 - 1) * cfg.cmdLat * cmdScale;
    if (out.length > 2) out[2] = (Math.random() * 2 - 1) * cfg.cmdAng * cmdScale;
  } else {
    out.fill(0);
    out[0] = 0.25 * cmdScale;
  }
}

// ── v2.2: KI-Code-Term (identisch zu src/lib/md/customcode.ts) ──
const FORBIDDEN = /\b(import|require|eval|Function|fetch|XMLHttpRequest|localStorage|sessionStorage|indexedDB|document|window|globalThis|self|postMessage|Worker|WebSocket)\b/;
let customCacheKey = null;
let customFn = null;
let customWarned = false;

function getCustomFn(code) {
  if (customCacheKey === code) return customFn;
  customCacheKey = code;
  customFn = null;
  try {
    if (code && code.trim() && !FORBIDDEN.test(code)) {
      const compiled = new Function("api", `"use strict";\n${code}\n`);
      // Kaltstart-Check mit Dummy-API
      const n = 8;
      const probe = compiled({
        h: 0.7, height: 0.7, upZ: 1, gz: -1, vx: 0, vy: 0, vz: 0, omega: 0,
        angles: new Float32Array(n), act: new Float32Array(n), prevAct: new Float32Array(n),
        qpos: new Float32Array(16), qvel: new Float32Array(16), qacc: new Float32Array(16),
        torso: [0, 0, 0.7], target: null, cmd: new Float32Array([0, 0, 0]),
        imitDelta: null, imitTarget: null, dt: 0.02, t: 0, step: 0,
      });
      if (typeof probe === "number" && Number.isFinite(probe)) customFn = compiled;
    }
  } catch {
    customFn = null;
  }
  return customFn;
}

function evalCustomFn(fn, api) {
  try {
    const v = fn(api);
    return Number.isFinite(v) ? v : 0;
  } catch (err) {
    if (!customWarned) {
      customWarned = true;
      console.warn("[custom-reward] Laufzeitfehler (Term liefert 0):", err);
    }
    return 0;
  }
}

const BALL_RADIUS = 0.05;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function boot(payload) {
  const mod = await import("/wasm/mujoco.js");
  const factory = mod.default ?? mod.loadMujoco ?? mod;
  mujoco = await factory({
    locateFile: (p) => (p.endsWith(".wasm") ? "/wasm/mujoco.wasm" : p),
  });

  // VFS selbst aufbauen: gleiche Bytes wie der Main-Thread (meshBase + Dateiliste)
  const vfs = new mujoco.MjVFS();
  await Promise.all(
    payload.meshFiles.map(async (f) => {
      const r = await fetch(`${payload.meshBase}/${f}`, { cache: "force-cache" });
      if (!r.ok) throw new Error(`Mesh-Fetch fehlgeschlagen: ${f} (${r.status})`);
      vfs.addBuffer(`${payload.vfsPrefix}/${f}`, new Uint8Array(await r.arrayBuffer()));
    }),
  );
  model = mujoco.MjModel.from_xml_string(payload.xml, vfs);
  data = new mujoco.MjData(model);

  ctx = {
    modelId: payload.modelId,
    obsDim: payload.obsDim,
    actionDim: payload.actionDim,
    cmdSize: payload.cmdSize,
    obsType: payload.obsType,
    decimation: payload.decimation,
    actionScale: payload.actionScale,
    defaultPose: payload.defaultPose,
    jointNames: payload.jointNames,
    torsoBody: payload.torsoBody,
    gyroSensor: payload.gyroSensor,
    obsExtra: payload.obsExtra ?? 0,
    targetHeight: payload.targetHeight ?? (payload.modelId === "microduck" ? 0.12 : 0.72),
    imitation: payload.imitation ?? null,
  };
  if (ctx.imitation) {
    imit = {
      dim: ctx.imitation.dim,
      frames: ctx.imitation.frames,
      fps: ctx.imitation.fps,
      duration: ctx.imitation.duration,
      targets: new Float32Array(ctx.imitation.targets),
      rootY: new Float32Array(ctx.imitation.rootY),
      baseY: ctx.imitation.baseY,
      center: ctx.imitation.center ? new Float32Array(ctx.imitation.center) : null,
      scaleY: ctx.imitation.scaleY ?? 1,
    };
    imitBuf = new Float32Array(imit.dim);
  }
  policyDt = ctx.decimation * (payload.modelId === "microduck" ? 0.005 : 0.002);

  const MJ_OBJ_BODY = mujoco.mjtObj.mjOBJ_BODY.value ?? 1;
  const MJ_OBJ_KEY = mujoco.mjtObj.mjOBJ_KEY.value ?? 6;
  const torsoId = mujoco.mj_name2id(model, MJ_OBJ_BODY, ctx.torsoBody);
  keyId = mujoco.mj_name2id(model, MJ_OBJ_KEY, payload.keyframe);
  const qposAdr = ctx.jointNames.map((n) => model.jnt(n).qposadr);
  const dofAdr = ctx.jointNames.map((n) => model.jnt(n).dofadr);
  let gyroAdr = -1;
  try {
    if (ctx.gyroSensor) gyroAdr = model.sensor(ctx.gyroSensor).adr;
  } catch {
    gyroAdr = -1;
  }
  const nu = model.nu;
  let ctrlRange = null;
  try {
    // HINWEIS: actuator_ctrllimited ist in @mujoco/mujoco 3.13 als
    // memory_view<bool> gebunden (Zugriff wirft!) → Limitierung über
    // ctrlrange erschließen (0/0 oder min==max = frei).
    const rng = model.actuator_ctrlrange;
    if (rng && rng.length >= nu * 2) {
      ctrlRange = new Float32Array(nu * 2);
      for (let i = 0; i < nu; i++) {
        const lo = rng[i * 2], hi = rng[i * 2 + 1];
        const limited = lo !== hi && !(lo === 0 && hi === 0);
        ctrlRange[i * 2] = limited ? lo : -1e6;
        ctrlRange[i * 2 + 1] = limited ? hi : 1e6;
      }
    }
  } catch {
    ctrlRange = null;
  }
  // jnt_limited: gleiches Bindungsproblem → [0,0] in jnt_range = unbegrenzt.
  const jntRangeArr = model.jnt_range;
  const jntRange = ctx.jointNames.map((n) => {
    const j = model.jnt(n);
    if (jntRangeArr && jntRangeArr.length > j.id * 2 + 1) {
      return [jntRangeArr[j.id * 2], jntRangeArr[j.id * 2 + 1]];
    }
    return [0, 0];
  });
  addrs = { qposAdr, dofAdr, gyroAdr, torsoId, ctrlRange, jntRange };
  resetToKeyframe();
}

function resetToKeyframe() {
  mujoco.mj_resetDataKeyframe(model, data, keyId);
  // v2.3: G1-Arme natürlicher (Ellbogen ~26° statt 90°) – identisch zur Engine
  if (ctx.modelId === "unitree_g1") {
    const relax = {
      left_shoulder_pitch_joint: 0.35, right_shoulder_pitch_joint: 0.35,
      left_shoulder_roll_joint: 0.1, right_shoulder_roll_joint: -0.1,
      left_elbow_joint: 0.45, right_elbow_joint: 0.45,
    };
    const qpos = data.qpos;
    for (let j = 0; j < ctx.jointNames.length; j++) {
      const r = relax[ctx.jointNames[j]];
      if (r !== undefined) qpos[addrs.qposAdr[j]] = r;
    }
    mujoco.mj_forward(model, data);
  } else {
    mujoco.mj_forward(model, data);
  }
  const qpos = data.qpos;
  standPose = new Float32Array(ctx.actionDim);
  for (let j = 0; j < ctx.actionDim; j++) standPose[j] = qpos[addrs.qposAdr[j]];
  applyCtrlFromPose(standPose);
}

function applyCtrlFromPose(targets) {
  const ctrl = data.ctrl;
  const lim = addrs.ctrlRange;
  for (let j = 0; j < ctx.actionDim; j++) {
    let v = targets[j];
    if (lim) v = Math.min(lim[j * 2 + 1], Math.max(lim[j * 2], v));
    ctrl[j] = v;
  }
}

function stepWithAction(action) {
  const ctrl = data.ctrl;
  const lim = addrs.ctrlRange;
  for (let j = 0; j < ctx.actionDim; j++) {
    const base = ctx.modelId === "microduck" ? ctx.defaultPose[j] : standPose[j];
    let v = base + action[j] * ctx.actionScale;
    if (lim) v = Math.min(lim[j * 2 + 1], Math.max(lim[j * 2], v));
    ctrl[j] = v;
  }
  for (let s = 0; s < ctx.decimation; s++) mujoco.mj_step(model, data);
}

function projGravity() {
  const xq = data.body(addrs.torsoId).xquat; // wxyz
  const w = xq[0], x = xq[1], y = xq[2], z = xq[3];
  return [
    -2 * (x * z - w * y),
    -2 * (y * z + w * x),
    -(1 - 2 * (x * x + y * y)),
  ];
}

function projGravZ() {
  const xq = data.body(addrs.torsoId).xquat;
  const x = xq[1], y = xq[2];
  return -(1 - 2 * (x * x + y * y));
}

function buildObs(lastAction, cmd, obs) {
  const qpos = data.qpos;
  const qvel = data.qvel;
  let i = 0;
  const g = projGravity();
  if (ctx.obsType === "new-cmd-obs") {
    for (let a = 0; a < 3; a++) obs[i++] = data.sensordata[addrs.gyroAdr + a];
    obs[i++] = g[0]; obs[i++] = g[1]; obs[i++] = g[2];
    for (let j = 0; j < ctx.actionDim; j++) obs[i++] = qpos[addrs.qposAdr[j]] - ctx.defaultPose[j];
    for (let j = 0; j < ctx.actionDim; j++) obs[i++] = qvel[addrs.dofAdr[j]];
    for (let j = 0; j < ctx.actionDim; j++) obs[i++] = lastAction[j];
    for (let c = 0; c < ctx.cmdSize; c++) obs[i++] = cmd[c];
  } else {
    obs[i++] = g[0]; obs[i++] = g[1]; obs[i++] = g[2];
    obs[i++] = qpos[2];
    let av0 = qvel[3], av1 = qvel[4], av2 = qvel[5];
    if (addrs.gyroAdr >= 0) {
      av0 = data.sensordata[addrs.gyroAdr];
      av1 = data.sensordata[addrs.gyroAdr + 1];
      av2 = data.sensordata[addrs.gyroAdr + 2];
    }
    obs[i++] = av0; obs[i++] = av1; obs[i++] = av2;
    for (let j = 0; j < ctx.actionDim; j++) obs[i++] = qpos[addrs.qposAdr[j]] - standPose[j];
    for (let j = 0; j < ctx.actionDim; j++) obs[i++] = qvel[addrs.dofAdr[j]];
    for (let j = 0; j < ctx.actionDim; j++) obs[i++] = lastAction[j];
    for (let c = 0; c < ctx.cmdSize; c++) obs[i++] = cmd[c];
  }
  if (ctx.obsExtra >= 2 && obs.length >= i + 2) {
    obs[i++] = imitPhase[0];
    obs[i++] = imitPhase[1];
  }
  return obs;
}

// ── v2.1: Imitations-Sampling (identisch zu src/lib/md/es.ts) ──
let imitPhase = [0, 1];
let imitTarget = null;
let imitRootDelta = null;

function imitSampleAt(d, t, out) {
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
    out[j] = Number.isFinite(v) ? v : (center ? center[j] : 0);
  }
  const raw = (d.rootY[f0] + (d.rootY[f1] - d.rootY[f0]) * u) - d.baseY;
  const s = d.scaleY ?? 1;
  return Number.isFinite(raw) ? raw * s : 0;
}

function isFallen() {
  const z = data.qpos[2];
  const gz = projGravZ();
  const fallHeight = ctx.modelId === "microduck" ? 0.06 : 0.5;
  if (!Number.isFinite(z) || !Number.isFinite(gz)) return true;
  return z < fallHeight || gz > -0.5;
}

// MLP (identisch zu src/lib/md/policy.ts)
function mlpForward(layout, w1, b1, w2, b2, obs, hiddenBuf, act) {
  const o = layout.obsDim, h = layout.hidden, a = layout.actionDim;
  for (let j = 0; j < h; j++) {
    let s = b1[j];
    const base = j * o;
    for (let k = 0; k < o; k++) s += w1[base + k] * obs[k];
    hiddenBuf[j] = Math.tanh(s);
  }
  for (let i = 0; i < a; i++) {
    let s = b2[i];
    const base = i * h;
    for (let k = 0; k < h; k++) s += w2[base + k] * hiddenBuf[k];
    act[i] = Math.tanh(s);
  }
  return act;
}

function runRollout(theta, layout, reward, steps, targetPoint, runCfg, cmdScale) {
  runCfg = sanitizeRunCfg(runCfg);
  cmdScale = typeof cmdScale === "number" && Number.isFinite(cmdScale)
    ? Math.min(1, Math.max(0.1, cmdScale)) : 0.4;
  resetToKeyframe();
  // v2.3: Reset-Rauschen (jeder Start leicht anders)
  if (runCfg.noiseReset) {
    const qp = data.qpos;
    const qv = data.qvel;
    for (let j = 0; j < addrs.qposAdr.length; j++) qp[addrs.qposAdr[j]] += (Math.random() * 2 - 1) * 0.04;
    for (let j = 0; j < Math.min(6, qv.length); j++) qv[j] += (Math.random() * 2 - 1) * 0.08;
    mujoco.mj_forward(model, data);
  }
  // v2.3: Zufalls-Bewegungsbefehl für diese Runde
  const cmd = new Float32Array(ctx.cmdSize >= 3 ? ctx.cmdSize : 3);
  drawCmd(runCfg, cmdScale, cmd);
  const o = layout.obsDim, h = layout.hidden;
  const w1 = theta.slice(0, h * o);
  const b1 = theta.slice(h * o, h * o + h);
  const w2 = theta.slice(h * o + h, h * o + h + layout.actionDim * h);
  const b2 = theta.slice(h * o + h + layout.actionDim * h);
  const obs = new Float32Array(o);
  const hiddenBuf = new Float32Array(h);
  const act = new Float32Array(layout.actionDim);
  const prevAct = new Float32Array(layout.actionDim);
  const lastAction = new Float32Array(layout.actionDim);
  const cmd = new Float32Array(ctx.cmdSize);
  const T = reward.terms;
  // v2.2: wiederverwendetes API-Objekt für den KI-Code-Term (kein GC-Druck)
  const customTerm = reward.custom;
  const customUse = !!(customTerm && customTerm.enabled && customTerm.code);
  const customFnLocal = customUse ? getCustomFn(customTerm.code) : null;
  const api = customUse && customFnLocal ? {
    h: 0, height: 0, upZ: 0, gz: 0, vx: 0, vy: 0, vz: 0, omega: 0,
    angles: new Float32Array(layout.actionDim),
    act, prevAct,
    qpos: null, qvel: null, qacc: null,
    torso: [0, 0, 0], target: null,
    cmd,
    imitDelta: null, imitTarget: null,
    dt: policyDt, t: 0, step: 0,
  } : null;
  let sum = 0, n = 0, fell = false;
  let tImit = 0; // Imitations-Zeit (s)
  let pushTimer = 60 + Math.floor(Math.random() * 60);
  let poisoned = false;
  for (let t = 0; t < steps && !poisoned; t++) {
    if (imit && imitBuf) {
      imitRootDelta = imitSampleAt(imit, tImit, imitBuf);
      imitTarget = imitBuf;
      const ph = (imit.duration > 0 ? (tImit % imit.duration) / imit.duration : 0) * 2 * Math.PI;
      imitPhase = [Math.sin(ph), Math.cos(ph)];
    }
    buildObs(lastAction, cmd, obs);
    mlpForward(layout, w1, b1, w2, b2, obs, hiddenBuf, act);
    // v2.3: Aktions-EMA-Glättung (gegen Zittern)
    const alpha = runCfg.actionSmooth;
    if (alpha < 1) {
      for (let j = 0; j < act.length; j++) {
        act[j] = alpha * act[j] + (1 - alpha) * prevAct[j];
      }
    }
    lastAction.set(act);
    stepWithAction(act);
    n++;
    // v2.3: Zufalls-Stöße (Domain Randomization)
    if (runCfg.pushes && n > 40) {
      if (--pushTimer <= 0) {
        pushTimer = 80 + Math.floor(Math.random() * 120);
        const qv = data.qvel;
        if (qv.length >= 6) {
          qv[0] += (Math.random() * 2 - 1) * 0.35;
          qv[1] += (Math.random() * 2 - 1) * 0.35;
          qv[5] += (Math.random() * 2 - 1) * 0.28;
        }
      }
    }
    // ── Reward-Terme (identisch zur Main-Thread-Variante) ──
    const gz = projGravZ();
    const upZ = Math.max(0, Math.min(1, -gz));
    const hh = data.qpos[2];
    const qpos0 = data.qpos;
    const yaw = Math.atan2(
      2 * (qpos0[3] * qpos0[6] + qpos0[4] * qpos0[5]),
      1 - 2 * (qpos0[5] * qpos0[5] + qpos0[6] * qpos0[6]),
    );
    const qvel = data.qvel;
    const vx = Math.cos(yaw) * qvel[0] + Math.sin(yaw) * qvel[1];
    const vy = -Math.sin(yaw) * qvel[0] + Math.cos(yaw) * qvel[1];
    const omega = qvel[5];
    let val = 0;
    if (T.upright?.enabled) val += T.upright.weight * upZ;
    if (T.height?.enabled) {
      val += T.height.weight * (1 - Math.abs(hh - T.height.param) / Math.max(0.05, T.height.param));
    }
    if (T.forward?.enabled) {
      val += T.forward.weight * Math.max(0, 1 - Math.abs(vx - T.forward.param) / 0.5);
    }
    if (T.lateral?.enabled) val += T.lateral.weight * Math.max(-1, Math.min(1, vy));
    if (T.yaw?.enabled) val += T.yaw.weight * (1 - Math.min(1, Math.abs(omega - cmd[2])));
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
      const qacc = data.qacc;
      let s = 0;
      for (let j = 0; j < addrs.dofAdr.length; j++) s += Math.abs(qacc[addrs.dofAdr[j]]);
      val -= T.jointAccel.weight * (s / addrs.dofAdr.length / 1000);
    }
    if (T.contact?.enabled) {
      const qacc = data.qacc;
      let s = 0;
      for (let j = 0; j < 6; j++) s += Math.abs(qacc[j]);
      val -= T.contact.weight * (s / 6 / 1000);
    }
    if (T.jointLimit?.enabled) {
      const angles = addrs.qposAdr.map((a) => qpos0[a]);
      let pen = 0, cnt = 0;
      for (let j = 0; j < angles.length; j++) {
        const r = addrs.jntRange[j] || [0, 0];
        const lim = Math.max(Math.abs(r[0]), Math.abs(r[1]));
        if (lim < 1e-6) continue;
        cnt++;
        const over = Math.max(0, (Math.abs(angles[j]) - 0.9 * lim) / (0.1 * lim));
        pen += Math.min(1, over);
      }
      if (cnt > 0) val -= T.jointLimit.weight * (pen / cnt);
    }
    // ── v2.1: Imitation ──
    if (T.imitate?.enabled && imitTarget) {
      let err = 0;
      for (let j = 0; j < addrs.qposAdr.length; j++) {
        err += Math.abs(qpos0[addrs.qposAdr[j]] - imitTarget[j]);
      }
      err /= Math.max(1, addrs.qposAdr.length);
      val += T.imitate.weight * Math.max(0, 1 - err / Math.max(0.05, T.imitate.param));
    }
    if (T.imitHeight?.enabled && imitRootDelta !== null) {
      const zt = ctx.targetHeight + imitRootDelta;
      val -= T.imitHeight.weight
        * Math.min(1, Math.abs(data.qpos[2] - zt) / Math.max(0.05, T.imitHeight.param));
    }
    // ── v2.1: Punkt-Modus ──
    if ((T.pointChase?.enabled || T.pointAvoid?.enabled) && pointSnapshot) {
      const px = data.body(addrs.torsoId).xpos;
      const d = Math.hypot(px[0] - pointSnapshot[0], px[1] - pointSnapshot[1]);
      if (T.pointChase?.enabled) {
        val += T.pointChase.weight * Math.max(0, 1 - d / Math.max(0.05, T.pointChase.param));
      }
      if (T.pointAvoid?.enabled) {
        val += T.pointAvoid.weight * Math.min(1, d / Math.max(0.05, T.pointAvoid.param));
      }
    }
    // ── v2.2: KI-Code-Term (identisch zur Main-Thread-Variante) ──
    if (api && customFnLocal) {
      const torsoX = data.body(addrs.torsoId).xpos;
      api.h = hh; api.height = hh;
      api.upZ = upZ; api.gz = gz;
      api.vx = vx; api.vy = vy; api.vz = qvel[2] ?? 0; api.omega = omega;
      const ang = api.angles;
      for (let j = 0; j < addrs.qposAdr.length; j++) ang[j] = qpos0[addrs.qposAdr[j]];
      api.qpos = qpos0; api.qvel = qvel; api.qacc = data.qacc;
      api.torso[0] = torsoX[0]; api.torso[1] = torsoX[1]; api.torso[2] = torsoX[2];
      api.target = pointSnapshot ? [pointSnapshot[0], pointSnapshot[1]] : null;
      api.imitDelta = imitRootDelta;
      api.imitTarget = imitTarget;
      api.t = tImit; api.step = n;
      val += customTerm.weight * evalCustomFn(customFnLocal, api);
    }
    sum += val;
    tImit += policyDt;
    prevAct.set(act);
    // v2.3: NaN-Schutz – divergierte Physik beendet die Runde sofort
    if (!Number.isFinite(sum) || !Number.isFinite(data.qpos[2])) {
      poisoned = true;
      fell = true;
      sum -= T.fall?.enabled ? T.fall.weight : 5;
      break;
    }
    if (isFallen()) {
      fell = true;
      if (T.fall?.enabled) sum -= T.fall.weight;
      break;
    }
  }
  let fitness;
  if (n <= 0) {
    fitness = 0;
  } else if (runCfg.fitnessMode === "sum") {
    fitness = (sum / Math.max(1, steps)) * 100 + (n / Math.max(1, steps)) * 20;
  } else {
    fitness = sum / n;
  }
  return { fitness: Number.isFinite(fitness) ? fitness : -100, fell, steps: n };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await boot(msg);
      post({ type: "ready", workerId: msg.workerId });
    } else if (msg.type === "eval") {
      const theta = new Float32Array(msg.theta);
      pointSnapshot = msg.targetPoint ? msg.targetPoint : null;
      const r = runRollout(
        theta, msg.layout, msg.reward, msg.rolloutSteps, pointSnapshot,
        msg.runCfg, msg.cmdScale,
      );
      post({ type: "result", jobId: msg.jobId, ...r });
    }
  } catch (err) {
    if (msg.type === "init") {
      post({ type: "bootError", message: String(err?.message || err) });
    } else {
      post({ type: "result", jobId: msg.jobId, fitness: -100, fell: true, steps: 0 });
    }
  }
};

void BALL_RADIUS;
