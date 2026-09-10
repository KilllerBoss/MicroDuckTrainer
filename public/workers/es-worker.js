// ── MicroDuck Trainer v2.0 – ES-Worker (headless Rollouts) ───────────────────
// Module-Worker: lädt MuJoCo nativ via import('/wasm/mujoco.js') (die Datei ist
// ein echtes ES-Modul – Entscheidung im Worklog dokumentiert). Kein Rendering:
// reine Physik + MLP-Policy + Reward-Berechnung, identisch zu src/lib/md/es.ts.

let mujoco = null;
let model = null;
let keyId = -1;
let ctx = null; // { modelId, obsDim, actionDim, cmdSize, obsType, decimation, actionScale, defaultPose }
let addrs = null; // { qposAdr, dofAdr, gyroAdr, torsoId, ballQposAdr, ballDofAdr, ctrlRange, jntRange }
let standPose = null;
let data = null;

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
  };

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
  mujoco.mj_forward(model, data);
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
  return obs;
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

function runRollout(theta, layout, reward, steps) {
  resetToKeyframe();
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
  let sum = 0, n = 0, fell = false;
  for (let t = 0; t < steps; t++) {
    buildObs(lastAction, cmd, obs);
    mlpForward(layout, w1, b1, w2, b2, obs, hiddenBuf, act);
    lastAction.set(act);
    stepWithAction(act);
    n++;
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
    sum += val;
    prevAct.set(act);
    if (isFallen()) {
      fell = true;
      if (T.fall?.enabled) sum -= T.fall.weight * n;
      break;
    }
  }
  return { fitness: n > 0 ? sum / n : 0, fell, steps: n };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await boot(msg);
      post({ type: "ready", workerId: msg.workerId });
    } else if (msg.type === "eval") {
      const theta = new Float32Array(msg.theta);
      const r = runRollout(theta, msg.layout, msg.reward, msg.rolloutSteps);
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
