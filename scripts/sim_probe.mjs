// Node-Rollout-Probe: repliziert die App-Physik (buildDuckXml + VFS + Keyframe)
// und lässt die extrahierte Warm-Start-Policy laufen. Damit kann der Trainings-
// Rollout Schritt für Schritt untersucht werden.
import { readFileSync, readdirSync } from "fs";

const pub = "/home/z/my-project/public";

// ── MuJoCo-WASM laden (Emscripten-ESM) ──
const mod = await import(pub + "/wasm/mujoco.js");
const factory = mod.default ?? mod.loadMujoco ?? mod;
const mujoco = await factory({
  locateFile: (p) => (p.endsWith(".wasm") ? pub + "/wasm/mujoco.wasm" : p),
});

// ── buildDuckXml replizieren (String-Operationen, identisches Ergebnis) ──
let src = readFileSync(pub + "/robot/mjlab/robot_allcollisions.xml", "utf8");
// Visual-Geoms entfernen (self-closing)
src = src.replace(/<geom[^>]*class="visual"[^>]*\/>/g, "");
// option/timestep
src = src.replace("</mujoco>", '<option timestep="0.005"/></mujoco>');
// floor + walls + ball + keyframe
const BALL_PARK = "50 0 0.05";
const qposFree = "-0.6 0 0.12 1 0 0 0";
const DUCK_DEFAULT_POSE = [
  0, -0.08726646259971647, -0.457924, -0.00494, 0.452984,
  0.3490658503988659, 0.3490658503988659, 0, 0,
  0, 0.08726646259971647, 0.457924, 0.00494, -0.452984,
];
// Gelenk-Reihenfolge aus dem XML (body > joint in qpos-Reihenfolge)
const jointRe = /<body[^>]*>(?:(?!<\/body>)[\s\S])*?<joint[^>]*name="([^"]+)"[^>]*\/>/g;
// Einfacher: alle joint-name in Dateireihenfolge, aber nur die im worldbody-Body-Baum:
const jointNames = [];
{
  // Zähle Verschachtelung via einfacher Token-Scan
  const tokens = src.match(/<body\b|<joint\b[^>]*>|<\/body>/g) ?? [];
  const stack = [];
  for (const t of tokens) {
    if (t === "<body\b".replace("\\b", "") || t.startsWith("<body")) stack.push(1);
    else if (t === "</body>") stack.pop();
    else if (t.startsWith("<joint")) {
      const m = t.match(/name="([^"]+)"/);
      if (m && stack.length > 0) jointNames.push(m[1]);
    }
  }
}
const poseByName = new Map([
  "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
].map((n, i) => [n, DUCK_DEFAULT_POSE[i]]));
const qposJoints = jointNames.map((n) => poseByName.get(n) ?? 0).join(" ");
const wbInject = `
<geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0"/>
<geom name="wall_px" type="box" pos="1.525 0 0.125" size="0.025 1.55 0.125"/>
<geom name="wall_nx" type="box" pos="-1.525 0 0.125" size="0.025 1.55 0.125"/>
<geom name="wall_py" type="box" pos="0 1.525 0.125" size="1.55 0.025 0.125"/>
<geom name="wall_ny" type="box" pos="0 -1.525 0.125" size="1.55 0.025 0.125"/>
<body name="ball" pos="${BALL_PARK}"><freejoint name="ball_freejoint"/><geom name="ball_geom" type="sphere" size="0.05" mass="0.03" friction="0.4 0.01 0.003" solref="0.03 0.4" condim="6"/></body>
</worldbody>`;
src = src.replace("</worldbody>", wbInject);
src = src.replace("</mujoco>", `<keyframe><key name="STAND" qpos="${qposFree} ${qposJoints} ${BALL_PARK} 1 0 0 0" ctrl="${DUCK_DEFAULT_POSE.join(" ")}"/></keyframe></mujoco>`);

// meshdir anpassen (VFS-Prefix "assets")
src = src.replace('meshdir="assets"', 'meshdir="assets"');

// ── VFS füllen ──
const vfs = new mujoco.MjVFS();
const meshFiles = [...src.matchAll(/mesh\s+[^>]*file="([^"]+)"/g)].map((m) => m[1]);
const uniq = [...new Set(meshFiles.map((f) => f.replace(/^assets\//, "")))];
for (const f of uniq) {
  const bytes = readFileSync(pub + "/robot/mjlab/meshes/" + f);
  vfs.addBuffer("assets/" + f, new Uint8Array(bytes));
}
console.log("Meshes im VFS:", uniq.length);

const model = mujoco.MjModel.from_xml_string(src, vfs);
const data = new mujoco.MjData(model);
console.log("nq", model.nq, "nu", model.nu, "nkey", model.nkey);
if (process.env.CLAMP === "1") { for (let j = 0; j < 14; j += 1) console.log("ctrlrange", j, model.actuator_ctrlrange[j*2].toFixed(2), model.actuator_ctrlrange[j*2+1].toFixed(2)); }

const MJ_OBJ_KEY = mujoco.mjtObj.mjOBJ_KEY.value ?? 6;
const keyId = mujoco.mj_name2id(model, MJ_OBJ_KEY, "STAND");

// Adressen
const JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"];
const qposAdr = JOINTS.map((n) => model.jnt(n).qposadr);
const dofAdr = JOINTS.map((n) => model.jnt(n).dofadr);
const gyroAdr = model.sensor("imu_ang_vel").adr;
const trunkId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value ?? 1, "trunk_base");

// ── Warm-Start-Theta extrahieren (aus scripts/final_check.mjs übernommen) ──
function parseOnnx(path) {
  const b = readFileSync(path);
  const u8 = b.constructor === Uint8Array ? b : new Uint8Array(b);
  function rv(p) { let r = 0, s = 0; for (;;) { const x = u8[p++]; r += (x & 0x7f) * Math.pow(2, s); if (!(x & 0x80)) break; s += 7; } return [r, p]; }
  function ff(start, end, cb) {
    let pos = start;
    while (pos < end) {
      const [key, p1] = rv(pos);
      const f = key >>> 3, w = key & 7;
      if (w === 2) { const [l, q] = rv(p1); cb(f, w, q, l); pos = q + l; }
      else { let p2; if (w === 0) p2 = rv(p1)[1]; else if (w === 1) p2 = p1 + 8; else if (w === 5) p2 = p1 + 4; else return; cb(f, w, p1, p2 - p1); pos = p2; }
    }
  }
  const tensors = [];
  ff(0, u8.length, (f, w, pos, len) => {
    if (f !== 7 || w !== 2) return;
    ff(pos, pos + len, (gf, gw, gpos, glen) => {
      if (gf !== 5 || gw !== 2) return;
      let name = "";
      ff(gpos, gpos + glen, (tf, tw, tpos, tlen) => {
        if (tf === 8 && tw === 2) name = new TextDecoder().decode(u8.slice(tpos, tpos + tlen));
      });
      const t = { name, dims: [], floats: new Float32Array(0) };
      ff(gpos, gpos + glen, (tf, tw, tpos, tlen) => {
        if (tf === 1 && tw === 0) t.dims.push(rv(tpos)[0]);
        else if (tf === 1 && tw === 2) { let p = tpos; while (p < tpos + tlen) { const [v, np] = rv(p); t.dims.push(v); p = np; } }
        else if (tf === 9 && tw === 2) t.floats = new Float32Array(u8.slice(tpos, tpos + tlen).buffer, 0, tlen / 4);
      });
      tensors.push(t);
    });
  });
  return tensors;
}
const tensors = parseOnnx(pub + "/policies/BEST_alpha_walking.onnx");
const byName = new Map(tensors.map((t) => [t.name, t]));
const layers = [];
for (const t of byName.values()) {
  const m = t.name.match(/mlp\.(\d+)\.weight/);
  if (m) layers.push({ idx: +m[1], w: t });
}
layers.sort((a, b) => a.idx - b.idx);
for (const L of layers) L.b = byName.get(L.w.name.replace(/\.weight$/, ".bias"));
const meanT = [...byName.values()].find((t) => /_mean/i.test(t.name));
const stdT = [...byName.values()].find((t) => t !== meanT && /Div/i.test(t.name));
const hs = layers.map((L) => L.w.dims[0]).slice(0, -1);
const dims = [61, ...hs, 14];
const theta = new Float32Array(dims.slice(0, -1).reduce((a, d, i) => a + dims[i + 1] * d + dims[i + 1], 0));
{
  let off = 0;
  for (let l = 0; l < layers.length; l++) {
    const { w, b } = layers[l];
    const outDim = w.dims[0], inDim = w.dims[1];
    for (let i = 0; i < outDim; i++) {
      let corr = 0;
      for (let k = 0; k < inDim; k++) {
        const wv = w.floats[i * inDim + k];
        if (l === 0) { theta[off + i * inDim + k] = wv / stdT.floats[k]; corr += wv * meanT.floats[k] / stdT.floats[k]; }
        else theta[off + i * inDim + k] = wv;
      }
      theta[off + outDim * inDim + i] = l === 0 ? b.floats[i] - corr : b.floats[i];
    }
    off += outDim * inDim + outDim;
  }
}
console.log("theta:", theta.length);

function forward(obs) {
  let src2 = obs, off = 0;
  for (let l = 0; l < dims.length - 1; l++) {
    const inDim = dims[l], outDim = dims[l + 1];
    const w = theta.subarray(off, off + outDim * inDim); off += outDim * inDim;
    const b = theta.subarray(off, off + outDim); off += outDim;
    const out = new Float32Array(outDim);
    for (let i = 0; i < outDim; i++) {
      let s = b[i];
      const base = i * inDim;
      for (let k = 0; k < inDim; k++) s += w[base + k] * src2[k];
      out[i] = s;
    }
    if (l < dims.length - 2) for (let i = 0; i < outDim; i++) if (out[i] < 0) out[i] = Math.expm1(out[i]);
    src2 = out;
  }
  return src2;
}

// ── Rollout ──
function gauss() { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function reset() {
  mujoco.mj_resetDataKeyframe(model, data, keyId);
  mujoco.mj_forward(model, data);
  if (process.env.NOISE === "1") {
    const qpos = data.qpos, qvel = data.qvel;
    for (let j = 0; j < 14; j++) { qpos[qposAdr[j]] += gauss() * 0.03; qvel[dofAdr[j]] += gauss() * 0.1; }
    qvel[0] += gauss() * 0.05; qvel[1] += gauss() * 0.05;
    mujoco.mj_forward(model, data);
    for (let j = 0; j < 14; j++) data.ctrl[j] = DUCK_DEFAULT_POSE[j];
  }
}
reset();
const cmd = new Float32Array(13);
const obs = new Float32Array(61);
const lastAction = new Float32Array(14);
let action = new Float32Array(14);
const SMOOTH = parseFloat(process.env.SMOOTH ?? "1");
const CMDX = parseFloat(process.env.CMDX ?? "0");
const STEPS = parseInt(process.env.STEPS ?? "200");
cmd[0] = CMDX;
cmd[2] = parseFloat(process.env.CMDZ ?? "0");
for (let t = 0; t < STEPS; t++) {
  // obs
  let i = 0;
  const qpos = data.qpos, qvel = data.qvel;
  for (let a = 0; a < 3; a++) obs[i++] = data.sensordata[gyroAdr + a];
  const xq = data.body(trunkId).xquat;
  const w = xq[0], x = xq[1], y = xq[2], z = xq[3];
  obs[i++] = -2 * (x * z - w * y);
  obs[i++] = -2 * (y * z + w * x);
  obs[i++] = -(1 - 2 * (x * x + y * y));
  for (let j = 0; j < 14; j++) obs[i++] = qpos[qposAdr[j]] - DUCK_DEFAULT_POSE[j];
  for (let j = 0; j < 14; j++) obs[i++] = qvel[dofAdr[j]];
  for (let j = 0; j < 14; j++) obs[i++] = lastAction[j];
  for (let c = 0; c < 13; c++) obs[i++] = cmd[c];
  const raw = process.env.ZEROACT === "1" ? new Float32Array(14) : forward(obs);
  const prev = action;
  action = new Float32Array(14);
  for (let j = 0; j < 14; j++) action[j] = SMOOTH < 0.999 ? SMOOTH * raw[j] + (1 - SMOOTH) * prev[j] : raw[j];
  lastAction.set(action);
  // ctrl = defaultPose + action
  const ctrl = data.ctrl;
  if (process.env.CLAMP === "1") {
    for (let j = 0; j < 14; j++) {
      const lo = model.actuator_ctrlrange[j * 2], hi = model.actuator_ctrlrange[j * 2 + 1];
      let v = DUCK_DEFAULT_POSE[j] + action[j];
      ctrl[j] = v < lo ? lo : v > hi ? hi : v;
    }
  } else {
    for (let j = 0; j < 14; j++) ctrl[j] = DUCK_DEFAULT_POSE[j] + action[j];
  }
  for (let s = 0; s < 4; s++) mujoco.mj_step(model, data);
  if (process.env.PUSH === "1" && t > 0 && t % 55 === 0) {
    const qv = data.qvel;
    qv[0] += gauss() * 0.35; qv[1] += gauss() * 0.35;
    qv[3] += gauss() * 0.2; qv[4] += gauss() * 0.2; qv[5] += gauss() * 0.3;
  }
  if (t % 10 === 0 || t < 8 || t === STEPS - 1) {
    const zz = qpos[2].toFixed(3);
    const vx = qvel[0].toFixed(3);
    const a0 = action[0].toFixed(2);
    const range = Math.max(...action).toFixed(2), minv = Math.min(...action).toFixed(2);
    console.log(`t=${String(t).padStart(3)} z=${zz} vx=${vx} act=[${a0}..${minv}..${range}] x=${qpos[0].toFixed(2)}`);
  }
  if (qpos[2] < 0.06) { console.log(`GESTÜRZT bei t=${t} z=${qpos[2].toFixed(3)}`); break; }
}
console.log("Endposition x =", data.qpos[0].toFixed(3), "z =", data.qpos[2].toFixed(3));
