// Probe: Extraktion + Normalizer-Folding + Aktivierungs-Kalibrierung der
// Original-ONNX-Policy — isoliert in Node (onnxruntime-web/wasm).
import * as ort from "onnxruntime-web/wasm";
import { readFileSync } from "fs";

ort.env.wasm.wasmPaths = "file:///home/z/my-project/node_modules/onnxruntime-web/dist/";
ort.env.wasm.numThreads = 1;

const URL = "/home/z/my-project/public/policies/BEST_alpha_walking.onnx";
const OBS_DIM = 61, ACT_DIM = 14;

// ── Protobuf-Minimalparser (wie src/lib/md/policy.ts) ──
function readVarint(buf, pos) {
  let result = 0, shift = 0;
  for (;;) {
    const b = buf[pos++];
    result += (b & 0x7f) * 2 ** shift;
    if ((b & 0x80) === 0) break;
    shift += 7;
  }
  return [result, pos];
}
function forEachField(buf, start, end, cb) {
  let pos = start;
  while (pos < end) {
    const [key, p1] = readVarint(buf, pos);
    const field = key >>> 3, wire = key & 7;
    if (wire === 2) {
      const [len, q] = readVarint(buf, p1);
      cb(field, wire, q, len);
      pos = q + len;
    } else {
      let p2;
      if (wire === 0) p2 = readVarint(buf, p1)[1];
      else if (wire === 1) p2 = p1 + 8;
      else if (wire === 5) p2 = p1 + 4;
      else return;
      cb(field, wire, p1, p2 - p1);
      pos = p2;
    }
  }
}
function parseTensor(buf, start, end, name) {
  const t = { name, dims: [], floats: new Float32Array(0) };
  forEachField(buf, start, end, (field, wire, pos, len) => {
    if (field === 1 && wire === 0) t.dims.push(readVarint(buf, pos)[0]);
    else if (field === 1 && wire === 2) { let p = pos; while (p < pos + len) { const [v, np] = readVarint(buf, p); t.dims.push(v); p = np; } }
    else if (field === 9 && wire === 2) t.floats = new Float32Array(buf.slice(pos, pos + len).buffer, 0, len / 4);
    else if (field === 4 && wire === 2) {
      const n = len / 4, f = new Float32Array(n);
      for (let i = 0; i < n; i++) f[i] = new DataView(buf.buffer, buf.byteOffset + pos + i * 4).getFloat32(0, true);
      t.floats = f;
    }
  });
  return t;
}

const buf = new Uint8Array(readFileSync(URL));
const tensors = [];
forEachField(buf, 0, buf.length, (f, w, pos, len) => {
  if (f !== 7 || w !== 2) return;
  forEachField(buf, pos, pos + len, (gf, gw, gpos, glen) => {
    if (gf !== 5 || gw !== 2) return;
    let name = "";
    forEachField(buf, gpos, gpos + glen, (tf, tw, tpos, tlen) => {
      if (tf === 8 && tw === 2) name = new TextDecoder().decode(buf.slice(tpos, tpos + tlen));
    });
    tensors.push(parseTensor(buf, gpos, gpos + glen, name));
  });
});
const byName = new Map(tensors.map((t) => [t.name, t]));
console.log("Initialisierer:", [...byName.keys()].join(", "));

// Schichten
const layers = [];
for (const t of byName.values()) {
  const m = t.name.match(/mlp\.(\d+)\.weight/);
  if (m && t.dims.length === 2) layers.push({ idx: +m[1], w: t });
}
layers.sort((a, b) => a.idx - b.idx);
for (const L of layers) {
  L.b = byName.get(L.w.name.replace(/\.weight$/, ".bias"));
  console.log(`Layer mlp.${L.idx}: [${L.w.dims.join(",")}] bias=${!!L.b}`);
}

// Normalizer
const meanT = [...byName.values()].find((t) => /normalizer.*mean|_mean/i.test(t.name) && t.floats.length === OBS_DIM);
const stdT = [...byName.values()].find((t) => t !== meanT && t.floats.length === OBS_DIM && (/div|std/i.test(t.name) || t.name.includes("onnx::")));
console.log("mean:", meanT?.name, "std:", stdT?.name);
if (!meanT || !stdT) { console.log("KEIN NORMALIZER GEFUNDEN"); process.exit(1); }
console.log("mean[0..4]:", Array.from(meanT.floats.slice(0, 4)).map((v) => v.toFixed(3)).join(", "));
console.log("std[0..4]:", Array.from(stdT.floats.slice(0, 4)).map((v) => v.toFixed(3)).join(", "));

// Folden
const folded = layers.map((L) => null);
{
  const w = layers[0].w, b = layers[0].b;
  const outDim = w.dims[0], inDim = w.dims[1];
  const W = new Float32Array(outDim * inDim), B = new Float32Array(outDim);
  for (let i = 0; i < outDim; i++) {
    let corr = 0;
    for (let k = 0; k < inDim; k++) {
      const wv = w.floats[i * inDim + k];
      W[i * inDim + k] = wv / stdT.floats[k];
      corr += wv * meanT.floats[k] / stdT.floats[k];
    }
    B[i] = b.floats[i] - corr;
  }
  folded[0] = { W, B, outDim, inDim };
}
for (let l = 1; l < layers.length; l++) {
  const w = layers[l].w, b = layers[l].b;
  folded[l] = { W: w.floats, B: b.floats, outDim: w.dims[0], inDim: w.dims[1] };
}

function mlpForward(obs, act) {
  let src = obs;
  for (let l = 0; l < folded.length; l++) {
    const { W, B, outDim, inDim } = folded[l];
    const buf2 = new Float32Array(outDim);
    for (let i = 0; i < outDim; i++) {
      let s = B[i];
      const base = i * inDim;
      for (let k = 0; k < inDim; k++) s += W[base + k] * src[k];
      buf2[i] = s;
    }
    const isLast = l === folded.length - 1;
    if (!isLast) {
      if (act === "tanh") for (let i = 0; i < buf2.length; i++) buf2[i] = Math.tanh(buf2[i]);
      else if (act === "relu") for (let i = 0; i < buf2.length; i++) if (buf2[i] < 0) buf2[i] = 0;
      else if (act === "elu") for (let i = 0; i < buf2.length; i++) if (buf2[i] < 0) buf2[i] = Math.expm1(buf2[i]);
      else if (act === "sigmoid") for (let i = 0; i < buf2.length; i++) buf2[i] = 1 / (1 + Math.exp(-buf2[i]));
      else if (act === "silu") for (let i = 0; i < buf2.length; i++) { const x = buf2[i]; buf2[i] = x / (1 + Math.exp(-x)); }
    }
    src = buf2;
  }
  return src;
}

// ONNX-Session
const session = await ort.InferenceSession.create(buf, {
  executionProviders: ["wasm"], graphOptimizationLevel: "all",
});
console.log("ONNX inputs:", session.inputNames, "outputs:", session.outputNames);

const N = 6;
const errs = { tanh: 0, relu: 0, elu: 0, sigmoid: 0, silu: 0 };
for (const act of Object.keys(errs)) {
  let e = 0;
  for (let n = 0; n < N; n++) {
    const obs = new Float32Array(OBS_DIM);
    for (let k = 0; k < OBS_DIM; k++) obs[k] = meanT.floats[k] + (Math.random() * 2 - 1) * 0.5;
    const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", obs, [1, OBS_DIM]) };
    const r = await session.run(feeds);
    const ref = r[session.outputNames[0]].data;
    const out = mlpForward(obs, act);
    let de = 0;
    for (let k = 0; k < out.length; k++) de += Math.abs(out[k] - ref[k]);
    e += de / out.length;
    if (n === 0 && act === "tanh") {
      console.log("ONNX out[0..5]:", Array.from(ref.slice(0, 5)).map((v) => v.toFixed(3)).join(", "));
      console.log("MLP(tanh) out :", Array.from(out.slice(0, 5)).map((v) => v.toFixed(3)).join(", "));
    }
  }
  errs[act] = e / N;
}
console.log("Kalibrierung:", JSON.stringify(Object.fromEntries(Object.entries(errs).map(([k, v]) => [k, +v.toFixed(4)]))));

// ── Hypothesen-Test: Divisor-Deutung von onnx::Div_24 ──
const D = stdT.floats;
const variants = {
  "W/D": (wv, d, k) => wv / d[k],
  "W*D": (wv, d, k) => wv * d[k],
  "W/sqrt(D)": (wv, d, k) => wv / Math.sqrt(d[k]),
  "W*sqrt(D)": (wv, d, k) => wv * Math.sqrt(d[k]),
};
for (const [name, f] of Object.entries(variants)) {
  for (const act of ["elu", "tanh"]) {
    // Folding mit Variante
    const w0 = layers[0].w, b0 = layers[0].b;
    const outDim0 = w0.dims[0], inDim0 = w0.dims[1];
    const W = new Float32Array(outDim0 * inDim0), B = new Float32Array(outDim0);
    for (let i = 0; i < outDim0; i++) {
      let corr = 0;
      for (let k = 0; k < inDim0; k++) {
        const wv = w0.floats[i * inDim0 + k];
        W[i * inDim0 + k] = f(wv, D, k);
        corr += wv * meanT.floats[k] * (name === "W*D" ? D[k] : name === "W*sqrt(D)" ? Math.sqrt(D[k]) : name === "W/sqrt(D)" ? Math.sqrt(D[k]) : D[k]);
      }
      B[i] = b0.floats[i] - corr;
    }
    const save = folded[0];
    folded[0] = { W, B, outDim: outDim0, inDim: inDim0 };
    let e = 0;
    for (let n = 0; n < N; n++) {
      const obs = new Float32Array(OBS_DIM);
      for (let k = 0; k < OBS_DIM; k++) obs[k] = meanT.floats[k] + (Math.random() * 2 - 1) * 0.5;
      const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", obs, [1, OBS_DIM]) };
      const r = await session.run(feeds);
      const ref = r[session.outputNames[0]].data;
      const out = mlpForward(obs, act);
      let de = 0;
      for (let k = 0; k < out.length; k++) de += Math.abs(out[k] - ref[k]);
      e += de / out.length;
    }
    folded[0] = save;
    console.log(`${name} + ${act}: ${(e / N).toFixed(5)}`);
  }
}

// ── Bisektion: (A) ONNX(norm(x)) vs rohes MLP(x_norm) ──
{
  const w0 = layers[0].w, b0 = layers[0].b;
  const outDim0 = w0.dims[0], inDim0 = w0.dims[1];
  const Wraw = w0.floats, Braw = b0.floats;
  folded[0] = { W: Wraw, B: Braw, outDim: outDim0, inDim: inDim0 };
  let eA = 0;
  for (let n = 0; n < 5; n++) {
    const obs = new Float32Array(OBS_DIM);
    for (let k = 0; k < OBS_DIM; k++) obs[k] = meanT.floats[k] + (Math.random() * 2 - 1) * 0.5;
    const norm = new Float32Array(OBS_DIM);
    for (let k = 0; k < OBS_DIM; k++) norm[k] = (obs[k] - meanT.floats[k]) / D[k];
    const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", norm, [1, OBS_DIM]) };
    const r = await session.run(feeds);
    const ref = r[session.outputNames[0]].data;
    const out = mlpForward(norm, "elu");
    let de = 0;
    for (let k = 0; k < out.length; k++) de += Math.abs(out[k] - ref[k]);
    eA += de / out.length;
  }
  console.log("A) roh-MLP(norm(x)) vs ONNX(norm(x)):", (eA / 5).toFixed(6));
}

// ── Zero-Input-Test: ONNX(0) vs MLP(0) + Layer-Statistiken ──
{
  const zero = new Float32Array(OBS_DIM);
  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", zero, [1, OBS_DIM]) };
  const r = await session.run(feeds);
  const ref = r[session.outputNames[0]].data;
  const out = mlpForward(zero, "elu");
  console.log("ONNX(0):", Array.from(ref.slice(0, 6)).map((v) => v.toFixed(3)).join(", "));
  console.log("MLP(0): ", Array.from(out.slice(0, 6)).map((v) => v.toFixed(3)).join(", "));
  // Layer-Statistiken meines Netzes bei 0-Input
  let src = zero;
  for (let l = 0; l < folded.length; l++) {
    const { W, B, outDim, inDim } = folded[l];
    const buf2 = new Float32Array(outDim);
    for (let i = 0; i < outDim; i++) {
      let s = B[i];
      const base = i * inDim;
      for (let k = 0; k < inDim; k++) s += W[base + k] * src[k];
      buf2[i] = s;
    }
    const pre = buf2;
    if (l < folded.length - 1) for (let i = 0; i < buf2.length; i++) buf2[i] = Math.expm1(Math.min(0, buf2[i])) + Math.max(0, buf2[i]);
    console.log(`L${l}: pre mean=${(pre.reduce((a,b)=>a+b,0)/outDim).toFixed(3)} absmax=${Math.max(...pre.map(Math.abs)).toFixed(2)} W-absmean=${(W.reduce((a,b)=>a+Math.abs(b),0)/W.length).toFixed(4)}`);
    src = buf2;
  }
}
