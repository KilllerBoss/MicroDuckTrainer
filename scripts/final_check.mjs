// Finaler Validierungstest: policy.ts-Logik (theta-Layout + MlpPolicy-Forward)
// gegen ort mit ELU aus dem Graph.
import * as ort from "onnxruntime-web/wasm";
import { readFileSync } from "fs";

ort.env.wasm.wasmPaths = "file:///home/z/my-project/node_modules/onnxruntime-web/dist/";
ort.env.wasm.numThreads = 1;

const MODEL = "/home/z/my-project/public/policies/BEST_alpha_walking.onnx";
const OBS_DIM = 61, ACT_DIM = 14;

function readVarint(b, p) { let r = 0, s = 0; for (;;) { const x = b[p++]; r += (x & 0x7f) * Math.pow(2, s); if (!(x & 0x80)) break; s += 7; } return [r, p]; }
function forEachField(b, start, end, cb) {
  let pos = start;
  while (pos < end) {
    const [key, p1] = readVarint(b, pos);
    const f = key >>> 3, w = key & 7;
    if (w === 2) { const [len, q] = readVarint(b, p1); cb(f, w, q, len); pos = q + len; }
    else { let p2; if (w === 0) p2 = readVarint(b, p1)[1]; else if (w === 1) p2 = p1 + 8; else if (w === 5) p2 = p1 + 4; else return; cb(f, w, p1, p2 - p1); pos = p2; }
  }
}
function parseTensor(b, start, len, name) {
  const t = { name, dims: [], floats: new Float32Array(0) };
  forEachField(b, start, start + len, (f, w, pos, l) => {
    if (f === 1 && w === 0) t.dims.push(readVarint(b, pos)[0]);
    else if (f === 1 && w === 2) { let p = pos; while (p < pos + l) { const [v, np] = readVarint(b, p); t.dims.push(v); p = np; } }
    else if (f === 9 && w === 2) t.floats = new Float32Array(b.slice(pos, pos + l).buffer, 0, l / 4);
  });
  return t;
}

const buf = new Uint8Array(readFileSync(MODEL));
const tensors = [];
forEachField(buf, 0, buf.length, (f, w, pos, len) => {
  if (f !== 7 || w !== 2) return;
  forEachField(buf, pos, pos + len, (gf, gw, gpos, glen) => {
    if (gf !== 5 || gw !== 2) return;
    let name = "";
    forEachField(buf, gpos, gpos + glen, (tf, tw, tpos, tlen) => {
      if (tf === 8 && tw === 2) name = new TextDecoder().decode(buf.slice(tpos, tpos + tlen));
    });
    tensors.push(parseTensor(buf, gpos, glen, name));
  });
});
const byName = new Map(tensors.map((t) => [t.name, t]));
console.log("Tensoren:", byName.size);

// Graph-Ops lesen → Aktivierung
let firstAct = "tanh";
forEachField(buf, 0, buf.length, (f, w, pos, len) => {
  if (f !== 7 || w !== 2) return;
  forEachField(buf, pos, pos + len, (gf, gw, gpos, glen) => {
    if (gf !== 1 || gw !== 2) return; // node
    let op = "";
    forEachField(buf, gpos, gpos + glen, (nf, nw, npos, nlen) => {
      if (nf === 4 && nw === 2) op = new TextDecoder().decode(buf.slice(npos, npos + nlen));
    });
    if (["Elu", "Tanh", "Relu", "Sigmoid", "Silu"].includes(op) && firstAct === "tanh") firstAct = op.toLowerCase();
  });
});
console.log("Aktivierung aus Graph:", firstAct);

// theta nach policy.ts-Logik
const layers = [];
for (const t of byName.values()) {
  const m = t.name.match(/mlp\.(\d+)\.weight/);
  if (m && t.dims.length === 2) layers.push({ idx: +m[1], w: t });
}
layers.sort((a, b) => a.idx - b.idx);
for (const L of layers) L.b = byName.get(L.w.name.replace(/\.weight$/, ".bias"));
const meanT = [...byName.values()].find((t) => /_mean/i.test(t.name));
const stdT = [...byName.values()].find((t) => t !== meanT && /Div/i.test(t.name));
const mean = meanT.floats, std = stdT.floats;
const hs = layers.map((L) => L.w.dims[0]).slice(0, -1);
const dims = [OBS_DIM, ...hs, ACT_DIM];
let theta = new Float32Array(dims.slice(0, -1).reduce((a, d, i) => a + dims[i + 1] * d + dims[i + 1], 0));
{
  let off = 0;
  for (let l = 0; l < layers.length; l++) {
    const { w, b } = layers[l];
    const outDim = w.dims[0], inDim = w.dims[1];
    for (let i = 0; i < outDim; i++) {
      let corr = 0;
      for (let k = 0; k < inDim; k++) {
        const wv = w.floats[i * inDim + k];
        if (l === 0) { theta[off + i * inDim + k] = wv / std[k]; corr += wv * mean[k] / std[k]; }
        else theta[off + i * inDim + k] = wv;
      }
      theta[off + outDim * inDim + i] = l === 0 ? b.floats[i] - corr : b.floats[i];
    }
    off += outDim * inDim + outDim;
  }
}

// Forward exakt wie MlpPolicy (slices, act)
function forward(obs) {
  let src = obs;
  let off = 0;
  for (let l = 0; l < dims.length - 1; l++) {
    const inDim = dims[l], outDim = dims[l + 1];
    const w = theta.subarray(off, off + outDim * inDim); off += outDim * inDim;
    const b = theta.subarray(off, off + outDim); off += outDim;
    const buf2 = new Float32Array(outDim);
    for (let i = 0; i < outDim; i++) {
      let s = b[i];
      const base = i * inDim;
      for (let k = 0; k < inDim; k++) s += w[base + k] * src[k];
      buf2[i] = s;
    }
    if (l < dims.length - 2) {
      if (firstAct === "elu") for (let i = 0; i < outDim; i++) if (buf2[i] < 0) buf2[i] = Math.expm1(buf2[i]);
      else if (firstAct === "relu") for (let i = 0; i < outDim; i++) if (buf2[i] < 0) buf2[i] = 0;
      else for (let i = 0; i < outDim; i++) buf2[i] = Math.tanh(buf2[i]);
    }
    src = buf2;
  }
  return src;
}

const session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
let e = 0;
const N = 8;
for (let n = 0; n < N; n++) {
  const obs = new Float32Array(OBS_DIM);
  for (let k = 0; k < OBS_DIM; k++) obs[k] = mean[k] + (Math.random() * 2 - 1) * 0.5;
  const r = await session.run({ obs: new ort.Tensor("float32", obs, [1, OBS_DIM]) });
  const ref = r.actions.data;
  const out = forward(obs);
  let de = 0;
  for (let k = 0; k < out.length; k++) de += Math.abs(out[k] - ref[k]);
  e += de / out.length;
}
console.log("policy.ts-Logik (ELU, gefaltet) Fehler vs ort:", (e / N).toExponential(2));
