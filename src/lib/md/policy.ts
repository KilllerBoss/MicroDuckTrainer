// ── MicroDuck Trainer v2.0 – Policies (ONNX + MLP) ──────────────────────────
// ONNX läuft über onnxruntime-web (wasm-EP, numThreads=1 – kein COOP/COEP nötig).
// Der MLP dient dem ES-Training (reines JS, schnell genug); der Warm-Start
// extrahiert die Gewichte offline aus der ONNX-Datei (minimaler Protobuf-Parser).

import * as ort from "onnxruntime-web/wasm";
import type { ModelMeta } from "./models";

// WASM-Binary statisch aus /wasm/ (Static Export; das JS-Glue ist im Bundle).
ort.env.wasm.wasmPaths = { wasm: "/wasm/ort-wasm-simd-threaded.wasm" };
ort.env.wasm.numThreads = 1;

export class OnnxPolicy {
  session: ort.InferenceSession;
  obsDim: number;
  actionDim: number;

  private constructor(session: ort.InferenceSession, obsDim: number, actionDim: number) {
    this.session = session;
    this.obsDim = obsDim;
    this.actionDim = actionDim;
  }

  static async create(url: string): Promise<OnnxPolicy> {
    const session = await ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    // Obs-/Aktionsgröße ergibt sich implizit aus dem übergebenen Tensor;
    // die Session validiert die Form selbst.
    return new OnnxPolicy(session, 0, 0);
  }

  async forward(obs: Float32Array): Promise<Float32Array> {
    const feeds = { [this.session.inputNames[0]]: new ort.Tensor("float32", obs, [1, obs.length]) };
    const out = await this.session.run(feeds);
    return out[this.session.outputNames[0]].data as Float32Array;
  }
}

// ── Minimaler MLP (obs → hidden → action, Tanh) ─────────────────────────────

export interface MlpLayout {
  obsDim: number;
  hidden: number;
  actionDim: number;
}

export class MlpPolicy {
  layout: MlpLayout;
  theta: Float32Array; // [W1 (h×o), b1 (h), W2 (a×h), b2 (a)]
  private w1: Float32Array;
  private b1: Float32Array;
  private w2: Float32Array;
  private b2: Float32Array;
  private hiddenBuf: Float32Array;

  constructor(layout: MlpLayout, theta: Float32Array) {
    const { obsDim: o, hidden: h, actionDim: a } = layout;
    const expected = h * o + h + a * h + a;
    if (theta.length !== expected) {
      throw new Error(`Theta-Größe ${theta.length} ≠ erwartet ${expected}`);
    }
    this.layout = layout;
    this.theta = theta;
    this.w1 = theta.slice(0, h * o);
    this.b1 = theta.slice(h * o, h * o + h);
    this.w2 = theta.slice(h * o + h, h * o + h + a * h);
    this.b2 = theta.slice(h * o + h + a * h);
    this.hiddenBuf = new Float32Array(h);
  }

  static thetaSize(o: number, h: number, a: number): number {
    return h * o + h + a * h + a;
  }

  static randomTheta(o: number, h: number, a: number, scale = 0.3): Float32Array {
    const n = MlpPolicy.thetaSize(o, h, a);
    const t = new Float32Array(n);
    for (let i = 0; i < n; i++) t[i] = (Math.random() * 2 - 1) * scale;
    return t;
  }

  forward(obs: Float32Array, out?: Float32Array): Float32Array {
    const { obsDim: o, hidden: h, actionDim: a } = this.layout;
    const hb = this.hiddenBuf;
    for (let j = 0; j < h; j++) {
      let s = this.b1[j];
      const base = j * o;
      for (let k = 0; k < o; k++) s += this.w1[base + k] * obs[k];
      hb[j] = Math.tanh(s);
    }
    const act = out ?? new Float32Array(a);
    for (let i = 0; i < a; i++) {
      let s = this.b2[i];
      const base = i * h;
      for (let k = 0; k < h; k++) s += this.w2[base + k] * hb[k];
      act[i] = Math.tanh(s);
    }
    return act;
  }
}

// ── ONNX-Gewichtsextraktion (offline, minimaler Protobuf-Parser) ─────────────

interface TensorProto {
  name: string;
  dims: number[];
  floats: Float32Array;
}

function readVarint(buf: Uint8Array, pos: number): [number, number] {
  let result = 0, shift = 0;
  for (;;) {
    const b = buf[pos++];
    result += (b & 0x7f) * Math.pow(2, shift);
    if ((b & 0x80) === 0) break;
    shift += 7;
  }
  return [result, pos];
}

function skipField(buf: Uint8Array, pos: number, wireType: number): number {
  switch (wireType) {
    case 0: return readVarint(buf, pos)[1];
    case 1: return pos + 8;
    case 2: {
      const [len, p] = readVarint(buf, pos);
      return p + len;
    }
    case 5: return pos + 4;
    default: throw new Error(`Protobuf: unbekannter Wire-Type ${wireType}`);
  }
}

/** Durchläuft eine Protobuf-Nachricht und ruft cb(fieldNo, wireType, valuePos, valueLen) auf. */
function forEachField(
  buf: Uint8Array, start: number, end: number,
  cb: (field: number, wire: number, pos: number, len: number) => void,
): void {
  let pos = start;
  while (pos < end) {
    const [key, p1] = readVarint(buf, pos);
    const field = key >>> 3, wire = key & 7;
    if (wire === 2) {
      const [len, p2] = readVarint(buf, p1);
      cb(field, wire, p2, len);
      pos = p2 + len;
    } else {
      const p2 = skipField(buf, p1, wire);
      cb(field, wire, p1, p2 - p1);
      pos = p2;
    }
  }
}

function parseTensor(buf: Uint8Array, start: number, end: number, name: string): TensorProto {
  const t: TensorProto = { name, dims: [], floats: new Float32Array(0) };
  forEachField(buf, start, end, (field, wire, pos, len) => {
    if (field === 1 && wire === 2) {
      // dims: packed int64
      let p = pos;
      while (p < pos + len) {
        const [v, np] = readVarint(buf, p);
        t.dims.push(v);
        p = np;
      }
    } else if (field === 9 && wire === 2) {
      // raw_data
      t.floats = new Float32Array(buf.slice(pos, pos + len).buffer, 0, len / 4);
    } else if (field === 4 && wire === 2) {
      // float_data (packed)
      const n = len / 4;
      const f = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const dv = new DataView(buf.buffer, buf.byteOffset + pos + i * 4);
        f[i] = dv.getFloat32(0, true);
      }
      t.floats = f;
    }
  });
  return t;
}

/**
 * Lädt eine ONNX-Datei und extrahiert ein 2-Layer-MLP (obs→hidden→action).
 * Layout-Erkennung über die Tensorformen: W1=[hidden,obsDim], W2=[actDim,hidden].
 * Gibt null zurück, wenn kein passendes Layout gefunden wird (→ Random-Init).
 */
export async function extractMlpFromOnnx(
  url: string,
  obsDim: number,
  actionDim: number,
): Promise<{ theta: Float32Array; layout: MlpLayout; source: string } | null> {
  try {
    const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
    const tensors: TensorProto[] = [];
    // ModelProto: field 7 = graph
    forEachField(buf, 0, buf.length, (field, wire, pos, len) => {
      if (field !== 7 || wire !== 2) return;
      // GraphProto: field 5 = initializer
      forEachField(buf, pos, pos + len, (gf, gw, gpos, glen) => {
        if (gf !== 5 || gw !== 2) return;
        // TensorProto: field 8 = name (string)
        let name = "";
        forEachField(buf, gpos, gpos + glen, (tf, tw, tpos, tlen) => {
          if (tf === 8 && tw === 2) {
            name = new TextDecoder().decode(buf.slice(tpos, tpos + tlen));
          }
        });
        tensors.push(parseTensor(buf, gpos, gpos + glen, name));
      });
    });
    const byName = new Map(tensors.map((t) => [t.name, t]));
    // Kandidaten: Tensoren mit 2 Dims = Gewichtsmatrizen
    const weights = tensors.filter((t) => t.dims.length === 2);
    for (const w1 of weights) {
      if (w1.dims[1] !== obsDim || w1.floats.length !== w1.dims[0] * w1.dims[1]) continue;
      const hidden = w1.dims[0];
      if (hidden <= 0 || hidden > 512) continue;
      const b1 = byName.get(w1.name.replace(/weight/i, "bias"));
      if (!b1 || b1.dims.length !== 1 || b1.dims[0] !== hidden) continue;
      const w2 = weights.find(
        (t) => t !== w1 && t.dims[0] === actionDim && t.dims[1] === hidden,
      );
      if (!w2) continue;
      const b2 = byName.get(w2.name.replace(/weight/i, "bias"));
      if (!b2 || b2.dims[0] !== actionDim) continue;
      const layout: MlpLayout = { obsDim, hidden, actionDim };
      const theta = new Float32Array(MlpPolicy.thetaSize(obsDim, hidden, actionDim));
      theta.set(w1.floats, 0);
      theta.set(b1.floats, hidden * obsDim);
      theta.set(w2.floats, hidden * obsDim + hidden);
      theta.set(b2.floats, hidden * obsDim + hidden + actionDim * hidden);
      return { theta, layout, source: w1.name };
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Warm-Start: versucht, die Standard-Policy als MLP zu laden; sonst Zufall.
 */
export async function warmStartMlp(meta: ModelMeta): Promise<{
  theta: Float32Array;
  layout: MlpLayout;
  warm: boolean;
}> {
  const policy = meta.policies.find((p) => p.id === meta.defaultOnnx) ?? meta.policies[0];
  if (policy) {
    const r = await extractMlpFromOnnx(policy.url, meta.obsDim, meta.actionDim);
    if (r) return { ...r, warm: true };
  }
  return {
    theta: MlpPolicy.randomTheta(meta.obsDim, 32, meta.actionDim),
    layout: { obsDim: meta.obsDim, hidden: 32, actionDim: meta.actionDim },
    warm: false,
  };
}
