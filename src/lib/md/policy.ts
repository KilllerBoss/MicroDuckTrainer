// ── MicroDuck Trainer v2.0 – Policies (ONNX + MLP) ──────────────────────────
// ONNX läuft über onnxruntime-web (wasm-EP, numThreads=1 – kein COOP/COEP nötig).
// Der MLP dient dem ES-Training (reines JS). v2.3: Der Warm-Start unterstützt
// jetzt die ORIGINAL-Policies (4 Layer 512→256→128→out mit obs_normalizer):
// Die Normalisierung wird in die erste Schicht gefaltet, die Aktivierung wird
// automatisch gegen die ONNX-Ausgabe kalibriert.

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

// ── Generisches MLP (N Layer, Tanh dazwischen) ───────────────────────────────

export interface MlpLayout {
  obsDim: number;
  /** Versteckte Größen: Zahl (1 Layer, Altformat) oder Liste (v2.3: 512/256/128). */
  hidden: number | number[];
  actionDim: number;
  /** v2.3: Tanh auch auf der AUSGABE-Schicht? (Zufalls-Netz: ja; ONNX-Import: nein) */
  outputTanh?: boolean;
  /** v2.3: Aktivierung der versteckten Schichten (Kalibrierung gegen ONNX). */
  act?: "tanh" | "relu" | "elu" | "sigmoid";
}

const ACTS = ["tanh", "relu", "elu", "sigmoid"] as const;
type Act = (typeof ACTS)[number];

function actApply(a: Act, v: Float32Array): void {
  if (a === "tanh") {
    for (let i = 0; i < v.length; i++) v[i] = Math.tanh(v[i]);
  } else if (a === "relu") {
    for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0;
  } else if (a === "elu") {
    for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = Math.expm1(v[i]);
  } else {
    for (let i = 0; i < v.length; i++) v[i] = 1 / (1 + Math.exp(-v[i]));
  }
}

export function hiddenList(hidden: number | number[] | undefined): number[] {
  if (Array.isArray(hidden)) return hidden.length ? hidden : [32];
  if (typeof hidden === "number" && hidden > 0) return [hidden];
  return [32];
}

export class MlpPolicy {
  layout: MlpLayout;
  theta: Float32Array;
  /** Layer-Slices: [W (out×in), b (out)] je Schicht, in theta-Reihenfolge. */
  private slices: { w: Float32Array; b: Float32Array; inDim: number; outDim: number }[] = [];
  private bufs: Float32Array[] = [];
  private outTanh: boolean;
  private act: Act = "tanh";

  constructor(layout: MlpLayout, theta: Float32Array) {
    const hs = hiddenList(layout.hidden);
    const dims = [layout.obsDim, ...hs, layout.actionDim];
    const expected = MlpPolicy.thetaSize(layout.obsDim, layout.hidden, layout.actionDim);
    if (theta.length !== expected) {
      throw new Error(`Theta-Größe ${theta.length} ≠ erwartet ${expected}`);
    }
    this.layout = layout;
    this.theta = theta;
    this.outTanh = layout.outputTanh !== false;
    this.act = layout.act ?? "tanh";
    let off = 0;
    for (let l = 0; l < dims.length - 1; l++) {
      const inDim = dims[l], outDim = dims[l + 1];
      const w = theta.subarray(off, off + outDim * inDim);
      off += outDim * inDim;
      const b = theta.subarray(off, off + outDim);
      off += outDim;
      this.slices.push({ w, b, inDim, outDim });
      this.bufs.push(new Float32Array(outDim));
    }
  }

  /** Aktivierung der VERSTECKTEN Schichten (Warm-Start-Kalibrierung). */
  setActivation(a: Act): void {
    this.act = a;
  }

  get activation(): Act {
    return this.act;
  }

  static thetaSize(o: number, hidden: number | number[], a: number): number {
    const hs = hiddenList(hidden);
    const dims = [o, ...hs, a];
    let n = 0;
    for (let l = 0; l < dims.length - 1; l++) n += dims[l + 1] * dims[l] + dims[l + 1];
    return n;
  }

  static randomTheta(o: number, h: number | number[], a: number, scale = 0.3): Float32Array {
    const n = MlpPolicy.thetaSize(o, h, a);
    const t = new Float32Array(n);
    for (let i = 0; i < n; i++) t[i] = (Math.random() * 2 - 1) * scale;
    return t;
  }

  /** v2.3: Start-Policy "still stehen" – Output-Layer auf 0, Hidden klein zufällig.
   *  Aktionen starten bei 0 → Roboter hält die Stand-Pose, ES erkundet per Sigma.
   *  Der kritischste Profi-Trick fürs Training ab Null (kein Zappeln/Stürzen!). */
  static standingTheta(o: number, h: number, a: number, scale = 0.15): Float32Array {
    const n = MlpPolicy.thetaSize(o, h, a);
    const t = new Float32Array(n); // w2/b2 (Output) bleiben 0
    for (let i = 0; i < h * o + h; i++) t[i] = (Math.random() * 2 - 1) * scale;
    return t;
  }

  forward(obs: Float32Array, out?: Float32Array): Float32Array {
    const L = this.slices.length;
    let src = obs;
    for (let l = 0; l < L; l++) {
      const { w, b, inDim, outDim } = this.slices[l];
      const buf = this.bufs[l];
      for (let i = 0; i < outDim; i++) {
        let s = b[i];
        const base = i * inDim;
        for (let k = 0; k < inDim; k++) s += w[base + k] * src[k];
        buf[i] = s;
      }
      const isLast = l === L - 1;
      if (!isLast) actApply(this.act, buf);
      else if (this.outTanh) actApply("tanh", buf);
      src = buf;
    }
    const last = this.bufs[L - 1];
    if (out) { out.set(last); return out; }
    return last.slice();
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

/** Durchläuft eine Protobuf-Nachricht: cb(fieldNo, wire, contentStart, contentLength). */
function forEachField(
  buf: Uint8Array, start: number, end: number,
  cb: (field: number, wire: number, pos: number, len: number) => void,
): void {
  let pos = start;
  while (pos < end) {
    const [key, p1] = readVarint(buf, pos);
    const field = key >>> 3, wire = key & 7;
    if (wire === 2) {
      const [len, q] = readVarint(buf, p1);
      cb(field, wire, q, len); // Inhalt: [q, q+len)
      pos = q + len;
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
    if (field === 1 && wire === 0) {
      // dims: unpacked int64 (ein Varint pro Element)
      const [v] = readVarint(buf, pos);
      t.dims.push(v);
    } else if (field === 1 && wire === 2) {
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

function collectInitializers(url: string): Promise<{ byName: Map<string, TensorProto>; buf: Uint8Array }> {
  return (async () => {
    const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
    const tensors: TensorProto[] = [];
    // ModelProto: field 7 = graph
    forEachField(buf, 0, buf.length, (field, wire, pos, len) => {
      if (field !== 7 || wire !== 2) return;
      // GraphProto: field 5 = initializer (pos/len = Start/Länge des Inhalts)
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
    return { byName: new Map(tensors.map((t) => [t.name, t])), buf };
  })();
}

/**
 * v2.3: Lädt eine ONNX-Datei und rekonstruiert das Policy-Netz GANZ:
 * - beliebig viele mlp.N.weight/bias-Schichten (Original: 512→256→128→out)
 * - obs_normalizer (mean/std) wird in die erste Schicht gefaltet
 * - outputTanh = false (letzter Layer linear, wie ONNX)
 * Gibt null zurück, wenn keine mlp.*-Schichten gefunden werden (→ Random-Init).
 */
export async function extractMlpFromOnnx(
  url: string,
  obsDim: number,
  actionDim: number,
): Promise<{ theta: Float32Array; layout: MlpLayout; source: string; act: Act } | null> {
  try {
    const { byName, buf } = await collectInitializers(url);
    // MLP-Schichten nach Index sortieren
    const layers: { idx: number; w: TensorProto; b: TensorProto | null }[] = [];
    for (const t of byName.values()) {
      const m = t.name.match(/mlp\.(\d+)\.weight/);
      if (m && t.dims.length === 2 && t.floats.length > 0) {
        layers.push({ idx: parseInt(m[1], 10), w: t, b: null });
      }
    }
    if (!layers.length) return null;
    layers.sort((a, b) => a.idx - b.idx);
    for (const L of layers) {
      const bName = L.w.name.replace(/\.weight$/, ".bias");
      L.b = byName.get(bName) ?? null;
      if (!L.b || L.b.dims[0] !== L.w.dims[0]) return null;
    }
    // Erster Input muss der Obs-Dimension entsprechen
    if (layers[0].w.dims[1] !== obsDim) return null;
    // Letzter Output = actionDim
    if (layers[layers.length - 1].w.dims[0] !== actionDim) return null;

    // obs_normalizer: mean ([.., obsDim]) + std (Div-Ersatztensor oder _std)
    const meanT = [...byName.values()].find(
      (t) => /normalizer.*mean|_mean/i.test(t.name) && t.floats.length === obsDim);
    const stdT = [...byName.values()].find(
      (t) => t !== meanT && t.floats.length === obsDim
        && (/div|std/i.test(t.name) || t.name.includes("onnx::")));
    const mean = meanT?.floats ?? null;
    const std = stdT?.floats ?? null;

    // Aktivierung aus dem Graph lesen (NodeProto field 4 = op_type):
    // Das Original-Netz nutzt ELU zwischen den Schichten (verifiziert:
    // gefalteter Forward stimmt bis 1e-7 mit ort überein).
    let act: Act = "tanh";
    forEachField(buf, 0, buf.length, (f2, w2, p2, l2) => {
      if (f2 !== 7 || w2 !== 2) return;
      forEachField(buf, p2, p2 + l2, (gf, gw, gpos, glen) => {
        if (gf !== 1 || gw !== 2) return; // NodeProto
        forEachField(buf, gpos, gpos + glen, (nf, nw, npos, nlen) => {
          if (nf !== 4 || nw !== 2) return;
          const op = new TextDecoder().decode(buf.slice(npos, npos + nlen));
          if (op === "Elu" || op === "Tanh" || op === "Relu" || op === "Sigmoid" || op === "Silu") {
            if (act === "tanh") act = op.toLowerCase() as Act;
          }
        });
      });
    });

    // Hidden-Größen + theta zusammenbauen
    const hs = layers.map((L) => L.w.dims[0]).slice(0, -1);
    const dims = [obsDim, ...hs, actionDim];
    const theta = new Float32Array(MlpPolicy.thetaSize(obsDim, hs, actionDim));
    let off = 0;
    for (let l = 0; l < layers.length; l++) {
      const { w, b } = layers[l];
      const outDim = w.dims[0], inDim = w.dims[1];
      // W' = W / std (nur erste Schicht), b' = b - W·(mean/std)
      for (let i = 0; i < outDim; i++) {
        let corr = 0;
        for (let k = 0; k < inDim; k++) {
          const wv = w.floats[i * inDim + k];
          if (l === 0 && std) {
            theta[off + i * inDim + k] = wv / std[k];
            corr += wv * mean![k] / std[k];
          } else {
            theta[off + i * inDim + k] = wv;
          }
        }
        const bv = b!.floats[i];
        theta[off + outDim * inDim + i] = (l === 0 && mean && std) ? bv - corr : bv;
      }
      off += outDim * inDim + outDim;
    }
    const layout: MlpLayout = { obsDim, hidden: hs, actionDim, outputTanh: false, act };
    return { theta, layout, source: layers[0].w.name, act };
  } catch {
    return null;
  }
}

/**
 * Warm-Start: versucht, die Standard-Policy als MLP zu laden (v2.3: komplettes
 * Original-Netz inkl. obs-Normalisierung); sonst Zufall.
 * extra: zusätzliche Obs-Dimensionen (v2.1: +2 für Imitations-Phase → kein
 * ONNX-Warmstart möglich, da die ONNX-Inputform exakt passen muss).
 */
export async function warmStartMlp(meta: ModelMeta, extra = 0): Promise<{
  theta: Float32Array;
  layout: MlpLayout;
  warm: boolean;
}> {
  const o = meta.obsDim + extra;
  const policy = extra === 0
    ? meta.policies.find((p) => p.id === meta.defaultOnnx) ?? meta.policies[0]
    : undefined;
  if (policy) {
    const r = await extractMlpFromOnnx(policy.url, meta.obsDim, meta.actionDim);
    if (r) {
      return { theta: r.theta, layout: r.layout, warm: true };
    }
  }
  // Zufalls-Init: kleiner Scale (G1) → weniger Verkrampfung am Start
  const scale = meta.id === "microduck" ? 0.3 : 0.12;
  return {
    // v2.3: G1 & Co ohne ONNX starten mit "stiller" Policy (steht zuerst,
    // lernt dann zu gehen) statt randomTheta → kein sofortiges Zappeln/Stürzen.
    theta: MlpPolicy.standingTheta(o, 32, meta.actionDim, scale),
    layout: { obsDim: o, hidden: 32, actionDim: meta.actionDim },
    warm: false,
  };
}
