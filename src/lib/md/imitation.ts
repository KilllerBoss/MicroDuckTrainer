// ── MicroDuck Trainer v2.1 – GLB-Animations-Imitation ────────────────────────
// Nutzer lädt .glb mit Animationen hoch → Clips werden geparst, Knochen werden
// per Heuristik auf Roboter-Gelenke gemappt (Mixamo-/VRM-artige Namen) und pro
// Frame als Gelenk-Zielwinkel gesampelt. Reward „imitate" matcht diese Ziele,
// Phase (sin/cos) geht in die Observation – so kann die ES zeitabhängiges
// Nachahmen lernen. Test-Modus spielt die Animation endlos ohne Reset.

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ModelId, ModelMeta } from "./models";

export interface ClipInfo {
  index: number;
  name: string;
  duration: number;
  bones: number;
}

export interface JointMapEntry {
  jointIdx: number;
  boneName: string;
  axis: 0 | 1 | 2; // Euler-Komponente (XYZ) der Bone-Relativrotation
  sign: number;
  rest: THREE.Quaternion;
  track: THREE.QuaternionKeyframeTrack;
}

export interface ImitClip {
  name: string;
  duration: number;
  fps: number; // Samplerate der Zielpose (30 Hz)
  frames: number;
  dim: number; // actionDim
  targets: Float32Array; // frames × dim – v2.3: RELATIV zur Roboter-Ruhelage
  /** v2.3: Ruhelage (Gelenkwinkel) des Ziel-Roboters – absolut = rel + center. */
  center: Float32Array;
  rootY: Float32Array; // frames – Wurzel-Höhe (GLB y-up, Meter)
  baseY: number;
  /** v2.3: Roboterhöhe / Clip-Wurzelhöhe – skaliert Root-Delta cross-species. */
  scaleY: number;
  mapped: number; // Anzahl gemappter Gelenke
  mapping: { joint: string; bone: string }[];
}

// ── Knochen-Heuristik ────────────────────────────────────────────────────────
// side: "l"/"r" · part: hips|knee|ankle|foot|shoulder|elbow|wrist|spine|neck|head

interface BoneKey {
  side: "l" | "r" | null;
  part: string;
}

function classifyBone(name: string): BoneKey | null {
  const n = name.toLowerCase().replace(/mixamorig[:_]?/g, "").replace(/[\s:_-]+/g, "");
  let side: "l" | "r" | null = null;
  if (/^left|_l$|_l_/.test(n)) side = "l";
  else if (/^right|_r$|_r_/.test(n)) side = "r";
  const strip = n.replace(/^(left|right)/, "");
  let part: string | null = null;
  const rules: [RegExp, string][] = [
    [/upleg|thigh|hip$|hipjoint/, "hip"],
    [/shin|calf|knee|lowerleg|^leg$/, "knee"],
    [/foot|ankle|toebase/, "ankle"],
    [/shoulder|clavicle|arm(?!$)/, "shoulder"],
    [/forearm|elbow|arm$/, "elbow"],
    [/hand|wrist/, "wrist"],
    [/spine\d?|chest|torso|waist|upperchest/, "spine"],
    [/neck/, "neck"],
    [/head(?!top)/, "head"],
    [/hips$|pelvis|root/, "hips"],
  ];
  for (const [re, p] of rules) {
    if (re.test(strip)) { part = p; break; }
  }
  if (!part) return null;
  return { side, part };
}

/** part+axis → Roboter-Gelenk-Index (Heuristik pro Roboter). */
function jointFor(part: string, axis: 0 | 1 | 2, side: "l" | "r" | null, meta: ModelMeta): number {
  const s = side === "r" ? "right" : "left";
  const find = (name: string): number => {
    const i = meta.jointNames.indexOf(name);
    return i;
  };
  const g1 = meta.id === "unitree_g1";
  if (part === "hip") {
    if (axis === 0) return find(g1 ? `${s}_hip_pitch_joint` : `${s}_hip_pitch`);
    if (axis === 2) return find(g1 ? `${s}_hip_roll_joint` : `${s}_hip_roll`);
    return find(g1 ? `${s}_hip_yaw_joint` : `${s}_hip_yaw`);
  }
  if (part === "knee") return find(g1 ? `${s}_knee_joint` : `${s}_knee`);
  if (part === "ankle") {
    if (axis === 2) return find(g1 ? `${s}_ankle_roll_joint` : `${s}_ankle`);
    return find(g1 ? `${s}_ankle_pitch_joint` : `${s}_ankle`);
  }
  if (part === "shoulder") {
    if (axis === 0) return find(g1 ? `${s}_shoulder_pitch_joint` : "neck_pitch");
    if (axis === 2) return find(g1 ? `${s}_shoulder_roll_joint` : "head_roll");
    return find(g1 ? `${s}_shoulder_yaw_joint` : "head_yaw");
  }
  if (part === "elbow") return find(g1 ? `${s}_elbow_joint` : "head_pitch");
  if (part === "wrist") {
    if (axis === 0) return find(g1 ? `${s}_wrist_roll_joint` : -1 as unknown as string);
    if (axis === 1) return find(g1 ? `${s}_wrist_pitch_joint` : -1 as unknown as string);
    return find(g1 ? `${s}_wrist_yaw_joint` : -1 as unknown as string);
  }
  if (part === "spine") {
    if (axis === 1) return find(g1 ? "waist_yaw_joint" : "neck_pitch");
    if (axis === 2) return find(g1 ? "waist_roll_joint" : "head_roll");
    return find(g1 ? "waist_pitch_joint" : "head_pitch");
  }
  if (part === "neck") return find(g1 ? -1 as unknown as string : "neck_pitch");
  if (part === "head") {
    if (axis === 0) return find(g1 ? -1 as unknown as string : "head_pitch");
    if (axis === 1) return find(g1 ? -1 as unknown as string : "head_yaw");
    return find(g1 ? -1 as unknown as string : "head_roll");
  }
  return -1;
}

// Gelenk-spezifische Vorzeichen (best-effort, zur Laufzeit via Spiegel-Option korrigierbar)
function axisSign(part: string, axis: 0 | 1 | 2, side: "l" | "r" | null): number {
  const s = side === "r" ? -1 : 1;
  if (part === "hip" && axis === 0) return 1;       // pitch: beide gleich
  if (part === "hip" && axis === 2) return s;       // roll gespiegelt
  if (part === "knee") return -1;
  if (part === "ankle") return 1;
  if (part === "shoulder" && axis === 0) return 1;
  if (part === "elbow") return 1;
  if (part === "spine") return 1;
  return s;
}

// ── Parsen ───────────────────────────────────────────────────────────────────

export async function listGlbAnimations(buffer: ArrayBuffer): Promise<ClipInfo[]> {
  const gltf = await new GLTFLoader().parseAsync(buffer, "");
  const root = gltf.scene;
  let bones = 0;
  root.traverse((o) => { if ((o as any).isBone || o.type === "Bone") bones++; });
  return gltf.animations.map((c, index) => ({
    index,
    name: c.name || `Clip ${index + 1}`,
    duration: c.duration,
    bones,
  }));
}

/**
 * Sampelt einen Clip zu Gelenk-Zielwinkeln. mirror = links/rechts tauschen.
 * Euler-Extraktion: rel = q(t) · q_rest⁻¹ (XYZ-Order), Komponente axis · sign.
 * v2.3: Ziele werden RELATIV zur Roboter-Ruhelage gespeichert (`center`) und
 * das Root-Höhen-Delta mit scaleY skaliert – so kann z. B. die Ente aus
 * Human-Animationen lernen (Cross-Species), ohne von absurden Absolutwinkeln
 * zerlegt zu werden. center: Roboter-Ruhelage (z. B. engine.standPose).
 */
export async function buildImitClip(
  buffer: ArrayBuffer,
  clipIndex: number,
  meta: ModelMeta,
  mirror: boolean,
  center?: Float32Array | number[] | null,
  robotHeight?: number,
): Promise<ImitClip> {
  const gltf = await new GLTFLoader().parseAsync(buffer, "");
  const clip = gltf.animations[clipIndex];
  if (!clip) throw new Error("Clip nicht gefunden");
  const fps = 30;
  const frames = Math.max(2, Math.ceil(clip.duration * fps) + 1);
  const dim = meta.actionDim;

  // Bones sammeln: normalisierter Name → Node (robust gegen Präfixe/Sanitizing)
  const norm = (v: string) => v.toLowerCase().replace(/[^a-z0-9]/g, "");
  const nodeByNorm = new Map<string, THREE.Object3D>();
  gltf.scene.traverse((o) => {
    if (!o.name) return;
    nodeByNorm.set(norm(o.name), o);
    nodeByNorm.set(norm(o.name.replace(/^mixamorig[:_]?/i, "")), o);
  });
  const trackByNode = new Map<THREE.Object3D, THREE.QuaternionKeyframeTrack>();
  for (const t of clip.tracks) {
    if (!(t instanceof THREE.QuaternionKeyframeTrack)) continue;
    const m = t.name.match(/^(.*)\.quaternion$/);
    if (!m) continue;
    const raw = m[1];
    const node =
      nodeByNorm.get(norm(raw)) ??
      nodeByNorm.get(norm(raw.replace(/^mixamorig[:_]?/i, "")));
    if (node) trackByNode.set(node, t as unknown as THREE.QuaternionKeyframeTrack);
  }

  const entries: JointMapEntry[] = [];
  const used = new Set<number>();
  const mapping: { joint: string; bone: string }[] = [];
  for (const [node, track] of trackByNode) {
    const key = classifyBone(node.name || "");
    if (!key || key.part === "hips") continue;
    // Mirror: Seiten tauschen
    const side = mirror ? (key.side === "l" ? "r" : key.side === "r" ? "l" : null) : key.side;
    for (const axis of [0, 1, 2] as const) {
      const idx = jointFor(key.part, axis, side, meta);
      if (idx < 0 || used.has(idx)) continue;
      used.add(idx);
      entries.push({
        jointIdx: idx,
        boneName: node.name || "?",
        axis,
        sign: axisSign(key.part, axis, side),
        rest: node.quaternion.clone(),
        track,
      });
      mapping.push({ joint: meta.jointNames[idx], bone: node.name || "?" });
    }
  }

  const targets = new Float32Array(frames * dim);
  const centerArr = center && center.length === dim
    ? Float32Array.from(center)
    : new Float32Array(meta.defaultPose);
  const rootTrack = clip.tracks.find(
    (t) => /hips|pelvis|root/i.test(t.name) && t.name.endsWith(".position"),
  ) as unknown as THREE.VectorKeyframeTrack | undefined;
  const rootY = new Float32Array(frames);
  const rootInterp = rootTrack ? (rootTrack as any).createInterpolant() : null;
  const tmp = new THREE.Quaternion();
  const rel = new THREE.Quaternion();
  const _rest = new THREE.Quaternion();
  const eul = new THREE.Euler(0, 0, 0, "XYZ");
  let baseY = 0;

  for (let f = 0; f < frames; f++) {
    const t = Math.min(clip.duration, f / fps);
    if (rootInterp) {
      const v = (rootInterp as any).evaluate(t);
      rootY[f] = Number.isFinite(v[1]) ? v[1] : 0; // GLB y-up
      if (f === 0) baseY = rootY[0];
    }
    for (const e of entries) {
      const v = (e.track as any).createInterpolant().evaluate(t);
      // v2.3: Quaternionen normalisieren (Zero-Längen-Keys in manchen GLBs
      // erzeugen sonst NaN beim Slerp → REWARD-NaN!).
      const qlen = Math.hypot(v[0], v[1], v[2], v[3]);
      if (!Number.isFinite(qlen) || qlen < 1e-8) continue; // Kaputter Key → Ruhelage
      tmp.set(v[0] / qlen, v[1] / qlen, v[2] / qlen, v[3] / qlen);
      rel.copy(tmp).multiply(_rest.copy(e.rest).invert());
      eul.setFromQuaternion(rel, "XYZ");
      const comp = e.axis === 0 ? eul.x : e.axis === 1 ? eul.y : eul.z;
      if (!Number.isFinite(comp)) continue;
      // v2.3: RELATIV zur Ruhelage speichern (robust cross-species)
      targets[f * dim + e.jointIdx] = e.sign * comp;
    }
  }

  // v2.3: Höhen-Skalierung Roboter ↔ Clip (Ente 0.12 m, G1 0.72 m …)
  const targetH = Number.isFinite(robotHeight) && robotHeight! > 0.02
    ? robotHeight! : meta.targetHeight;
  const scaleY = baseY > 0.05 ? Math.min(3, Math.max(0.2, targetH / baseY)) : 1;
  // NaN-Sweep: nie nicht-finite Werte rausgeben
  for (let i = 0; i < targets.length; i++) {
    if (!Number.isFinite(targets[i])) targets[i] = 0;
  }
  for (let i = 0; i < rootY.length; i++) {
    if (!Number.isFinite(rootY[i])) rootY[i] = baseY;
  }

  return {
    name: clip.name || `Clip ${clipIndex + 1}`,
    duration: clip.duration,
    fps,
    frames,
    dim,
    targets,
    center: centerArr,
    rootY,
    baseY,
    scaleY,
    mapped: entries.length,
    mapping,
  };
}

/** Ziel-Gelenkwinkel (ABSOLUT = rel + Ruhelage) zum Zeitpunkt t (loop). */
export function targetAt(clip: ImitClip, t: number, out: Float32Array): void {
  const tt = clip.duration > 0 ? ((t % clip.duration) + clip.duration) % clip.duration : 0;
  const x = tt * clip.fps;
  const f0 = Math.min(clip.frames - 1, Math.floor(x));
  const f1 = Math.min(clip.frames - 1, f0 + 1);
  const u = x - f0;
  const c = clip.center;
  for (let j = 0; j < clip.dim; j++) {
    const a = clip.targets[f0 * clip.dim + j];
    const b = clip.targets[f1 * clip.dim + j];
    let v = a + (b - a) * u;
    if (c && j < c.length) v += c[j];
    out[j] = Number.isFinite(v) ? v : (c && j < c.length ? c[j] : 0);
  }
}

/** Wurzel-Höhen-Delta zum Zeitpunkt t (loop, relativ zum Clip-Start, skaliert). */
export function rootDeltaAt(clip: ImitClip, t: number): number {
  const tt = clip.duration > 0 ? ((t % clip.duration) + clip.duration) % clip.duration : 0;
  const x = tt * clip.fps;
  const f0 = Math.min(clip.frames - 1, Math.floor(x));
  const f1 = Math.min(clip.frames - 1, f0 + 1);
  const u = x - f0;
  const raw = (clip.rootY[f0] + (clip.rootY[f1] - clip.rootY[f0]) * u) - clip.baseY;
  return Number.isFinite(raw) ? raw * (clip.scaleY || 1) : 0;
}

/** Persistenz der letzten Auswahl (Index + Mirror) – GLB selbst bleibt Session-Daten. */
export function saveImitPrefs(modelId: ModelId, clipName: string, mirror: boolean): void {
  try {
    localStorage.setItem(`mdt_v2_imit_${modelId}`, JSON.stringify({ clipName, mirror }));
  } catch { /* ignore */ }
}

export function loadImitPrefs(modelId: ModelId): { clipName: string; mirror: boolean } | null {
  try {
    const raw = localStorage.getItem(`mdt_v2_imit_${modelId}`);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return null;
}
