// ── MicroDuck Trainer v2.1 – Random-Weltgenerator ────────────────────────────
// Erzeugt prozedurale Testwelten (MuJoCo-Geoms + three.js-Meshes): Treppen,
// Hügel, Löcher (Fliesen-Plattform), Hindernisse, Balancierstange. Deterministisch
// per Seed – „Neue Welt würfeln" ändert nur den Seed.

import * as THREE from "three";
import type { ModelId } from "./models";

export type WorldGeomType = "box" | "sphere" | "cylinder";

export interface WorldGeom {
  name: string;
  type: WorldGeomType;
  /** box: Halbe Kanten (hx,hy,hz) · sphere: (r,0,0) · cylinder: (r,halbeHöhe,0) */
  size: [number, number, number];
  pos: [number, number, number];
  /** Quaternion (w,x,y,z) – z. B. Rampen */
  quat?: [number, number, number, number];
  rgba: [number, number, number, number];
}

export interface WorldFeatures {
  treppen: boolean;
  huegel: boolean;
  loecher: boolean;
  hindernisse: boolean;
  stange: boolean;
}

export interface WorldConfig {
  enabled: boolean;
  seed: number;
  difficulty: number; // 0..1 – Höhen/Größen
  density: number; // 0..1 – Anzahl Objekte
  features: WorldFeatures;
}

/** Ergebnis: Geoms für Physik+Rendering; holes=true → Boden wird zu Fliesen. */
export interface WorldBuild {
  geoms: WorldGeom[];
  holes: boolean;
  tileSize: number;
}

export function defaultWorldFeatures(): WorldFeatures {
  return {
    treppen: true, huegel: true, loecher: false, hindernisse: true, stange: false,
  };
}

export function defaultWorldConfig(): WorldConfig {
  return {
    enabled: false,
    seed: (Math.random() * 0xffffffff) >>> 0,
    difficulty: 0.4,
    density: 0.5,
    features: defaultWorldFeatures(),
  };
}

export function randomSeed(): number {
  return (Math.random() * 0xffffffff) >>> 0;
}

// Deterministischer RNG (mulberry32)
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ARENA: Record<ModelId, number> = { microduck: 1.5, unitree_g1: 4.0 };
/** Freihalten der Startzone, damit der Roboter nie in einem Objekt spawnt. */
const START_CLEAR = 0.55;

export function arenaHalf(modelId: ModelId): number {
  return ARENA[modelId];
}

const CYAN_TILE: [number, number, number, number] = [0.10, 0.16, 0.22, 1];
const AMBER: [number, number, number, number] = [0.85, 0.55, 0.15, 1];
const STEEL: [number, number, number, number] = [0.45, 0.5, 0.58, 1];
const ORANGE: [number, number, number, number] = [0.95, 0.45, 0.1, 1];

/** Erzeugt die Welt-Geoms (deterministisch). */
export function generateWorld(cfg: WorldConfig, modelId: ModelId): WorldBuild {
  const geoms: WorldGeom[] = [];
  const holes = cfg.enabled && cfg.features.loecher;
  const half = ARENA[modelId] - 0.1;
  const duck = modelId === "microduck";
  const d = Math.max(0, Math.min(1, cfg.difficulty));
  const dens = Math.max(0.05, Math.min(1, cfg.density));
  const rand = rng(cfg.seed);

  const clearOfStart = (x: number, y: number, m: number) =>
    Math.hypot(x, y) > START_CLEAR + m;

  /** Versucht count Objekte; bis zu 8 Kandidaten pro Objekt. */
  const place = (
    count: number,
    margin: number,
    make: (x: number, y: number, i: number) => WorldGeom,
  ): void => {
    let placed = 0;
    for (let i = 0; i < count * 8 && placed < count; i++) {
      const x = (rand() * 2 - 1) * (half - margin - 0.05);
      const y = (rand() * 2 - 1) * (half - margin - 0.05);
      if (!clearOfStart(x, y, margin)) continue;
      geoms.push(make(x, y, placed));
      placed++;
    }
  };

  // ── Löcher: Fliesenboden (füllt die GANZE Arena, Rand immer fest) ──────────
  const tileSize = duck ? 0.4 : 0.6;
  if (holes) {
    const n = Math.ceil((half * 2) / tileSize);
    const t = (half * 2) / n; // effektive Kantenlänge (exakt füllend)
    const gapChance = 0.12 + 0.2 * dens;
    for (let ix = 0; ix < n; ix++) {
      for (let iy = 0; iy < n; iy++) {
        const x = -half + t / 2 + ix * t;
        const y = -half + t / 2 + iy * t;
        const border = ix === 0 || iy === 0 || ix === n - 1 || iy === n - 1;
        const inStart = Math.hypot(x, y) < START_CLEAR + t / 2;
        // Löcher nur AUSSERHALB der Startzone (Spawn bleibt immer fest!)
        if (!border && !inStart && rand() < gapChance) continue;
        geoms.push({
          name: `tile_${ix}_${iy}`,
          type: "box",
          size: [t / 2 - 0.005, t / 2 - 0.005, 0.05],
          pos: [x, y, -0.05],
          rgba: CYAN_TILE,
        });
      }
    }
  }

  if (!cfg.enabled) return { geoms, holes, tileSize };

  // ── Treppe (aufsteigend, in einem Quadranten) ────────────────────────────
  if (cfg.features.treppen) {
    const steps = 3 + Math.round(rand() * 2 + d); // 3..5
    const stepH = (0.035 + 0.05 * d) * (duck ? 0.8 : 1);
    const stepD = duck ? 0.2 : 0.26;
    const w = duck ? 0.45 : 0.7;
    const dir = Math.floor(rand() * 4); // Quadrant
    const sx = dir === 0 || dir === 3 ? 1 : -1;
    const sy = dir < 2 ? 1 : -1;
    const baseX = sx * half * 0.52;
    const baseY = sy * half * 0.52;
    for (let i = 0; i < steps; i++) {
      const x = baseX + sx * i * stepD;
      const y = baseY;
      if (!clearOfStart(x - sx * stepD / 2, y, stepD)) continue;
      geoms.push({
        name: `stufe_${i}`,
        type: "box",
        size: [stepD / 2, w / 2, ((i + 1) * stepH) / 2],
        pos: [x, y, ((i + 1) * stepH) / 2],
        rgba: STEEL,
      });
    }
  }

  // ── Hügel: halb eingegrabene Kugeln ──────────────────────────────────────
  if (cfg.features.huegel) {
    const count = Math.round((duck ? 1 : 2) + 3 * dens);
    place(count, 0.15, (x, y, i) => {
      const r = (0.18 + 0.28 * (0.4 + 0.6 * d)) * (duck ? 0.7 : 1) * (0.8 + 0.5 * rand());
      return {
        name: `huegel_${i}`, type: "sphere", size: [r, 0, 0],
        pos: [x, y, -r * (0.45 + 0.2 * rand())] as [number, number, number],
        rgba: [0.3, 0.34, 0.42, 1] as [number, number, number, number],
      };
    });
  }

  // ── Hindernisse: Boxen + Zylinder ────────────────────────────────────────
  if (cfg.features.hindernisse) {
    const count = Math.round((duck ? 2 : 3) + 4 * dens);
    place(count, 0.12, (x, y, i) => {
      const cyl = rand() < 0.4;
      const h = (0.07 + 0.28 * d) * (duck ? 0.7 : 1) * (0.7 + 0.7 * rand());
      const w = (0.07 + 0.12 * rand()) * (duck ? 0.8 : 1);
      if (cyl) {
        return {
          name: `hind_cyl_${i}`, type: "cylinder", size: [w, h / 2, 0],
          pos: [x, y, h / 2] as [number, number, number], rgba: AMBER,
        };
      }
      return {
        name: `hind_box_${i}`, type: "box",
        size: [w / 2, (0.1 + 0.16 * rand()) / 2, h / 2],
        pos: [x, y, h / 2] as [number, number, number],
        quat: yawQuat(rand() * Math.PI),
        rgba: AMBER,
      };
    });
  }

  // ── Balancierstange: schmale lange Box quer über die Arena ────────────────
  if (cfg.features.stange) {
    const len = Math.min(1.5 + 1.1 * rand(), half * 1.5);
    const wide = 0.05 + 0.05 * (1 - d);
    const h = 0.06 + 0.09 * d;
    const sy = rand() < 0.5 ? 1 : -1;
    const cx = (rand() * 2 - 1) * half * 0.25;
    const cy = sy * half * 0.55;
    if (Math.hypot(cx, cy) - wide > START_CLEAR * 0.8) {
      geoms.push({
        name: "balancierstange",
        type: "box",
        size: [len / 2, wide / 2, h / 2],
        pos: [cx, cy, h / 2],
        quat: yawQuat((rand() - 0.5) * 0.9), // leicht gedreht
        rgba: ORANGE,
      });
    }
  }

  return { geoms, holes, tileSize };
}

function yawQuat(yaw: number): [number, number, number, number] {
  return [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
}

// ── XML-Erzeugung (geteilt mit xml.ts, auch für ES-Worker-Payload) ───────────

export function worldGeomXmlAttr(g: WorldGeom): Record<string, string> {
  const attrs: Record<string, string> = {
    name: g.name,
    type: g.type,
    size: `${fmt(g.size[0])} ${fmt(g.size[1])} ${fmt(g.size[2])}`,
    pos: `${fmt(g.pos[0])} ${fmt(g.pos[1])} ${fmt(g.pos[2])}`,
    rgba: `${g.rgba[0]} ${g.rgba[1]} ${g.rgba[2]} ${g.rgba[3]}`,
  };
  if (g.quat) attrs.quat = `${fmt(g.quat[0])} ${fmt(g.quat[1])} ${fmt(g.quat[2])} ${fmt(g.quat[3])}`;
  if (g.type !== "box") attrs.friction = "0.9 0.02 0.004";
  return attrs;
}

function fmt(v: number): string {
  return String(Math.round(v * 1e4) / 1e4);
}

// ── three.js-Visualisierung ──────────────────────────────────────────────────

const quatTmp = new THREE.Quaternion();

/** Baut die drei.js-Repräsentation der Welt-Geoms (MJCF z-up → three y-up). */
export function buildWorldMeshes(build: WorldBuild): THREE.Group {
  const group = new THREE.Group();
  group.name = "world_props";
  for (const g of build.geoms) {
    let geo: THREE.BufferGeometry;
    if (g.type === "box") {
      geo = new THREE.BoxGeometry(g.size[0] * 2, g.size[2] * 2, g.size[1] * 2);
    } else if (g.type === "sphere") {
      geo = new THREE.SphereGeometry(g.size[0], 20, 14);
    } else {
      geo = new THREE.CylinderGeometry(g.size[0], g.size[0], g.size[1] * 2, 18);
    }
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]),
      roughness: 0.8,
      metalness: 0.15,
    });
    const mesh = new THREE.Mesh(geo, mat);
    // MJCF (x,y,z) → three (x, z, -y)
    mesh.position.set(g.pos[0], g.pos[2], -g.pos[1]);
    if (g.quat) {
      quatTmp.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
      // Rotation in three-Frame konvertieren: q_three = M * q_mjcf * M⁻¹ (M = Achsen-Tausch)
      const m = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
      mesh.quaternion.copy(m).multiply(quatTmp).multiply(m.clone().invert());
    }
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    group.add(mesh);
  }
  return group;
}
