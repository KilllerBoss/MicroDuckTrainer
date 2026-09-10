// ── MicroDuck Trainer v2.0 – MJCF-Aufbereitung ───────────────────────────────
// Wird sowohl vom Engine-Boot (Main-Thread) als auch vom ES-Worker benutzt,
// damit Physik-XML und VFS-Dateiliste identisch sind. Muster: reference/game.js.

import type { ModelMeta, ModelId } from "./models";

const BALL_RADIUS = 0.05;
const BALL_PARK = "50 0 0.05"; // geparkt = abwesend

/** Ente: bewährtes Muster aus reference/game.js buildPhysicsXml. */
export function buildPhysicsXml(
  modelId: ModelId,
  src: string,
  meta: ModelMeta,
): { xml: string; meshFiles: string[] } {
  if (modelId === "unitree_g1") return buildG1Xml(src, meta);
  return buildDuckXml(src, meta);
}

/** Für den ES-Worker: identisches Ergebnis inkl. Metadaten-Paket. */
export function buildPhysicsXmlForWorker(
  modelId: ModelId,
  src: string,
  meta: ModelMeta,
): {
  xml: string; meshBase: string; vfsPrefix: string; meshFiles: string[];
  modelId: ModelId; obsDim: number; actionDim: number;
} {
  const { xml, meshFiles } = buildPhysicsXml(modelId, src, meta);
  return {
    xml, meshFiles,
    meshBase: meta.meshBase,
    vfsPrefix: meta.vfsPrefix,
    modelId,
    obsDim: meta.obsDim,
    actionDim: meta.actionDim,
  };
}

function buildDuckXml(src: string, meta: ModelMeta): { xml: string; meshFiles: string[] } {
  const doc = new DOMParser().parseFromString(src, "text/xml");
  // Visual-Geoms raus → VFS braucht nur die Kollisions-Meshes (~10 Dateien)
  for (const g of [...doc.querySelectorAll('geom[class="visual"]')]) g.remove();
  const usedMeshes = new Set(
    [...doc.querySelectorAll("geom[mesh]")].map((g) => g.getAttribute("mesh")),
  );
  for (const m of [...doc.querySelectorAll("asset > mesh")]) {
    const name = m.getAttribute("name") ?? m.getAttribute("file")!.replace(/\.stl$/i, "");
    if (!usedMeshes.has(name)) m.remove();
  }
  const root = doc.documentElement;
  const el = (tag: string, attrs: Record<string, string>) => {
    const e = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };
  root.appendChild(el("option", { timestep: String(meta.timestep) }));
  doc.querySelector("worldbody")!.appendChild(
    el("geom", { name: "floor", type: "plane", size: "0 0 0.05", pos: "0 0 0" }),
  );
  // Arenen-Wände wie im Original (halten Ente + Ball im 3×3-m-Feld)
  const ht = 0.05 / 2, hh = 0.25 / 2;
  const off = 1.5 + ht, span = 1.5 + 0.05;
  for (const w of [
    { name: "wall_px", pos: `${off} 0 ${hh}`, size: `${ht} ${span} ${hh}` },
    { name: "wall_nx", pos: `${-off} 0 ${hh}`, size: `${ht} ${span} ${hh}` },
    { name: "wall_py", pos: `0 ${off} ${hh}`, size: `${span} ${ht} ${hh}` },
    { name: "wall_ny", pos: `0 ${-off} ${hh}`, size: `${span} ${ht} ${hh}` },
  ]) {
    doc.querySelector("worldbody")!.appendChild(
      el("geom", { name: w.name, type: "box", pos: w.pos, size: w.size }),
    );
  }
  // Kickbarer Ball (Free-Sphere, Beach-Ball-Gefühl) – NACH dem Roboter
  // angehängt, damit der Trunk-Freejoint in qpos vorne bleibt.
  const ballBody = el("body", { name: "ball", pos: BALL_PARK });
  ballBody.appendChild(el("freejoint", { name: "ball_freejoint" }));
  ballBody.appendChild(el("geom", {
    name: "ball_geom", type: "sphere", size: String(BALL_RADIUS),
    mass: "0.03", friction: "0.4 0.01 0.003", solref: "0.03 0.4", condim: "6",
  }));
  doc.querySelector("worldbody")!.appendChild(ballBody);
  // STAND-Keyframe (mjlab scene_walk.xml): qpos = Freejoint + Gelenke + Ball.
  const qposFree = "-0.6 0 0.12 1 0 0 0";
  const poseByName = new Map(meta.jointNames.map((n, i) => [n, meta.defaultPose[i]]));
  const qposJoints = [...doc.querySelectorAll("body > joint")]
    .map((j) => poseByName.get(j.getAttribute("name")!) ?? 0)
    .join(" ");
  const kf = doc.createElement("keyframe");
  kf.appendChild(el("key", {
    name: "STAND",
    qpos: `${qposFree} ${qposJoints} ${BALL_PARK} 1 0 0 0`,
    ctrl: meta.defaultPose.join(" "),
  }));
  root.appendChild(kf);
  const meshFiles = [...doc.querySelectorAll("asset > mesh")].map((m) => m.getAttribute("file")!);
  return { xml: new XMLSerializer().serializeToString(doc), meshFiles };
}

function buildG1Xml(src: string, meta: ModelMeta): { xml: string; meshFiles: string[] } {
  const doc = new DOMParser().parseFromString(src, "text/xml");
  const root = doc.documentElement;
  const el = (tag: string, attrs: Record<string, string>) => {
    const e = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };
  const opt = doc.querySelector("option");
  if (opt) opt.setAttribute("timestep", String(meta.timestep));
  else root.appendChild(el("option", { timestep: String(meta.timestep) }));
  // Boden + große Arena (Humanoid braucht mehr Platz als die Ente)
  const wb = doc.querySelector("worldbody")!;
  wb.appendChild(el("geom", { name: "floor", type: "plane", size: "0 0 0.05", pos: "0 0 0" }));
  const A = 4.0, ht = 0.05 / 2, hh = 0.4 / 2;
  const off = A + ht, span = A + 0.05;
  for (const w of [
    { name: "wall_px", pos: `${off} 0 ${hh}`, size: `${ht} ${span} ${hh}` },
    { name: "wall_nx", pos: `${-off} 0 ${hh}`, size: `${ht} ${span} ${hh}` },
    { name: "wall_py", pos: `0 ${off} ${hh}`, size: `${span} ${ht} ${hh}` },
    { name: "wall_ny", pos: `0 ${-off} ${hh}`, size: `${span} ${ht} ${hh}` },
  ]) {
    wb.appendChild(el("geom", { name: w.name, type: "box", pos: w.pos, size: w.size }));
  }
  // Kickbarer Ball wie bei der Ente
  const ballBody = el("body", { name: "ball", pos: BALL_PARK });
  ballBody.appendChild(el("freejoint", { name: "ball_freejoint" }));
  ballBody.appendChild(el("geom", {
    name: "ball_geom", type: "sphere", size: String(BALL_RADIUS),
    mass: "0.03", friction: "0.4 0.01 0.003", solref: "0.03 0.4", condim: "6",
  }));
  wb.appendChild(ballBody);
  // Bestehende Keyframes um die 7 Ball-qpos-Werte erweitern (nq muss exakt passen)
  for (const key of [...doc.querySelectorAll("keyframe > key")]) {
    const q = key.getAttribute("qpos");
    if (q) key.setAttribute("qpos", `${q} ${BALL_PARK} 1 0 0 0`);
  }
  const meshFiles = [...doc.querySelectorAll("asset > mesh")].map((m) => m.getAttribute("file")!);
  return { xml: new XMLSerializer().serializeToString(doc), meshFiles };
}
