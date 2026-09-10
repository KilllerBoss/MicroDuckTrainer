// ── MicroDuck Trainer v2.0 – Rendering (three.js) ────────────────────────────
// Tron-Look (#05060a, Cyan-Grid), Chase-Cam per Touch drehbar.
// Ente: GLB-Rig exakt nach reference/duck.js (kinematics.json + toCreasedNormals)
// mit Sync über Freejoint-qpos + setJoint. G1: STL pro MuJoCo-Body, Sync direkt
// aus xpos/xquat (Body-Baum aus mjModel).

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { mergeVertices, toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";


export const ACCENT = 0x22d3ee;

// ── Duck-Rig (Port von reference/duck.js) ────────────────────────────────────

const CREASE = Math.PI / 5;
let glbGeomsPromise: Promise<Map<string, { display: THREE.BufferGeometry; welded: THREE.BufferGeometry }>> | null = null;

function loadGlbGeometries() {
  if (!glbGeomsPromise) {
    glbGeomsPromise = new GLTFLoader()
      .loadAsync("/robot/mjlab/microduck.glb")
      .then((gltf) => {
        const map = new Map();
        gltf.scene.traverse((o) => {
          if (!o.isMesh || !o.geometry) return;
          const name = o.userData.meshFile || o.name || o.geometry.name;
          if (!name || map.has(name)) return;
          const welded = o.geometry.clone();
          welded.deleteAttribute("normal");
          // toCreasedNormals hashed auf 0.01-Einheiten: Meshes (m) → mm skalieren
          const scaled = welded.clone();
          scaled.scale(1000, 1000, 1000);
          const display = toCreasedNormals(scaled, CREASE);
          display.scale(1e-3, 1e-3, 1e-3);
          map.set(name, { display, welded });
        });
        return map;
      });
  }
  return glbGeomsPromise;
}

export function geometryToBinaryStl(geometry: THREE.BufferGeometry): ArrayBuffer {
  const pos = geometry.attributes.position;
  const idx = geometry.index;
  const triCount = (idx ? idx.count : pos.count) / 3;
  const buf = new ArrayBuffer(84 + triCount * 50);
  const view = new DataView(buf);
  view.setUint32(80, triCount, true);
  let off = 84;
  const vx = (i: number) => {
    const j = idx ? idx.getX(i) : i;
    return [pos.getX(j), pos.getY(j), pos.getZ(j)];
  };
  for (let t = 0; t < triCount; t++) {
    const a = vx(t * 3);
    const b = vx(t * 3 + 1);
    const c = vx(t * 3 + 2);
    const nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    const ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    const nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const len = Math.hypot(nx, ny, nz) || 1;
    view.setFloat32(off, nx / len, true);
    view.setFloat32(off + 4, ny / len, true);
    view.setFloat32(off + 8, nz / len, true);
    view.setFloat32(off + 12, a[0], true);
    view.setFloat32(off + 16, a[1], true);
    view.setFloat32(off + 20, a[2], true);
    view.setFloat32(off + 24, b[0], true);
    view.setFloat32(off + 28, b[1], true);
    view.setFloat32(off + 32, b[2], true);
    view.setFloat32(off + 36, c[0], true);
    view.setFloat32(off + 40, c[1], true);
    view.setFloat32(off + 44, c[2], true);
    view.setUint16(off + 48, 0, true);
    off += 50;
  }
  return buf;
}

interface DuckJoint {
  body: THREE.Group;
  axis: THREE.Vector3;
  baseQuat: THREE.Quaternion;
  range: [number, number] | null;
}

export interface DuckRig {
  kind: "duck";
  placer: THREE.Group;
  root: THREE.Group;
  bodies: Map<string, THREE.Group>;
  joints: Map<string, DuckJoint>;
}

async function buildDuckRig(): Promise<DuckRig> {
  const k = await (await fetch("/robot/mjlab/kinematics.json")).json();
  const placer = new THREE.Group();
  placer.name = "duck_placer";
  const root = new THREE.Group();
  root.name = "duck_root";
  root.rotation.x = -Math.PI / 2; // MJCF z-up → three y-up
  placer.add(root);

  const bodies = new Map<string, THREE.Group>();
  const joints = new Map<string, DuckJoint>();
  const geomByName = await loadGlbGeometries();
  // Roller-only Meshes (nicht im GLB): STL-Fallback
  const stlCache = new Map<string, Promise<{ display: THREE.BufferGeometry; welded: THREE.BufferGeometry }>>();
  const loadMesh = (name: string) => {
    const entry = geomByName.get(name);
    if (entry) return Promise.resolve(entry);
    if (!stlCache.has(name)) {
      stlCache.set(
        name,
        new STLLoader().loadAsync(`/robot/mjlab/meshes/${name}`).then((raw) => {
          raw.deleteAttribute("normal");
          const welded = mergeVertices(raw, 1e-4);
          welded.scale(1000, 1000, 1000);
          const display = toCreasedNormals(welded, CREASE);
          display.scale(1e-3, 1e-3, 1e-3);
          welded.scale(1e-3, 1e-3, 1e-3);
          return { display, welded };
        }),
      );
    }
    return stlCache.get(name)!;
  };

  for (const b of k.bodies) {
    const g = new THREE.Group();
    g.name = b.name;
    g.position.set(b.pos[0], b.pos[1], b.pos[2]);
    g.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
    bodies.set(b.name, g);
  }
  for (const b of k.bodies) {
    const g = bodies.get(b.name)!;
    if (b.parent && bodies.has(b.parent)) bodies.get(b.parent)!.add(g);
    else root.add(g);
  }
  for (const b of k.bodies) {
    if (!b.joint || (b.joint.type && b.joint.type !== "hinge")) continue;
    const g = bodies.get(b.name)!;
    joints.set(b.joint.name, {
      body: g,
      axis: new THREE.Vector3(...b.joint.axis).normalize(),
      baseQuat: g.quaternion.clone(),
      range: b.joint.range ?? null,
    });
  }

  const matCache = new Map<string, THREE.MeshStandardMaterial>();
  const matFor = (rgba: number[]) => {
    const key = rgba.join(",");
    const cached = matCache.get(key);
    if (cached) return cached;
    const m = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      roughness: 0.55,
      metalness: 0.1,
      transparent: (rgba[3] ?? 1) < 1,
      opacity: rgba[3] ?? 1,
    });
    matCache.set(key, m);
    return m;
  };

  const seen = new Set<string>();
  const pending: Promise<void>[] = [];
  for (const b of k.bodies) {
    const g = bodies.get(b.name);
    if (!g) continue;
    for (const geom of b.geoms) {
      if (geom.type && geom.type !== "mesh") continue;
      if (!geom.mesh) continue;
      const dupKey = `${b.name}|${geom.mesh}|${geom.pos}|${geom.quat}`;
      if (seen.has(dupKey)) continue;
      seen.add(dupKey);
      pending.push(
        loadMesh(geom.mesh).then(({ display }) => {
          const rgba = geom.color
            ? [geom.color[0], geom.color[1], geom.color[2], geom.color[3] ?? 1]
            : [0.85, 0.85, 0.85, 1];
          const m = new THREE.Mesh(display, matFor(rgba));
          if (geom.pos) m.position.set(...geom.pos);
          if (geom.quat) m.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
          g.add(m);
        }),
      );
    }
  }
  await Promise.all(pending);
  return { kind: "duck", placer, root, bodies, joints };
}

const _q = new THREE.Quaternion();
function setDuckJoint(rig: DuckRig, name: string, angle: number) {
  const j = rig.joints.get(name);
  if (!j) return;
  let a = angle;
  if (j.range) a = Math.min(j.range[1], Math.max(j.range[0], a));
  const rot = _q.setFromAxisAngle(j.axis, a);
  j.body.quaternion.copy(j.baseQuat).multiply(rot);
}

// ── G1-Rig (STL pro Body, Posen aus mjModel) ─────────────────────────────────

export interface G1Rig {
  kind: "g1";
  root: THREE.Group;
  bodyGroups: THREE.Group[]; // Index = MuJoCo-Body-ID (0 = world → skip)
}

interface G1BodySpec {
  name: string;
  meshes: { file: string; pos?: number[]; quat?: number[]; rgba: number[] }[];
}

async function buildG1Rig(xmlSrc: string): Promise<G1Rig> {
  const doc = new DOMParser().parseFromString(xmlSrc, "text/xml");
  const materials = new Map<string, number[]>();
  for (const m of [...doc.querySelectorAll("asset > material")]) {
    const rgba = (m.getAttribute("rgba") ?? "0.7 0.7 0.7 1").split(/\s+/).map(Number);
    materials.set(m.getAttribute("name")!, rgba);
  }
  const meshFileByName = new Map<string, string>();
  for (const m of [...doc.querySelectorAll("asset > mesh")]) {
    const name = m.getAttribute("name") ?? m.getAttribute("file")!.replace(/\.STL$/i, "");
    meshFileByName.set(name, m.getAttribute("file")!);
  }

  const specs: G1BodySpec[] = [];
  const walk = (el: Element) => {
    if (el.tagName === "body") {
      const name = el.getAttribute("name") ?? `body_${specs.length}`;
      const meshes: G1BodySpec["meshes"] = [];
      for (const g of [...el.children].filter((c) => c.tagName === "geom")) {
        const cls = g.getAttribute("class") ?? "";
        if (cls !== "visual") continue;
        const meshName = g.getAttribute("mesh");
        if (!meshName) continue;
        const file = meshFileByName.get(meshName);
        if (!file) continue;
        const matName = g.getAttribute("material");
        const rgba = matName && materials.get(matName) ? materials.get(matName)! : [0.7, 0.7, 0.7, 1];
        meshes.push({
          file,
          pos: g.getAttribute("pos") ? g.getAttribute("pos")!.split(/\s+/).map(Number) : undefined,
          quat: g.getAttribute("quat") ? g.getAttribute("quat")!.split(/\s+/).map(Number) : undefined,
          rgba,
        });
      }
      specs.push({ name, meshes });
    }
    for (const child of [...el.children]) walk(child);
  };
  walk(doc.querySelector("worldbody")!);

  const root = new THREE.Group();
  root.name = "g1_root";
  root.rotation.x = -Math.PI / 2; // MJCF z-up → three y-up
  const bodyGroups: THREE.Group[] = [];
  const bodyIndexByName = new Map<string, number>();

  const loader = new STLLoader();
  const pending: Promise<void>[] = [];
  specs.forEach((spec, i) => {
    const g = new THREE.Group();
    g.name = spec.name;
    bodyGroups.push(g);
    bodyIndexByName.set(spec.name, i);
    root.add(g);
    for (const mesh of spec.meshes) {
      pending.push(
        loader.loadAsync(`/robot/unitree_g1/assets/${mesh.file}`).then((geo) => {
          const rgba = mesh.rgba;
          const mat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
            roughness: 0.5,
            metalness: 0.35,
            transparent: (rgba[3] ?? 1) < 1,
            opacity: rgba[3] ?? 1,
          });
          const m = new THREE.Mesh(geo, mat);
          if (mesh.pos) m.position.set(...mesh.pos);
          if (mesh.quat) m.quaternion.set(mesh.quat[1], mesh.quat[2], mesh.quat[3], mesh.quat[0]);
          g.add(m);
        }),
      );
    }
  });
  await Promise.all(pending);
  return { kind: "g1", root, bodyGroups };
}

// ── Welt / Szene ─────────────────────────────────────────────────────────────

export interface World {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  robotRoot: THREE.Group;
  duckRig: DuckRig | null;
  g1Rig: G1Rig | null;
  ball: THREE.Mesh;
  setCameraTarget: (x: number, y: number, z: number) => void;
  updateCamera: (dt: number) => void;
  /** v2.1: Welt-Objekte (Treppen, Hügel …) als Gruppe setzen/entfernen. */
  setWorldProps: (group: THREE.Group | null) => void;
  /** v2.1: Joystick-Punkt-Marker setzen (Punkt-Modus). */
  setTargetPoint: (x: number, y: number, visible: boolean) => void;
  dispose: () => void;
}

export async function createWorld(container: HTMLElement): Promise<World> {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05060a);
  scene.fog = new THREE.Fog(0x05060a, 5, 22);

  const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
  camera.position.set(1.6, 1.2, 1.9);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = false;
  container.appendChild(renderer.domElement);
  renderer.domElement.style.touchAction = "none";
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";

  // Lichter
  const hemi = new THREE.HemisphereLight(0x8fd8ff, 0x0a0e18, 0.9);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(3, 6, 2);
  scene.add(dir);
  const rim = new THREE.DirectionalLight(ACCENT, 0.35);
  rim.position.set(-4, 2, -3);
  scene.add(rim);

  // Boden: dunkle Scheibe + Cyan-Gitter (1 m)
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x070b12, roughness: 0.9, metalness: 0.1,
  });
  const floor = new THREE.Mesh(new THREE.CircleGeometry(14, 48), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.002;
  scene.add(floor);
  const grid = new THREE.GridHelper(20, 20, ACCENT, 0x0e7490);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.28;
  grid.position.y = 0.001;
  scene.add(grid);
  const gridFine = new THREE.GridHelper(20, 100, 0x155e75, 0x155e75);
  (gridFine.material as THREE.Material).transparent = true;
  (gridFine.material as THREE.Material).opacity = 0.12;
  gridFine.position.y = 0.0005;
  scene.add(gridFine);

  // Kickbarer Beach-Ball (Pose kommt aus der Physik)
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.05, 24, 18),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 }),
  );
  ball.visible = false;
  scene.add(ball);
  // Ball-Muster: cyan Streifen via zweitem Ring
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.05, 0.008, 10, 32),
    new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.4 }),
  );
  ring.rotation.x = Math.PI / 2;
  ball.add(ring);

  // Roboter-Root (wird beim Modellwechsel ausgetauscht)
  const robotRoot = new THREE.Group();
  scene.add(robotRoot);

  // v2.1: Punkt-Marker (glühender Ring + Kern) für den Punkt-Modus
  const marker = new THREE.Group();
  const markerRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.09, 0.014, 10, 36),
    new THREE.MeshStandardMaterial({ color: 0xff9a3c, emissive: 0xff6a00, emissiveIntensity: 1.4, roughness: 0.4 }),
  );
  markerRing.rotation.x = Math.PI / 2;
  const markerDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.03, 14, 10),
    new THREE.MeshStandardMaterial({ color: 0xffd28a, emissive: 0xff8c00, emissiveIntensity: 1.8 }),
  );
  marker.add(markerRing, markerDot);
  marker.position.y = 0.03;
  marker.visible = false;
  scene.add(marker);
  const worldProps = new THREE.Group();
  worldProps.name = "world_props_root";
  scene.add(worldProps);
  let markerPulse = 0;

  let duckRig: DuckRig | null = null;
  let g1Rig: G1Rig | null = null;

  // ── Chase-Cam (Lerp folgt Torso; Touch-Drag dreht) ──
  const target = new THREE.Vector3(0, 0.3, 0);
  const camSph = { yaw: 0.6, pitch: 0.42, dist: 1.6 };
  const smoothTarget = new THREE.Vector3(0, 0.3, 0);
  let dragging = false;
  let lastX = 0, lastY = 0;
  const el = renderer.domElement;
  el.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    camSph.yaw -= dx * 0.006;
    camSph.pitch = Math.max(0.05, Math.min(1.35, camSph.pitch + dy * 0.005));
  });
  const endDrag = () => {
    dragging = false;
  };
  el.addEventListener("pointerup", endDrag);
  el.addEventListener("pointercancel", endDrag);
  el.addEventListener("wheel", (e) => {
    camSph.dist = Math.max(0.4, Math.min(8, camSph.dist + e.deltaY * 0.002));
  }, { passive: true });

  const resize = () => {
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  };
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  const world: World = {
    scene, camera, renderer, robotRoot, duckRig, g1Rig, ball,
    setCameraTarget(x, y, z) {
      target.set(x, y, z);
    },
    setWorldProps(group) {
      for (const c of [...worldProps.children]) worldProps.remove(c);
      if (group) worldProps.add(group);
    },
    setTargetPoint(x, y, visible) {
      marker.visible = visible;
      if (visible) marker.position.set(x, 0.03, -y);
    },
    updateCamera(dt) {
      if (marker.visible) {
        markerPulse += dt * 5;
        const k = 1 + Math.sin(markerPulse) * 0.12;
        markerRing.scale.setScalar(k);
      }
      smoothTarget.lerp(target, Math.min(1, dt * 6));
      const cp = new THREE.Vector3(
        smoothTarget.x + camSph.dist * Math.cos(camSph.pitch) * Math.cos(camSph.yaw),
        smoothTarget.y + camSph.dist * Math.sin(camSph.pitch),
        smoothTarget.z + camSph.dist * Math.cos(camSph.pitch) * Math.sin(camSph.yaw),
      );
      camera.position.lerp(cp, Math.min(1, dt * 8));
      camera.lookAt(smoothTarget);
    },
    dispose() {
      ro.disconnect();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
  return world;
}

/** Baut das Rig für ein Modell und ersetzt das alte (dispose alles Alt). */
export async function mountRig(
  world: World,
  modelId: "microduck" | "unitree_g1",
): Promise<DuckRig | G1Rig | null> {
  // Altes Rig entfernen (Geometrien/Materialien bleiben im Cache – Wiederverwendung)
  for (const child of [...world.robotRoot.children]) {
    world.robotRoot.remove(child);
  }
  world.duckRig = null;
  world.g1Rig = null;
  if (modelId === "microduck") {
    const rig = await buildDuckRig();
    world.robotRoot.add(rig.placer);
    world.duckRig = rig;
    return rig;
  }
  const src = await (await fetch("/robot/unitree_g1/g1.xml")).text();
  const rig = await buildG1Rig(src);
  world.robotRoot.add(rig.root);
  world.g1Rig = rig;
  return rig;
}

/** Sync Duck: Freejoint-qpos → trunk, Gelenke → setJoint (wie Original syncRig). */
export function syncDuck(world: World, engine: any): void {
  const rig = world.duckRig;
  if (!rig) return;
  const trunkGroup = rig.bodies.get(engine.meta.torsoBody);
  if (!trunkGroup) return;
  const qpos = engine.data.qpos as Float32Array;
  trunkGroup.position.set(qpos[0], qpos[1], qpos[2]);
  trunkGroup.quaternion.set(qpos[4], qpos[5], qpos[6], qpos[3]);
  for (let j = 0; j < engine.meta.jointNames.length; j++) {
    setDuckJoint(rig, engine.meta.jointNames[j], qpos[engine.qposAdrOf(j)]);
  }
}

/** Sync G1: xpos/xquat je Body direkt auf die Weltgruppen. */
export function syncG1(world: World, engine: any, mujoco: any): void {
  const rig = world.g1Rig;
  if (!rig) return;
  const model = engine.model;
  const n = model.nbody;
  const MJ_OBJ_BODY = (mujoco.mjtObj.mjOBJ_BODY as any).value ?? 1;
  for (let id = 1; id < n; id++) {
    const name = mujoco.mj_id2name(model, MJ_OBJ_BODY, id);
    if (!name) continue;
    // Specs liegen in Dokumentreihenfolge = Body-ID-Reihenfolge, Index = id-1
    const g = rig.bodyGroups[id - 1];
    if (!g || g.name !== name) {
      // Fallback über Namen (robust bei Umordnung)
      const idx = rig.bodyGroups.findIndex((b) => b.name === name);
      if (idx < 0) continue;
      const grp = rig.bodyGroups[idx];
      const p = engine.data.body(id).xpos as Float32Array;
      const q = engine.data.body(id).xquat as Float32Array;
      grp.position.set(p[0], p[1], p[2]);
      grp.quaternion.set(q[1], q[2], q[3], q[0]);
      continue;
    }
    const p = engine.data.body(id).xpos as Float32Array;
    const q = engine.data.body(id).xquat as Float32Array;
    g.position.set(p[0], p[1], p[2]);
    g.quaternion.set(q[1], q[2], q[3], q[0]);
  }
}

export function syncBall(world: World, engine: any): void {
  const p = engine.ballPose();
  if (!p) {
    world.ball.visible = false;
    return;
  }
  world.ball.visible = true;
  world.ball.position.set(p[0], p[2], -p[1]);
}
