// ── MicroDuck Trainer v2.0 – MuJoCo-Engine-Wrapper ──────────────────────────
// Bauptaue: reference/game.js (Boot-Kette WASM → MJCF+VFS → MjData → Keyframe).
// Die App lädt die Runtimes ausschließlich via /wasm/... URLs (Static Export).

import type { ModelMeta, ModelId } from "./models";
import { getModel } from "./models";
import type { WorldBuild } from "./worldgen";

// Minimale strukturelle Typen für die Emscripten-Bindings (@mujoco/mujoco).
type MujocoModule = any;

interface EngineAddrs {
  qposAdr: number[]; // qpos-Adresse je Gelenk (Aktuator-Reihenfolge)
  dofAdr: number[]; // dof-Adresse je Gelenk
  gyroAdr: number; // Sensor-Startadresse Basis-Winkelgeschwindigkeit
  torsoId: number;
  keyId: number;
  ballQposAdr: number;
  ballDofAdr: number;
  ctrlRange: Float32Array | null; // [min,max] je Aktuator (G1)
  jntRange: [number, number][]; // Gelenklimits je Aktuator (0,0 = unbegrenzt)
}

let runtimePromise: Promise<MujocoModule> | null = null;

/** Lädt die MuJoCo-WASM-Fabrik nativ aus /wasm/mujoco.js (ESM, kein Bundler-Eingriff). */
export function loadMujocoRuntime(): Promise<MujocoModule> {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      const mod: any = await import(
        /* webpackIgnore: true */ /* turbopackIgnore: true */ "/wasm/mujoco.js"
      );
      const factory = mod.default ?? mod.loadMujoco ?? mod;
      return factory({
        locateFile: (p: string) => (p.endsWith(".wasm") ? "/wasm/mujoco.wasm" : p),
      });
    })();
  }
  return runtimePromise;
}

const BALL_RADIUS = 0.05; // Beach-Ball-Radius (wie Original)

// ── XML-Aufbereitung ─────────────────────────────────────────────────────────
// Gemeinsame Logik mit dem ES-Worker: src/lib/md/xml.ts (buildPhysicsXml).

import { buildPhysicsXml } from "./xml";

/** Holt alle Physik-Meshes in das MuJoCo-VFS: Name-Muster "assets/<Datei>". */
async function fillVfs(mujoco: MujocoModule, meshBase: string, vfsPrefix: string, files: string[]) {
  const vfs = new mujoco.MjVFS();
  await Promise.all(
    files.map(async (f) => {
      const r = await fetch(`${meshBase}/${f}`, { cache: "force-cache" });
      if (!r.ok) throw new Error(`Mesh-Fetch fehlgeschlagen: ${f} (${r.status})`);
      vfs.addBuffer(`${vfsPrefix}/${f}`, new Uint8Array(await r.arrayBuffer()));
    }),
  );
  return vfs;
}

function meta_obsDim(meta: ModelMeta | null): number {
  return meta ? meta.obsDim : 61;
}

// ── Engine ───────────────────────────────────────────────────────────────────

export class Engine {
  meta!: ModelMeta;
  model: any = null;
  data: any = null;
  private mujoco: MujocoModule | null = null;
  private addrs!: EngineAddrs;
  private vfs: any = null;

  obs = new Float32Array(0);
  lastAction = new Float32Array(0);
  cmd = new Float32Array(0);
  standPose: Float32Array = new Float32Array(0); // Gelenk-Sollwerte des Keyframes
  loadedId: ModelId | null = null;

  // ── v2.1: Welt, Punkt-Ziel, Imitation ──
  private worldKey: string | null = null;
  /** Joystick-Punkt (Welt-X/Y) für Reward pointChase/pointAvoid. */
  targetPoint: [number, number] | null = null;
  /** Gelenk-Ziele der Animation (Imitation); null = aus. */
  imitTarget: Float32Array | null = null;
  /** Wurzel-Höhen-Delta der Animation (m); null = aus. */
  imitRootDelta: number | null = null;
  /** Aktuelle Phase [sin, cos] der Animation (0..1 → 2π). */
  imitPhase: [number, number] | null = null;

  /** Kompiliert das Modell eines Roboters; alte Instanz wird freigegeben.
   *  world: Random-Welt (Geoms werden ins MJCF injiziert) – Änderung erzwingt Rebuild. */
  async load(modelId: ModelId, world?: WorldBuild | null): Promise<void> {
    const meta = getModel(modelId);
    const wk = world ? `${world.holes ? "h" : "f"}:${world.geoms.length}` : null;
    if (this.loadedId === modelId && this.model && wk === this.worldKey) {
      this.resetToKeyframe();
      return;
    }
    const mujoco = await loadMujocoRuntime();
    this.mujoco = mujoco;
    const src = await (await fetch(meta.mjcf)).text();
    const built = buildPhysicsXml(modelId, src, meta, world ?? null);
    this.worldKey = wk;

    this.dispose();
    this.vfs = await fillVfs(mujoco, meta.meshBase, meta.vfsPrefix, built.meshFiles);
    const model = mujoco.MjModel.from_xml_string(built.xml, this.vfs);
    const data = new mujoco.MjData(model);
    this.meta = meta;
    this.model = model;
    this.data = data;
    this.obs = new Float32Array(meta.obsDim);
    this.lastAction = new Float32Array(meta.actionDim);
    this.cmd = new Float32Array(meta.cmdSize);

    const nm = (type: number, name: string) =>
      mujoco.mj_name2id(model, type, name);
    const MJ_OBJ_BODY = (mujoco.mjtObj.mjOBJ_BODY as any).value ?? 1;
    const MJ_OBJ_KEY = (mujoco.mjtObj.mjOBJ_KEY as any).value ?? 6;
    const MJ_OBJ_SENSOR = (mujoco.mjtObj.mjOBJ_SENSOR as any).value ?? 8;

    const qposAdr: number[] = [];
    const dofAdr: number[] = [];
    for (const n of meta.jointNames) {
      qposAdr.push(model.jnt(n).qposadr);
      dofAdr.push(model.jnt(n).dofadr);
    }
    let gyroAdr = -1;
    try {
      if (meta.gyroSensor) gyroAdr = model.sensor(meta.gyroSensor).adr;
    } catch {
      gyroAdr = -1; // Fallback in baseAngVel(): Freejoint-DoFs (Weltframe)
    }

    let ctrlRange: Float32Array | null = null;
    const nCtrl = model.nu;
    // HINWEIS: actuator_ctrllimited ist in @mujoco/mujoco 3.13 als
    // memory_view<bool> gebunden und wirft beim Zugriff eine BindingError
    // (bool-View ist im WASM nicht registriert). Ob ein Aktuator limitiert
    // ist, wird daher über ctrlrange erschlossen (0/0 oder min==max = frei).
    const rng = model.actuator_ctrlrange as Float32Array;
    if (nCtrl > 0 && rng && rng.length >= nCtrl * 2) {
      ctrlRange = new Float32Array(nCtrl * 2);
      for (let i = 0; i < nCtrl; i++) {
        const lo = rng[i * 2], hi = rng[i * 2 + 1];
        const limited = lo !== hi && !(lo === 0 && hi === 0);
        ctrlRange[i * 2] = limited ? lo : -1e6;
        ctrlRange[i * 2 + 1] = limited ? hi : 1e6;
      }
    }

    // Gelenklimits: jnt_limited hat dasselbe Bindungsproblem → ebenfalls
    // über jnt_range erschließen ([0,0] = unbegrenzt).
    const jntRangeArr = model.jnt_range as Float32Array;
    const jntRange: [number, number][] = meta.jointNames.map((n) => {
      const j = model.jnt(n);
      if (jntRangeArr && jntRangeArr.length > j.id * 2 + 1) {
        return [jntRangeArr[j.id * 2], jntRangeArr[j.id * 2 + 1]];
      }
      return [0, 0];
    });

    this.addrs = {
      qposAdr,
      dofAdr,
      gyroAdr,
      torsoId: nm(MJ_OBJ_BODY, meta.torsoBody),
      keyId: nm(MJ_OBJ_KEY, meta.keyframe),
      ballQposAdr: model.jnt("ball_freejoint").qposadr,
      ballDofAdr: model.jnt("ball_freejoint").dofadr,
      ctrlRange,
      jntRange,
    };
    this.loadedId = modelId;
    this.resetToKeyframe();
  }

  resetToKeyframe(): void {
    if (!this.model || !this.data) return;
    this.mujoco!.mj_resetDataKeyframe(this.model, this.data, this.addrs.keyId);
    this.mujoco!.mj_forward(this.model, this.data);
    // Stand-Pose aus dem Keyframe lesen (G1: Ellbogen 1.28 rad etc.)
    const qpos = this.data.qpos as Float32Array;
    this.standPose = new Float32Array(this.meta.actionDim);
    for (let j = 0; j < this.meta.actionDim; j++) {
      this.standPose[j] = qpos[this.addrs.qposAdr[j]];
    }
    this.lastAction.fill(0);
    this.cmd.fill(0);
    this.imitPhase = null;
    // Positionaktuatoren (G1, kp=500): sonst zieht ctrl=0 die Gelenke nach 0!
    this.applyCtrlFromPose(this.standPose);
  }

  /** ctrl direkt auf Gelenk-Sollwerte setzen (Manuell-Modus / Recovery). */
  applyCtrlFromPose(targets: Float32Array): void {
    const ctrl = this.data.ctrl as Float32Array;
    const lim = this.addrs.ctrlRange;
    for (let j = 0; j < this.meta.actionDim; j++) {
      let v = targets[j];
      if (lim) v = Math.min(lim[j * 2 + 1], Math.max(lim[j * 2], v));
      ctrl[j] = v;
    }
  }

  /** Zusätzliches MjData auf demselben Modell (Main-Thread-Interleaving). */
  newData(): any {
    return new this.mujoco!.MjData(this.model);
  }

  /** Gelenklimits je Aktuator ([min,max]; [0,0] = unbegrenzt). */
  get jntRanges(): [number, number][] {
    return this.addrs.jntRange;
  }

  /** Aktuator-ctrl-Grenzen (flach [min,max] je Aktuator) oder null. */
  get ctrlLimits(): Float32Array | null {
    return this.addrs.ctrlRange;
  }

  /** Zugriff auf das MuJoCo-Modul (für Rig-Sync, mj_id2name etc.). */
  get mujoco(): MujocoModule {
    return this.mujoco;
  }

  /** qpos-Adresse des j-ten Aktuator-Gelenks (für Rig-Sync). */
  qposAdrOf(j: number): number {
    return this.addrs.qposAdr[j];
  }

  /** Aktiven Datenkontext wechseln (für Interleaved-Rollouts). */
  setData(d: any): void {
    this.data = d;
  }

  /** Policy-Aktion in ctrl übersetzen (je Modell) und decimation-fach steppen. */
  stepWithAction(action: Float32Array): Float32Array {
    const meta = this.meta;
    const ctrl = this.data.ctrl as Float32Array;
    const lim = this.addrs.ctrlRange;
    for (let j = 0; j < meta.actionDim; j++) {
      let v: number;
      if (meta.id === "microduck") {
        v = meta.defaultPose[j] + action[j] * meta.actionScale;
      } else {
        v = this.standPose[j] + action[j] * meta.actionScale;
      }
      if (lim) v = Math.min(lim[j * 2 + 1], Math.max(lim[j * 2], v));
      ctrl[j] = v;
    }
    return this.stepPhysics();
  }

  /** Physik decimation-fach advanceieren und neue Obs bauen. */
  stepPhysics(): Float32Array {
    for (let s = 0; s < this.meta.decimation; s++) {
      this.mujoco!.mj_step(this.model, this.data);
    }
    return this.buildObs();
  }

  /** Observation gemäß Modell-Layout (Rohansichten immer frisch lesen!). */
  buildObs(): Float32Array {
    const meta = this.meta;
    const data = this.data;
    const qpos = data.qpos as Float32Array;
    const qvel = data.qvel as Float32Array;
    const sens = data.sensordata as Float32Array;
    const obs = this.obs;
    let i = 0;
    const g = this.projGravity();
    if (meta.obsType === "new-cmd-obs") {
      for (let a = 0; a < 3; a++) obs[i++] = sens[this.addrs.gyroAdr + a];
      obs[i++] = g[0]; obs[i++] = g[1]; obs[i++] = g[2];
      for (let j = 0; j < meta.actionDim; j++) obs[i++] = qpos[this.addrs.qposAdr[j]] - meta.defaultPose[j];
      for (let j = 0; j < meta.actionDim; j++) obs[i++] = qvel[this.addrs.dofAdr[j]];
      for (let j = 0; j < meta.actionDim; j++) obs[i++] = this.lastAction[j];
      for (let c = 0; c < meta.cmdSize; c++) obs[i++] = this.cmd[c];
    } else {
      // g1-v1: projGrav(3), height(1), angVel(3), qpos(29), qvel(29), lastAct(29), cmd(3)
      obs[i++] = g[0]; obs[i++] = g[1]; obs[i++] = g[2];
      obs[i++] = this.height();
      const av = this.baseAngVel();
      obs[i++] = av[0]; obs[i++] = av[1]; obs[i++] = av[2];
      for (let j = 0; j < meta.actionDim; j++) obs[i++] = qpos[this.addrs.qposAdr[j]] - this.standPose[j];
      for (let j = 0; j < meta.actionDim; j++) obs[i++] = qvel[this.addrs.dofAdr[j]];
      for (let j = 0; j < meta.actionDim; j++) obs[i++] = this.lastAction[j];
      for (let c = 0; c < meta.cmdSize; c++) obs[i++] = this.cmd[c];
    }
    if (this.imitPhase && obs.length >= i + 2) {
      obs[i++] = this.imitPhase[0];
      obs[i++] = this.imitPhase[1];
    }
    return obs;
  }

  /** Obs-Buffer vergrößern (Imitations-Phase +2) oder zurücksetzen. */
  setImitObs(on: boolean): void {
    const need = meta_obsDim(this.meta) + (on ? 2 : 0);
    if (this.obs.length !== need) this.obs = new Float32Array(need);
    if (!on) this.imitPhase = null;
  }

  /** Sicht auf die Obs: ohne Phase (ONNX braucht exakte Dimension) oder mit. */
  obsFor(extra: boolean): Float32Array {
    const base = meta_obsDim(this.meta);
    if (!extra) {
      const v = this.obs.subarray(0, base);
      return v.length === base ? v : this.obs.subarray(0, base);
    }
    return this.obs;
  }

  // ── Zustands-Helfer ────────────────────────────────────────────────────────

  /** Welt-z-Gravitation in Trunk-Frame projiziert (aufrecht ≈ -1). */
  projGravity(): [number, number, number] {
    const xq = this.data.body(this.addrs.torsoId).xquat as Float32Array; // wxyz
    const w = xq[0], x = xq[1], y = xq[2], z = xq[3];
    // q⁻¹ · (0,0,-1)  ==  -(dritte Zeile der Rotationsmatrix)
    const gx = -2 * (x * z - w * y);
    const gy = -2 * (y * z + w * x);
    const gz = -(1 - 2 * (x * x + y * y));
    return [gx, gy, gz];
  }

  projGravZ(): number {
    const xq = this.data.body(this.addrs.torsoId).xquat as Float32Array;
    const x = xq[1], y = xq[2];
    return -(1 - 2 * (x * x + y * y));
  }

  /** Basis-Winkelgeschwindigkeit (Body-Frame via Gyro-Sensor, sonst Weltframe). */
  baseAngVel(): [number, number, number] {
    const sens = this.data.sensordata as Float32Array;
    if (this.addrs.gyroAdr >= 0 && sens && sens.length > this.addrs.gyroAdr + 2) {
      return [sens[this.addrs.gyroAdr], sens[this.addrs.gyroAdr + 1], sens[this.addrs.gyroAdr + 2]];
    }
    const qvel = this.data.qvel as Float32Array;
    return [qvel[3], qvel[4], qvel[5]];
  }

  height(): number {
    return this.data.qpos[2];
  }

  torsoPos(): [number, number, number] {
    const p = this.data.body(this.addrs.torsoId).xpos as Float32Array;
    return [p[0], p[1], p[2]];
  }

  torsoQuat(): [number, number, number, number] {
    const q = this.data.body(this.addrs.torsoId).xquat as Float32Array;
    return [q[0], q[1], q[2], q[3]];
  }

  /** Gierwinkel des Trunk (Freejoint-Quaternion, MJCF-Bodenkoordinaten). */
  yaw(): number {
    const qpos = this.data.qpos as Float32Array;
    return Math.atan2(
      2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
      1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
    );
  }

  jointAngles(): number[] {
    const qpos = this.data.qpos as Float32Array;
    return this.addrs.qposAdr.map((a) => qpos[a]);
  }

  jointVelocities(): number[] {
    const qvel = this.data.qvel as Float32Array;
    return this.addrs.dofAdr.map((a) => qvel[a]);
  }

  jointAccel(): number[] {
    const qacc = this.data.qacc as Float32Array;
    return this.addrs.dofAdr.map((a) => qacc[a]);
  }

  isFallen(): boolean {
    if (!this.data) return false;
    const z = this.data.qpos[2];
    const gz = this.projGravZ();
    if (!Number.isFinite(z) || !Number.isFinite(gz)) return true;
    return z < this.meta.fallHeight || gz > -this.meta.fallUpZ;
  }

  /** Ball ~0.4 m vor dem Roboter spawnen (beach-ball feel, wie Original). */
  spawnBallAhead(): void {
    const qpos = this.data.qpos as Float32Array;
    const qvel = this.data.qvel as Float32Array;
    const yaw = this.yaw();
    const heading = yaw + (Math.random() - 0.5) * 0.7;
    const dist = 0.4 + Math.random() * 0.2;
    const a = this.addrs.ballQposAdr, d = this.addrs.ballDofAdr;
    qpos[a] = qpos[0] + Math.cos(heading) * dist;
    qpos[a + 1] = qpos[1] + Math.sin(heading) * dist;
    qpos[a + 2] = BALL_RADIUS + 0.02;
    qpos[a + 3] = 1; qpos[a + 4] = 0; qpos[a + 5] = 0; qpos[a + 6] = 0;
    for (let i = 0; i < 6; i++) qvel[d + i] = 0;
    this.mujoco!.mj_forward(this.model, this.data);
  }

  ballPose(): [number, number, number] | null {
    if (!this.data) return null;
    const a = this.addrs.ballQposAdr;
    const q = this.data.qpos as Float32Array;
    if (Math.abs(q[a]) > 20) return null; // geparkt
    return [q[a], q[a + 1], q[a + 2]];
  }

  dispose(): void {
    try {
      this.data?.delete?.();
      this.model?.delete?.();
      this.vfs?.delete?.();
    } catch {
      // Emscripten-Handles ignorieren
    }
    this.data = null;
    this.model = null;
    this.vfs = null;
    this.loadedId = null;
  }
}
