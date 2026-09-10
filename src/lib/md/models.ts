// ── MicroDuck Trainer v2.0 – Modell-Registry ────────────────────────────────
// Metadaten für alle simulierten Roboter. Die Ente folgt exakt der Blaupause
// aus reference/game.js (Obs 61D "new-cmd-obs", 50 Hz Policy, decimation 4);
// der Unitree G1 hat eine eigene Obs-Definition (keine trainierte ONNX-Policy).

export type ModelId = "microduck" | "unitree_g1";

export interface PolicyDef {
  id: string; // Dateiname ohne Endung, z. B. "BEST_alpha_walking"
  label: string;
  url: string; // public-URL der ONNX-Datei
}

export interface PosePreset {
  id: string;
  label: string;
  targets: Record<string, number>; // Joint-Name -> Zielwinkel (rad)
}

export interface ModelMeta {
  id: ModelId;
  label: string;
  emoji: string;
  mjcf: string; // public-URL des MJCF
  meshBase: string; // Basis-URL für VFS-Meshes (ohne Dateiname)
  vfsPrefix: string; // Präfix im MuJoCo-VFS (MJCF meshdir)
  actionDim: number;
  obsDim: number;
  cmdSize: number;
  timestep: number;
  decimation: number; // Physik-Substeps pro Policy-Step → 50 Hz Policy
  actionScale: number; // ctrl = standPose + action * actionScale
  obsType: "new-cmd-obs" | "g1-v1";
  torsoBody: string; // Basis-Body (Freejoint-Träger)
  gyroSensor: string | null; // Sensor für Basis-Winkelgeschwindigkeit
  jointNames: string[]; // Reihenfolge = Aktuator-Reihenfolge
  defaultPose: number[]; // Neutralpose der Gelenke (Obs-Offset)
  keyframe: string; // Keyframe für Reset
  fallHeight: number; // Torso-Höhe darunter = gefallen
  fallUpZ: number; // projizierte Gravitation darüber (betragsmäßig) = gekippt
  targetHeight: number; // Soll-Höhe für Reward "Höhe"
  velocityLimit: { fwd: number; back: number; ang: number };
  /** v2.5: Anfahr-Boden (m/s) – ab diesem cmd fährt das Modell real an.
   *  Ente: ~0.26 (Node-Sim gemessen: 0.22 steht, 0.24 läuft). Darunter
   *  belohnt der Tracking-Reward STEHEN → die Stehen-Falle im Training. */
  cmdFloor: number;
  policies: PolicyDef[];
  defaultOnnx: string; // Policy-ID für den Boot
}

// ── Ente (MicroDuck / pollen-robotics alpha) ───────────────────────────────
const DUCK_JOINTS = [
  "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
];
// Aus reference/constants.js (STAND-Keyframe der mjlab-Szene, ONNX-Metadaten).
const DUCK_DEFAULT_POSE = [
  0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
  0.3490658503988659, 0.3490658503988659, 0, 0,
  0, 0.08726646259971647, 0.457924, 0.004940, -0.452984,
];

const DUCK_POLICIES: PolicyDef[] = [
  { id: "BEST_alpha_walking", label: "Gehen", url: "/policies/BEST_alpha_walking.onnx" },
  { id: "BEST_alpha_stand", label: "Aufstehen", url: "/policies/BEST_alpha_stand.onnx" },
  { id: "BEST_alpha_sitstand", label: "Sitzen/Aufstehen", url: "/policies/BEST_alpha_sitstand.onnx" },
  { id: "BEST_roller", label: "Roller fahren", url: "/policies/BEST_roller.onnx" },
  { id: "BEST_roller_crouch", label: "Roller ducken", url: "/policies/BEST_roller_crouch.onnx" },
  { id: "alpha_ground_pick", label: "Aufpicken", url: "/policies/alpha_ground_pick.onnx" },
  { id: "ball_kick_left", label: "Tritt links", url: "/policies/ball_kick_left.onnx" },
  { id: "ball_kick_right", label: "Tritt rechts", url: "/policies/ball_kick_right.onnx" },
  { id: "roulade", label: "Roulade", url: "/policies/roulade.onnx" },
];

// ── Unitree G1 (29 DoF, Reihenfolge = Aktuatoren im MJCF) ──────────────────
const G1_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
];

export const MODELS: Record<ModelId, ModelMeta> = {
  microduck: {
    id: "microduck",
    label: "Ente (MicroDuck)",
    emoji: "🦆",
    mjcf: "/robot/mjlab/robot_allcollisions.xml",
    meshBase: "/robot/mjlab/meshes",
    vfsPrefix: "assets",
    actionDim: 14,
    obsDim: 61, // [ang_vel(3), proj_grav(3), qpos(14), qvel(14), last_act(14), cmd(13)]
    cmdSize: 13,
    timestep: 0.005,
    decimation: 4, // 50 Hz Policy
    actionScale: 1.0,
    obsType: "new-cmd-obs",
    torsoBody: "trunk_base",
    gyroSensor: "imu_ang_vel",
    jointNames: DUCK_JOINTS,
    defaultPose: DUCK_DEFAULT_POSE,
    keyframe: "STAND",
    fallHeight: 0.06, // Trunk steht auf ~0.12 m (Original-Schwelle qpos[2]<0.02-Nähe)
    fallUpZ: 0.5, // projizierte Gravitation > -0.5 → >60° gekippt (wie Original)
    targetHeight: 0.12,
    velocityLimit: { fwd: 0.25, back: -0.2, ang: 1.0 },
    cmdFloor: 0.26,
    policies: DUCK_POLICIES,
    defaultOnnx: "BEST_alpha_walking",
  },
  unitree_g1: {
    id: "unitree_g1",
    label: "Mensch (Unitree G1)",
    emoji: "🧍",
    mjcf: "/robot/unitree_g1/g1.xml",
    meshBase: "/robot/unitree_g1/assets",
    vfsPrefix: "assets",
    actionDim: 29,
    // Eigene Definition (kein trainiertes ONNX): projGrav(3) + height(1) +
    // baseAngVel(3) + jointPos(29) + jointVel(29) + lastAction(29) + cmd(3) = 97
    obsDim: 97,
    cmdSize: 3,
    timestep: 0.002,
    decimation: 10, // 50 Hz Policy
    actionScale: 0.5,
    obsType: "g1-v1",
    torsoBody: "torso_link",
    gyroSensor: "imu-pelvis-angular-velocity",
    jointNames: G1_JOINTS,
    defaultPose: new Array(29).fill(0),
    keyframe: "stand",
    fallHeight: 0.5,
    fallUpZ: 0.5,
    targetHeight: 0.72,
    velocityLimit: { fwd: 0.5, back: -0.4, ang: 1.0 },
    cmdFloor: 0.12, // G1 trainiert ohne Warm-Start → moderater Boden
    policies: [],
    defaultOnnx: "",
  },
};

export function getModel(id: ModelId): ModelMeta {
  return MODELS[id];
}

export const MODEL_IDS: ModelId[] = ["microduck", "unitree_g1"];
