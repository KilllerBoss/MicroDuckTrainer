// ── MicroDuck Trainer v2.0 – Action-Mapping (Gamepad → Roboter) ──────────────
// Pro Modell editierbar und persistiert (mdt_v2_map_<modelId>).

import type { ModelId } from "./models";
import { getModel } from "./models";

export type SourceId = "joyX" | "joyY" | "A" | "B" | "C" | "D";
export type TargetType = "gelenk" | "cmd" | "policy" | "pose";

export interface MappingEntry {
  source: SourceId;
  targetType: TargetType;
  /** je nach Typ: Joint-Namen (kommagetrennt), cmd_x/cmd_y/cmd_yaw, Policy-ID oder Pose-ID */
  target: string;
  gain: number;
}

export const SOURCES: { id: SourceId; label: string; isStick: boolean }[] = [
  { id: "joyX", label: "Joystick X", isStick: true },
  { id: "joyY", label: "Joystick Y", isStick: true },
  { id: "A", label: "A", isStick: false },
  { id: "B", label: "B", isStick: false },
  { id: "C", label: "C", isStick: false },
  { id: "D", label: "D", isStick: false },
];

// ── Defaults ─────────────────────────────────────────────────────────────────

export function defaultMapping(modelId: ModelId): MappingEntry[] {
  if (modelId === "microduck") {
    return [
      { source: "joyX", targetType: "cmd", target: "cmd_yaw", gain: 2 },
      { source: "joyY", targetType: "cmd", target: "cmd_x", gain: 1.5 },
      { source: "A", targetType: "policy", target: "BEST_alpha_walking", gain: 1 },
      { source: "B", targetType: "policy", target: "BEST_alpha_stand", gain: 1 },
      { source: "C", targetType: "policy", target: "BEST_roller", gain: 1 },
      { source: "D", targetType: "policy", target: "ball_kick_left", gain: 1 },
    ];
  }
  return [
    {
      source: "joyY", targetType: "gelenk",
      target: "left_hip_pitch_joint,right_hip_pitch_joint", gain: 0.4,
    },
    { source: "joyX", targetType: "gelenk", target: "waist_yaw_joint", gain: 0.6 },
    {
      source: "A", targetType: "pose", target: "wave", gain: 1,
    },
    { source: "B", targetType: "pose", target: "squat", gain: 1 },
    { source: "C", targetType: "pose", target: "arms_up", gain: 1 },
    { source: "D", targetType: "pose", target: "neutral", gain: 1 },
  ];
}

// ── Pose-Presets (G1) ────────────────────────────────────────────────────────

import type { PosePreset } from "./models";

export const G1_POSES: PosePreset[] = [
  {
    id: "wave", label: "Winken",
    targets: {
      right_shoulder_pitch_joint: -1.6,
      right_shoulder_roll_joint: -0.2,
      right_elbow_joint: -0.5,
      left_shoulder_pitch_joint: 0,
      left_elbow_joint: 1.28,
    },
  },
  {
    id: "squat", label: "Hocke",
    targets: {
      left_knee_joint: 0.9, right_knee_joint: 0.9,
      left_hip_pitch_joint: -0.9, right_hip_pitch_joint: -0.9,
      left_ankle_pitch_joint: 0.35, right_ankle_pitch_joint: 0.35,
    },
  },
  {
    id: "arms_up", label: "Arme hoch",
    targets: {
      left_shoulder_pitch_joint: -2.8, right_shoulder_pitch_joint: -2.8,
      left_elbow_joint: 0.2, right_elbow_joint: 0.2,
    },
  },
  {
    id: "neutral", label: "Grundstellung",
    targets: {}, // alle Gelenke → 0
  },
];

export function posesFor(modelId: ModelId): PosePreset[] {
  if (modelId === "unitree_g1") return G1_POSES;
  // Ente: Sitz-Pose als Preset (aus reference/duck.js SITTING_POSE)
  return [
    {
      id: "sit", label: "Sitz",
      targets: {
        left_hip_yaw: 0.0, left_hip_roll: 0.0, left_hip_pitch: -0.5236,
        left_knee: 1.0472, left_ankle: 0.0,
        neck_pitch: 1.02, head_pitch: 0.9, head_yaw: 0.0, head_roll: 0.0,
        right_hip_yaw: 0.0, right_hip_roll: 0.0, right_hip_pitch: 0.5236,
        right_knee: -1.0472, right_ankle: 0.0,
      },
    },
    {
      id: "stand", label: "Grundstellung",
      targets: {},
    },
  ];
}

export function poseById(modelId: ModelId, id: string): PosePreset | null {
  return posesFor(modelId).find((p) => p.id === id) ?? null;
}

// ── Persistenz ───────────────────────────────────────────────────────────────

const mapKey = (modelId: ModelId) => `mdt_v2_map_${modelId}`;

export function loadMapping(modelId: ModelId): MappingEntry[] {
  try {
    const raw = localStorage.getItem(mapKey(modelId));
    if (raw) {
      const parsed = JSON.parse(raw) as MappingEntry[];
      if (Array.isArray(parsed) && parsed.length >= 2) {
        // Quellen auffüllen, falls eine fehlt
        const base = defaultMapping(modelId);
        for (const b of base) {
          if (!parsed.some((p) => p.source === b.source)) parsed.push(b);
        }
        return parsed;
      }
    }
  } catch {
    // ignorieren
  }
  return defaultMapping(modelId);
}

export function saveMapping(modelId: ModelId, mapping: MappingEntry[]): void {
  try {
    localStorage.setItem(mapKey(modelId), JSON.stringify(mapping));
  } catch {
    // ignorieren
  }
}

/** Ziel-Optionen für den Mapping-Editor. */
export function targetOptions(modelId: ModelId): { type: TargetType; value: string; label: string }[] {
  const meta = getModel(modelId);
  const opts: { type: TargetType; value: string; label: string }[] = [];
  if (modelId === "microduck") {
    opts.push(
      { type: "cmd", value: "cmd_x", label: "Command: Vorwärts (cmd_x)" },
      { type: "cmd", value: "cmd_y", label: "Command: Seitwärts (cmd_y)" },
      { type: "cmd", value: "cmd_yaw", label: "Command: Gieren (cmd_yaw)" },
    );
    for (const p of meta.policies) {
      opts.push({ type: "policy", value: p.id, label: `Policy: ${p.label}` });
    }
    for (const p of posesFor(modelId)) {
      opts.push({ type: "pose", value: p.id, label: `Pose: ${p.label}` });
    }
  } else {
    for (const j of meta.jointNames) {
      opts.push({ type: "gelenk", value: j, label: `Gelenk: ${j}` });
    }
    for (const p of posesFor(modelId)) {
      opts.push({ type: "pose", value: p.id, label: `Pose: ${p.label}` });
    }
  }
  return opts;
}
