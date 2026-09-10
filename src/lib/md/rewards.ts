// ── MicroDuck Trainer v2.0 – Konfigurierbare Bewertung (Reward) ──────────────
// Alle Terme ein/aus + Gewicht-Slider; Persistenz pro Modell im localStorage.

import type { ModelId } from "./models";
import { getModel } from "./models";

export interface TermState {
  enabled: boolean;
  weight: number;
  param: number; // bedeutung je Term (z. B. Zielhöhe, Ziel-Tempo)
}

export interface RewardConfig {
  version: 2;
  terms: Record<string, TermState>;
}

export interface RewardTermDef {
  id: string;
  label: string;
  desc: string;
  defaultWeight: number;
  hasParam: boolean;
  paramLabel: string;
  paramMin: number;
  paramMax: number;
  paramStep: number;
  defaultParam: (modelId: ModelId) => number;
  /** Gewicht < 0 = Strafterm (wird in der UI farblich markiert) */
  penalty?: boolean;
}

export const REWARD_TERMS: RewardTermDef[] = [
  {
    id: "upright", label: "Aufrecht", desc: "Aufrechte Körperhaltung (Up-Vektor z-Anteil)",
    defaultWeight: 1.5, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 1, paramStep: 0.01,
    defaultParam: () => 0,
  },
  {
    id: "height", label: "Höhe", desc: "Abweichung von der Ziel-Körperhöhe",
    defaultWeight: 0.8, hasParam: true, paramLabel: "Zielhöhe (m)", paramMin: 0.05, paramMax: 1.2, paramStep: 0.01,
    defaultParam: (m) => getModel(m).targetHeight,
  },
  {
    id: "forward", label: "Vorwärtstempo", desc: "Vorwärtsgeschwindigkeit in Laufrichtung",
    defaultWeight: 2.0, hasParam: true, paramLabel: "Ziel-Tempo (m/s)", paramMin: -1, paramMax: 1.5, paramStep: 0.05,
    defaultParam: (m) => (m === "microduck" ? 0.3 : 0.4),
  },
  {
    id: "lateral", label: "Seitwärts", desc: "Seitliche Geschwindigkeit (Strafe oder Belohnung)",
    defaultWeight: 0, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 1, paramStep: 0.01,
    defaultParam: () => 0,
  },
  {
    id: "yaw", label: "Gieren zum Ziel", desc: "Dreht actual ↔ kommandierte Gierrate (Training: keine Drift)",
    defaultWeight: 0.5, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 1, paramStep: 0.01,
    defaultParam: () => 0,
  },
  {
    id: "alive", label: "Lebendbonus", desc: "Kleiner Bonus pro überlebtem Schritt",
    defaultWeight: 0.5, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 1, paramStep: 0.01,
    defaultParam: () => 0,
  },
  {
    id: "energy", label: "Energie-Strafe", desc: "Strafe für hohe Aktionen (Stromsparen)",
    defaultWeight: 0.5, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 1, paramStep: 0.01,
    penalty: true, defaultParam: () => 0,
  },
  {
    id: "smoothness", label: "Aktionsglätte", desc: "Strafe für ruckartige Aktionswechsel",
    defaultWeight: 0.3, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 1, paramStep: 0.01,
    penalty: true, defaultParam: () => 0,
  },
  {
    id: "jointAccel", label: "Gelenkbeschleunigung", desc: "Strafe für hektische Gelenkbewegungen",
    defaultWeight: 0.2, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 1, paramStep: 0.01,
    penalty: true, defaultParam: () => 0,
  },
  {
    id: "jointLimit", label: "Gelenklimit-Strafe", desc: "Strafe nahe der Gelenkanschläge",
    defaultWeight: 0.5, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 1, paramStep: 0.01,
    penalty: true, defaultParam: () => 0,
  },
  {
    id: "contact", label: "Kontakt-Strafe", desc: "Impuls-Spitzen (harte Aufprälle) bestrafen – Näherung über Basis-Beschleunigung",
    defaultWeight: 0.2, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 1, paramStep: 0.01,
    penalty: true, defaultParam: () => 0,
  },
  {
    id: "fall", label: "Fall-Strafe", desc: "Einmalige große Strafe bei einem Sturz",
    defaultWeight: 10, hasParam: false, paramLabel: "", paramMin: 0, paramMax: 20, paramStep: 0.5,
    penalty: true, defaultParam: () => 0,
  },
];

export const TERM_IDS = new Set(REWARD_TERMS.map((t) => t.id));

export function defaultRewardConfig(modelId: ModelId): RewardConfig {
  const terms: Record<string, TermState> = {};
  for (const def of REWARD_TERMS) {
    terms[def.id] = {
      enabled: def.defaultWeight !== 0,
      weight: def.defaultWeight,
      param: def.defaultParam(modelId),
    };
  }
  return { version: 2, terms };
}

// ── Presets pro Modell ───────────────────────────────────────────────────────

export type PresetId = "gehen" | "stehen" | "robust";

export const PRESET_LABELS: Record<PresetId, string> = {
  gehen: "Gehen",
  stehen: "Stehen",
  robust: "Robust",
};

export function presetConfig(modelId: ModelId, preset: PresetId): RewardConfig {
  const cfg = defaultRewardConfig(modelId);
  const set = (id: string, enabled: boolean, weight?: number, param?: number) => {
    const t = cfg.terms[id];
    t.enabled = enabled;
    if (weight !== undefined) t.weight = weight;
    if (param !== undefined) t.param = param;
  };
  if (preset === "gehen") {
    set("upright", true, 1.5);
    set("height", true, 0.3);
    set("forward", true, 2.5, modelId === "microduck" ? 0.3 : 0.45);
    set("lateral", false, 0);
    set("yaw", true, 0.5);
    set("alive", true, 0.5);
    set("energy", true, 0.5);
    set("smoothness", true, 0.3);
    set("jointAccel", true, 0.2);
    set("jointLimit", true, 0.5);
    set("contact", true, 0.2);
    set("fall", true, 10);
  } else if (preset === "stehen") {
    set("upright", true, 2.0);
    set("height", true, 2.0, modelId === "microduck" ? 0.12 : 0.72);
    set("forward", false, 0);
    set("lateral", false, 0);
    set("yaw", false, 0);
    set("alive", true, 1.0);
    set("energy", true, 1.0);
    set("smoothness", true, 0.5);
    set("jointAccel", true, 0.5);
    set("jointLimit", true, 0.5);
    set("contact", true, 0.5);
    set("fall", true, 10);
  } else {
    set("upright", true, 2.5);
    set("height", true, 0.5);
    set("forward", true, 1.0);
    set("lateral", false, 0);
    set("yaw", true, 0.5);
    set("alive", true, 1.0);
    set("energy", true, 0.8);
    set("smoothness", true, 0.8);
    set("jointAccel", true, 0.4);
    set("jointLimit", true, 1.0);
    set("contact", true, 0.5);
    set("fall", true, 15);
  }
  return cfg;
}

// ── Persistenz ───────────────────────────────────────────────────────────────

const key = (modelId: ModelId) => `mdt_v2_reward_${modelId}`;

export function loadRewardConfig(modelId: ModelId): RewardConfig {
  try {
    const raw = localStorage.getItem(key(modelId));
    if (raw) {
      const parsed = JSON.parse(raw) as RewardConfig;
      if (parsed?.version === 2 && parsed.terms) {
        // Fehlende Terme ergänzen (Vorwärtskompatibilität)
        const base = defaultRewardConfig(modelId);
        for (const id of TERM_IDS) {
          if (!parsed.terms[id]) parsed.terms[id] = base.terms[id];
        }
        return parsed;
      }
    }
  } catch {
    // kaputte Daten → Default
  }
  return defaultRewardConfig(modelId);
}

export function saveRewardConfig(modelId: ModelId, cfg: RewardConfig): void {
  try {
    localStorage.setItem(key(modelId), JSON.stringify(cfg));
  } catch {
    // Speicher voll / privater Modus
  }
}
