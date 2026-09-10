// ── MicroDuck Trainer v2.0 – KI-Bewertungsgenerator (OFFLINE) ────────────────
// Erzeugt Reward-Konfigurationen aus Ziel-Chips per Regel-Templates und
// schlägt adaptive Anpassungen aus Trainingsstatistiken vor. Kein Netzwerk!

import type { ModelId } from "./models";
import type { RewardConfig } from "./rewards";
import { defaultRewardConfig } from "./rewards";

export const GOAL_CHIPS = [
  { id: "walk", label: "Vorwärts gehen", emoji: "🚶" },
  { id: "run", label: "Schnell laufen", emoji: "🏃" },
  { id: "upright", label: "Aufrecht bleiben", emoji: "🧍" },
  { id: "energy", label: "Wenig Energie", emoji: "🔋" },
  { id: "calm", label: "Ruhige Bewegung", emoji: "🧘" },
  { id: "tall", label: "Hoch stehen", emoji: "📏" },
  { id: "dance", label: "Tanzen/Hüpfen", emoji: "💃" },
  { id: "backward", label: "Rückwärts", emoji: "↩️" },
  { id: "sideways", label: "Seitwärts", emoji: "↔️" },
  { id: "balance", label: "Balancieren", emoji: "⚖️" },
] as const;

export type GoalId = (typeof GOAL_CHIPS)[number]["id"];

export interface AiResult {
  config: RewardConfig;
  explanation: string;
}

function apply(cfg: RewardConfig, id: string, patch: Partial<{ enabled: boolean; weight: number; param: number }>) {
  const t = cfg.terms[id];
  if (!t) return;
  Object.assign(t, patch);
}

export function generateReward(goals: GoalId[], modelId: ModelId): AiResult {
  const cfg = defaultRewardConfig(modelId);
  const reasons: string[] = [];
  const duck = modelId === "microduck";

  // Grundgerüst: robust und stabil
  apply(cfg, "upright", { enabled: true, weight: 1.5 });
  apply(cfg, "alive", { enabled: true, weight: 0.5 });
  apply(cfg, "fall", { enabled: true, weight: 10 });
  apply(cfg, "energy", { enabled: true, weight: 0.5 });
  reasons.push("Stabilisator-Grundgerüst (aufrecht, Lebendbonus, Fall-Strafe) ist immer aktiv.");

  if (goals.includes("walk")) {
    apply(cfg, "forward", { enabled: true, weight: 2.5, param: 0.3 });
    apply(cfg, "yaw", { enabled: true, weight: 0.5 });
    reasons.push("„Vorwärts gehen“ → Vorwärtstempo 2.5 mit moderatem Ziel 0.3 m/s, Gierraten-Abgleich an.");
  }
  if (goals.includes("run")) {
    apply(cfg, "forward", { enabled: true, weight: 3.5, param: duck ? 0.55 : 0.8 });
    apply(cfg, "energy", { enabled: true, weight: Math.max(0.2, (cfg.terms.energy?.weight ?? 0.5) - 0.2) });
    reasons.push("„Schnell laufen“ → höheres Zieltempo und höhere Belohnung, Energie-Strafe etwas gelockert.");
  }
  if (goals.includes("upright") || goals.includes("balance")) {
    apply(cfg, "upright", { enabled: true, weight: (cfg.terms.upright?.weight ?? 1.5) + 1.0 });
    apply(cfg, "contact", { enabled: true, weight: 0.4 });
    apply(cfg, "fall", { enabled: true, weight: (cfg.terms.fall?.weight ?? 10) + 4 });
    reasons.push("„Aufrecht/Balancieren“ → Aufrecht-Gewicht +1.0, Fall-Strafe +4, Kontakt-Strafe an.");
  }
  if (goals.includes("energy")) {
    apply(cfg, "energy", { enabled: true, weight: (cfg.terms.energy?.weight ?? 0.5) + 0.8 });
    apply(cfg, "jointAccel", { enabled: true, weight: 0.5 });
    reasons.push("„Wenig Energie“ → Energie-Strafe +0.8, Gelenkbeschleunigung an.");
  }
  if (goals.includes("calm")) {
    apply(cfg, "smoothness", { enabled: true, weight: 1.0 });
    apply(cfg, "jointAccel", { enabled: true, weight: 0.6 });
    apply(cfg, "forward", { enabled: true, weight: Math.min(cfg.terms.forward?.weight ?? 1, 1.5), param: 0.15 });
    reasons.push("„Ruhige Bewegung“ → Aktionsglätte 1.0, sanftes Tempo, hektische Gelenke bestraft.");
  }
  if (goals.includes("tall")) {
    apply(cfg, "height", { enabled: true, weight: 2.0, param: duck ? 0.13 : 0.78 });
    apply(cfg, "upright", { enabled: true, weight: (cfg.terms.upright?.weight ?? 1.5) + 0.5 });
    reasons.push(`„Hoch stehen“ → Zielhöhe ${duck ? "0.13 m (Enten-Trunk max.)" : "0.78 m (Pelvis)"}, Aufrecht +0.5.`);
  }
  if (goals.includes("dance")) {
    apply(cfg, "forward", { enabled: false, weight: 0 });
    apply(cfg, "jointLimit", { enabled: true, weight: 0.3 });
    apply(cfg, "energy", { enabled: true, weight: 0.3 });
    apply(cfg, "smoothness", { enabled: true, weight: 0.4 });
    apply(cfg, "alive", { enabled: true, weight: 1.0 });
    reasons.push("„Tanzen/Hüpfen“ → Tempo aus, Lebendbonus hoch, nur weiche Grenzen – die ES findet eigene Rhythmen.");
  }
  if (goals.includes("backward")) {
    apply(cfg, "forward", { enabled: true, weight: 2.5, param: -0.2 });
    reasons.push("„Rückwärts“ → Ziel-Tempo −0.2 m/s (negatives Ziel = rückwärts).");
  }
  if (goals.includes("sideways")) {
    apply(cfg, "lateral", { enabled: true, weight: 2.0 });
    apply(cfg, "forward", { enabled: true, weight: 0.5, param: 0.1 });
    reasons.push("„Seitwärts“ → Seitwärts-Geschwindigkeit belohnt (2.0), Vorwärts nur schwach.");
  }
  if (goals.length === 0) {
    reasons.push("Keine Chips gewählt → es bleibt das robuste Grundgerüst; wähle Ziele für spezifischere Bewertung.");
  }

  const explanation =
    `KI hat diese Bewertung vorgeschlagen, weil: ${reasons.join(" ")}`;
  return { config: cfg, explanation };
}

// ── Adaptive Vorschläge aus Trainingsstatistiken ─────────────────────────────

export interface TrainingStats {
  fallRate: number; // Anteil fehlgeschlagener Rollouts (0..1)
  fitnessHistory: number[]; // beste Fitness je Generation
  meanFallHeightRatio: number; // optionaler Hinweis (0..1)
}

export interface AiSuggestion {
  id: string;
  text: string;
  patch: (cfg: RewardConfig) => void;
}

export function analyzeTraining(modelId: ModelId, stats: TrainingStats): AiSuggestion[] {
  const out: AiSuggestion[] = [];
  const hist = stats.fitnessHistory;
  const last = hist.slice(-8);
  const improving = last.length >= 4 && last[last.length - 1] > last[0] * 1.02;

  if (stats.fallRate > 0.4) {
    out.push({
      id: "falling",
      text: modelId === "microduck"
        ? "Die Ente fällt oft → Gewicht „aufrecht“ +0.5 und Fall-Strafe +2."
        : "G1 fällt oft → Gewicht „aufrecht“ +0.5 und Fall-Strafe +2.",
      patch: (cfg) => {
        cfg.terms.upright.enabled = true;
        cfg.terms.upright.weight = Math.min(5, cfg.terms.upright.weight + 0.5);
        cfg.terms.fall.enabled = true;
        cfg.terms.fall.weight = Math.min(20, cfg.terms.fall.weight + 2);
      },
    });
  }
  if (!improving && stats.fallRate < 0.15 && hist.length >= 4) {
    out.push({
      id: "plateau",
      text: "Fitness stagniert bei stabilen Läufen → mehr Tempo wagen: „Vorwärtstempo“ +0.5, Energie-Strafe −0.2.",
      patch: (cfg) => {
        cfg.terms.forward.enabled = true;
        cfg.terms.forward.weight += 0.5;
        cfg.terms.energy.weight = Math.max(0, cfg.terms.energy.weight - 0.2);
      },
    });
  }
  if (hist.length >= 6) {
    const drop = hist[hist.length - 1] - hist[hist.length - 6];
    if (drop < -1) {
      out.push({
        id: "regression",
        text: "Fitness sinkt → Bewertung überladen? Aktionsglätte und Gelenkbeschleunigung −0.2 each.",
        patch: (cfg) => {
          cfg.terms.smoothness.weight = Math.max(0, cfg.terms.smoothness.weight - 0.2);
          cfg.terms.jointAccel.weight = Math.max(0, cfg.terms.jointAccel.weight - 0.2);
        },
      });
    }
  }
  if (stats.meanFallHeightRatio > 0 && stats.meanFallHeightRatio < 0.5 && stats.fallRate > 0.2) {
    out.push({
      id: "knee",
      text: "Viele Stürze mit tiefem Körper → „Höhe“-Gewicht +0.5 (früher aufstehen).",
      patch: (cfg) => {
        cfg.terms.height.enabled = true;
        cfg.terms.height.weight += 0.5;
      },
    });
  }
  return out.slice(0, 3);
}
