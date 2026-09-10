"use client";

// ── MicroDuck Trainer v2.0 – Bewertungs-Panel (Reward-Terme + KI) ────────────

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { Telemetry } from "@/lib/md/app-core";
import {
  REWARD_TERMS,
  presetConfig,
  type PresetId,
  type RewardConfig,
} from "@/lib/md/rewards";
import {
  GOAL_CHIPS,
  generateReward,
  analyzeTraining,
  type GoalId,
} from "@/lib/md/rewardAI";
import { Sparkles, Wand2 } from "lucide-react";

interface RewardPanelProps {
  tel: Telemetry;
  onChange: (cfg: RewardConfig) => void;
}

const PRESETS: PresetId[] = ["gehen", "stehen", "robust"];

export default function RewardPanel({ tel, onChange }: RewardPanelProps) {
  const cfg = tel.reward;
  const [goals, setGoals] = useState<GoalId[]>([]);
  const [aiText, setAiText] = useState<string>("");

  const modelId = tel.modelId ?? "microduck";

  const suggestions = useMemo(() => {
    const es = tel.es;
    if (!es || es.history.length < 2) return [];
    return analyzeTraining(modelId, {
      fallRate: es.fellRate,
      fitnessHistory: es.history,
      meanFallHeightRatio: 0,
    });
  }, [tel.es, modelId]);

  const toggleGoal = (id: GoalId) =>
    setGoals((g) => (g.includes(id) ? g.filter((x) => x !== id) : [...g, id]));

  const patchTerm = (id: string, patch: Partial<{ enabled: boolean; weight: number; param: number }>) => {
    const next: RewardConfig = {
      version: 2,
      terms: { ...cfg.terms, [id]: { ...cfg.terms[id], ...patch } },
    };
    onChange(next);
  };

  return (
    <div className="space-y-4">
      {/* Presets */}
      <div className="flex gap-2">
        {PRESETS.map((p) => (
          <Button
            key={p}
            variant="outline"
            size="sm"
            className="h-10 flex-1 border-cyan-500/30 text-cyan-300 capitalize hover:bg-cyan-500/10"
            onClick={() => onChange(presetConfig(modelId, p))}
          >
            {p === "gehen" ? "Gehen" : p === "stehen" ? "Stehen" : "Robust"}
          </Button>
        ))}
      </div>

      {/* Terme */}
      <div className="max-h-72 space-y-2 overflow-y-auto pr-1 md:max-h-none">
        {REWARD_TERMS.map((def) => {
          const t = cfg.terms[def.id];
          if (!t) return null;
          return (
            <div
              key={def.id}
              className={`rounded-lg border p-2.5 transition-colors ${
                t.enabled ? "border-cyan-500/25 bg-black/30" : "border-slate-700/50 bg-black/20 opacity-70"
              }`}
            >
              <div className="flex items-center gap-2">
                <Switch
                  checked={t.enabled}
                  onCheckedChange={(v) => patchTerm(def.id, { enabled: v })}
                  aria-label={`${def.label} ein/aus`}
                />
                <span className={`text-sm ${def.penalty ? "text-amber-300" : "text-slate-200"}`}>
                  {def.label}
                </span>
                <span className="ml-auto font-mono text-xs text-cyan-300">
                  {t.weight >= 0 ? "+" : ""}
                  {t.weight.toFixed(1)}
                </span>
              </div>
              <p className="mt-1 pl-11 text-[11px] leading-snug text-slate-500">{def.desc}</p>
              <div className="mt-2 flex items-center gap-2 pl-11">
                <Slider
                  value={[t.weight]}
                  min={-5}
                  max={5}
                  step={0.1}
                  onValueChange={(v) => patchTerm(def.id, { weight: v[0], enabled: v[0] !== 0 })}
                  className="flex-1 [&_[data-slot=slider-range]]:bg-cyan-400 [&_[data-slot=slider-thumb]]:border-cyan-300"
                />
              </div>
              {def.hasParam && (
                <div className="mt-2 flex items-center gap-2 pl-11">
                  <span className="w-24 shrink-0 text-[10px] uppercase tracking-wide text-slate-500">
                    {def.paramLabel}
                  </span>
                  <Slider
                    value={[t.param]}
                    min={def.paramMin}
                    max={def.paramMax}
                    step={def.paramStep}
                    onValueChange={(v) => patchTerm(def.id, { param: v[0] })}
                    className="flex-1 [&_[data-slot=slider-range]]:bg-amber-300 [&_[data-slot=slider-thumb]]:border-amber-200"
                  />
                  <span className="w-12 shrink-0 text-right font-mono text-xs text-amber-300">
                    {t.param.toFixed(2)}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* KI-Bewertung */}
      <div className="rounded-lg border border-fuchsia-500/25 bg-fuchsia-500/5 p-3">
        <div className="flex items-center gap-2">
          <Wand2 className="h-4 w-4 text-fuchsia-400" />
          <h3 className="text-sm font-semibold text-fuchsia-300">KI-Bewertung (offline)</h3>
        </div>
        <p className="mt-1 text-[11px] text-slate-400">
          Ziele antippen – die KI baut daraus eine passende Bewertung. Funktioniert komplett
          offline per Regel-Templates.
        </p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {GOAL_CHIPS.map((g) => (
            <button
              key={g.id}
              type="button"
              onClick={() => toggleGoal(g.id)}
              className={`min-h-[36px] rounded-full border px-3 text-xs transition-colors ${
                goals.includes(g.id)
                  ? "border-fuchsia-400/60 bg-fuchsia-500/20 text-fuchsia-200"
                  : "border-slate-600/60 text-slate-400 hover:border-fuchsia-400/40 hover:text-fuchsia-200"
              }`}
            >
              {g.emoji} {g.label}
            </button>
          ))}
        </div>
        <Button
          onClick={() => {
            const r = generateReward(goals, modelId);
            onChange(r.config);
            setAiText(r.explanation);
          }}
          className="mt-3 h-11 w-full gap-1.5 bg-fuchsia-500 font-semibold text-white hover:bg-fuchsia-400"
        >
          <Sparkles className="h-4 w-4" /> Bewertung generieren
        </Button>
        {aiText && (
          <p className="mt-2 rounded-md bg-black/40 p-2 text-[11px] leading-relaxed text-fuchsia-200/90">
            {aiText}
          </p>
        )}
      </div>

      {/* Adaptive Vorschläge */}
      {suggestions.length > 0 && (
        <div className="space-y-2">
          <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" /> Vorschläge aus dem Training
          </h3>
          {suggestions.map((s) => (
            <div
              key={s.id}
              className="flex items-center gap-2 rounded-lg border border-cyan-500/20 bg-black/30 p-2.5"
            >
              <p className="flex-1 text-[11px] leading-snug text-slate-300">{s.text}</p>
              <Button
                size="sm"
                onClick={() => {
                  const next: RewardConfig = JSON.parse(JSON.stringify(cfg));
                  s.patch(next);
                  onChange(next);
                }}
                className="h-9 shrink-0 bg-cyan-500 text-black hover:bg-cyan-400"
              >
                Übernehmen
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
