"use client";

// ── MicroDuck Trainer v2.1 – Welt-Panel (Random-World-Generator) ─────────────

import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import type { Telemetry } from "@/lib/md/app-core";
import type { WorldConfig, WorldFeatures } from "@/lib/md/worldgen";
import { Dices, Mountain } from "lucide-react";

interface WorldPanelProps {
  tel: Telemetry;
  onApply: (cfg: WorldConfig) => void;
  onReroll: () => void;
}

const FEATURE_LABELS: { key: keyof WorldFeatures; label: string; emoji: string }[] = [
  { key: "treppen", label: "Treppen", emoji: "🪜" },
  { key: "huegel", label: "Hügel", emoji: "⛰️" },
  { key: "loecher", label: "Löcher", emoji: "🕳️" },
  { key: "hindernisse", label: "Hindernisse", emoji: "🧱" },
  { key: "stange", label: "Balancierstange", emoji: "🤸" },
];

export default function WorldPanel({ tel, onApply, onReroll }: WorldPanelProps) {
  const cfg = tel.world;
  const patch = (p: Partial<WorldConfig>) => onApply({ ...cfg, ...p });
  const patchFeature = (k: keyof WorldFeatures, v: boolean) =>
    onApply({ ...cfg, features: { ...cfg.features, [k]: v } });

  return (
    <div className="space-y-4">
      {/* An/Aus */}
      <div className="flex items-center justify-between rounded-lg border border-cyan-500/20 bg-black/30 p-3">
        <div className="flex items-center gap-2.5">
          <Mountain className="h-5 w-5 text-cyan-300" />
          <div>
            <div className="text-sm text-slate-200">Random-Welt</div>
            <div className="text-[11px] text-slate-500">
              Prozedurale Hindernisse in der Physik
            </div>
          </div>
        </div>
        <Switch
          checked={cfg.enabled}
          onCheckedChange={(v) => patch({ enabled: v })}
          aria-label="Welt ein/aus"
        />
      </div>

      {/* Features */}
      <div className={`space-y-2 ${cfg.enabled ? "" : "pointer-events-none opacity-50"}`}>
        {FEATURE_LABELS.map((f) => (
          <div
            key={f.key}
            className="flex items-center justify-between rounded-lg border border-slate-700/50 bg-black/20 p-2.5"
          >
            <span className="text-sm text-slate-200">
              {f.emoji} {f.label}
            </span>
            <Switch
              checked={cfg.features[f.key]}
              onCheckedChange={(v) => patchFeature(f.key, v)}
              aria-label={`${f.label} ein/aus`}
            />
          </div>
        ))}
      </div>

      {/* Schwierigkeit + Dichte */}
      <div className={`space-y-3 rounded-lg border border-slate-700/50 bg-black/20 p-3 ${cfg.enabled ? "" : "pointer-events-none opacity-50"}`}>
        <div>
          <div className="mb-1.5 flex items-center justify-between text-xs">
            <span className="text-slate-400">Schwierigkeit</span>
            <span className="font-mono text-cyan-300">{(cfg.difficulty * 100).toFixed(0)}%</span>
          </div>
          <Slider
            value={[cfg.difficulty]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={(v) => patch({ difficulty: v[0] })}
            className="[&_[data-slot=slider-range]]:bg-amber-400 [&_[data-slot=slider-thumb]]:border-amber-300"
          />
        </div>
        <div>
          <div className="mb-1.5 flex items-center justify-between text-xs">
            <span className="text-slate-400">Objekt-Dichte</span>
            <span className="font-mono text-cyan-300">{(cfg.density * 100).toFixed(0)}%</span>
          </div>
          <Slider
            value={[cfg.density]}
            min={0.05}
            max={1}
            step={0.05}
            onValueChange={(v) => patch({ density: v[0] })}
            className="[&_[data-slot=slider-range]]:bg-cyan-400 [&_[data-slot=slider-thumb]]:border-cyan-300"
          />
        </div>
      </div>

      {/* Seed + Aktionen */}
      <div className={`space-y-2 ${cfg.enabled ? "" : "pointer-events-none opacity-50"}`}>
        <div className="flex items-center justify-between rounded-lg border border-slate-700/50 bg-black/20 px-3 py-2">
          <span className="text-xs text-slate-400">Seed</span>
          <span className="font-mono text-xs text-slate-300">{cfg.seed}</span>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={onReroll}
            className="h-11 flex-1 gap-1.5 bg-cyan-500 font-semibold text-black hover:bg-cyan-400"
          >
            <Dices className="h-4 w-4" /> Neue Welt würfeln
          </Button>
        </div>
        <p className="text-[11px] leading-snug text-slate-500">
          Änderungen bauen die Physik neu (kurzer Ladevorgang). Der Roboter startet
          immer in der freien Zone in der Mitte. Löcher ersetzen den Boden durch
          Fliesen – wer reinfällt, wird als Sturz gewertet.
        </p>
      </div>

      {/* Trainings-Hinweis */}
      <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-3 text-[11px] leading-relaxed text-slate-400">
        <b className="text-cyan-300">Tipp:</b> Welt aktivieren, dann im
        Training-Panel mit Turbo 16×+ trainieren – die Evolution Strategy lernt
        robuste Bewegungen über Hindernisse. Für Balancierstange zusätzlich das
        Ziel „Aufrecht bleiben" in der Bewertung hochgewichten.
      </div>
    </div>
  );
}
