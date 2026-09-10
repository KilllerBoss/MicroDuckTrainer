"use client";

// ── MicroDuck Trainer v2.0 – Trainings-Panel (OpenAI-ES) ─────────────────────

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import FitnessChart from "./FitnessChart";
import type { Telemetry } from "@/lib/md/app-core";
import { TURBO_LEVELS, type TurboLevel } from "@/lib/md/es";
import { Play, Square, Eye, Save, FolderOpen } from "lucide-react";

interface TrainingPanelProps {
  tel: Telemetry;
  turbo: TurboLevel;
  onTurbo: (t: TurboLevel) => void;
  onStart: () => void;
  onStop: () => void;
  onShowBest: () => void;
  onSave: () => void;
  onLoad: () => void;
  hasSaved: boolean;
}

function fmt(n: number | undefined | null, digits = 2): string {
  if (n === undefined || n === null || !Number.isFinite(n)) return "–";
  return n.toFixed(digits);
}

export default function TrainingPanel(props: TrainingPanelProps) {
  const { tel, turbo, onTurbo, onStart, onStop, onShowBest, onSave, onLoad, hasSaved } = props;
  const es = tel.es;
  const [watchBest, setWatchBest] = useState(false);

  return (
    <div className="space-y-4">
      {/* Start/Stop + Turbo */}
      <div className="flex flex-wrap items-center gap-2">
        {tel.training ? (
          <Button
            onClick={onStop}
            className="h-11 min-w-[110px] gap-1.5 bg-red-500/90 text-white hover:bg-red-500"
          >
            <Square className="h-4 w-4" /> Stoppen
          </Button>
        ) : (
          <Button
            onClick={onStart}
            disabled={tel.loading || !tel.booted}
            className="h-11 min-w-[110px] gap-1.5 bg-cyan-500 font-semibold text-black hover:bg-cyan-400"
          >
            <Play className="h-4 w-4" /> Starten
          </Button>
        )}
        <div className="flex overflow-hidden rounded-lg border border-cyan-500/25">
          {TURBO_LEVELS.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => onTurbo(t)}
              className={`h-11 px-2.5 font-mono text-xs transition-colors ${
                turbo === t
                  ? "bg-cyan-500/25 text-cyan-200"
                  : "text-slate-400 hover:bg-white/5"
              } ${t === 64 ? "font-bold" : ""}`}
              title={t === 64 ? "Hyper-Modus (96 Individuen)" : `${t}× Turbo`}
            >
              {t === 64 ? "Hyper" : `${t}×`}
            </button>
          ))}
        </div>
      </div>

      {/* Status */}
      <div className="flex items-center gap-2 text-xs">
        {tel.training ? (
          <Badge className="gap-1.5 border-cyan-400/40 bg-cyan-500/15 text-cyan-300">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-300" />
            Training läuft…
          </Badge>
        ) : (
          <Badge variant="outline" className="border-slate-600 text-slate-400">
            Gestoppt
          </Badge>
        )}
        {es && (
          <Badge variant="outline" className="border-slate-600 font-mono text-slate-400">
            {es.evaluator === "worker"
              ? `${es.workersReady} Worker`
              : "Main-Thread"}
          </Badge>
        )}
        {tel.esReady && (
          <Badge variant="outline" className="border-emerald-500/40 text-emerald-400">
            ES-Policy bereit
          </Badge>
        )}
      </div>

      {/* Stats-Raster */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: "Generation", value: es ? String(es.generation) : "–" },
          { label: "Best (Gen)", value: es ? fmt(es.bestFitness) : "–" },
          { label: "Best ever", value: es ? fmt(es.bestEver) : "–" },
          { label: "Sigma", value: es ? fmt(es.sigma, 3) : "0.080" },
          { label: "Schritte/s", value: es ? fmt(es.stepsPerSec, 0) : "–" },
          { label: "Sturzrate", value: es ? fmt(es.fellRate, 2) : "–" },
        ].map((s) => (
          <div key={s.label} className="rounded-lg border border-cyan-500/15 bg-black/30 p-2">
            <div className="text-[10px] uppercase tracking-wide text-slate-500">{s.label}</div>
            <div className="font-mono text-sm text-cyan-200">{s.value}</div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <FitnessChart history={es?.history ?? []} />

      {/* Aktionen */}
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant="outline"
          onClick={onShowBest}
          disabled={!tel.esReady}
          className="h-11 gap-1.5 border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"
        >
          <Eye className="h-4 w-4" /> Beste zeigen
        </Button>
        <div className="flex items-center justify-between rounded-lg border border-cyan-500/15 bg-black/30 px-3">
          <span className="text-xs text-slate-300">Live verfolgen</span>
          <Switch
            checked={watchBest}
            onCheckedChange={(v) => {
              setWatchBest(v);
              if (v) onShowBest();
            }}
          />
        </div>
        <Button
          variant="outline"
          onClick={onSave}
          disabled={!tel.es && !tel.esReady}
          className="h-11 gap-1.5 border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"
        >
          <Save className="h-4 w-4" /> Speichern
        </Button>
        <Button
          variant="outline"
          onClick={onLoad}
          disabled={!hasSaved}
          className="h-11 gap-1.5 border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"
        >
          <FolderOpen className="h-4 w-4" /> Laden
        </Button>
      </div>
      <p className="text-[11px] leading-relaxed text-slate-500">
        OpenAI-ES mit antithetischem Sampling (24–96 Individuen je Turbo-Stufe), adaptivem
        Sigma und Rank-Shaping. „Speichern/Laden“ nutzt den lokalen Speicher des Geräts –
        perfekt für die APK.
      </p>
    </div>
  );
}
