"use client";

// ── MicroDuck Trainer v2.3 – Trainings-Panel (OpenAI-ES + Profi-Tricks) ─────

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import FitnessChart from "./FitnessChart";
import type { Telemetry } from "@/lib/md/app-core";
import { TURBO_LEVELS, type TurboLevel, type RunCfg } from "@/lib/md/es";
import {
  Play, Square, Eye, Save, FolderOpen, Download, Upload, Timer, Gauge, Wand2,
} from "lucide-react";

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
  // v2.2
  onTrainCfg: (patch: Partial<Telemetry["trainCfg"]>) => void;
  onExport: () => void;
  onImport: () => void;
}

function fmt(n: number | undefined | null, digits = 2): string {
  if (n === undefined || n === null || !Number.isFinite(n)) return "–";
  return n.toFixed(digits);
}

const SPEED_LEVELS: (1 | 2 | 4 | 8)[] = [1, 2, 4, 8];

export default function TrainingPanel(props: TrainingPanelProps) {
  const {
    tel, turbo, onTurbo, onStart, onStop, onShowBest, onSave, onLoad, hasSaved,
    onTrainCfg, onExport, onImport,
  } = props;
  const es = tel.es;
  const [watchBest, setWatchBest] = useState(false);
  const [showPro, setShowPro] = useState(false);
  const cfg = tel.trainCfg;
  const rc: RunCfg = cfg.runCfg;
  const genProgress = cfg.maxGenerations > 0 && es
    ? Math.min(100, (es.generation / cfg.maxGenerations) * 100)
    : null;
  const setRun = (p: Partial<RunCfg>) => onTrainCfg({ runCfg: { ...rc, ...p } });

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

      {/* v2.3: Sichtbares Sim-Tempo (Physik + Roboter synchron schneller) */}
      <div className="flex items-center gap-2">
        <Gauge className="h-4 w-4 shrink-0 text-amber-300" />
        <span className="text-[11px] uppercase tracking-wide text-slate-500">Tempo</span>
        <div className="flex overflow-hidden rounded-lg border border-amber-500/25">
          {SPEED_LEVELS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => onTrainCfg({ speedMult: s })}
              className={`h-9 px-3 font-mono text-xs transition-colors ${
                cfg.speedMult === s
                  ? "bg-amber-500/25 text-amber-200"
                  : "text-slate-400 hover:bg-white/5"
              }`}
              title={`Simulation läuft ${s}× so schnell (Physik, Roboter und Umgebung synchron)`}
            >
              {s}×
            </button>
          ))}
        </div>
        <span className="text-[10px] leading-tight text-slate-600">
          Roboter &amp; Physik laufen<br />synchron schneller
        </span>
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
          { label: "Sigma", value: es ? fmt(es.sigma, 3) : fmt(cfg.sigma, 3) },
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

      {/* v2.2: Ziel-Generationen Fortschritt */}
      {genProgress !== null && (
        <div>
          <div className="mb-1 flex justify-between text-[11px] text-slate-400">
            <span className="flex items-center gap-1">
              <Timer className="h-3 w-3" /> Ziel: {cfg.maxGenerations} Generationen
            </span>
            <span className="font-mono text-cyan-300">{es?.generation ?? 0} / {cfg.maxGenerations}</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-black/40">
            <div className="h-full rounded-full bg-cyan-400 transition-all" style={{ width: `${genProgress}%` }} />
          </div>
        </div>
      )}

      {/* v2.2: Runden & Hyperparameter */}
      <div className="space-y-3 rounded-lg border border-cyan-500/20 bg-black/20 p-3">
        <div className="text-xs font-semibold text-cyan-300">Runden &amp; Hyperparameter</div>

        <div>
          <div className="mb-1 flex justify-between text-[11px] text-slate-400">
            <span>Rundenlänge (Schritte je Runde)</span>
            <span className="font-mono text-cyan-300">{cfg.rolloutSteps}</span>
          </div>
          <Slider
            value={[cfg.rolloutSteps]}
            min={30}
            max={600}
            step={10}
            onValueChange={(v) => onTrainCfg({ rolloutSteps: v[0] })}
          />
          <p className="mt-0.5 text-[10px] text-slate-600">
            Längere Runden = reiferes Verhalten, aber langsamer je Generation.
          </p>
        </div>

        <div>
          <div className="mb-1 flex justify-between text-[11px] text-slate-400">
            <span>Ziel-Generationen (Auto-Stopp)</span>
            <span className="font-mono text-cyan-300">
              {cfg.maxGenerations === 0 ? "∞ (manuell stoppen)" : String(cfg.maxGenerations)}
            </span>
          </div>
          <Slider
            value={[cfg.maxGenerations]}
            min={0}
            max={3000}
            step={25}
            onValueChange={(v) => onTrainCfg({ maxGenerations: v[0] })}
          />
          <p className="mt-0.5 text-[10px] text-slate-600">
            Training stoppt automatisch, wenn das Ziel erreicht ist. 0 = unbegrenzt.
          </p>
        </div>

        <div>
          <div className="mb-1 flex justify-between text-[11px] text-slate-400">
            <span>Lernrate</span>
            <span className="font-mono text-cyan-300">{cfg.lr.toFixed(3)}</span>
          </div>
          <Slider
            value={[cfg.lr]}
            min={0.005}
            max={0.1}
            step={0.005}
            onValueChange={(v) => onTrainCfg({ lr: v[0] })}
          />
        </div>

        <div>
          <div className="mb-1 flex justify-between text-[11px] text-slate-400">
            <span>Rauschen (Sigma)</span>
            <span className="font-mono text-cyan-300">{cfg.sigma.toFixed(3)}</span>
          </div>
          <Slider
            value={[cfg.sigma]}
            min={0.01}
            max={0.2}
            step={0.005}
            onValueChange={(v) => onTrainCfg({ sigma: v[0] })}
          />
          <p className="mt-0.5 text-[10px] text-slate-600">
            Mehr Rauschen = mehr Entdeckung, weniger Feinschliff.
          </p>
        </div>
      </div>

      {/* v2.3: Profi-Tricks (auf/zu) */}
      <div className="rounded-lg border border-violet-500/25 bg-violet-500/5">
        <button
          type="button"
          onClick={() => setShowPro(!showPro)}
          className="flex w-full items-center justify-between px-3 py-2.5"
        >
          <span className="flex items-center gap-2 text-xs font-semibold text-violet-300">
            <Wand2 className="h-4 w-4" /> Profi-Tricks (aktiv)
          </span>
          <span className="text-[10px] text-violet-400/70">{showPro ? "▲" : "▼"}</span>
        </button>
        {showPro && (
          <div className="space-y-3 border-t border-violet-500/20 p-3">
            <p className="text-[10px] leading-relaxed text-slate-500">
              Genau die Techniken, mit denen Profi-Teams (Unitree, ANYmal, Berkeley)
              Lauf-Roboter trainieren – hier ein/aus schaltbar.
            </p>
            {[
              { k: "cmdTrain" as const, label: "Zufalls-Tempo je Runde", desc: "Jede Runde neues Tempo/Richtung → Roboter lernt überall zu gehen (Joystick-fähig)" },
              { k: "curriculum" as const, label: "Curriculum", desc: "Zu viele Stürze → Tempo automatisch runter; läuft es → automatisch rauf" },
              { k: "pushes" as const, label: "Zufalls-Stöße", desc: "Rempelt den Roboter an → wird robust gegen Stöße (Domain Randomization)" },
              { k: "noiseReset" as const, label: "Start-Rauschen", desc: "Jeder Start leicht anders → keine Memory-Löcher, robustere Gelenkwinkel" },
            ].map((row) => (
              <div key={row.k} className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-xs text-slate-200">{row.label}</div>
                  <div className="text-[10px] leading-snug text-slate-500">{row.desc}</div>
                </div>
                <Switch
                  checked={rc[row.k]}
                  onCheckedChange={(v) => setRun({ [row.k]: v } as Partial<RunCfg>)}
                  aria-label={row.label}
                />
              </div>
            ))}
            <div>
              <div className="mb-1 flex justify-between text-[11px] text-slate-400">
                <span>Aktions-Glättung (gegen Zittern)</span>
                <span className="font-mono text-violet-300">{rc.actionSmooth.toFixed(2)}</span>
              </div>
              <Slider
                value={[rc.actionSmooth]}
                min={0.3}
                max={1}
                step={0.05}
                onValueChange={(v) => setRun({ actionSmooth: v[0] })}
              />
              <p className="mt-0.5 text-[10px] text-slate-600">
                Niedriger = ruhigere Bewegung (0.6 empfohlen), 1.0 = aus.
              </p>
            </div>
            <div>
              <div className="mb-1 flex justify-between text-[11px] text-slate-400">
                <span>Ziel-Tempo beim Training (m/s)</span>
                <span className="font-mono text-violet-300">{rc.cmdFwd.toFixed(2)}</span>
              </div>
              <Slider
                value={[rc.cmdFwd]}
                min={0.1}
                max={0.8}
                step={0.05}
                onValueChange={(v) => setRun({ cmdFwd: v[0] })}
              />
            </div>
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-xs text-slate-200">Überlebens-Fitness</div>
                <div className="text-[10px] leading-snug text-slate-500">Wer länger nicht fällt, gewinnt (statt Durchschnitt)</div>
              </div>
              <Switch
                checked={rc.fitnessMode === "sum"}
                onCheckedChange={(v) => setRun({ fitnessMode: v ? "sum" : "mean" })}
                aria-label="Überlebens-Fitness"
              />
            </div>
          </div>
        )}
      </div>

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

      {/* v2.2: Export/Import als Datei (APK: Downloads-Ordner / Dateimanager) */}
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant="outline"
          onClick={onExport}
          className="h-11 gap-1.5 border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/10"
        >
          <Download className="h-4 w-4" /> Export (.json)
        </Button>
        <Button
          variant="outline"
          onClick={onImport}
          className="h-11 gap-1.5 border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/10"
        >
          <Upload className="h-4 w-4" /> Import (.json)
        </Button>
      </div>
      <p className="text-[11px] leading-relaxed text-slate-500">
        OpenAI-ES mit antithetischem Sampling (24–96 Individuen je Turbo-Stufe), adaptivem
        Sigma und Rank-Shaping. „Speichern/Laden“ nutzt den lokalen Speicher; „Export/Import“
        schreibt vollständige Trainingsstände (Policy + Regeln + Runden-Einstellungen) in den
        Downloads-Ordner bzw. liest sie aus dem Dateimanager – perfekt zum Sichern und Teilen.
      </p>
    </div>
  );
}
