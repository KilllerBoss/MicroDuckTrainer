"use client";

// ── MicroDuck Trainer v2.2 – Trainings-Panel (OpenAI-ES) ─────────────────────

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import FitnessChart from "./FitnessChart";
import type { Telemetry } from "@/lib/md/app-core";
import { TURBO_LEVELS, type TurboLevel } from "@/lib/md/es";
import {
  Play, Square, Eye, Save, FolderOpen, Download, Upload, Timer,
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

export default function TrainingPanel(props: TrainingPanelProps) {
  const {
    tel, turbo, onTurbo, onStart, onStop, onShowBest, onSave, onLoad, hasSaved,
    onTrainCfg, onExport, onImport,
  } = props;
  const es = tel.es;
  const [watchBest, setWatchBest] = useState(false);
  const cfg = tel.trainCfg;
  const genProgress = cfg.maxGenerations > 0 && es
    ? Math.min(100, (es.generation / cfg.maxGenerations) * 100)
    : null;

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

      {/* v2.3: Profi-Tricks (Warum fällt/zittert der Roboter? → das hier löst es) */}
      <div className="space-y-3 rounded-lg border border-emerald-500/25 bg-emerald-500/5 p-3">
        <div className="text-xs font-semibold text-emerald-300">Profi-Tricks (empfohlen: alle AN)</div>

        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[13px] text-slate-200">Befehle trainieren</div>
            <p className="text-[10px] leading-snug text-slate-500">
              Zufalls-Tempo pro Runde + Belohnung fürs Folgen. Ohne diesen Trick
              weiß die Policy nie, dass sie LAUFEN soll – sie steht nur rum.
            </p>
          </div>
          <Switch checked={cfg.cmdTrain} onCheckedChange={(v) => onTrainCfg({ cmdTrain: v })} />
        </div>

        {cfg.cmdTrain && (
          <div>
            <div className="mb-1 flex justify-between text-[11px] text-slate-400">
              <span>Befehls-Tempo (m/s)</span>
              <span className="font-mono text-emerald-300">{cfg.cmdFwd.toFixed(2)}</span>
            </div>
            <Slider
              value={[cfg.cmdFwd]}
              min={0.05}
              max={0.6}
              step={0.05}
              onValueChange={(v) => onTrainCfg({ cmdFwd: v[0] })}
            />
            <p className="mt-0.5 text-[10px] text-slate-600">
              Ente läuft natürlich ~0,25 m/s, Mensch ~0,5 m/s. Trainierte Policies
              folgen danach auch dem Joystick im Test.
            </p>
          </div>
        )}

        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[13px] text-slate-200">Curriculum (Tempo-Treppe)</div>
            <p className="text-[10px] leading-snug text-slate-500">
              Startet langsam, erhöht das Tempo automatisch, wenn die Sturzrate sinkt.
              Profi-Standard für stabiles Laufenlernen.
            </p>
          </div>
          <Switch checked={cfg.curriculum} onCheckedChange={(v) => onTrainCfg({ curriculum: v })} />
        </div>

        <div>
          <div className="mb-1 flex justify-between text-[11px] text-slate-400">
            <span>Aktions-Glättung (killt Zittern)</span>
            <span className="font-mono text-emerald-300">
              {cfg.actionSmooth >= 0.999 ? "aus" : cfg.actionSmooth.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[cfg.actionSmooth]}
            min={0.3}
            max={1}
            step={0.05}
            onValueChange={(v) => onTrainCfg({ actionSmooth: v[0] })}
          />
          <p className="mt-0.5 text-[10px] text-slate-600">
            Tiefpassfilter auf den Gelenkbefehlen (wie in echten Robotern). Klein
            = ruhig + standfest, 1 = ungefiltert.
          </p>
        </div>

        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[13px] text-slate-200">Zufalls-Stöße</div>
            <p className="text-[10px] leading-snug text-slate-500">
              Schubser während der Runden – Policies werden robust statt glashaus-stabil.
            </p>
          </div>
          <Switch checked={cfg.pushes} onCheckedChange={(v) => onTrainCfg({ pushes: v })} />
        </div>

        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[13px] text-slate-200">Reset-Rauschen</div>
            <p className="text-[10px] leading-snug text-slate-500">
              Jede Runde startet leicht anders (Gelenke/Tempo variieren) statt immer exakt gleich.
            </p>
          </div>
          <Switch checked={cfg.noiseReset} onCheckedChange={(v) => onTrainCfg({ noiseReset: v })} />
        </div>

        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[13px] text-slate-200">Überlebens-Zählung</div>
            <p className="text-[10px] leading-snug text-slate-500">
              Fitness = Reward-SUMME: länger stehen/laufen bringt mehr. Vorher zählte
              nur der Schnitt – Sturzzeitpunkt war egal (→ Roboter fiel gern mal).
            </p>
          </div>
          <Switch
            checked={cfg.fitnessMode === "sum"}
            onCheckedChange={(v) => onTrainCfg({ fitnessMode: v ? "sum" : "mean" })}
          />
        </div>

        <div>
          <div className="mb-1 flex justify-between text-[11px] text-slate-400">
            <span>Gewichtsbremse (gegen Verkrampfung)</span>
            <span className="font-mono text-emerald-300">{cfg.weightDecay.toFixed(3)}</span>
          </div>
          <Slider
            value={[cfg.weightDecay]}
            min={0}
            max={0.05}
            step={0.005}
            onValueChange={(v) => onTrainCfg({ weightDecay: v[0] })}
          />
          <p className="mt-0.5 text-[10px] text-slate-600">
            Hält Netzgewichte klein → keine saturierten Ausgänge → kein Zappeln. 0 = aus.
          </p>
        </div>

        {es && (
          <div className="flex items-center justify-between rounded-md bg-black/30 px-2.5 py-1.5 text-[11px]">
            <span className="text-slate-400">Curriculum-Tempo-Stufe</span>
            <span className="font-mono text-emerald-300">{(es.speedScale * 100).toFixed(0)}%</span>
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
