"use client";

// ── MicroDuck Trainer v2.2 – Steuerungs-Panel (Gamepad + Mappings + Punkt) ───

import { useState } from "react";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import type { Telemetry, PointCfg, PointModeState } from "@/lib/md/app-core";
import type { MappingEntry } from "@/lib/md/mapping";
import MappingEditor from "./MappingEditor";
import { Gamepad2, ShieldCheck, Crosshair } from "lucide-react";

interface ControlPanelProps {
  tel: Telemetry;
  gamepadOn: boolean;
  autoRecovery: boolean;
  onGamepad: (v: boolean) => void;
  onAutoRecovery: (v: boolean) => void;
  onMappings: (m: MappingEntry[]) => void;
  onPointCfg: (patch: Partial<PointCfg>) => void;
}

const POINT_MODES: { id: PointModeState; label: string; hint: string }[] = [
  { id: "aus", label: "Aus", hint: "Kein Punkt – Joystick steuert den Roboter direkt." },
  { id: "frei", label: "Frei", hint: "Punkt läuft frei in der Arena; der Roboter verfolgt ihn." },
  { id: "umkreis", label: "Umkreis", hint: "Der Punkt bleibt immer im einstellbaren Umkreis um den Roboter." },
  { id: "pfad", label: "Pfad", hint: "Pfad Roboter → Punkt wird physikalisch mit Schwung (Momentum) berechnet – der Roboter geht ihm nach." },
];

export default function ControlPanel(props: ControlPanelProps) {
  const { tel, gamepadOn, autoRecovery, onGamepad, onAutoRecovery, onMappings, onPointCfg } = props;
  const [showEditor, setShowEditor] = useState(false);
  const modelId = tel.modelId ?? "microduck";
  const pc: PointCfg = tel.pointCfg ?? {
    mode: "aus", radius: 2, speedByDist: true,
    maxSpeed: modelId === "unitree_g1" ? 0.6 : 0.25, fullDist: 1.5,
  };
  const activeHint = POINT_MODES.find((m) => m.id === pc.mode)?.hint ?? "";
  const maxSpeedMax = modelId === "unitree_g1" ? 1.2 : 0.25;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between rounded-lg border border-cyan-500/20 bg-black/30 p-3">
        <div className="flex items-center gap-2.5">
          <Gamepad2 className="h-5 w-5 text-cyan-300" />
          <div>
            <div className="text-sm text-slate-200">Touch-Gamepad</div>
            <div className="text-[11px] text-slate-500">
              Overlay anzeigen (im Manuell-Modus immer aktiv)
            </div>
          </div>
        </div>
        <Switch checked={gamepadOn} onCheckedChange={onGamepad} aria-label="Gamepad ein/aus" />
      </div>

      {/* ── Joystick-Punkt (v2.2) ── */}
      <div className="rounded-lg border border-amber-500/25 bg-black/30 p-3">
        <div className="flex items-center gap-2.5">
          <Crosshair className="h-5 w-5 text-amber-300" />
          <div>
            <div className="text-sm text-slate-200">Joystick-Punkt</div>
            <div className="text-[11px] text-slate-500">
              Kamera-relativ: Vorne ist immer dort, wohin du schaust
            </div>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-4 gap-1">
          {POINT_MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => onPointCfg({ mode: m.id })}
              aria-pressed={pc.mode === m.id}
              className={`h-9 rounded-md border text-xs transition-colors ${
                pc.mode === m.id
                  ? "border-amber-400/70 bg-amber-500/20 font-semibold text-amber-200"
                  : "border-slate-700/60 text-slate-400 hover:border-amber-400/40 hover:text-amber-200"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <p className="mt-1.5 text-[11px] leading-snug text-slate-500">{activeHint}</p>

        {pc.mode !== "aus" && (
          <div className="mt-3 space-y-3 border-t border-amber-500/10 pt-3">
            {(pc.mode === "umkreis" || pc.mode === "pfad") && (
              <div>
                <div className="mb-1.5 flex items-center justify-between text-xs">
                  <span className="text-slate-400">Umkreis-Radius</span>
                  <span className="font-mono text-amber-300">{pc.radius.toFixed(1)} m</span>
                </div>
                <Slider
                  value={[pc.radius]}
                  min={0.3}
                  max={6}
                  step={0.1}
                  onValueChange={(v) => onPointCfg({ radius: v[0] })}
                  className="[&_[data-slot=slider-range]]:bg-amber-400 [&_[data-slot=slider-thumb]]:border-amber-300"
                />
              </div>
            )}

            <div className="flex items-center justify-between gap-3">
              <div className="text-xs text-slate-400">
                Weiterer Punkt = schneller laufen
              </div>
              <Switch
                checked={pc.speedByDist}
                onCheckedChange={(v) => onPointCfg({ speedByDist: v })}
                aria-label="Tempo mit Distanz skalieren"
              />
            </div>

            {pc.speedByDist && (
              <>
                <div>
                  <div className="mb-1.5 flex items-center justify-between text-xs">
                    <span className="text-slate-400">Max-Tempo</span>
                    <span className="font-mono text-amber-300">{pc.maxSpeed.toFixed(2)} m/s</span>
                  </div>
                  <Slider
                    value={[pc.maxSpeed]}
                    min={0.05}
                    max={maxSpeedMax}
                    step={0.05}
                    onValueChange={(v) => onPointCfg({ maxSpeed: v[0] })}
                    className="[&_[data-slot=slider-range]]:bg-cyan-400 [&_[data-slot=slider-thumb]]:border-cyan-300"
                  />
                </div>
                <div>
                  <div className="mb-1.5 flex items-center justify-between text-xs">
                    <span className="text-slate-400">Volles Tempo ab</span>
                    <span className="font-mono text-amber-300">{pc.fullDist.toFixed(1)} m</span>
                  </div>
                  <Slider
                    value={[pc.fullDist]}
                    min={0.3}
                    max={6}
                    step={0.1}
                    onValueChange={(v) => onPointCfg({ fullDist: v[0] })}
                    className="[&_[data-slot=slider-range]]:bg-cyan-400 [&_[data-slot=slider-thumb]]:border-cyan-300"
                  />
                </div>
              </>
            )}

            <p className="text-[10px] leading-snug text-slate-500">
              Wie der Roboter auf den Punkt reagiert, stellst du im
              Bewertungs-Panel ein („Zum Punkt laufen“ / „Vom Punkt weg“).
            </p>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between rounded-lg border border-cyan-500/20 bg-black/30 p-3">
        <div className="flex items-center gap-2.5">
          <ShieldCheck className="h-5 w-5 text-emerald-300" />
          <div>
            <div className="text-sm text-slate-200">Auto-Recovery (Test)</div>
            <div className="text-[11px] text-slate-500">
              Nach Sturz sanft zur Stand-Pose blenden – kein Teleport
            </div>
          </div>
        </div>
        <Switch
          checked={autoRecovery}
          onCheckedChange={onAutoRecovery}
          aria-label="Auto-Recovery ein/aus"
        />
      </div>

      <div className="rounded-lg border border-cyan-500/20 bg-black/30 p-3">
        <button
          type="button"
          onClick={() => setShowEditor((v) => !v)}
          className="flex w-full items-center justify-between text-left"
        >
          <div>
            <div className="text-sm text-slate-200">Mapping-Editor</div>
            <div className="text-[11px] text-slate-500">
              Joystick &amp; A–D frei belegen (pro Roboter gespeichert)
            </div>
          </div>
          <span className={`font-mono text-xs text-cyan-300 transition-transform ${showEditor ? "rotate-180" : ""}`}>
            ▼
          </span>
        </button>
        {showEditor && (
          <div className="mt-3 border-t border-cyan-500/10 pt-3">
            <MappingEditor modelId={modelId} mappings={tel.mappings} onChange={onMappings} />
          </div>
        )}
      </div>
    </div>
  );
}
