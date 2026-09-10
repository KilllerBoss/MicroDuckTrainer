"use client";

// ── MicroDuck Trainer v2.0 – Steuerungs-Panel (Gamepad + Mappings) ───────────

import { useState } from "react";
import { Switch } from "@/components/ui/switch";
import type { Telemetry } from "@/lib/md/app-core";
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
  onPointMode: (v: boolean) => void;
}

export default function ControlPanel(props: ControlPanelProps) {
  const { tel, gamepadOn, autoRecovery, onGamepad, onAutoRecovery, onMappings, onPointMode } = props;
  const [showEditor, setShowEditor] = useState(false);
  const modelId = tel.modelId ?? "microduck";

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

      <div className="flex items-center justify-between rounded-lg border border-amber-500/25 bg-black/30 p-3">
        <div className="flex items-center gap-2.5">
          <Crosshair className="h-5 w-5 text-amber-300" />
          <div>
            <div className="text-sm text-slate-200">Punkt-Modus</div>
            <div className="text-[11px] text-slate-500">
              Joystick bewegt einen 3D-Punkt – Reaktion in der Bewertung einstellen
              („Zum Punkt laufen“ / „Vom Punkt weg“)
            </div>
          </div>
        </div>
        <Switch
          checked={tel.pointMode}
          onCheckedChange={onPointMode}
          aria-label="Punkt-Modus ein/aus"
        />
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
