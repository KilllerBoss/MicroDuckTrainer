"use client";

// ── MicroDuck Trainer v2.0 – Mapping-Editor (Gamepad → Roboter) ──────────────

import type { MappingEntry, SourceId, TargetType } from "@/lib/md/mapping";
import {
  SOURCES,
  defaultMapping,
  targetOptions,
} from "@/lib/md/mapping";
import type { ModelId } from "@/lib/md/models";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { RotateCcw } from "lucide-react";

interface MappingEditorProps {
  modelId: ModelId;
  mappings: MappingEntry[];
  onChange: (m: MappingEntry[]) => void;
}

export default function MappingEditor({ modelId, mappings, onChange }: MappingEditorProps) {
  const options = targetOptions(modelId);

  const update = (source: SourceId, patch: Partial<MappingEntry>) => {
    onChange(
      SOURCES.map((s) => {
        const cur = mappings.find((m) => m.source === s.id) ?? {
          source: s.id, targetType: "cmd" as TargetType, target: "", gain: 1,
        };
        return s.id === source ? { ...cur, ...patch } : cur;
      }),
    );
  };

  const reset = () => onChange(defaultMapping(modelId));

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-400">
          Joystick &amp; Tasten A–D auf Commands, Policies, Posen oder Gelenke legen.
        </p>
        <Button
          variant="outline"
          size="sm"
          onClick={reset}
          className="h-9 gap-1 border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"
        >
          <RotateCcw className="h-3.5 w-3.5" /> Reset
        </Button>
      </div>
      <div className="max-h-64 space-y-2 overflow-y-auto pr-1 md:max-h-none">
        {SOURCES.map((s) => {
          const m = mappings.find((e) => e.source === s.id);
          return (
            <div
              key={s.id}
              className="rounded-lg border border-cyan-500/15 bg-black/30 p-2.5"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`flex h-8 min-w-8 items-center justify-center rounded-md px-1.5 font-mono text-xs font-bold ${
                    s.isStick ? "bg-cyan-500/15 text-cyan-300" : "bg-white/10 text-white"
                  }`}
                >
                  {s.label}
                </span>
                <select
                  aria-label={`Ziel für ${s.label}`}
                  value={m ? `${m.targetType}:${m.target}` : ""}
                  onChange={(e) => {
                    const [type, ...rest] = e.target.value.split(":");
                    update(s.id, { targetType: type as TargetType, target: rest.join(":") });
                  }}
                  className="h-9 min-w-0 flex-1 rounded-md border border-cyan-500/20 bg-[#0b0f18] px-2 text-xs text-slate-200 outline-none focus:border-cyan-400/50"
                >
                  <option value="" disabled>
                    – Ziel wählen –
                  </option>
                  {["cmd", "policy", "pose", "gelenk"].map((group) => {
                    const groupOpts = options.filter((o) => o.type === group);
                    if (!groupOpts.length) return null;
                    const label =
                      group === "cmd" ? "Command"
                      : group === "policy" ? "Policy (ONNX)"
                      : group === "pose" ? "Pose"
                      : "Gelenk";
                    return (
                      <optgroup key={group} label={label}>
                        {groupOpts.map((o) => (
                          <option key={`${o.type}:${o.value}`} value={`${o.type}:${o.value}`}>
                            {o.label}
                          </option>
                        ))}
                      </optgroup>
                    );
                  })}
                </select>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <span className="w-10 shrink-0 font-mono text-[10px] uppercase tracking-wide text-slate-500">
                  Gain
                </span>
                <Slider
                  value={[m?.gain ?? 1]}
                  min={0}
                  max={3}
                  step={0.05}
                  onValueChange={(v) => update(s.id, { gain: v[0] })}
                  className="flex-1 [&_[data-slot=slider-range]]:bg-cyan-400 [&_[data-slot=slider-thumb]]:border-cyan-300"
                />
                <span className="w-10 shrink-0 text-right font-mono text-xs text-cyan-300">
                  {(m?.gain ?? 1).toFixed(2)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
