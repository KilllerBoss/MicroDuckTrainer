"use client";

// ── MicroDuck Trainer v2.1 – Gemini-Panel (KI passt Trainingsregeln an) ──────
// Ziel in natürlicher Sprache → Gemini erzeugt Reward/Welt/Punkt/Turbo-Patch.
// Modelle: gemini-robotics-er-2-preview · gemini-3.5-flash-lite · gemini-3.8-flash

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import type { Telemetry } from "@/lib/md/app-core";
import {
  GEMINI_MODELS, loadGeminiPrefs, saveGeminiPrefs,
  applyGoalWithGemini, type GeminiModelId,
} from "@/lib/md/gemini";
import { toast } from "@/hooks/use-toast";
import { Bot, Sparkles, Loader2, KeyRound, Wand2 } from "lucide-react";

interface GeminiPanelProps {
  tel: Telemetry;
  /** Wendet den Patch im Kern an (Reward/Punkt/Welt/Turbo). */
  onApplyPatch: (actions: {
    reward?: Telemetry["reward"];
    pointMode?: boolean;
    pointCfg?: Partial<Telemetry["pointCfg"]>;
    world?: Partial<Telemetry["world"]>;
    turbo?: number;
    resetFirst?: boolean;
  }) => void;
  onTurbo: (t: number) => void;
}

const EXAMPLES = [
  "Ich will, dass der Roboter lernt zu springen",
  "Robuster über Treppen und Hindernisse laufen",
  "Auf der Balancierstange balancieren",
  "Dem Joystick-Punkt schnell nachlaufen",
  "Energiesparend und ruhig gehen",
];

export default function GeminiPanel({ tel, onApplyPatch, onTurbo }: GeminiPanelProps) {
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState<GeminiModelId>("gemini-3.5-flash-lite");
  const [goal, setGoal] = useState("");
  const [busy, setBusy] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    const p = loadGeminiPrefs();
    setApiKey(p.apiKey);
    setModel(p.model);
  }, []);

  const run = async () => {
    if (!goal.trim()) {
      toast({ title: "Ziel fehlt", description: "Beschreibe, was der Roboter lernen soll." });
      return;
    }
    if (!apiKey.trim()) {
      toast({
        title: "API-Key fehlt",
        description: "Gemini API-Key von aistudio.google.com eintragen (kostenlos).",
        variant: "destructive",
      });
      return;
    }
    setBusy(true);
    setExplanation(null);
    try {
      saveGeminiPrefs(apiKey.trim(), model);
      const { patch } = await applyGoalWithGemini(apiKey.trim(), model, {
        modelId: tel.modelId ?? "microduck",
        reward: tel.reward,
        imitationActive: !!tel.imitation?.playing,
        imitationName: tel.imitation?.name ?? null,
        worldEnabled: tel.world.enabled,
        pointMode: tel.pointCfg?.mode ?? "aus",
        pointRadius: tel.pointCfg?.radius ?? 2,
        generation: tel.es?.generation ?? 0,
      }, goal.trim());

      // Reward-Patch anwenden
      let reward: Telemetry["reward"] | undefined;
      if (patch.reward) {
        const terms = { ...tel.reward.terms } as Telemetry["reward"]["terms"];
        for (const [k, v] of Object.entries(patch.reward)) {
          const cur = terms[k];
          if (!cur) continue;
          terms[k] = {
            ...cur,
            enabled: v.enabled ?? cur.enabled,
            weight: v.weight ?? cur.weight,
            param: v.param ?? cur.param,
          };
        }
        reward = { version: 2, terms };
      }
      onApplyPatch({
        reward,
        pointMode: patch.pointMode,
        pointCfg: patch.point,
        world: patch.world
          ? {
              enabled: patch.world.enabled,
              difficulty: patch.world.difficulty,
              density: patch.world.density,
              features: patch.world.features as Telemetry["world"]["features"] | undefined,
            }
          : undefined,
        turbo: patch.turbo,
        resetFirst: patch.resetFirst,
      });
      if (patch.turbo && [1, 4, 16, 32, 64].includes(patch.turbo)) onTurbo(patch.turbo);
      setExplanation(patch.explanation || "Regeln wurden angepasst.");
      toast({ title: "Gemini hat die Regeln angepasst", description: "Siehe Erklärung im Panel." });
    } catch (err: any) {
      toast({ title: "Gemini-Fehler", description: err?.message || String(err), variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* API-Key */}
      <div className="rounded-lg border border-slate-700/50 bg-black/20 p-3">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <KeyRound className="h-3.5 w-3.5" /> Gemini API-Key (lokal gespeichert)
        </div>
        <div className="mt-2 flex gap-2">
          <input
            type={showKey ? "text" : "password"}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="AIza…"
            className="h-10 flex-1 rounded-md border border-slate-700 bg-black/40 px-3 font-mono text-xs text-slate-200 outline-none focus:border-cyan-500/60"
          />
          <Button
            variant="outline"
            size="sm"
            className="h-10 border-slate-700 px-2 text-slate-400"
            onClick={() => setShowKey((v) => !v)}
            aria-label="Key anzeigen"
          >
            👁
          </Button>
        </div>
        <p className="mt-1.5 text-[10px] text-slate-500">
          Standard-Key ist eingebaut – du musst nichts eintragen. Eigener Key:
          kostenlos auf aistudio.google.com → „Get API key“.
        </p>
      </div>

      {/* Modell */}
      <div className="rounded-lg border border-slate-700/50 bg-black/20 p-3">
        <div className="mb-2 flex items-center gap-2 text-xs text-slate-400">
          <Bot className="h-3.5 w-3.5" /> Modell
        </div>
        <div className="grid gap-1.5">
          {GEMINI_MODELS.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setModel(m.id)}
              className={`flex items-center justify-between rounded-md border px-3 py-2 text-left text-xs transition-colors ${
                model === m.id
                  ? "border-fuchsia-400/60 bg-fuchsia-500/15 text-fuchsia-200"
                  : "border-slate-700/60 text-slate-300 hover:border-fuchsia-400/40"
              }`}
            >
              <span>{m.label}</span>
              <span className="font-mono text-[9px] text-slate-500">{m.id}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Ziel */}
      <div className="rounded-lg border border-fuchsia-500/25 bg-fuchsia-500/5 p-3">
        <div className="mb-2 flex items-center gap-2 text-xs text-fuchsia-300">
          <Wand2 className="h-3.5 w-3.5" /> Was soll der Roboter lernen?
        </div>
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          rows={3}
          placeholder="z. B. Ich will, dass der Roboter nach dem Training wirklich springen kann…"
          className="w-full resize-none rounded-md border border-slate-700 bg-black/40 p-3 text-xs text-slate-200 outline-none focus:border-fuchsia-500/60"
        />
        <div className="mt-2 flex flex-wrap gap-1.5">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => setGoal(ex)}
              className="min-h-[30px] rounded-full border border-slate-600/60 px-2.5 text-[10px] text-slate-400 hover:border-fuchsia-400/50 hover:text-fuchsia-200"
            >
              {ex}
            </button>
          ))}
        </div>
        <Button
          onClick={() => void run()}
          disabled={busy}
          className="mt-3 h-11 w-full gap-1.5 bg-fuchsia-500 font-semibold text-white hover:bg-fuchsia-400 disabled:opacity-60"
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          {busy ? "Gemini denkt…" : "Regeln automatisch anpassen"}
        </Button>
      </div>

      {/* Erklärung */}
      {explanation && (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
          <div className="mb-1 text-xs font-semibold text-emerald-300">Warum diese Regeln?</div>
          <p className="text-[11px] leading-relaxed text-emerald-100/90">{explanation}</p>
        </div>
      )}

      <p className="text-[11px] leading-snug text-slate-500">
        Gemini kann Reward-Gewichte, den Joystick-Punkt (Modus „Aus / Frei /
        Umkreis / Pfad“ inkl. Tempo-Regeln), die Random-Welt und das
        Turbo-Level auf einmal anpassen – du musst nichts programmieren. Die
        Ergebnisse erscheinen als Toast; prüfe die Änderungen im
        Bewertungs-Panel.
      </p>
    </div>
  );
}
