"use client";

// ── MicroDuck Trainer v2.2 – Gemini-Panel (Regeln + Code-Experte) ────────────
// Tab 1 (Regeln): Ziel in natürlicher Sprache → Gemini erzeugt Reward/Welt/
//   Punkt/Turbo-Patch. Kein Code nötig.
// Tab 2 (Code-Experte): Gemini SCHREIBT SELBST JavaScript-Trainingscode
//   (Reward-Term), entscheidet Rundenlänge, Ziel-Generationen, Turbo, Reset –
//   der ganze Trainingscode. Vorschau + Anwenden + Download.
// Modelle: gemini-robotics-er-2-preview · gemini-3.5-flash-lite · gemini-3.8-flash

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import type { Telemetry } from "@/lib/md/app-core";
import type { RewardConfig } from "@/lib/md/rewards";
import {
  GEMINI_MODELS, loadGeminiPrefs, saveGeminiPrefs,
  applyGoalWithGemini, applyCodeWithGemini, type GeminiModelId, type GeminiPatch,
} from "@/lib/md/gemini";
import { saveToDevice } from "@/lib/md/fileio";
import { toast } from "@/hooks/use-toast";
import {
  Bot, Sparkles, Loader2, KeyRound, Wand2, Code2, Trash2, Download,
  Copy, Check, SlidersHorizontal,
} from "lucide-react";

interface GeminiPanelProps {
  tel: Telemetry;
  /** Wendet den Patch im Kern an (Reward/Punkt/Welt/Turbo/Runden/Custom-Code). */
  onApplyPatch: (actions: {
    reward?: Telemetry["reward"];
    pointMode?: boolean;
    pointCfg?: Partial<Telemetry["pointCfg"]>;
    world?: Partial<Telemetry["world"]>;
    turbo?: number;
    resetFirst?: boolean;
    custom?: RewardConfig["custom"] | { clear: true };
    training?: Partial<Telemetry["trainCfg"]>;
  }) => void;
  onTurbo: (t: number) => void;
}

const EXAMPLES_RULES = [
  "Ich will, dass der Roboter lernt zu springen",
  "Robuster über Treppen und Hindernisse laufen",
  "Auf der Balancierstange balancieren",
  "Dem Joystick-Punkt schnell nachlaufen",
  "Energiesparend und ruhig gehen",
];

const EXAMPLES_CODE = [
  "Der Roboter soll hoch springen – schreib den Code dafür",
  "Schnell zum Joystick-Punkt rennen (Code-Term)",
  "Ruhig und stabil stehen, kaum Schwitzen",
  "Sitzt-Mimik: Höhe kurz senken, dann wieder aufstehen",
];

interface CodeResult {
  name: string;
  weight: number;
  code: string;
  patch: GeminiPatch;
  raw: string;
}

export default function GeminiPanel({ tel, onApplyPatch, onTurbo }: GeminiPanelProps) {
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState<GeminiModelId>("gemini-3.5-flash-lite");
  const [tab, setTab] = useState<"regeln" | "code">("regeln");
  const [goal, setGoal] = useState("");
  const [busy, setBusy] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [codeResult, setCodeResult] = useState<CodeResult | null>(null);
  const [codeWeight, setCodeWeight] = useState(1.5);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const p = loadGeminiPrefs();
    setApiKey(p.apiKey);
    setModel(p.model);
  }, []);

  const buildCtx = () => ({
    modelId: tel.modelId ?? "microduck",
    reward: tel.reward,
    imitationActive: !!tel.imitation?.playing,
    imitationName: tel.imitation?.name ?? null,
    worldEnabled: tel.world.enabled,
    pointMode: tel.pointCfg?.mode ?? "aus",
    pointRadius: tel.pointCfg?.radius ?? 2,
    generation: tel.es?.generation ?? 0,
  });

  const applyCommonPatch = (patch: GeminiPatch) => {
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
    const tr = patch.training;
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
      turbo: tr?.turbo ?? patch.turbo,
      resetFirst: tr?.resetFirst ?? patch.resetFirst,
      training: tr || patch.runCfg
        ? {
            ...(tr
              ? {
                  rolloutSteps: tr.rolloutSteps,
                  maxGenerations: tr.generations,
                  lr: tr.lr,
                  sigma: tr.sigma,
                }
              : {}),
            // v2.3: Profi-Tricks aus Gemini-Patch
            ...(patch.runCfg ? { runCfg: { ...tel.trainCfg.runCfg, ...patch.runCfg } } : {}),
          }
        : undefined,
    });
    const turbo = tr?.turbo ?? patch.turbo;
    if (turbo && [1, 4, 16, 32, 64].includes(turbo)) onTurbo(turbo);
  };

  const runRules = async () => {
    if (!goal.trim()) {
      toast({ title: "Ziel fehlt", description: "Beschreibe, was der Roboter lernen soll." });
      return;
    }
    setBusy(true);
    setExplanation(null);
    try {
      saveGeminiPrefs(apiKey.trim(), model);
      const { patch } = await applyGoalWithGemini(apiKey.trim(), model, buildCtx(), goal.trim());
      applyCommonPatch(patch);
      setExplanation(patch.explanation || "Regeln wurden angepasst.");
      toast({ title: "Gemini hat die Regeln angepasst", description: "Siehe Erklärung im Panel." });
    } catch (err: any) {
      toast({ title: "Gemini-Fehler", description: err?.message || String(err), variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  const runCode = async () => {
    if (!goal.trim()) {
      toast({ title: "Ziel fehlt", description: "Beschreibe, was Gemini in Code umsetzen soll." });
      return;
    }
    setBusy(true);
    setExplanation(null);
    try {
      saveGeminiPrefs(apiKey.trim(), model);
      const current = tel.reward.custom?.enabled ? tel.reward.custom.code : null;
      const { patch, raw } = await applyCodeWithGemini(
        apiKey.trim(), model, buildCtx(), goal.trim(), current,
      );
      const res: CodeResult = {
        name: patch.codeName || "KI-Code",
        weight: patch.codeWeight ?? 1.5,
        code: patch.code!,
        patch,
        raw,
      };
      setCodeResult(res);
      setCodeWeight(res.weight);
      setExplanation(patch.explanation || "Code bereit – prüfe ihn und tippe auf Anwenden.");
      toast({ title: "Gemini hat Trainingscode geschrieben", description: "Vorschau prüfen → Anwenden." });
    } catch (err: any) {
      toast({ title: "Gemini-Fehler", description: err?.message || String(err), variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  const applyCode = () => {
    if (!codeResult) return;
    onApplyPatch({ custom: { enabled: true, weight: codeWeight, name: codeResult.name, code: codeResult.code } });
    applyCommonPatch(codeResult.patch);
    toast({ title: "Code übernommen", description: `${codeResult.name} ist jetzt ein aktiver Reward-Term.` });
  };

  const downloadCode = async () => {
    if (!codeResult) return;
    const content = `// MicroDuck Trainer – KI-Reward-Code\n// Name: ${codeResult.name}\n// Gewicht: ${codeWeight}\n// api: h, upZ, vz, vx, vy, omega, angles, act, prevAct, qpos, qvel, qacc, torso, target, cmd, imitDelta, dt, t, step\nexport default function reward(api) {\n${codeResult.code}\n}\n`;
    const way = await saveToDevice(`mdt-reward-${codeResult.name.replace(/\W+/g, "_").toLowerCase()}.js`, content, "text/javascript");
    toast({
      title: "Code gespeichert",
      description: way === "bridge" ? "Im Downloads-Ordner der App." : "Download gestartet.",
    });
  };

  const copyCode = async () => {
    if (!codeResult) return;
    try {
      await navigator.clipboard.writeText(codeResult.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* Clipboard verweigert */ }
  };

  const activeCustom = tel.customCode
    ? (tel.reward.custom && tel.reward.custom.enabled ? tel.reward.custom : null)
    : null;

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

      {/* Tabs */}
      <div className="flex overflow-hidden rounded-lg border border-fuchsia-500/30" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "regeln"}
          onClick={() => setTab("regeln")}
          className={`h-10 flex-1 text-xs transition-colors ${
            tab === "regeln" ? "bg-fuchsia-500/25 font-semibold text-fuchsia-100" : "text-slate-400 hover:bg-white/5"
          }`}
        >
          <SlidersHorizontal className="mr-1 inline h-3.5 w-3.5" /> Regeln
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "code"}
          onClick={() => setTab("code")}
          className={`h-10 flex-1 text-xs transition-colors ${
            tab === "code" ? "bg-fuchsia-500/25 font-semibold text-fuchsia-100" : "text-slate-400 hover:bg-white/5"
          }`}
        >
          <Code2 className="mr-1 inline h-3.5 w-3.5" /> Code-Experte
        </button>
      </div>

      {/* Aktiver KI-Code-Term */}
      {activeCustom && (
        <div className="rounded-lg border border-emerald-500/40 bg-emerald-500/5 p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-300">
                <Check className="h-3.5 w-3.5 shrink-0" /> Aktiv: {activeCustom.name}
              </div>
              <div className="text-[10px] text-slate-500">
                Gewicht {activeCustom.weight.toFixed(1)} · läuft in jedem Reward-Schritt
              </div>
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={() => { onApplyPatch({ custom: { clear: true } }); toast({ title: "KI-Code entfernt" }); }}
              aria-label="KI-Code entfernen"
              className="h-9 shrink-0 border-red-400/40 px-2 text-red-300 hover:bg-red-500/10"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      {/* Ziel */}
      <div className="rounded-lg border border-fuchsia-500/25 bg-fuchsia-500/5 p-3">
        <div className="mb-2 flex items-center gap-2 text-xs text-fuchsia-300">
          <Wand2 className="h-3.5 w-3.5" />
          {tab === "regeln" ? "Was soll der Roboter lernen?" : "Was soll der KI-Code bewirken?"}
        </div>
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          rows={3}
          placeholder={tab === "regeln"
            ? "z. B. Ich will, dass der Roboter nach dem Training wirklich springen kann…"
            : "z. B. Schreibe den Trainingscode so, dass der Roboter hoch springt…"}
          className="w-full resize-none rounded-md border border-slate-700 bg-black/40 p-3 text-xs text-slate-200 outline-none focus:border-fuchsia-500/60"
        />
        <div className="mt-2 flex flex-wrap gap-1.5">
          {(tab === "regeln" ? EXAMPLES_RULES : EXAMPLES_CODE).map((ex) => (
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
          onClick={() => void (tab === "regeln" ? runRules() : runCode())}
          disabled={busy}
          className="mt-3 h-11 w-full gap-1.5 bg-fuchsia-500 font-semibold text-white hover:bg-fuchsia-400 disabled:opacity-60"
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          {busy ? "Gemini denkt…"
            : tab === "regeln" ? "Regeln automatisch anpassen"
            : "Trainingscode schreiben lassen"}
        </Button>
      </div>

      {/* Code-Vorschau (Code-Experte) */}
      {tab === "code" && codeResult && (
        <div className="rounded-lg border border-cyan-500/30 bg-black/30 p-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-cyan-300">
              <Code2 className="h-3.5 w-3.5" /> {codeResult.name}
            </div>
            <div className="flex gap-1">
              <Button
                size="sm" variant="outline"
                onClick={() => void copyCode()}
                aria-label="Code kopieren"
                className="h-8 border-slate-700 px-2 text-slate-400"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              </Button>
              <Button
                size="sm" variant="outline"
                onClick={() => void downloadCode()}
                aria-label="Code herunterladen"
                className="h-8 border-slate-700 px-2 text-slate-400"
              >
                <Download className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
          <pre className="max-h-56 overflow-auto rounded-md border border-slate-800 bg-[#05060a] p-2.5 font-mono text-[10.5px] leading-relaxed text-emerald-200/90">
{codeResult.code}
          </pre>

          {/* Gewicht */}
          <div className="mt-3">
            <div className="mb-1 flex justify-between text-[11px] text-slate-400">
              <span>Gewicht des Code-Terms</span>
              <span className="font-mono text-cyan-300">{codeWeight.toFixed(1)}</span>
            </div>
            <Slider
              value={[codeWeight]}
              min={-5}
              max={10}
              step={0.1}
              onValueChange={(v) => setCodeWeight(v[0])}
              className="[&_[data-slot=slider-range]]:bg-cyan-400"
            />
          </div>

          {/* Runden-Entscheidungen */}
          {codeResult.patch.training && (
            <div className="mt-3 grid grid-cols-2 gap-1.5 text-[10px]">
              {codeResult.patch.training.rolloutSteps !== undefined && (
                <div className="rounded border border-slate-700/60 bg-black/30 px-2 py-1.5">
                  <span className="text-slate-500">Rundenlänge: </span>
                  <span className="font-mono text-slate-300">{codeResult.patch.training.rolloutSteps} Schritte</span>
                </div>
              )}
              {codeResult.patch.training.generations !== undefined && (
                <div className="rounded border border-slate-700/60 bg-black/30 px-2 py-1.5">
                  <span className="text-slate-500">Ziel: </span>
                  <span className="font-mono text-slate-300">
                    {codeResult.patch.training.generations === 0 ? "∞ Generationen" : `${codeResult.patch.training.generations} Gen.`}
                  </span>
                </div>
              )}
              {codeResult.patch.training.turbo !== undefined && (
                <div className="rounded border border-slate-700/60 bg-black/30 px-2 py-1.5">
                  <span className="text-slate-500">Turbo: </span>
                  <span className="font-mono text-slate-300">{codeResult.patch.training.turbo}×</span>
                </div>
              )}
              {codeResult.patch.training.resetFirst !== undefined && (
                <div className="rounded border border-slate-700/60 bg-black/30 px-2 py-1.5">
                  <span className="text-slate-500">Reset zuerst: </span>
                  <span className="font-mono text-slate-300">{codeResult.patch.training.resetFirst ? "ja" : "nein"}</span>
                </div>
              )}
            </div>
          )}

          <Button
            onClick={applyCode}
            className="mt-3 h-11 w-full bg-emerald-500 font-semibold text-black hover:bg-emerald-400"
          >
            <Check className="h-4 w-4" /> Anwenden &amp; übernehmen
          </Button>
        </div>
      )}

      {/* Erklärung */}
      {explanation && (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
          <div className="mb-1 text-xs font-semibold text-emerald-300">
            {tab === "code" ? "Geminis Erklärung" : "Warum diese Regeln?"}
          </div>
          <p className="text-[11px] leading-relaxed text-emerald-100/90">{explanation}</p>
        </div>
      )}

      <p className="text-[11px] leading-snug text-slate-500">
        {tab === "regeln"
          ? "Gemini kann Reward-Gewichte, den Joystick-Punkt (Aus/Frei/Umkreis/Pfad), die Random-Welt und Turbo auf einmal anpassen – ganz ohne Programmieren."
          : "Im Code-Modus schreibt Gemini echten JavaScript-Trainingscode (sieht ihn vorher in der Vorschau), entscheidet Rundenlänge, Ziel-Generationen und Turbo. Der Code läuft lokal in der Trainings-Engine – validiert und ohne Netz-Zugriff."}
      </p>
    </div>
  );
}
