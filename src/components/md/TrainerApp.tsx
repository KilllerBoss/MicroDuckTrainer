"use client";

// ── MicroDuck Trainer v2.0 – Haupt-App ───────────────────────────────────────
// Topbar (mobil-kompakt), 3D-View, Touch-Gamepad, Panels als Desktop-Karte /
// Mobile-Bottom-Sheet. Alles Client-only (kein SSR, keine Hydration-Fehler).

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { toast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import TrainerCore, { type Telemetry, type PointCfg, type TrainCfg } from "@/lib/md/app-core";
import type { ModelId } from "@/lib/md/models";
import { getModel } from "@/lib/md/models";
import { TURBO_LEVELS, type TurboLevel } from "@/lib/md/es";
import type { MappingEntry } from "@/lib/md/mapping";
import type { RewardConfig } from "@/lib/md/rewards";
import { saveToDevice, initFileSavedListener } from "@/lib/md/fileio";
import GamepadOverlay from "./GamepadOverlay";
import TrainingPanel from "./TrainingPanel";
import RewardPanel from "./RewardPanel";
import ControlPanel from "./ControlPanel";
import WorldPanel from "./WorldPanel";
import AnimationPanel from "./AnimationPanel";
import GeminiPanel from "./GeminiPanel";
import {
  Maximize, Gamepad2, Dumbbell, Star, SlidersHorizontal, RotateCcw,
  AlertTriangle, Loader2, Mountain, Film, Bot, Menu, Download, Upload,
} from "lucide-react";

type PanelId = "training" | "reward" | "control" | "world" | "animation" | "ki" | null;

/** v2.2: alle Panels im Vollbild-Men – Topbar bleibt schlank (Mobil). */
const MENU_ITEMS: {
  id: Exclude<PanelId, null>;
  label: string;
  desc: string;
  icon: typeof Dumbbell;
  ki?: boolean;
}[] = [
  { id: "training", label: "Training", desc: "Start/Stopp, Turbo, Runden, Export/Import", icon: Dumbbell },
  { id: "ki", label: "KI-Trainingsregeln", desc: "Gemini passt Regeln an oder schreibt selbst Code", icon: Bot, ki: true },
  { id: "reward", label: "Bewertung", desc: "Reward-Terme & Gewichte", icon: Star },
  { id: "control", label: "Steuerung", desc: "Joystick-Punkt, Aktion-Mapping, Gamepad", icon: SlidersHorizontal },
  { id: "world", label: "Welt", desc: "Zufallswelt: Treppen, Hügel, Löcher, Hindernisse", icon: Mountain },
  { id: "animation", label: "Animation", desc: "GLB hochladen & nachmachen lernen", icon: Film },
];

const MODE_LABELS: Record<string, string> = {
  manuell: "Manuell",
  training: "Training",
  test: "Test",
};

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

const INITIAL_TEL: Telemetry = {
  booted: false, loading: false, loadingText: "", error: null,
  modelId: null, mode: "manuell", source: "onnx", onnxId: "", overridePolicy: null,
  esReady: false, speed: 0, height: 0, ctrlHz: 0, recovering: false, fallen: false,
  testUptime: 0, testReward: 0, training: false, es: null, mappings: [],
  reward: { version: 2, terms: {} },
  trainCfg: { rolloutSteps: 200, maxGenerations: 0, lr: 0.03, sigma: 0.08,
    cmdTrain: true, cmdFwd: 0.25, curriculum: true, actionSmooth: 0.6,
    pushes: true, noiseReset: true, fitnessMode: "sum", weightDecay: 0.02 },
  trainPreview: true,
  previewSpeed: 1,
  customCode: null,
  world: { enabled: false, seed: 0, difficulty: 0.4, density: 0.5,
    features: { treppen: true, huegel: true, loecher: false, hindernisse: true, stange: false } },
  pointMode: false,
  pointCfg: { mode: "aus", radius: 2, speedByDist: true, maxSpeed: 0.25, fullDist: 1.5 },
  point: [0.8, 0],
  imitation: null,
};

const emptySubscribe = () => () => {};

export default function TrainerApp() {
  // Client-Erkennung ohne setState-im-Effect (Hydration-sicher)
  const mounted = useSyncExternalStore(emptySubscribe, () => true, () => false);
  const [tel, setTel] = useState<Telemetry>(INITIAL_TEL);
  const [panel, setPanel] = useState<PanelId>(null);
  const [gamepadOn, setGamepadOn] = useState(false);
  const [autoRecovery, setAutoRecovery] = useState(true);
  const [turbo, setTurbo] = useState<TurboLevel>(1);
  const [hasSaved, setHasSaved] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const rootRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const coreRef = useRef<TrainerCore | null>(null);
  const loadingShown = useRef(false);
  const importRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!mounted || !containerRef.current || coreRef.current) return;
    const core = new TrainerCore();
    coreRef.current = core;
    core.onTelemetry = (t) => setTel(t);
    core.onNotice = (title, message) => toast({ title, description: message });
    void core.boot(containerRef.current);
    return () => {
      core.dispose();
      coreRef.current = null;
    };
  }, [mounted]);

  // v2.2: APK-Bridge meldet gespeicherte Dateien (Downloads-Ordner)
  useEffect(() => {
    return initFileSavedListener((ev) => {
      toast({
        title: ev.ok ? "Datei gespeichert" : "Speichern fehlgeschlagen",
        description: ev.info,
        variant: ev.ok ? undefined : "destructive",
      });
    });
  }, []);

  // Lade-Toasts ("Lade Ente…" / "Lade G1 (29 Gelenke)…")
  useEffect(() => {
    if (tel.loading && tel.loadingText) {
      if (!loadingShown.current) {
        loadingShown.current = true;
        toast({ title: tel.loadingText, description: "MuJoCo wird kompiliert…" });
      }
    } else if (!tel.loading && loadingShown.current) {
      loadingShown.current = false;
      toast({ title: "Bereit", description: "Roboter geladen – viel Erfolg beim Training!" });
    }
  }, [tel.loading, tel.loadingText]);

  // Fehler-Toast
  useEffect(() => {
    if (tel.error) {
      toast({
        title: "Fehler",
        description: tel.error,
        variant: "destructive",
      });
    }
  }, [tel.error]);

  const core = () => coreRef.current;

  const switchModel = useCallback((id: ModelId) => {
    const c = core();
    if (!c || tel.loading || id === tel.modelId) return;
    void c.loadModel(id);
  }, [tel.loading, tel.modelId]);

  const setMode = useCallback((mode: Telemetry["mode"]) => {
    core()?.setMode(mode);
  }, []);

  const onStartTraining = useCallback(async () => {
    const c = core();
    if (!c) return;
    await c.startTraining();
    setHasSaved(c.hasSavedTheta());
  }, []);

  const onSaveTheta = useCallback(async () => {
    const ok = await core()?.saveTheta();
    toast({
      title: ok ? "Gespeichert" : "Speichern fehlgeschlagen",
      description: ok
        ? "Beste Policy im lokalen Speicher abgelegt."
        : "Kein Trainingsstand vorhanden.",
      variant: ok ? undefined : "destructive",
    });
  }, []);

  const onLoadTheta = useCallback(async () => {
    const c = core();
    if (!c) return;
    const ok = await c.loadTheta();
    toast({
      title: ok ? "Geladen" : "Laden fehlgeschlagen",
      description: ok ? "Trainierte Policy wiederhergestellt." : "Kein Speicherstand gefunden.",
      variant: ok ? undefined : "destructive",
    });
  }, []);

  const onMappings = useCallback((m: MappingEntry[]) => {
    core()?.setMappings(m);
  }, []);

  const openPanel = useCallback((id: Exclude<PanelId, null>) => {
    setPanel((cur) => (cur === id ? null : id));
    setMenuOpen(false);
    if (id === "training") {
      try { setHasSaved(coreRef.current?.hasSavedTheta() ?? false); } catch { setHasSaved(false); }
    }
  }, []);

  const onReward = useCallback((cfg: RewardConfig) => {
    core()?.setReward(cfg);
  }, []);

  const onJoystick = useCallback((x: number, y: number) => {
    core()?.setJoystick(x, y);
  }, []);

  // ── v2.1-Callbacks ──
  const onApplyWorld = useCallback((cfg: Telemetry["world"]) => {
    void core()?.setWorld(cfg);
  }, []);

  const onRerollWorld = useCallback(() => {
    core()?.rerollWorld();
  }, []);

  const onActivateAnim = useCallback(async (
    buffer: ArrayBuffer, clipIndex: number, mirror: boolean,
  ): Promise<{ ok: boolean; message: string }> => {
    const c = core();
    const id = c ? (tel.modelId ?? "microduck") : "microduck";
    if (!c) return { ok: false, message: "App noch nicht bereit." };
    return c.activateAnimation(buffer, clipIndex, mirror, id);
  }, [tel.modelId]);

  const onClearAnim = useCallback(() => core()?.clearAnimation(), []);
  const onPlayingAnim = useCallback((v: boolean) => core()?.setImitPlaying(v), []);
  const onManualAnim = useCallback((v: boolean) => core()?.setImitManual(v), []);

  const onImitWeight = useCallback((w: number) => {
    const c = core();
    const cur = c ? tel.reward : null;
    if (!c || !cur?.terms.imitate) return;
    const next = {
      version: 2 as const,
      terms: { ...cur.terms, imitate: { ...cur.terms.imitate, weight: w, enabled: w !== 0 } },
    };
    c.setReward(next);
  }, [tel.reward]);

  const onPointCfg = useCallback((patch: Partial<PointCfg>) => core()?.setPointCfg(patch), []);

  // ── v2.2-Callbacks ──
  const onTrainCfg = useCallback((patch: Partial<TrainCfg>) => core()?.setTrainCfg(patch), []);

  // v2.2: Export als einfache Funktion (Telemetrie-Werte brauchen kein Memo)
  const onExport = async () => {
    const json = coreRef.current?.exportAll();
    if (!json) {
      toast({ title: "Export fehlgeschlagen", description: "App noch nicht bereit.", variant: "destructive" });
      return;
    }
    try {
      const gen = tel.es?.generation ?? 0;
      const name = `mdt-export-${tel.modelId ?? "model"}-gen${gen}.json`;
      const way = await saveToDevice(name, json, "application/json");
      toast({
        title: "Export erstellt",
        description: way === "bridge"
          ? `${name} landet im Downloads-Ordner.`
          : `${name} wird heruntergeladen.`,
      });
    } catch (err: any) {
      toast({ title: "Export fehlgeschlagen", description: err?.message || String(err), variant: "destructive" });
    }
  };

  const onImportFile = useCallback(async (f: File | null) => {
    if (!f) return;
    try {
      const txt = await f.text();
      const r = await coreRef.current?.importAll(txt);
      toast({
        title: r?.ok ? "Import abgeschlossen" : "Import fehlgeschlagen",
        description: r?.message || "Unbekanntes Ergebnis.",
        variant: r?.ok ? undefined : "destructive",
      });
      try { setHasSaved(coreRef.current?.hasSavedTheta() ?? false); } catch { /* ignore */ }
    } catch (err: any) {
      toast({ title: "Import fehlgeschlagen", description: err?.message || String(err), variant: "destructive" });
    }
  }, []);

  const onGeminiPatch = useCallback((actions: {
    reward?: Telemetry["reward"];
    pointMode?: boolean;
    pointCfg?: Partial<PointCfg>;
    world?: Partial<Telemetry["world"]>;
    turbo?: number;
    resetFirst?: boolean;
    custom?: RewardConfig["custom"] | { clear: true };
    training?: Partial<TrainCfg>;
  }) => {
    const c = core();
    if (!c) return;
    if (actions.reward) c.setReward(actions.reward);
    if (actions.custom) {
      if ("clear" in actions.custom) c.setCustomTerm(null);
      else c.setCustomTerm(actions.custom);
    }
    if (actions.training) c.setTrainCfg(actions.training);
    if (typeof actions.pointMode === "boolean") c.setPointMode(actions.pointMode);
    if (actions.pointCfg) c.setPointCfg(actions.pointCfg);
    if (actions.world) {
      void c.setWorld({ ...tel.world, ...actions.world,
        features: { ...tel.world.features, ...(actions.world.features ?? {}) } });
    }
    if (actions.resetFirst) {
      c.resetSim();
      toast({ title: "Zurückgesetzt", description: "Neues Bewegungsmuster – Training startet bei Gen 0." });
    }
  }, [tel.world]);

  const onButton = useCallback((source: "joyX" | "joyY" | "A" | "B" | "C" | "D", pressed: boolean) => {
    if (pressed) core()?.pressButton(source);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (rootRef.current) void TrainerCore.toggleFullscreen(rootRef.current);
  }, []);

  const modelId = tel.modelId;
  const meta = modelId ? getModel(modelId) : null;
  const isG1 = modelId === "unitree_g1";
  const showGamepad = gamepadOn || tel.mode === "manuell";

  if (!mounted) {
    return <div className="fixed inset-0 bg-[#05060a]" />;
  }

  return (
    <div
      ref={rootRef}
      className="fixed inset-0 flex flex-col overflow-hidden bg-[#05060a] text-slate-200 select-none"
    >
      {/* 3D-View */}
      <div ref={containerRef} className="absolute inset-0" aria-label="3D-Szene" />

      {/* ── Topbar (v2.2: kompakt – alle Panels im ☰-Menü) ── */}
      <header className="relative z-40 border-b border-cyan-500/15 bg-[#0b0f18]/95 backdrop-blur">
        <div className="flex items-center gap-1.5 px-2 py-1.5">
          <h1 className="mr-0.5 shrink-0 text-[13px] font-bold tracking-tight text-cyan-300 sm:text-sm">
            <span aria-hidden>🦆</span>
            <span className="ml-1 hidden sm:inline">MicroDuck Trainer</span>
          </h1>

          {/* Modell-Switch */}
          <div className="flex overflow-hidden rounded-lg border border-cyan-500/25" role="group" aria-label="Roboter wählen">
            <button
              type="button"
              onClick={() => switchModel("microduck")}
              disabled={tel.loading}
              className={`h-9 px-2.5 text-xs transition-colors sm:px-3 ${
                modelId === "microduck" ? "bg-cyan-500/25 text-cyan-200" : "text-slate-400 hover:bg-white/5"
              }`}
            >
              🦆<span className="ml-1 hidden sm:inline">Ente</span>
            </button>
            <button
              type="button"
              onClick={() => switchModel("unitree_g1")}
              disabled={tel.loading}
              className={`h-9 px-2.5 text-xs transition-colors sm:px-3 ${
                modelId === "unitree_g1" ? "bg-cyan-500/25 text-cyan-200" : "text-slate-400 hover:bg-white/5"
              }`}
            >
              🧍<span className="ml-1 hidden sm:inline">Mensch</span>
            </button>
          </div>

          <div className="flex-1" />

          {/* Gamepad / Vollbild / Menü */}
          <Button
            variant="outline"
            onClick={() => setGamepadOn((v) => !v)}
            aria-pressed={gamepadOn}
            aria-label="Gamepad ein/aus"
            className={`h-9 px-2 ${gamepadOn ? "border-cyan-400 bg-cyan-500/20 text-cyan-200" : "border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"}`}
          >
            <Gamepad2 className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            onClick={toggleFullscreen}
            aria-label="Vollbild"
            className="h-9 px-2 border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"
          >
            <Maximize className="h-4 w-4" />
          </Button>
          <Button
            variant={menuOpen ? "default" : "outline"}
            onClick={() => setMenuOpen((v) => !v)}
            aria-expanded={menuOpen}
            aria-label="Menü öffnen"
            className={`h-9 px-2 ${menuOpen ? "bg-cyan-500 text-black hover:bg-cyan-400" : "border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"}`}
          >
            <Menu className="h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* ── v2.2: Vollbild-Menü (große Touch-Ziele statt 8 Topbar-Buttons) ── */}
      {menuOpen && (
        <div
          className="absolute inset-0 z-50 bg-[#05060a]/70 backdrop-blur-sm"
          onClick={() => setMenuOpen(false)}
          role="presentation"
        >
          <nav
            className="absolute inset-x-2 top-12 max-h-[calc(100%-4rem)] overflow-y-auto rounded-2xl border border-cyan-500/25 bg-[#0b0f18] p-2 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
            aria-label="Hauptmenü"
            style={{ paddingBottom: "calc(0.5rem + env(safe-area-inset-bottom))" }}
          >
            <div className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Panels
            </div>
            {MENU_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = panel === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => openPanel(item.id)}
                  className={`flex min-h-[56px] w-full items-center gap-3 rounded-xl px-3 text-left transition-colors ${
                    active
                      ? (item.ki ? "bg-fuchsia-500/20 text-fuchsia-200" : "bg-cyan-500/20 text-cyan-100")
                      : "text-slate-200 hover:bg-white/5"
                  }`}
                >
                  <span
                    className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border ${
                      item.ki
                        ? "border-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-300"
                        : "border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
                    }`}
                  >
                    <Icon className="h-5 w-5" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-medium">{item.label}</span>
                    <span className="block truncate text-[11px] text-slate-500">{item.desc}</span>
                  </span>
                  {active && (
                    <span className={`shrink-0 rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase ${
                      item.ki ? "bg-fuchsia-500/30 text-fuchsia-200" : "bg-cyan-500/30 text-cyan-200"
                    }`}>
                      offen
                    </span>
                  )}
                </button>
              );
            })}
            <div className="mt-2 flex gap-2 border-t border-cyan-500/15 pt-2">
              <Button
                variant="outline"
                onClick={() => { core()?.resetSim(); setMenuOpen(false); }}
                disabled={tel.mode === "test" || tel.loading}
                className="h-11 flex-1 gap-1.5 border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"
              >
                <RotateCcw className="h-4 w-4" /> Reset
              </Button>
              <Button
                variant="outline"
                onClick={() => { importRef.current?.click(); setMenuOpen(false); }}
                className="h-11 flex-1 gap-1.5 border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"
              >
                <Upload className="h-4 w-4" /> Import
              </Button>
              <Button
                variant="outline"
                onClick={() => { void onExport(); setMenuOpen(false); }}
                className="h-11 flex-1 gap-1.5 border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"
              >
                <Download className="h-4 w-4" /> Export
              </Button>
            </div>
            {/* v2.3: Alles auf Original zurücksetzen (Original-Welten/-Regeln) */}
            <Button
              variant="outline"
              onClick={() => {
                try {
                  const keys: string[] = [];
                  for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (k && k.startsWith("mdt_v2_")) keys.push(k);
                  }
                  keys.forEach((k) => localStorage.removeItem(k));
                } catch { /* ignore */ }
                location.reload();
              }}
              className="h-11 w-full gap-1.5 border-amber-500/40 text-amber-300 hover:bg-amber-500/10"
            >
              <RotateCcw className="h-4 w-4" /> Alles auf Original zurücksetzen
            </Button>
            <p className="px-3 pt-1 text-[10px] leading-snug text-slate-600">
              Setzt Welten, Regeln, Training und Steuerung auf den Auslieferungszustand zurück.
            </p>
            <p className="px-3 pt-2 text-[10px] leading-snug text-slate-600">
              Tippe außerhalb des Menüs, um es zu schließen.
            </p>
          </nav>
        </div>
      )}

      {/* ── HUD (Modus + Quelle + Telemetrie) ── */}
      <div className="pointer-events-none absolute left-2 top-14 z-20 flex max-w-[calc(100%-1rem)] flex-wrap gap-1.5 sm:left-3 sm:top-16">
        {/* Modi */}
        <div className="pointer-events-auto flex overflow-hidden rounded-lg border border-cyan-500/25 bg-[#0b0f18]/90 backdrop-blur">
          {(["manuell", "training", "test"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              disabled={tel.loading}
              className={`h-9 px-2.5 text-xs transition-colors sm:px-3 ${
                tel.mode === m ? "bg-cyan-500/25 font-semibold text-cyan-200" : "text-slate-400 hover:bg-white/5"
              }`}
            >
              {MODE_LABELS[m]}
            </button>
          ))}
        </div>

        {/* Quelle */}
        <div className="pointer-events-auto flex overflow-hidden rounded-lg border border-cyan-500/25 bg-[#0b0f18]/90 backdrop-blur">
          <button
            type="button"
            onClick={() => core()?.setSource("onnx")}
            disabled={isG1}
            className={`h-9 px-2.5 text-xs transition-colors sm:px-3 ${
              tel.source === "onnx" ? "bg-cyan-500/25 font-semibold text-cyan-200" : "text-slate-400 hover:bg-white/5"
            } ${isG1 ? "opacity-40" : ""}`}
          >
            Original (ONNX)
          </button>
          <button
            type="button"
            onClick={() => core()?.setSource("es")}
            className={`h-9 px-2.5 text-xs transition-colors sm:px-3 ${
              tel.source === "es" ? "bg-cyan-500/25 font-semibold text-cyan-200" : "text-slate-400 hover:bg-white/5"
            }`}
          >
            ES-Policy
          </button>
          {tel.source === "onnx" && !isG1 && (
            <select
              aria-label="ONNX-Policy wählen"
              value={tel.overridePolicy ?? tel.onnxId}
              onChange={(e) => core()?.setOnnx(e.target.value)}
              className="h-9 max-w-[130px] border-l border-cyan-500/20 bg-[#0b0f18] px-1.5 text-xs text-slate-300 outline-none"
            >
              {(meta?.policies ?? []).map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          )}
        </div>

        {/* Telemetrie-Chips */}
        <div className="pointer-events-none flex flex-wrap gap-1.5">
          <Chip label="Speed" value={`${tel.speed.toFixed(2)} m/s`} />
          <Chip label="Höhe" value={`${tel.height.toFixed(2)} m`} />
          <Chip label="Hz" value={String(tel.ctrlHz)} />
          {tel.mode === "test" && (
            <>
              <Chip label="Laufzeit" value={fmtTime(tel.testUptime)} accent />
              <Chip label="Reward" value={tel.testReward.toFixed(1)} accent />
            </>
          )}
          {tel.mode === "training" && tel.es && (
            <Chip label="Gen" value={`${tel.es.generation} · ${Number.isFinite(tel.es.bestEver) ? tel.es.bestEver.toFixed(2) : "–"}`} accent />
          )}
          {tel.recovering && <Chip label="Recovery" value="aktiv" accent />}
          {tel.fallen && !tel.recovering && <Chip label="Sturz" value="erkannt" accent />}
          {tel.world.enabled && <Chip label="Welt" value="aktiv" />}
          {tel.pointMode && (
            <Chip
              label={`Punkt·${tel.pointCfg?.mode ?? "frei"}`}
              value={`${tel.point[0].toFixed(1)}, ${tel.point[1].toFixed(1)}`}
              accent
            />
          )}
          {tel.imitation?.playing && (
            <Chip label="Anim" value={`${tel.imitation.time.toFixed(1)}/${tel.imitation.duration.toFixed(1)}s`} accent />
          )}
        </div>
      </div>

      {/* Reset */}
      <div className="absolute right-2 top-14 z-20 sm:right-3 sm:top-16">
        <Button
          variant="outline"
          onClick={() => core()?.resetSim()}
          disabled={tel.mode === "test" || tel.loading}
          title={tel.mode === "test" ? "Im Test-Modus gibt es keinen Reset – nur Stop" : "Roboter zurücksetzen"}
          className="h-9 gap-1.5 border-cyan-500/30 bg-[#0b0f18]/90 px-2.5 text-xs text-cyan-300 backdrop-blur hover:bg-cyan-500/10"
        >
          <RotateCcw className="h-3.5 w-3.5" /> Reset
        </Button>
      </div>

      {/* v2.2: versteckter Import-Dialog (Android: öffnet den Dateimanager) */}
      <input
        ref={importRef}
        type="file"
        accept=".json,application/json"
        className="hidden"
        onChange={(e) => { void onImportFile(e.target.files?.[0] ?? null); e.target.value = ""; }}
      />

      {/* ES-Policy-Hinweis (Quelle ES ohne Modell) */}
      {tel.source === "es" && !tel.esReady && (
        <div className="absolute left-1/2 top-1/2 z-20 -translate-x-1/2 -translate-y-1/2 rounded-lg border border-amber-400/40 bg-[#0b0f18]/95 px-4 py-3 text-center text-xs text-amber-300 backdrop-blur">
          Noch keine ES-Policy vorhanden.
          <br />
          <span className="text-slate-400">Erst trainieren (oder Laden) – dann erneut wählen.</span>
        </div>
      )}

      {/* Test-Stoppen-Hinweis */}
      {tel.mode === "test" && (
        <div className="absolute bottom-2 left-1/2 z-20 -translate-x-1/2 rounded-full border border-cyan-500/30 bg-[#0b0f18]/90 px-3 py-1.5 text-[11px] text-cyan-300 backdrop-blur">
          Test läuft endlos – nur <b>Modus wechseln</b> beendet. Kein Reset, kein Teleport.
        </div>
      )}

      {/* ── Panel (Desktop: rechte Karte / Mobil: Bottom-Sheet) ── */}
      {panel && (
        <aside
          className="absolute z-40 flex flex-col border-cyan-500/20 bg-[#0b0f18]/95 shadow-2xl backdrop-blur
            inset-x-1 bottom-1 max-h-[60vh] rounded-xl border
            lg:inset-x-auto lg:bottom-3 lg:right-3 lg:top-32 lg:max-h-none lg:w-[400px] lg:rounded-xl"
          style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
          aria-label={`${panel === "training" ? "Trainings" : panel === "reward" ? "Bewertungs" : "Steuerungs"}-Panel`}
        >
          <div className="flex items-center justify-between border-b border-cyan-500/15 px-3 py-2">
            <h2 className="text-sm font-semibold text-cyan-300">
              {panel === "training" ? "Training (Evolution Strategy)"
                : panel === "reward" ? "Bewertung (Reward)"
                : panel === "control" ? "Steuerung"
                : panel === "world" ? "Welt (Random-Generator)"
                : panel === "animation" ? "Animation (GLB-Imitation)"
                : "KI-Trainingsregeln (Gemini)"}
            </h2>
            <button
              type="button"
              onClick={() => setPanel(null)}
              aria-label="Panel schließen"
              className="flex h-9 w-9 items-center justify-center rounded-md text-slate-400 hover:bg-white/10 hover:text-white"
            >
              ✕
            </button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-3 md-scroll">
            {panel === "training" && (
              <TrainingPanel
                tel={tel}
                turbo={turbo}
                hasSaved={hasSaved}
                onTurbo={(t) => { setTurbo(t); core()?.setTurbo(t); }}
                onStart={() => void onStartTraining()}
                onStop={() => core()?.stopTraining()}
                onShowBest={() => core()?.showBest()}
                onSave={() => void onSaveTheta()}
                onLoad={() => void onLoadTheta()}
                onTrainCfg={onTrainCfg}
                onTrainPreview={(on) => core()?.setTrainPreview(on)}
                onExport={() => void onExport()}
                onImport={() => importRef.current?.click()}
              />
            )}
            {panel === "reward" && <RewardPanel tel={tel} onChange={onReward} />}
            {panel === "control" && (
              <ControlPanel
                tel={tel}
                gamepadOn={gamepadOn}
                autoRecovery={autoRecovery}
                onGamepad={setGamepadOn}
                onAutoRecovery={(v) => { setAutoRecovery(v); core()?.setAutoRecovery(v); }}
                onMappings={onMappings}
                onPointCfg={onPointCfg}
              />
            )}
            {panel === "world" && (
              <WorldPanel tel={tel} onApply={onApplyWorld} onReroll={onRerollWorld} />
            )}
            {panel === "animation" && (
              <AnimationPanel
                tel={tel}
                onActivate={onActivateAnim}
                onClear={onClearAnim}
                onPlaying={onPlayingAnim}
                onManual={onManualAnim}
                onImitWeight={onImitWeight}
              />
            )}
            {panel === "ki" && (
              <GeminiPanel tel={tel} onApplyPatch={onGeminiPatch} onTurbo={(t) => {
                const lvl = ([1, 4, 16, 32, 64] as const).includes(t as 1 | 4 | 16 | 32 | 64)
                  ? (t as 1 | 4 | 16 | 32 | 64) : 1;
                setTurbo(lvl);
                core()?.setTurbo(lvl);
              }} />
            )}
          </div>
        </aside>
      )}

      {/* Gamepad-Overlay */}
      <GamepadOverlay visible={showGamepad} onJoystick={onJoystick} onButton={onButton} />

      {/* Lade-Overlay */}
      {tel.loading && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center gap-3 bg-[#05060a]/80 backdrop-blur-sm">
          <Loader2 className="h-8 w-8 animate-spin text-cyan-300" />
          <p className="font-mono text-sm text-cyan-200">{tel.loadingText}</p>
        </div>
      )}

      {/* Fehler-Overlay */}
      {tel.error && !tel.loading && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center gap-3 bg-[#05060a]/90 p-6 text-center">
          <AlertTriangle className="h-10 w-10 text-red-400" />
          <p className="max-w-md text-sm text-red-300">{tel.error}</p>
          <Button
            onClick={() => location.reload()}
            className="bg-cyan-500 font-semibold text-black hover:bg-cyan-400"
          >
            Neu laden
          </Button>
        </div>
      )}

      {/* Boot-Platzhalter */}
      {!tel.booted && !tel.loading && !tel.error && (
        <div className="absolute inset-0 z-40 flex items-center justify-center">
          <p className="font-mono text-sm text-slate-500">Initialisiere…</p>
        </div>
      )}
    </div>
  );
}

function Chip({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div
      className={`rounded-md border px-2 py-1 backdrop-blur ${
        accent ? "border-cyan-400/40 bg-cyan-500/15" : "border-slate-700/60 bg-[#0b0f18]/85"
      }`}
    >
      <span className="text-[9px] uppercase tracking-wide text-slate-500">{label} </span>
      <span className={`font-mono text-xs ${accent ? "text-cyan-200" : "text-slate-300"}`}>{value}</span>
    </div>
  );
}
