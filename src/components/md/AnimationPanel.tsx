"use client";

// ── MicroDuck Trainer v2.1 – Animations-Panel (GLB-Upload + Imitation) ───────
// GLB hochladen → Clips listen → Clip wählen → Roboter lernt sie per ES
// nachzumachen (Imitations-Reward + Phase in der Observation). Im Manuell-Modus
// optional direkt abspielen. Test-Modus spielt endlos ohne Reset.

import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import type { Telemetry } from "@/lib/md/app-core";
import type { ClipInfo } from "@/lib/md/imitation";
import { toast } from "@/hooks/use-toast";
import {
  Upload, Film, Play, Square, FlipHorizontal, Hand, Trash2, Loader2,
} from "lucide-react";

interface AnimationPanelProps {
  tel: Telemetry;
  onActivate: (buffer: ArrayBuffer, clipIndex: number, mirror: boolean) => Promise<{ ok: boolean; message: string }>;
  onClear: () => void;
  onPlaying: (v: boolean) => void;
  onManual: (v: boolean) => void;
  onImitWeight: (w: number) => void;
}

export default function AnimationPanel({
  tel, onActivate, onClear, onPlaying, onManual, onImitWeight,
}: AnimationPanelProps) {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [buffer, setBuffer] = useState<ArrayBuffer | null>(null);
  const [fileName, setFileName] = useState("");
  const [clips, setClips] = useState<ClipInfo[]>([]);
  const [selected, setSelected] = useState<number>(0);
  const [mirror, setMirror] = useState(false);
  const [parsing, setParsing] = useState(false);

  const imit = tel.imitation;

  const onFile = async (f: File | null) => {
    if (!f) return;
    if (!/\.glb$/i.test(f.name)) {
      toast({ title: "Falsches Format", description: "Bitte eine .glb-Datei wählen.", variant: "destructive" });
      return;
    }
    setParsing(true);
    try {
      const buf = await f.arrayBuffer();
      const { listGlbAnimations } = await import("@/lib/md/imitation");
      const list = await listGlbAnimations(buf);
      if (list.length === 0) {
        toast({ title: "Keine Animationen", description: "Die GLB enthält keine Animation-Clips.", variant: "destructive" });
        return;
      }
      setBuffer(buf);
      setFileName(f.name);
      setClips(list);
      setSelected(0);
      toast({ title: "GLB geladen", description: `${list.length} Clip(s) gefunden.` });
    } catch (err: any) {
      toast({ title: "Parse-Fehler", description: err?.message || String(err), variant: "destructive" });
    } finally {
      setParsing(false);
    }
  };

  const activate = async () => {
    if (!buffer) return;
    const r = await onActivate(buffer, selected, mirror);
    toast({
      title: r.ok ? "Imitation aktiv" : "Mapping fehlgeschlagen",
      description: r.message,
      variant: r.ok ? undefined : "destructive",
    });
  };

  return (
    <div className="space-y-4">
      {/* Upload */}
      <input
        ref={fileRef}
        type="file"
        accept=".glb,model/gltf-binary"
        className="hidden"
        onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
      />
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        className="flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed border-cyan-500/30 bg-cyan-500/5 p-5 transition-colors hover:border-cyan-400/60 hover:bg-cyan-500/10"
      >
        {parsing ? (
          <Loader2 className="h-6 w-6 animate-spin text-cyan-300" />
        ) : (
          <Upload className="h-6 w-6 text-cyan-300" />
        )}
        <span className="text-sm font-medium text-cyan-200">GLB-Animation hochladen</span>
        <span className="text-[11px] text-slate-500">
          z. B. Mixamo-Export (Humanoid-Skeleton) · Tasten: Springen, Tanzen, Winken…
        </span>
      </button>

      {/* Clips */}
      {clips.length > 0 && (
        <div className="space-y-2 rounded-lg border border-slate-700/50 bg-black/20 p-3">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Film className="h-3.5 w-3.5" /> {fileName} · {clips.length} Clip(s)
          </div>
          <div className="max-h-40 space-y-1.5 overflow-y-auto pr-1">
            {clips.map((c) => (
              <button
                key={c.index}
                type="button"
                onClick={() => setSelected(c.index)}
                className={`flex w-full items-center justify-between rounded-md border px-3 py-2 text-left text-xs transition-colors ${
                  selected === c.index
                    ? "border-cyan-400/60 bg-cyan-500/15 text-cyan-200"
                    : "border-slate-700/60 text-slate-300 hover:border-cyan-400/40"
                }`}
              >
                <span className="truncate">{c.name}</span>
                <span className="ml-2 shrink-0 font-mono text-[10px] text-slate-500">
                  {c.duration.toFixed(1)}s · {c.bones} Bones
                </span>
              </button>
            ))}
          </div>

          {/* Optionen */}
          <div className="flex items-center justify-between pt-1">
            <span className="flex items-center gap-1.5 text-xs text-slate-300">
              <FlipHorizontal className="h-3.5 w-3.5" /> Spiegeln (L/R tauschen)
            </span>
            <Switch checked={mirror} onCheckedChange={setMirror} aria-label="Spiegeln" />
          </div>

          <Button
            onClick={() => void activate()}
            className="h-11 w-full bg-fuchsia-500 font-semibold text-white hover:bg-fuchsia-400"
          >
            Nachmachen lernen (Imitation)
          </Button>
        </div>
      )}

      {/* Aktive Imitation */}
      {imit && (
        <div className="space-y-3 rounded-lg border border-fuchsia-500/30 bg-fuchsia-500/5 p-3">
          <div className="flex items-center justify-between">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-fuchsia-200">{imit.name}</div>
              <div className="text-[11px] text-slate-500">
                {imit.mapped} Gelenke gemappt · {imit.duration.toFixed(1)} s · t={imit.time.toFixed(1)}s
              </div>
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={onClear}
              aria-label="Animation entfernen"
              className="h-9 shrink-0 border-red-400/40 px-2 text-red-300 hover:bg-red-500/10"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-xs text-slate-300">
              {imit.playing ? <Play className="h-3.5 w-3.5" /> : <Square className="h-3.5 w-3.5" />}
              Wiedergabe (Training + Test)
            </span>
            <Switch
              checked={imit.playing}
              onCheckedChange={onPlaying}
              aria-label="Wiedergabe ein/aus"
            />
          </div>

          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-xs text-slate-300">
              <Hand className="h-3.5 w-3.5" /> Im Manuell-Modus direkt anwenden
            </span>
            <Switch
              checked={imit.manual}
              onCheckedChange={onManual}
              aria-label="Manuell anwenden"
            />
          </div>

          {/* Gewichte schnelleinstellen */}
          {tel.reward.terms.imitate && (
            <div>
              <div className="mb-1 flex justify-between text-[11px] text-slate-400">
                <span>Reward „Animation imitieren"</span>
                <span className="font-mono text-fuchsia-300">
                  {tel.reward.terms.imitate.weight.toFixed(1)}
                </span>
              </div>
              <Slider
                value={[tel.reward.terms.imitate.weight]}
                min={0}
                max={5}
                step={0.1}
                onValueChange={(v) => onImitWeight(v[0])}
                className="[&_[data-slot=slider-range]]:bg-fuchsia-400 [&_[data-slot=slider-thumb]]:border-fuchsia-300"
              />
            </div>
          )}

          <p className="text-[11px] leading-snug text-slate-500">
            Im Training bekommt die Policy die Animations-Phase (sin/cos) als
            zusätzliche Beobachtung – so lernt ES zeitabhängige Bewegungen. Der
            Test-Modus spielt alles kontinuierlich ohne Reset ab.
          </p>
        </div>
      )}

      {!imit && (
        <p className="rounded-lg border border-slate-700/50 bg-black/20 p-3 text-[11px] leading-relaxed text-slate-500">
          Ablauf: GLB hochladen → Clip wählen → „Nachmachen lernen". Die
          Knochen werden automatisch auf die Roboter-Gelenke gemappt (Mixamo- und
          ähnliche Humanoid-Skelette werden erkannt). Der Reward
          „Animation imitieren" wird automatisch aktiviert.
        </p>
      )}
    </div>
  );
}
