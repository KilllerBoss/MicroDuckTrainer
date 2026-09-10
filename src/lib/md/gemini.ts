// ── MicroDuck Trainer v2.1 – Gemini-Trainingsassistent ───────────────────────
// Nimmt ein Ziel in natürlicher Sprache („Ich will, dass der Roboter springen
// lernt") und passt Reward-Terme, Punkt-Modus, Welt und Turbo automatisch an.
// Modelle: gemini-robotics-er-2-preview · gemini-3.5-flash-lite · gemini-3.8-flash
// Läuft direkt im Client (generativelanguage.googleapis.com erlaubt CORS).

import type { ModelId } from "./models";
import { getModel } from "./models";
import { REWARD_TERMS, type RewardConfig } from "./rewards";
import type { WorldFeatures } from "./worldgen";

/** Standard-Key (vom Nutzer bereitgestellt); kann in der UI überschrieben werden. */
export const DEFAULT_GEMINI_KEY = "AIzaSyCl2mYBoobRRIneTUdJa2FFIF-BGj4iqrg";

export type GeminiPointMode = "aus" | "frei" | "umkreis" | "pfad";

export const GEMINI_MODELS = [
  { id: "gemini-robotics-er-2-preview", label: "Robotics ER 2 (Preview)" },
  { id: "gemini-3.5-flash-lite", label: "3.5 Flash Lite (schnell)" },
  { id: "gemini-3.8-flash", label: "3.8 Flash (smarter)" },
] as const;

export type GeminiModelId = (typeof GEMINI_MODELS)[number]["id"];

export interface GeminiPatch {
  reward?: Record<
    string,
    { enabled?: boolean; weight?: number; param?: number }
  >;
  pointMode?: boolean;
  point?: {
    mode?: GeminiPointMode;
    radius?: number;
    speedByDist?: boolean;
    maxSpeed?: number;
    fullDist?: number;
  };
  world?: {
    enabled?: boolean;
    difficulty?: number;
    density?: number;
    features?: Partial<WorldFeatures>;
  };
  turbo?: number; // 1 | 4 | 16 | 32 | 64
  resetFirst?: boolean; // Training von Null starten (z. B. bei neuem Bewegungsziel)
  explanation?: string;
}

export interface GeminiContext {
  modelId: ModelId;
  reward: RewardConfig;
  imitationActive: boolean;
  imitationName: string | null;
  worldEnabled: boolean;
  /** "aus" | "frei" | "umkreis" | "pfad" */
  pointMode: string;
  pointRadius: number;
  generation: number;
}

const TERM_DOCS = REWARD_TERMS.map(
  (t) =>
    `- "${t.id}" (${t.label}${t.penalty ? ", Strafterm" : ""}): ${t.desc}` +
    (t.hasParam ? ` · Param "${t.paramLabel}" ${t.paramMin}..${t.paramMax}` : ""),
).join("\n");

function buildPrompt(ctx: GeminiContext, goal: string): string {
  const meta = getModel(ctx.modelId);
  const cur = Object.entries(ctx.reward.terms)
    .filter(([, v]) => v.enabled)
    .map(([k, v]) => `${k}(w=${v.weight.toFixed(2)}${v.param ? `,p=${v.param.toFixed(2)}` : ""})`)
    .join(", ");
  return `Du bist der Trainingsregeln-Assistent einer Roboter-Simulation (MuJoCo + Evolution-Strategy-Training im Browser).
Roboter: ${meta.label} mit ${meta.actionDim} Gelenken (Positionsaktuatoren, Policy 50 Hz).
Aktive Reward-Terme: ${cur || "keine"}
Training: ${ctx.generation} Generationen bisher. Imitation einer GLB-Animation aktiv: ${ctx.imitationActive ? `ja (${ctx.imitationName})` : "nein"}.
Random-Welt (Treppen/Hügel/Löcher/Hindernisse/Balancierstange) aktiv: ${ctx.worldEnabled ? "ja" : "nein"}.
Punkt-Modus (Joystick bewegt einen 3D-Punkt, kamera-relativ): aktuell "${ctx.pointMode}" mit Radius ${ctx.pointRadius.toFixed(1)} m.
  - "aus": kein Punkt (klassische Joystick-Steuerung)
  - "frei": Punkt frei in der Arena, Roboter verfolgt ihn
  - "umkreis": Punkt bleibt im Radius um den Roboter
  - "pfad": Roboter folgt einem physikalisch berechneten Pfad mit Schwung (Momentum) zum Punkt

VERFÜGBARE REWARD-TERME:
${TERM_DOCS}

ZIEL DES NUTZERS: "${goal}"

Aufgabe: Passe die Trainingsregeln an, damit der Roboter nach dem Training das Gewünschte wirklich kann.
Wähle Gewichte konservativ, aber wirksam; deaktiviere Terme, die dem Ziel widersprechen.
Wenn das Ziel Bewegung über Zeit braucht (Springen, Tanzen, Aufstehen) und keine Imitation aktiv ist, erkläre es und setze trotzdem sinnvolle Terme.
Wenn Welt/Punkt-Modus helfen (z. B. Treppen → Welt mit Treppen, Ziel verfolgen → point.mode "frei" oder "pfad" + Reward-Term "pointChase"), setze sie.
turbo: 1=stabil, 16=schnell, 64=maximal (nur bei einfachen Zielen hoch setzen). resetFirst=true bei grundlegend neuem Bewegungsmuster.

Antworte AUSSCHLIESSLICH mit JSON (kein Markdown) nach diesem Schema:
{"reward":{"<termId>":{"enabled":bool,"weight":number,"param":number}},"point":{"mode":"frei","radius":1.5,"speedByDist":true,"maxSpeed":0.25,"fullDist":1.5},"world":{"enabled":bool,"difficulty":number,"density":number,"features":{"treppen":bool,"huegel":bool,"loecher":bool,"hindernisse":bool,"stange":bool}},"turbo":1,"resetFirst":bool,"explanation":"max. 4 Sätze, Deutsch, warum diese Regeln zum Ziel führen"}`;
}

export async function applyGoalWithGemini(
  apiKey: string,
  model: GeminiModelId,
  ctx: GeminiContext,
  goal: string,
): Promise<{ patch: GeminiPatch; raw: string }> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: buildPrompt(ctx, goal) }] }],
      generationConfig: {
        temperature: 0.4,
        responseMimeType: "application/json",
        maxOutputTokens: 2048,
      },
    }),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const j = await res.json();
      detail = j?.error?.message ?? detail;
    } catch { /* ignore */ }
    if (/location is not supported/i.test(detail)) {
      throw new Error(
        "Gemini: Dein aktueller Standort/Netz wird von der Gemini-API nicht unterstützt. "
        + "Wechsle ggf. ins WLAN oder nutze ein VPN mit passendem Land.",
      );
    }
    if (/API key not valid|API_KEY_INVALID/i.test(detail)) {
      throw new Error("Gemini: Der API-Key ist ungültig – prüfe ihn im KI-Panel.");
    }
    throw new Error(`Gemini-Fehler: ${detail}`);
  }
  const data = await res.json();
  const text: string =
    data?.candidates?.[0]?.content?.parts?.map((p: any) => p.text ?? "").join("") ?? "";
  if (!text.trim()) throw new Error("Gemini lieferte eine leere Antwort.");
  const patch = parsePatch(text);
  return { patch, raw: text };
}

const VALID_TERMS = new Set(REWARD_TERMS.map((t) => t.id));

export function parsePatch(text: string): GeminiPatch {
  let cleaned = text.trim();
  const fence = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fence) cleaned = fence[1].trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start >= 0 && end > start) cleaned = cleaned.slice(start, end + 1);
  let obj: any;
  try {
    obj = JSON.parse(cleaned);
  } catch {
    throw new Error("Antwort war kein gültiges JSON.");
  }
  const patch: GeminiPatch = {};
  if (obj.reward && typeof obj.reward === "object") {
    const reward: GeminiPatch["reward"] = {};
    for (const [k, v] of Object.entries(obj.reward)) {
      if (!VALID_TERMS.has(k) || typeof v !== "object" || v === null) continue;
      const tv = v as any;
      const entry: any = {};
      if (typeof tv.enabled === "boolean") entry.enabled = tv.enabled;
      if (Number.isFinite(tv.weight)) entry.weight = clampNum(tv.weight, -5, 20);
      if (Number.isFinite(tv.param)) entry.param = tv.param;
      if (Object.keys(entry).length) reward[k] = entry;
    }
    if (Object.keys(reward).length) patch.reward = reward;
  }
  if (typeof obj.pointMode === "boolean") patch.pointMode = obj.pointMode;
  if (obj.point && typeof obj.point === "object") {
    const p: NonNullable<GeminiPatch["point"]> = {};
    if (obj.point.mode === "aus" || obj.point.mode === "frei"
      || obj.point.mode === "umkreis" || obj.point.mode === "pfad") {
      p.mode = obj.point.mode;
    }
    if (Number.isFinite(obj.point.radius)) p.radius = clampNum(obj.point.radius, 0.3, 6);
    if (typeof obj.point.speedByDist === "boolean") p.speedByDist = obj.point.speedByDist;
    if (Number.isFinite(obj.point.maxSpeed)) p.maxSpeed = clampNum(obj.point.maxSpeed, 0.05, 1.2);
    if (Number.isFinite(obj.point.fullDist)) p.fullDist = clampNum(obj.point.fullDist, 0.3, 6);
    if (Object.keys(p).length) patch.point = p;
  }
  if (obj.world && typeof obj.world === "object") {
    const w: any = {};
    if (typeof obj.world.enabled === "boolean") w.enabled = obj.world.enabled;
    if (Number.isFinite(obj.world.difficulty)) w.difficulty = clampNum(obj.world.difficulty, 0, 1);
    if (Number.isFinite(obj.world.density)) w.density = clampNum(obj.world.density, 0, 1);
    if (obj.world.features && typeof obj.world.features === "object") {
      const feats: any = {};
      for (const f of ["treppen", "huegel", "loecher", "hindernisse", "stange"]) {
        if (typeof obj.world.features[f] === "boolean") feats[f] = obj.world.features[f];
      }
      if (Object.keys(feats).length) w.features = feats;
    }
    if (Object.keys(w).length) patch.world = w;
  }
  if ([1, 4, 16, 32, 64].includes(obj.turbo)) patch.turbo = obj.turbo;
  if (typeof obj.resetFirst === "boolean") patch.resetFirst = obj.resetFirst;
  if (typeof obj.explanation === "string") patch.explanation = obj.explanation;
  return patch;
}

function clampNum(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

// ── Persistenz (API-Key + Modellwahl) ────────────────────────────────────────

export function loadGeminiPrefs(): { apiKey: string; model: GeminiModelId } {
  try {
    return {
      apiKey: localStorage.getItem("mdt_v2_gemini_key") ?? DEFAULT_GEMINI_KEY,
      model: (localStorage.getItem("mdt_v2_gemini_model") as GeminiModelId) ?? "gemini-3.5-flash-lite",
    };
  } catch {
    return { apiKey: DEFAULT_GEMINI_KEY, model: "gemini-3.5-flash-lite" };
  }
}

export function saveGeminiPrefs(apiKey: string, model: GeminiModelId): void {
  try {
    localStorage.setItem("mdt_v2_gemini_key", apiKey);
    localStorage.setItem("mdt_v2_gemini_model", model);
  } catch { /* ignore */ }
}
