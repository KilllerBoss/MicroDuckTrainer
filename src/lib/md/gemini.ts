// ── MicroDuck Trainer v2.2 – Gemini-Trainingsassistent ───────────────────────
// Zwei Modi:
//  1) REGELN: Reward-Terme, Punkt-Modus, Welt, Turbo anpassen (kein Code).
//  2) CODE-EXPERTE: Gemini schreibt selbst JavaScript-Trainings-Code und
//     entscheidet Rundenlänge + Ziel-Generationen + Turbo + Reset.
// Modelle: gemini-robotics-er-2-preview · gemini-3.5-flash-lite · gemini-3.8-flash
// Läuft direkt im Client (generativelanguage.googleapis.com erlaubt CORS).

import type { ModelId } from "./models";
import { getModel } from "./models";
import { REWARD_TERMS, type RewardConfig } from "./rewards";
import type { WorldFeatures } from "./worldgen";
import { validateCustomCode } from "./customcode";

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
  // ── v2.3: Profi-Tricks (von Gemini steuerbar) ──
  runCfg?: {
    cmdTrain?: boolean;
    cmdFwd?: number;
    cmdLat?: number;
    cmdAng?: number;
    curriculum?: boolean;
    actionSmooth?: number;
    pushes?: boolean;
    noiseReset?: boolean;
    fitnessMode?: "sum" | "mean";
    weightDecay?: number;
  };
  // ── v2.2: Code-Experte ──
  /** Reward-Code (JS-Funktionskörper, `return` einer Zahl). */
  code?: string;
  codeName?: string;
  codeWeight?: number;
  /** Trainings-Einstellungen, die Gemini entscheiden darf. */
  training?: {
    rolloutSteps?: number;
    generations?: number;
    lr?: number;
    sigma?: number;
    turbo?: number;
    resetFirst?: boolean;
  };
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

// ── API-Dokumentation für den Code-Modus ──
const CODE_API_DOC = `Das Objekt "api" (jeder Aufruf = 1 Policy-Schritt, 50 Hz):
  api.h / api.height : Körperhöhe in Meter (Zahl)
  api.upZ            : 0..1, wie aufrecht der Roboter ist (1 = perfekt)
  api.gz             : -1..1, Schwerkraft-Projektion (-1 = aufrecht)
  api.vx             : Vorwärtsgeschwindigkeit m/s (Roboter-Sicht)
  api.vy             : Seitwärts-Geschwindigkeit m/s
  api.vz             : Vertikal-Geschwindigkeit m/s (positiv = steigt! für Sprünge)
  api.omega          : Gierrate rad/s
  api.angles         : Float32Array Gelenkwinkel (rad, ein Wert pro Gelenk)
  api.act / api.prevAct : Float32Array Aktionen -1..1 (aktuell / vorheriger Schritt)
  api.qpos, api.qvel, api.qacc : Float32Array volle MuJoCo-Zustände (api.qvel[2] = vz)
  api.torso          : [x, y, z] Position in der Welt (Meter)
  api.target         : [x, y] Joystick-Punkt in der Welt oder null
  api.cmd            : Float32Array [vx, vy, yaw] Kommandos
  api.imitDelta      : number | null – Root-Höhen-Delta der GLB-Animation
  api.imitTarget     : Float32Array | null – Ziel-Gelenkwinkel der GLB-Animation
  api.dt             : Sekunden pro Schritt (0.02)
  api.t              : Zeit im Rollout (Sekunden)
  api.step           : Schritt-Index im Rollout (0, 1, 2, ...)`;

function commonContext(ctx: GeminiContext): string {
  const meta = getModel(ctx.modelId);
  const cur = Object.entries(ctx.reward.terms)
    .filter(([, v]) => v.enabled)
    .map(([k, v]) => `${k}(w=${v.weight.toFixed(2)}${v.param ? `,p=${v.param.toFixed(2)}` : ""})`)
    .join(", ");
  return `Roboter: ${meta.label} mit ${meta.actionDim} Gelenken (Positionsaktuatoren, Policy 50 Hz).
Aktive Reward-Terme: ${cur || "keine"}
Training: ${ctx.generation} Generationen bisher. Imitation einer GLB-Animation aktiv: ${ctx.imitationActive ? `ja (${ctx.imitationName})` : "nein"}.
Random-Welt (Treppen/Hügel/Löcher/Hindernisse/Balancierstange) aktiv: ${ctx.worldEnabled ? "ja" : "nein"}.
Punkt-Modus (Joystick bewegt einen 3D-Punkt, kamera-relativ): aktuell "${ctx.pointMode}" mit Radius ${ctx.pointRadius.toFixed(1)} m.
  - "aus": kein Punkt (klassische Joystick-Steuerung)
  - "frei": Punkt frei in der Arena, Roboter verfolgt ihn
  - "umkreis": Punkt bleibt im Radius um den Roboter
  - "pfad": Roboter folgt einem physikalisch berechneten Pfad mit Schwung (Momentum) zum Punkt`;
}

function buildPrompt(ctx: GeminiContext, goal: string): string {
  return `Du bist der Trainingsregeln-Assistent einer Roboter-Simulation (MuJoCo + Evolution-Strategy-Training im Browser).
${commonContext(ctx)}

VERFÜGBARE REWARD-TERME:
${TERM_DOCS}

ZIEL DES NUTZERS: "${goal}"

Aufgabe: Passe die Trainingsregeln an, damit der Roboter nach dem Training das Gewünschte wirklich kann.
Wähle Gewichte konservativ, aber wirksam; deaktiviere Terme, die dem Ziel widersprechen.
Wenn das Ziel Bewegung über Zeit braucht (Springen, Tanzen, Aufstehen) und keine Imitation aktiv ist, erkläre es und setze trotzdem sinnvolle Terme.
Wenn Welt/Punkt-Modus helfen (z. B. Treppen → Welt mit Treppen, Ziel verfolgen → point.mode "frei" oder "pfad" + Reward-Term "pointChase"), setze sie.
turbo: 1=stabil, 16=schnell, 64=maximal (nur bei einfachen Zielen hoch setzen). resetFirst=true bei grundlegend neuem Bewegungsmuster.
PROFI-TRICKS (runCfg, optional): cmdTrain (bool, Zufalls-Tempo je Runde → Joystick-Fähigkeit), cmdFwd/cmdLat/cmdAng (Ziel-Tempo m/s bzw. rad/s), curriculum (bool, Tempo bei Stürzen automatisch reduzieren), actionSmooth (0.3=sehr glatt .. 1=aus; gegen Zittern), pushes (bool, Zufalls-Stöße für Robustheit), noiseReset (bool, varied Starts), fitnessMode ("sum"=Überleben zählt, empfohlen), weightDecay (0..0.05).

Antworte AUSSCHLIESSLICH mit JSON (kein Markdown) nach diesem Schema:
{"reward":{"<termId>":{"enabled":bool,"weight":number,"param":number}},"point":{"mode":"frei","radius":1.5,"speedByDist":true,"maxSpeed":0.25,"fullDist":1.5},"world":{"enabled":bool,"difficulty":number,"density":number,"features":{"treppen":bool,"huegel":bool,"loecher":bool,"hindernisse":bool,"stange":bool}},"turbo":1,"resetFirst":bool,"explanation":"max. 4 Sätze, Deutsch, warum diese Regeln zum Ziel führen"}`;
}

async function callGemini(apiKey: string, model: string, prompt: string, maxTokens: number): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0.4,
        responseMimeType: "application/json",
        maxOutputTokens: maxTokens,
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
  return text;
}

/** Modus 1: Trainingsregeln anpassen (kein Code). */
export async function applyGoalWithGemini(
  apiKey: string,
  model: GeminiModelId,
  ctx: GeminiContext,
  goal: string,
): Promise<{ patch: GeminiPatch; raw: string }> {
  const text = await callGemini(apiKey, model, buildPrompt(ctx, goal), 2048);
  return { patch: parsePatch(text), raw: text };
}

function buildCodePrompt(ctx: GeminiContext, goal: string, currentCode: string | null): string {
  return `Du bist der Trainings-Code-Autor einer Roboter-Simulation (MuJoCo + Evolution-Strategy im Browser). Du schreibst selbst JavaScript-Trainingscode und entscheidest über alles: Reward-Code, Rundenlänge (rolloutSteps), Anzahl Generationen (generations), Turbo und ob das Training bei Null starten muss.
${commonContext(ctx)}
${currentCode ? `AKTUELL AKTIVER CODE (vom vorherigen Lauf, kann verbessert werden):

\`\`\`${currentCode}
\`\`\`

` : ""}REWARD-CODE-API (dein Code erhält pro Schritt ein Objekt "api"):
${CODE_API_DOC}

REGELN FÜR DEN CODE:
- Der Code ist ein Funktionskörper; er MUSS mit "return" eine einzelne Zahl zurückgeben (Reward-Beitrag; typisch 0..3, negative Werte = Strafe).
- Maximal ~40 Zeilen, nur reines JavaScript (Math, if, for). VERBOTEN: import, fetch, eval, DOM, window, document, localStorage, while(true).
- Der Code wird pro Physik-Schritt (50 Hz) ausgeführt – halte ihn billig.
- Beispiele:
  Springen unterstützen:  let bonus = Math.max(0, api.vz) * 3; bonus += Math.max(0, api.h - 0.35) * 8; return bonus;
  Zum Punkt eilen:       if (!api.target) return 0; const dx = api.torso[0]-api.target[0], dy = api.torso[1]-api.target[1]; return Math.exp(-(dx*dx+dy*dy)/0.5) * 2;
  Ruhig stehen:          return api.upZ * 1.5 - Math.abs(api.vx) * 2 - Math.abs(api.omega) * 0.5;

ENTSCHEIDUNGEN (du entscheidest alles):
- rolloutSteps: Rundenlänge in Policy-Schritten (50 = kurz, 200 = Standard, 600 = sehr lang; für Sprünge 300-400).
- generations: Ziel-Generationen, nach denen das Training automatisch stoppt (einfach 300-800, schwer 1500-3000; 0 = unbegrenzt).
- turbo: 1=stabil, 4, 16=schnell, 32, 64=maximal.
- resetFirst: true bei grundlegend neuem Bewegungsmuster (z. B. Gehen → Springen).
- weight: Gewichtung des Code-Terms (1 = normal, 2-5 = dominanter, negativ = Strafe).

ZIEL DES NUTZERS: "${goal}"

Antworte AUSSCHLIESSLICH mit JSON (kein Markdown, Code als EIN JSON-String mit \\n-Zeilenumbrüchen):
{"name":"kurzer deutscher Name","weight":2.0,"code":"let bonus = ...; return bonus;","training":{"rolloutSteps":300,"generations":800,"turbo":16,"lr":0.03,"sigma":0.08,"resetFirst":true},"reward":{"<termId>":{"enabled":bool,"weight":number,"param":number}},"point":{"mode":"frei"},"world":{"enabled":bool},"explanation":"max. 4 Sätze Deutsch: was dein Code belohnt und warum die Runden-Einstellungen passen"}`;
}

/** Modus 2: Gemini schreibt den Trainings-Code selbst (v2.2). */
export async function applyCodeWithGemini(
  apiKey: string,
  model: GeminiModelId,
  ctx: GeminiContext,
  goal: string,
  currentCode: string | null,
): Promise<{ patch: GeminiPatch; raw: string }> {
  const text = await callGemini(apiKey, model, buildCodePrompt(ctx, goal, currentCode), 8192);
  const patch = parsePatch(text);
  if (!patch.code) {
    throw new Error("Gemini hat keinen Code geliefert – nochmal versuchen oder Regeln-Modus nutzen.");
  }
  const err = validateCustomCode(patch.code);
  if (err) {
    throw new Error(`Gemini-Code unbrauchbar: ${err} – erneut versuchen oder Modell wechseln.`);
  }
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
  // ── v2.3: Profi-Tricks ──
  if (obj.runCfg && typeof obj.runCfg === "object") {
    const r: NonNullable<GeminiPatch["runCfg"]> = {};
    if (typeof obj.runCfg.cmdTrain === "boolean") r.cmdTrain = obj.runCfg.cmdTrain;
    if (Number.isFinite(obj.runCfg.cmdFwd)) r.cmdFwd = clampNum(obj.runCfg.cmdFwd, 0.05, 1.2);
    if (Number.isFinite(obj.runCfg.cmdLat)) r.cmdLat = clampNum(obj.runCfg.cmdLat, 0, 0.6);
    if (Number.isFinite(obj.runCfg.cmdAng)) r.cmdAng = clampNum(obj.runCfg.cmdAng, 0, 2);
    if (typeof obj.runCfg.curriculum === "boolean") r.curriculum = obj.runCfg.curriculum;
    if (Number.isFinite(obj.runCfg.actionSmooth)) r.actionSmooth = clampNum(obj.runCfg.actionSmooth, 0.3, 1);
    if (typeof obj.runCfg.pushes === "boolean") r.pushes = obj.runCfg.pushes;
    if (typeof obj.runCfg.noiseReset === "boolean") r.noiseReset = obj.runCfg.noiseReset;
    if (obj.runCfg.fitnessMode === "sum" || obj.runCfg.fitnessMode === "mean") {
      r.fitnessMode = obj.runCfg.fitnessMode;
    }
    if (Number.isFinite(obj.runCfg.weightDecay)) r.weightDecay = clampNum(obj.runCfg.weightDecay, 0, 0.05);
    if (Object.keys(r).length) patch.runCfg = r;
  }
  if (typeof obj.explanation === "string") patch.explanation = obj.explanation;
  // ── v2.2: Code-Experte ──
  if (typeof obj.code === "string" && obj.code.trim()) {
    patch.code = obj.code.replace(/\r/g, "").slice(0, 4000);
  }
  if (typeof obj.name === "string" && obj.name.trim()) patch.codeName = obj.name.trim().slice(0, 40);
  if (Number.isFinite(obj.weight)) patch.codeWeight = clampNum(obj.weight, -20, 20);
  if (obj.training && typeof obj.training === "object") {
    const tr: NonNullable<GeminiPatch["training"]> = {};
    if (Number.isFinite(obj.training.rolloutSteps)) tr.rolloutSteps = clampNum(Math.round(obj.training.rolloutSteps), 30, 1000);
    if (Number.isFinite(obj.training.generations)) tr.generations = clampNum(Math.round(obj.training.generations), 0, 100000);
    if (Number.isFinite(obj.training.lr)) tr.lr = clampNum(obj.training.lr, 0.002, 0.2);
    if (Number.isFinite(obj.training.sigma)) tr.sigma = clampNum(obj.training.sigma, 0.005, 0.3);
    if ([1, 4, 16, 32, 64].includes(obj.training.turbo)) tr.turbo = obj.training.turbo;
    if (typeof obj.training.resetFirst === "boolean") tr.resetFirst = obj.training.resetFirst;
    if (Object.keys(tr).length) patch.training = tr;
  }
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
