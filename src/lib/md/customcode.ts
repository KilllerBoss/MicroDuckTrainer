// ── MicroDuck Trainer v2.2 – KI-geschriebener Reward-Code (Sandbox) ──────────
// Gemini (oder der Nutzer) schreibt einen JS-Funktionskörper, der pro Policy-
// Schritt einen Reward-Beitrag liefert. Kompiliert via new Function, gecacht
// pro Code-String, Laufzeitfehler → 0 (Term wird nicht deaktiviert, damit ein
// einmaliger Fußehler nicht das Training abwürgt – Fehler stehen in der Konsole).

export type CustomRewardFn = (api: Record<string, unknown>) => number;

const cache = new Map<string, CustomRewardFn | null>();
const warned = new WeakSet<CustomRewardFn>();

const FORBIDDEN = /\b(import|require|eval|Function|fetch|XMLHttpRequest|localStorage|sessionStorage|indexedDB|document|window|globalThis|self|postMessage|Worker|WebSocket)\b/;

/** Kompiliert (und cacht) den Reward-Code. null = unbrauchbar. */
export function getCustomFn(code: string): CustomRewardFn | null {
  if (!code || !code.trim()) return null;
  if (cache.has(code)) return cache.get(code) ?? null;
  let fn: CustomRewardFn | null = null;
  try {
    if (!FORBIDDEN.test(code)) {
      const compiled = new Function("api", `"use strict";\n${code}\n`) as CustomRewardFn;
      // Kaltstart-Check mit Dummy-API
      const probe = compiled(dummyApi());
      if (typeof probe === "number" && Number.isFinite(probe)) fn = compiled;
    }
  } catch {
    fn = null;
  }
  cache.set(code, fn);
  return fn;
}

/** Wertet den Term aus; Fehler → 0 + einmalige Konsole-Warnung. */
export function evalCustomFn(fn: CustomRewardFn, api: Record<string, unknown>): number {
  try {
    const v = fn(api);
    return Number.isFinite(v) ? v : 0;
  } catch (err) {
    if (!warned.has(fn)) {
      warned.add(fn);
      console.warn("[custom-reward] Laufzeitfehler (Term liefert 0):", err);
    }
    return 0;
  }
}

/** Prüft Fremdcode VOR dem Übernehmen (Gemini-Antwort). Liefert Fehlermeldung. */
export function validateCustomCode(code: string): string | null {
  if (!code || !code.trim()) return "Code ist leer.";
  if (FORBIDDEN.test(code)) {
    return "Code enthält verbotene Aufrufe (z. B. fetch/import/DOM).";
  }
  const fn = getCustomFn(code);
  if (!fn) return "Code kompiliert nicht oder liefert keine Zahl.";
  return null;
}

/** Dummy-API für Kaltstart-Check und Gemini-Antwort-Validierung. */
export function dummyApi(): Record<string, unknown> {
  const n = 8;
  return {
    h: 0.7, height: 0.7, upZ: 1, gz: -1,
    vx: 0, vy: 0, vz: 0, omega: 0,
    angles: new Float32Array(n),
    act: new Float32Array(n),
    prevAct: new Float32Array(n),
    qpos: new Float32Array(16),
    qvel: new Float32Array(16),
    qacc: new Float32Array(16),
    torso: [0, 0, 0.7],
    target: null,
    cmd: new Float32Array([0, 0, 0]),
    imitDelta: null,
    imitTarget: null,
    dt: 0.02, t: 0, step: 0,
  };
}

export function clearCustomFnCache(): void {
  cache.clear();
}
