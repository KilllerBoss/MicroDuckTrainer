// ── MicroDuck Trainer v2.2 – Datei-Upload/Download (APK-Bridge + Browser) ────
// In der APK schreibt window.MdtBridge.saveFile(name, base64, mime) direkt in
// den Downloads-Ordner (MediaStore). Im Browser fällt es auf <a download>
// zurück. Uploads laufen über <input type="file"> – in der APK öffnet der
// onShowFileChooser-Callback der Shell den Android-Dateimanager.

interface MdtAndroidBridge {
  saveFile(name: string, base64: string, mime: string): void;
}

export interface FileSavedEvent {
  ok: boolean;
  name: string;
  info: string;
}

export function hasAndroidBridge(): boolean {
  try {
    return typeof window !== "undefined" && !!(window as any).MdtBridge?.saveFile;
  } catch {
    return false;
  }
}

function toBase64(bytes: Uint8Array): string {
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

/**
 * Speichert eine Datei auf dem Gerät. Liefert den Weg zurück:
 * - "bridge": APK → Downloads-Ordner (Ergebnis kommt als 'mdt-file-saved'-Event)
 * - "browser": klassischer Download in den Browser-Download-Ordner
 */
export async function saveToDevice(
  name: string,
  content: string | Blob,
  mime = "application/octet-stream",
): Promise<"bridge" | "browser"> {
  const safe = name.replace(/[\\/:*?"<>|]/g, "_").trim() || "download.bin";

  let base64 = "";
  if (typeof content === "string") {
    base64 = toBase64(new TextEncoder().encode(content));
  } else {
    const buf = new Uint8Array(await content.arrayBuffer());
    base64 = toBase64(buf);
  }

  const bridge = (typeof window !== "undefined" ? (window as any).MdtBridge : null) as
    | MdtAndroidBridge
    | null;
  if (bridge?.saveFile) {
    bridge.saveFile(safe, base64, mime);
    return "bridge";
  }

  // Browser-Fallback
  const blob = typeof content === "string"
    ? new Blob([content], { type: mime })
    : content;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = safe;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  return "browser";
}

/**
 * Öffnet die Dateiauswahl (in der APK: Android-Dateimanager) und liefert die
 * gewählte Datei. Auf Mobilgeräten MUSS der Aufruf direkt aus einem
 * User-Gesture heraus erfolgen.
 */
export function openFilePicker(
  accept = "",
): Promise<File | null> {
  return new Promise((resolve) => {
    if (typeof document === "undefined") return resolve(null);
    const input = document.createElement("input");
    input.type = "file";
    if (accept) input.accept = accept;
    input.style.display = "none";
    let settled = false;
    const done = (f: File | null) => {
      if (settled) return;
      settled = true;
      input.remove();
      resolve(f);
    };
    input.addEventListener("change", () => {
      done(input.files && input.files.length > 0 ? input.files[0] : null);
    });
    // Abbruch (Dialog geschlossen) liefert kein change-Event → Focus-Heuristik
    window.addEventListener("focus", () => setTimeout(() => {
      if (!settled && (!input.files || input.files.length === 0)) {
        // kurz warten, falls change noch feuert
        setTimeout(() => {
          if (!settled) {
            done(input.files && input.files.length > 0 ? input.files[0] : null);
          }
        }, 800);
      }
    }, 300), { once: true });
    document.body.appendChild(input);
    input.click();
  });
}

/** Globaler Listener für APK-Rückmeldungen (Toast in der UI). */
export function initFileSavedListener(cb: (ev: FileSavedEvent) => void): () => void {
  if (typeof window === "undefined") return () => {};
  const handler = (e: Event) => {
    const msg = e as MessageEvent;
    if (msg.data && typeof msg.data === "object") {
      cb(msg.data as FileSavedEvent);
    }
  };
  window.addEventListener("mdt-file-saved", handler);
  return () => window.removeEventListener("mdt-file-saved", handler);
}
