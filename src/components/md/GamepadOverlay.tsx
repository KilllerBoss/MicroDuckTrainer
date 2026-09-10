"use client";

// ── MicroDuck Trainer v2.0 – Touch-Gamepad-Overlay ───────────────────────────
// LINKS: Joystick (~130 px, Deadzone 0.12, Pointer-Events, Rückstellung)
// RECHTS: 4 Buttons A(grün)/B(rot)/C(blau)/D(gelb) in Raute, Multi-Touch.
// Halbtransparent, touch-action: none, für Android-WebView (APK) optimiert.

import { useCallback, useRef, useState } from "react";
import type { SourceId } from "@/lib/md/mapping";

const JOY_SIZE = 130;
const KNOB_SIZE = 54;
const DEADZONE = 0.12;

interface GamepadOverlayProps {
  visible: boolean;
  onJoystick: (x: number, y: number) => void;
  onButton: (source: SourceId, pressed: boolean) => void;
}

const BUTTONS: { id: SourceId; label: string; color: string; ring: string; pos: string }[] = [
  { id: "A", label: "A", color: "bg-emerald-500/70", ring: "ring-emerald-300/60", pos: "left-1/2 top-0 -translate-x-1/2" },
  { id: "B", label: "B", color: "bg-red-500/70", ring: "ring-red-300/60", pos: "right-0 top-1/2 -translate-y-1/2" },
  { id: "C", label: "C", color: "bg-sky-500/70", ring: "ring-sky-300/60", pos: "left-1/2 bottom-0 -translate-x-1/2" },
  { id: "D", label: "D", color: "bg-amber-400/70", ring: "ring-amber-200/60", pos: "left-0 top-1/2 -translate-y-1/2" },
];

export default function GamepadOverlay({ visible, onJoystick, onButton }: GamepadOverlayProps) {
  const [knob, setKnob] = useState({ x: 0, y: 0 });
  const [active, setActive] = useState(false);
  const joyPointer = useRef<number | null>(null);
  const baseRef = useRef<HTMLDivElement | null>(null);

  const handleJoyMove = useCallback(
    (e: React.PointerEvent) => {
      if (joyPointer.current !== e.pointerId || !baseRef.current) return;
      const rect = baseRef.current.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const radius = rect.width / 2 - KNOB_SIZE / 2;
      let dx = e.clientX - cx;
      let dy = e.clientY - cy;
      const len = Math.hypot(dx, dy);
      if (len > radius) {
        dx = (dx / len) * radius;
        dy = (dy / len) * radius;
      }
      setKnob({ x: dx, y: dy });
      let nx = dx / radius;
      let ny = -dy / radius; // oben = +1
      const mag = Math.hypot(nx, ny);
      if (mag < DEADZONE) {
        nx = 0;
        ny = 0;
      }
      onJoystick(nx, ny);
    },
    [onJoystick],
  );

  const releaseJoy = useCallback(
    (e: React.PointerEvent) => {
      if (joyPointer.current !== e.pointerId) return;
      joyPointer.current = null;
      setActive(false);
      setKnob({ x: 0, y: 0 });
      onJoystick(0, 0);
    },
    [onJoystick],
  );

  if (!visible) return null;

  return (
    <div className="pointer-events-none absolute inset-0 z-30" style={{ touchAction: "none" }}>
      {/* Joystick unten links */}
      <div
        className="pointer-events-auto absolute bottom-5 left-4"
        style={{ width: JOY_SIZE, height: JOY_SIZE, paddingBottom: "env(safe-area-inset-bottom)" }}
      >
        <div
          ref={baseRef}
          role="application"
          aria-label="Bewegungs-Joystick"
          className={`relative h-full w-full rounded-full border-2 backdrop-blur-sm transition-colors ${
            active ? "border-cyan-300/70 bg-cyan-500/10" : "border-cyan-400/30 bg-black/30"
          }`}
          style={{ touchAction: "none" }}
          onPointerDown={(e) => {
            joyPointer.current = e.pointerId;
            (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
            setActive(true);
            handleJoyMove(e);
          }}
          onPointerMove={handleJoyMove}
          onPointerUp={releaseJoy}
          onPointerCancel={releaseJoy}
        >
          {/* Kreuz-Markierung */}
          <div className="absolute left-1/2 top-2 h-2 w-px -translate-x-1/2 bg-cyan-400/40" />
          <div className="absolute left-1/2 bottom-2 h-2 w-px -translate-x-1/2 bg-cyan-400/40" />
          <div className="absolute top-1/2 left-2 w-2 h-px -translate-y-1/2 bg-cyan-400/40" />
          <div className="absolute top-1/2 right-2 w-2 h-px -translate-y-1/2 bg-cyan-400/40" />
          <div
            className="absolute left-1/2 top-1/2 rounded-full border border-cyan-200/50 bg-cyan-400/30 shadow-[0_0_12px_rgba(34,211,238,0.45)]"
            style={{
              width: KNOB_SIZE,
              height: KNOB_SIZE,
              transform: `translate(calc(-50% + ${knob.x}px), calc(-50% + ${knob.y}px))`,
              transition: active ? "none" : "transform 0.15s ease-out",
            }}
          />
        </div>
      </div>

      {/* Buttons unten rechts (Raute) */}
      <div
        className="pointer-events-auto absolute right-4 bottom-5 h-[150px] w-[150px]"
        style={{ touchAction: "none", paddingBottom: "env(safe-area-inset-bottom)" }}
      >
        {BUTTONS.map((b) => (
          <button
            key={b.id}
            type="button"
            aria-label={`Taste ${b.label}`}
            className={`absolute ${b.pos} flex h-[52px] w-[52px] items-center justify-center rounded-full font-mono text-lg font-bold text-white/95 ring-2 backdrop-blur-sm transition-transform active:scale-90 ${b.color} ${b.ring}`}
            style={{ touchAction: "none" }}
            onPointerDown={(e) => {
              (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
              onButton(b.id, true);
            }}
            onPointerUp={() => onButton(b.id, false)}
            onPointerCancel={() => onButton(b.id, false)}
            onContextMenu={(e) => e.preventDefault()}
          >
            {b.label}
          </button>
        ))}
      </div>
    </div>
  );
}
