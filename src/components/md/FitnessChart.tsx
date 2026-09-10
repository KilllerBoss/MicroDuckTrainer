"use client";

// ── MicroDuck Trainer v2.0 – Fitness-Chart (Canvas, ohne Lib) ────────────────

import { useEffect, useRef } from "react";

interface FitnessChartProps {
  history: number[];
  height?: number;
}

export default function FitnessChart({ history, height = 120 }: FitnessChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 300;
    const h = height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Rahmen
    ctx.strokeStyle = "rgba(34,211,238,0.25)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    if (history.length < 2) {
      ctx.fillStyle = "rgba(148,163,184,0.7)";
      ctx.font = "11px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText("Noch keine Generationen – Fitnessverlauf erscheint hier", w / 2, h / 2);
      return;
    }

    const pad = 6;
    let min = Infinity;
    let max = -Infinity;
    for (const v of history) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (max - min < 1e-6) {
      max += 0.5;
      min -= 0.5;
    }
    const span = max - min;
    const px = (i: number) => pad + (i / (history.length - 1)) * (w - 2 * pad);
    const py = (v: number) => h - pad - ((v - min) / span) * (h - 2 * pad);

    // Nulllinie (falls im Bereich)
    if (min < 0 && max > 0) {
      ctx.strokeStyle = "rgba(148,163,184,0.3)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(pad, py(0));
      ctx.lineTo(w - pad, py(0));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Fläche
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(34,211,238,0.35)");
    grad.addColorStop(1, "rgba(34,211,238,0.02)");
    ctx.beginPath();
    ctx.moveTo(px(0), py(history[0]));
    for (let i = 1; i < history.length; i++) ctx.lineTo(px(i), py(history[i]));
    ctx.lineTo(px(history.length - 1), h - pad);
    ctx.lineTo(px(0), h - pad);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Linie
    ctx.beginPath();
    ctx.moveTo(px(0), py(history[0]));
    for (let i = 1; i < history.length; i++) ctx.lineTo(px(i), py(history[i]));
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // Letzter Punkt
    const lastX = px(history.length - 1);
    const lastY = py(history[history.length - 1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#a5f3fc";
    ctx.fill();

    // Min/Max-Labels
    ctx.fillStyle = "rgba(148,163,184,0.85)";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillText(max.toFixed(2), 4, 11);
    ctx.fillText(min.toFixed(2), 4, h - 4);
  }, [history, height]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-md bg-black/40"
      style={{ height }}
      aria-label="Fitness-Verlauf"
    />
  );
}
