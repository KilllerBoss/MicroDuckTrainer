'use client'

import dynamic from "next/dynamic";

// Client-only: MuJoCo-WASM + three.js dürfen nie SSR laufen (Hydration-Fehler).
const TrainerApp = dynamic(() => import("@/components/md/TrainerApp"), {
  ssr: false,
  loading: () => (
    <div className="fixed inset-0 flex items-center justify-center bg-[#05060a]">
      <p className="font-mono text-sm text-cyan-300">MicroDuck Trainer startet…</p>
    </div>
  ),
});

export default function Home() {
  return <TrainerApp />;
}
