#!/usr/bin/env python3
"""Merge-Conflicts auflösen: Standard = HEAD-Seite (v2.5), Ausnahmen per Regel."""
import re, sys

def resolve(path, keep="HEAD"):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    pattern = re.compile(r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [0-9a-f]+\n", re.DOTALL)
    n = [0]
    def repl(m):
        n[0] += 1
        return m.group(1) if keep == "HEAD" else m.group(2)
    out = pattern.sub(repl, src)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"{path}: {n[0]} Konflikte -> {keep}")

for p in [
    "src/lib/md/es.ts",
    "public/workers/es-worker.js",
    "src/components/md/GeminiPanel.tsx",
    "src/components/md/TrainingPanel.tsx",
    "src/components/md/WorldPanel.tsx",
    "src/lib/md/app-core.ts",
    "src/lib/md/rig.ts",
]:
    resolve(p, "HEAD")
