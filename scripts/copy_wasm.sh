#!/usr/bin/env bash
# MicroDuck Trainer v2.0 – kopiert die WASM-Runtimes statisch nach public/wasm/.
# Static Export (APK) lädt sie ausschließlich via /wasm/... URLs (keine Bundler-?url-Imports).
set -e
cd "$(dirname "$0")/.."
mkdir -p public/wasm

# MuJoCo WASM + ESM-JS-Glue (@mujoco/mujoco ist ein ES-Modul mit `export default loadMujoco`)
cp -f node_modules/@mujoco/mujoco/mujoco.wasm public/wasm/mujoco.wasm
cp -f node_modules/@mujoco/mujoco/mujoco.js  public/wasm/mujoco.js

# onnxruntime-web: simd-threaded Binary (executionProvider 'wasm', numThreads=1)
cp -f node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm public/wasm/ort-wasm-simd-threaded.wasm

echo "[copy_wasm] OK -> public/wasm/: $(ls public/wasm | tr '\n' ' ')"
