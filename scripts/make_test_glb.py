#!/usr/bin/env python3
"""Erzeugt eine minimale Test-GLB (Mixamo-artiges Skelett, Jump-Animation)
für die Imitations-QA: Hips + Left/RightUpLeg + Left/RightLeg (+Foot).
Ablauf 1.2 s: Hocken → Sprung → Landen."""
import json
import struct
import os

OUT = "/home/z/my-project/public/test/jump.glb"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── Keyframes ────────────────────────────────────────────────────────────────
times = [0.0, 0.3, 0.45, 0.7, 0.9, 1.2]
# Hips Y-Position (crouch → jump → land)
hips_y = [1.0, 0.78, 0.85, 1.35, 1.05, 1.0]
# Knie-Beugung (x-Rotation): hocken negativ, gestreckt 0
knee_rx = [0.0, -0.9, -0.5, 0.1, -0.2, 0.0]
hip_rx = [0.0, 0.5, 0.3, -0.1, 0.1, 0.0]

def quat(rx):
    # Quaternion um X: (x, y, z, w) – glTF-Order
    return [__import__("math").sin(rx / 2), 0.0, 0.0, __import__("math").cos(rx / 2)]

# ── Binärdaten ───────────────────────────────────────────────────────────────
bin_data = bytearray()
accessors = []
buffer_views = []

def add_view(data: bytes, target=None):
    offset = len(bin_data)
    while offset % 4:
        bin_data.append(0)
        offset = len(bin_data)
    bin_data.extend(data)
    while len(bin_data) % 4:
        bin_data.append(0)
    bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
    if target is not None:
        bv["target"] = target
    buffer_views.append(bv)
    return len(buffer_views) - 1

def add_accessor(data: bytes, comp_type, count, comp_count, target=None):
    bv = add_view(data, target)
    accessors.append({
        "bufferView": bv, "componentType": 5126, "count": count,
        "type": comp_type,
        "min": [0.0] * comp_count, "max": [0.0] * comp_count,
    })
    return len(accessors) - 1

import math
# times
t_data = struct.pack(f"<{len(times)}f", *times)
t_acc = add_accessor(t_data, "SCALAR", len(times), 1, 34962)

# Hips position (vec3) + quaternion (identity)
p_data = struct.pack(f"<{len(hips_y)*3}f", *sum(([0.0, y, 0.0] for y in hips_y), []))
p_acc = add_accessor(p_data, "VEC3", len(hips_y), 3, 34962)
h_q = [0.0, 0.0, 0.0, 1.0] * len(times)
h_acc = add_accessor(struct.pack(f"<{len(h_q)}f", *h_q), "VEC4", len(times), 4, 34962)

# Knie/Beine quaternionen
knee_acc, hip_acc = [], []
for side in ("Left", "Right"):
    q = sum((quat(a) for a in knee_rx), [])
    knee_acc.append(add_accessor(struct.pack(f"<{len(q)}f", *q), "VEC4", len(times), 4, 34962))
    q = sum((quat(a) for a in hip_rx), [])
    hip_acc.append(add_accessor(struct.pack(f"<{len(q)}f", *q), "VEC4", len(times), 4, 34962))

# Füße: Identity
f_acc = []
q = [0.0, 0.0, 0.0, 1.0] * len(times)
for _ in ("Left", "Right"):
    f_acc.append(add_accessor(struct.pack(f"<{len(q)}f", *q), "VEC4", len(times), 4, 34962))

# ── Node-Graph ───────────────────────────────────────────────────────────────
# 0 Hips · 1 LeftUpLeg · 2 RightUpLeg · 3 LeftLeg · 4 RightLeg · 5 LeftFoot · 6 RightFoot
nodes = [
    {"name": "mixamorig:Hips", "translation": [0, 1, 0], "children": [1, 2]},
    {"name": "mixamorig:LeftUpLeg", "translation": [0.1, -0.1, 0], "children": [3]},
    {"name": "mixamorig:RightUpLeg", "translation": [-0.1, -0.1, 0], "children": [4]},
    {"name": "mixamorig:LeftLeg", "translation": [0, -0.4, 0], "children": [5]},
    {"name": "mixamorig:RightLeg", "translation": [0, -0.4, 0], "children": [6]},
    {"name": "mixamorig:LeftFoot", "translation": [0, -0.4, 0]},
    {"name": "mixamorig:RightFoot", "translation": [0, -0.4, 0]},
]

def samplers_output(acc):  # helper
    return acc

channels, samplers = [], []
def add_track(node, t_acc_in, o_acc_in, path):
    si = len(samplers)
    samplers.append({"input": t_acc_in, "output": o_acc_in, "interpolation": "LINEAR"})
    channels.append({"sampler": si, "target": {"node": node, "path": path}})

add_track(0, t_acc, p_acc, "translation")
add_track(0, t_acc, h_acc, "rotation")
add_track(1, t_acc, hip_acc[0], "rotation")
add_track(2, t_acc, hip_acc[1], "rotation")
add_track(3, t_acc, knee_acc[0], "rotation")
add_track(4, t_acc, knee_acc[1], "rotation")
add_track(5, t_acc, f_acc[0], "rotation")
add_track(6, t_acc, f_acc[1], "rotation")

gltf = {
    "asset": {"version": "2.0", "generator": "mdt-test"},
    "scene": 0,
    "scenes": [{"nodes": [0]}],
    "nodes": nodes,
    "animations": [{
        "name": "JumpTest",
        "channels": channels,
        "samplers": samplers,
    }],
    "buffers": [{"byteLength": len(bin_data)}],
    "bufferViews": buffer_views,
    "accessors": accessors,
}

# ── GLB-Container ────────────────────────────────────────────────────────────
json_data = json.dumps(gltf).encode("utf-8")
while len(json_data) % 4:
    json_data += b" "
while len(bin_data) % 4:
    bin_data.append(0)

total = 12 + 8 + len(json_data) + 8 + len(bin_data)
with open(OUT, "wb") as f:
    f.write(struct.pack("<III", 0x46546C67, 2, total))
    f.write(struct.pack("<I", len(json_data)))
    f.write(struct.pack("<I", 0x4E4F534A))
    f.write(json_data)
    f.write(struct.pack("<I", len(bin_data)))
    f.write(struct.pack("<I", 0x004E4942))
    f.write(bin_data)

print(f"OK: {OUT} ({os.path.getsize(OUT)} bytes, {len(nodes)} nodes, {len(channels)} tracks)")
