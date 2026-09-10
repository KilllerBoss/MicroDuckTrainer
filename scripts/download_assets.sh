#!/bin/bash
# MicroDuck Trainer v2.0 - Asset-Download (Original HF-Space Assets + Unitree G1)
set -e
cd /home/z/my-project
BASE="https://huggingface.co/spaces/pollen-robotics/microduck-simulator/resolve/main/app"

echo "== 1) mjlab Roboter-Assets (Ente, Original) =="
mkdir -p public/robot/mjlab/meshes public/policies public/wasm
for f in robot_allcollisions.xml robot_allcollisions_rollers.xml kinematics.json kinematics_rollers.json microduck.glb; do
  curl -sL --retry 3 "$BASE/public/robot/mjlab/$f" -o "public/robot/mjlab/$f" && echo "ok $f"
done

echo "== 2) mjlab Meshes =="
MESHES="ankle_l_v1.stl ankle_left.stl ankle_r_v1.stl ankle_right.stl banana_pcb_locker.stl bearing_roll.stl bottom_head_shell.stl elec_rpi_robot_hat_pcb.stl face_part.stl foot_left.stl foot_right.stl hip_l.stl jaw.stl jaw_soft.stl left_shell.stl leg.stl lens.stl m12_lens_holder.stl motor_support.stl neck.stl neck_pitch.stl noenoeil.stl np_f970.stl pcb__raspberry_pi_zero_2_w.stl power_support.stl right_shell.stl rim.stl roller_blade.stl seeed_bearing__configuration__22x16x4.stl seeed_bearing__configuration_default.stl soft_mouth_top.stl sole_left.stl sole_right.stl speaker.stl tire.stl top_head_shell.stl trunk_base.stl upper_leg_left.stl upper_leg_right.stl upper_leg_rigidity_plate.stl xl330.stl yaw2roll.stl yaw_roll_motion.stl"
for m in $MESHES; do
  curl -sL --retry 3 "$BASE/public/robot/mjlab/meshes/$m" -o "public/robot/mjlab/meshes/$m"
done
echo "meshes: $(ls public/robot/mjlab/meshes | wc -l) Dateien"

echo "== 3) Policies (ONNX, alle 9) =="
for p in BEST_alpha_walking BEST_alpha_stand BEST_alpha_sitstand BEST_roller BEST_roller_crouch alpha_ground_pick ball_kick_left ball_kick_right roulade; do
  curl -sL --retry 3 "$BASE/public/policies/$p.onnx" -o "public/policies/$p.onnx" && echo "ok $p.onnx"
done

echo "== 4) Referenz-Quellcode des Originals =="
mkdir -p reference/controls
for f in src/game/arena.js src/game/ball-actor.js src/game/ball-visual.js src/game/variants.js src/store.js; do
  curl -sL --retry 3 "$BASE/$f" -o "reference/$(basename $f)"
done
for f in controller gamepad keyboard touch waypoint; do
  curl -sL --retry 3 "$BASE/src/game/controls/$f.js" -o "reference/controls/$f.js"
done
curl -sL --retry 3 "$BASE/src/ui/TouchOverlay.jsx" -o reference/TouchOverlay.jsx || true
echo "reference: $(ls reference)"

echo "== 5) Unitree G1 (mujoco_menagerie, sparse clone) =="
rm -rf /tmp/menagerie
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie 2>&1 | tail -2
cd /tmp/menagerie
git sparse-checkout set unitree_g1
git checkout 2>&1 | tail -2
mkdir -p /home/z/my-project/public/robot/unitree_g1
cp -r /tmp/menagerie/unitree_g1/* /home/z/my-project/public/robot/unitree_g1/
echo "G1 files: $(ls /home/z/my-project/public/robot/unitree_g1)"
echo "G1 meshes: $(ls /home/z/my-project/public/robot/unitree_g1/meshes 2>/dev/null | wc -l)"

echo "== FERTIG =="
du -sh /home/z/my-project/public/robot /home/z/my-project/public/policies
