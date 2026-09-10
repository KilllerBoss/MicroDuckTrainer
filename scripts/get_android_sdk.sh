#!/bin/bash
# Android Build-Tools ohne Android Studio (aapt2, apksigner, zipalign, android.jar, r8)
set -e
SDK=/home/z/my-project/scripts/android-sdk
mkdir -p $SDK/bt $SDK/plat $SDK/tmp
cd $SDK/tmp

if [ ! -f $SDK/bt/aapt2 ]; then
  echo "== build-tools r34 =="
  curl -sL --retry 3 -o bt34.zip https://dl.google.com/android/repository/build-tools_r34-linux.zip
  unzip -q -o bt34.zip -d btex
  # Archiv enthaelt Ordner android-14 (Codename) - finde aapt2-Ebene
  BTDIR=$(ls btex | head -1)
  cp btex/$BTDIR/aapt2 $SDK/bt/ 2>/dev/null || cp btex/*/aapt2 $SDK/bt/
  cp btex/$BTDIR/zipalign $SDK/bt/ 2>/dev/null || true
  cp btex/$BTDIR/apksigner $SDK/bt/ 2>/dev/null || true
  cp btex/$BTDIR/lib/apksigner.jar $SDK/bt/ 2>/dev/null || true
  cp btex/$BTDIR/lib/d8.jar $SDK/bt/ 2>/dev/null || true
fi

if [ ! -f $SDK/plat/android.jar ]; then
  echo "== platform-34 =="
  curl -sL --retry 3 -o p34.zip https://dl.google.com/android/repository/platform-34_r02.zip
  unzip -q -o p34.zip -d pex
  find pex -name android.jar -exec cp {} $SDK/plat/android.jar \;
fi

if [ ! -f $SDK/r8.jar ]; then
  echo "== r8 8.5.35 =="
  curl -sL --retry 3 -o $SDK/r8.jar https://dl.google.com/android/maven2/com/android/tools/r8/8.5.35/r8-8.5.35.jar
fi

rm -rf $SDK/tmp
echo "== Ergebnis =="
ls -la $SDK $SDK/bt $SDK/plat
$SDK/bt/aapt2 version 2>&1 | head -2 || true
java -cp $SDK/r8.jar com.android.tools.r8.D8 --version 2>&1 | head -2 || true
