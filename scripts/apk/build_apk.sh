#!/bin/bash
# MicroDuck Trainer v2.2 – APK-Build ohne Android Studio
# Schritte: Icons → javac → D8 → aapt2 link (+90MB Assets) → DEX einfügen → zipalign → signieren
set -e
cd /home/z/my-project
ROOT=$PWD
export JAVA_HOME=$PWD/scripts/android-sdk/jdk-21.0.12.1+1
BT=$PWD/scripts/android-sdk/bt
PLATFORM=$PWD/scripts/android-sdk/plat/android.jar
R8=$PWD/scripts/android-sdk/r8.jar
APK=scripts/apk
OUT=$PWD/out
BUILD=$PWD/scripts/apk/build-v2
VERSION_CODE=9
VERSION_NAME=2.7
FINAL=download/MicroDuckTrainer-v2.7.apk

rm -rf "$BUILD"
mkdir -p "$BUILD/classes" "$BUILD/dexout" download

echo "== [1/7] Launcher-Icons =="
python3 scripts/make_launcher_icons.py >/dev/null
echo "icons ok"

echo "== [2/7] javac =="
"$JAVA_HOME/bin/javac" --release 11 -classpath "$PLATFORM" -d "$BUILD/classes" \
  $APK/java/com/microduck/trainer/*.java 2>&1 | grep -v "bootstrap class path" || true
ls "$BUILD/classes/com/microduck/trainer/" | head -3

echo "== [3/7] D8 (R8 8.5.35) → classes.dex =="
java -Xmx2g -cp "$R8" com.android.tools.r8.D8 --release --lib "$PLATFORM" --min-api 26 \
  --output "$BUILD/dexout" $(find "$BUILD/classes" -name '*.class')
ls -la "$BUILD/dexout/"

echo "== [4/7] aapt2 compile + link (Assets: $OUT) =="
"$BT/aapt2" compile --dir $APK/res -o "$BUILD/res.zip"
"$BT/aapt2" link -o "$BUILD/app-unsigned.apk" \
  -I "$PLATFORM" \
  --manifest $APK/AndroidManifest.xml \
  -R "$BUILD/res.zip" \
  -A "$OUT" \
  --auto-add-overlay \
  --min-sdk-version 26 --target-sdk-version 34 \
  --version-code $VERSION_CODE --version-name "$VERSION_NAME"
ls -la "$BUILD/app-unsigned.apk"

echo "== [5/7] classes.dex einfügen =="
cd "$BUILD"
cp dexout/classes.dex .
python3 - <<'EOF'
import zipfile, shutil, os
src = "app-unsigned.apk"; dst = "app-with-dex.apk"
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zout:
    for item in zin.infolist():
        if item.filename == "classes.dex":
            continue
        zout.writestr(item, zin.read(item.filename))
    with open("classes.dex", "rb") as f:
        zout.writestr(zipfile.ZipInfo("classes.dex"), f.read())
print("dex eingefuegt:", os.path.getsize(dst) // (1024 * 1024), "MB")
EOF

echo "== [6/7] zipalign =="
LD_LIBRARY_PATH="$BT" "$BT/zipalign" -f 4 app-with-dex.apk app-aligned.apk
echo "aligned"

echo "== [7/7] Signieren (v1+v2) =="
# Keystorepersistent: liegt in download/ (gleiche Signatur wie v2.0 → Update ohne Deinstallation)
KS_SRC=$ROOT/download/microduck-trainer.keystore
if [ ! -f "$KS_SRC" ]; then
  "$JAVA_HOME/bin/keytool" -genkeypair -v \
    -keystore "$KS_SRC" -alias microduck \
    -keyalg RSA -keysize 2048 -validity 10950 \
    -storepass microduck2026 -keypass microduck2026 \
    -dname "CN=MicroDuck Trainer, O=MicroDuck, C=DE" >/dev/null 2>&1
  echo "Keystore NEU erstellt (andere Signatur!)"
fi
cp "$KS_SRC" microduck-trainer.keystore
"$BT/apksigner" sign \
  --ks microduck-trainer.keystore \
  --ks-pass pass:microduck2026 --key-pass pass:microduck2026 \
  --out "/home/z/my-project/$FINAL" app-aligned.apk
"$BT/apksigner" verify --print-certs "/home/z/my-project/$FINAL" | head -4
echo "== FERTIG: $FINAL =="
ls -la "/home/z/my-project/$FINAL"
