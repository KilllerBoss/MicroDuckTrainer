#!/usr/bin/env python3
"""Launcher-Icons für MicroDuck Trainer (Enten-Silhouette, Tron-Stil)."""
from PIL import Image, ImageDraw
import os

RES = "/home/z/my-project/scripts/apk/res"
SIZES = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}


def make_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (5, 6, 10, 255))
    d = ImageDraw.Draw(img)
    s = size / 100.0
    # Cyan-Ring
    d.ellipse([4 * s, 4 * s, 96 * s, 96 * s], outline=(34, 211, 238, 255), width=max(2, int(3 * s)))
    # Entenkoerper (gelb)
    d.ellipse([22 * s, 44 * s, 74 * s, 82 * s], fill=(250, 204, 21, 255))
    # Kopf
    d.ellipse([46 * s, 20 * s, 82 * s, 56 * s], fill=(250, 204, 21, 255))
    # Schnabel (orange)
    d.polygon([(78 * s, 32 * s), (94 * s, 38 * s), (78 * s, 44 * s)], fill=(249, 115, 22, 255))
    # Auge
    d.ellipse([64 * s, 28 * s, 72 * s, 36 * s], fill=(5, 6, 10, 255))
    d.ellipse([66 * s, 30 * s, 70 * s, 34 * s], fill=(34, 211, 238, 255))
    # Fluegel-Andeutung
    d.arc([28 * s, 52 * s, 62 * s, 78 * s], start=200, end=340, fill=(202, 138, 4, 255), width=max(2, int(2.5 * s)))
    return img


for dpi, px in SIZES.items():
    out_dir = os.path.join(RES, f"mipmap-{dpi}")
    os.makedirs(out_dir, exist_ok=True)
    make_icon(px).save(os.path.join(out_dir, "ic_launcher.png"))
    print(f"ic_launcher {dpi}: {px}px")

print("Fertig.")
