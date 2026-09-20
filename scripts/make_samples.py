"""Generate demo scenes and labelled training pairs.

    python scripts/make_samples.py

Writes
    data/samples/<kind>_<n>.png          384x384 demo scenes for the UI
    data/train/images/<kind>_<n>.png     128x128 labelled training images
    data/train/masks/<kind>_<n>.png      matching indexed masks (0..4)

The labelled pairs show the exact format to use for your own real SAR data:
drop image/mask pairs into those two folders and re-run `python train.py`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.ml.datagen import make_scene            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "data", "samples")
TRAIN_I = os.path.join(ROOT, "data", "train", "images")
TRAIN_M = os.path.join(ROOT, "data", "train", "masks")

# distinct colours so an indexed mask is also human-readable
PALETTE = [0, 40, 60,      # 0 Sea
           200, 60, 45,    # 1 Oil
           215, 170, 55,   # 2 Look-alike
           90, 210, 245,   # 3 Ship
           125, 115, 100]  # 4 Land
PALETTE += [0] * (768 - len(PALETTE))


def save_mask(arr, path):
    im = Image.fromarray(arr.astype(np.uint8), mode="P")
    im.putpalette(PALETTE)
    im.save(path)


def main():
    for d in (SAMPLES, TRAIN_I, TRAIN_M):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(20260917)

    # --- demo scenes for the UI -------------------------------------- #
    n_demo = {"spill": 4, "lookalike": 3, "clean": 3}
    for kind, count in n_demo.items():
        made = 0
        while made < count:
            img, mask = make_scene(rng, 384, kind)
            oil = float((mask == 1).mean())
            if kind == "spill" and oil < 0.02:
                continue                     # make sure demo spills are visible
            if kind == "clean" and oil > 0:
                continue
            made += 1
            Image.fromarray(img).save(os.path.join(SAMPLES, f"{kind}_{made}.png"))
    print(f"[samples] {sum(n_demo.values())} demo scenes -> data/samples/")

    # --- labelled training pairs ------------------------------------- #
    n_train = {"spill": 4, "lookalike": 2, "clean": 2}
    total = 0
    for kind, count in n_train.items():
        made = 0
        while made < count:
            img, mask = make_scene(rng, 128, kind)
            oil = float((mask == 1).mean())
            if kind == "spill" and oil < 0.02:
                continue
            if kind == "clean" and oil > 0:
                continue
            made += 1
            total += 1
            stem = f"{kind}_{made}"
            Image.fromarray(img).save(os.path.join(TRAIN_I, stem + ".png"))
            save_mask(mask, os.path.join(TRAIN_M, stem + ".png"))
    print(f"[train]   {total} labelled image/mask pairs -> data/train/")
    print("          classes: 0=Sea 1=Oil 2=Look-alike 3=Ship 4=Land")


if __name__ == "__main__":
    main()
