"""Loading real labelled SAR data, and splitting it WITHOUT leakage.

Two things here decide whether a reported score means anything.

SCENE-LEVEL SPLITTING.  SAR training patches are cut from a much smaller
number of source scenes.  Patches from one scene share the same sea state,
incidence angle, speckle statistics and often the same slick.  If patches from
one scene land in both train and test, the model is graded on sea it has
already memorised and the score is inflated -- this single mistake accounts
for most implausible accuracy numbers in the literature and in student work.
`scene_split` guarantees a scene appears on exactly one side.

MASK DECODING.  Published datasets ship masks in three different shapes and
guessing wrong silently trains on garbage.  `decode_mask` handles all three
and reports which it used, so a wrong guess is visible rather than silent.

Expected layouts, in the order they are looked for:

    data/real/images/<name>.png + data/real/masks/<name>.png    preferred
    data/train/images/<name>.png + data/train/masks/<name>.png  legacy
    data/real/spill/*.png, data/real/nospill/*.png              labels only

Scene grouping: by default the scene id is the filename with any trailing
_<digits> removed, so `S1A_20230114_patch_07.png` and `..._patch_08.png` are
one scene.  If your dataset does not encode the scene in the filename, write
data/real/scenes.json as {"<filename>": "<scene id>", ...} and that mapping
wins.  Ungrouped files are each treated as their own scene, which is the
conservative choice only when they really are separate acquisitions.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image

from .unet import LAND, LOOKALIKE, N_CLASSES, OIL, SEA, SHIP

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Colour-coded masks: the palette used by the Krestenitis et al. Sentinel-1
# oil-spill dataset, whose five classes are exactly ours.
PALETTE = {
    (0, 0, 0): SEA,
    (0, 255, 255): OIL,
    (255, 0, 0): LOOKALIKE,
    (153, 76, 0): SHIP,
    (0, 153, 0): LAND,
}

_TRAILING_INDEX = re.compile(r"[_-]\d+$")


def scene_id(filename: str, mapping: dict | None = None) -> str:
    """Which acquisition did this patch come from?"""
    base = os.path.splitext(os.path.basename(filename))[0]
    if mapping and os.path.basename(filename) in mapping:
        return str(mapping[os.path.basename(filename)])
    if mapping and base in mapping:
        return str(mapping[base])
    stripped = _TRAILING_INDEX.sub("", base)
    return stripped or base


def decode_mask(path, size) -> tuple[np.ndarray, str]:
    """Read a label mask into class indices 0..4.  Returns (mask, how)."""
    im = Image.open(path)
    if im.mode in ("RGB", "RGBA", "P"):
        rgb = np.asarray(im.convert("RGB").resize((size, size), Image.NEAREST))
        flat = rgb.reshape(-1, 3)
        uniq = {tuple(c) for c in np.unique(flat, axis=0)}
        if uniq and uniq <= set(PALETTE) | {(255, 255, 255)}:
            out = np.zeros(flat.shape[0], np.int8)
            for colour, cls in PALETTE.items():
                out[(flat == np.array(colour)).all(1)] = cls
            return out.reshape(size, size), "rgb palette"
        # not our palette -- fall through to intensity handling
        im = im.convert("L")

    g = np.asarray(im.convert("L").resize((size, size), Image.NEAREST)).astype(np.int16)
    vals = np.unique(g)
    if vals.max() <= N_CLASSES - 1:
        return g.astype(np.int8), "class indices"
    if set(vals.tolist()) <= {0, 255}:
        return ((g > 127).astype(np.int8) * OIL), "binary oil mask"
    # Neither form. Guessing here silently trains the model on wrong labels,
    # which is worse than having no labels at all -- so this is reported as a
    # failure and the pair is dropped by discover(), not quietly thresholded.
    return None, f"UNDECODABLE - values {vals[:6].tolist()}{'...' if len(vals) > 6 else ''}"


def discover(root=None, verbose=True):
    """Find every (image, mask) pair, tagged with its scene."""
    root = root or ROOT
    mapping = {}
    mp = os.path.join(root, "data", "real", "scenes.json")
    if os.path.exists(mp):
        try:
            mapping = json.load(open(mp))
        except Exception:
            mapping = {}

    pairs = []
    for idir, mdir in (
        (os.path.join(root, "data", "real", "images"),
         os.path.join(root, "data", "real", "masks")),
        (os.path.join(root, "data", "train", "images"),
         os.path.join(root, "data", "train", "masks")),
    ):
        if not os.path.isdir(idir) or not os.path.isdir(mdir):
            continue
        masks = {os.path.splitext(f)[0]: os.path.join(mdir, f)
                 for f in os.listdir(mdir) if f.lower().endswith(IMG_EXT)}
        for fn in sorted(os.listdir(idir)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(fn)[0]
            if stem not in masks:
                continue
            pairs.append({"image": os.path.join(idir, fn),
                          "mask": masks[stem],
                          "scene": scene_id(fn, mapping)})

    if verbose and pairs:
        scenes = {p["scene"] for p in pairs}
        print(f"  found {len(pairs)} labelled patches from {len(scenes)} scenes")
        if len(scenes) < 3:
            print("  WARNING: fewer than 3 distinct scenes. A held-out score "
                  "from this data will not generalise -- it is measuring "
                  "memorisation of one acquisition.")
    return pairs


def scene_split(pairs, val_frac=0.25, seed=0, verbose=True):
    """Split so that no scene appears on both sides."""
    by_scene = defaultdict(list)
    for p in pairs:
        by_scene[p["scene"]].append(p)
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)

    n_val = max(1, int(round(len(scenes) * val_frac))) if len(scenes) > 1 else 0
    val_scenes = set(scenes[:n_val])

    train = [p for s in scenes if s not in val_scenes for p in by_scene[s]]
    val = [p for s in val_scenes for p in by_scene[s]]

    assert not ({p["scene"] for p in train} & {p["scene"] for p in val}), \
        "scene leaked across the split"

    if verbose:
        print(f"  split by scene: {len(scenes)-len(val_scenes)} train scenes "
              f"({len(train)} patches) / {len(val_scenes)} val scenes "
              f"({len(val)} patches)")
        if not val:
            print("  WARNING: only one scene available, so there is no "
                  "held-out set. Any score printed is training performance.")
    return train, val


def load_pairs(pairs, size=128, verbose=True):
    """(N,2,size,size) inputs and (N,size,size) masks, via the shared prep path."""
    from .preprocess import prepare_array

    xs, ys, hows = [], [], defaultdict(int)
    dropped = []
    for p in pairs:
        m, how = decode_mask(p["mask"], size)
        hows[how] += 1
        if m is None:                     # unreadable labels are dropped, never guessed
            dropped.append((p["mask"], how))
            continue
        raw = np.asarray(Image.open(p["image"]).convert("L"), np.float32)
        im, _, _ = prepare_array(raw, size)
        xs.append(im.astype(np.float32))
        ys.append(m.astype(np.int8))

    if verbose and hows:
        print("  mask decoding: " +
              ", ".join(f"{k} x{v}" for k, v in sorted(hows.items())))
    if dropped:
        print(f"  DROPPED {len(dropped)} pair(s) whose masks are neither class "
              f"indices (0-{N_CLASSES-1}), binary 0/255, nor the RGB palette.")
        for path, how in dropped[:3]:
            print(f"    {os.path.basename(path)}: {how}")
        print("  Training on guessed labels is worse than training on fewer, "
              "so these are excluded rather than thresholded.")

    if not xs:
        return (np.zeros((0, 2, size, size), np.float32),
                np.zeros((0, size, size), np.int8))
    return np.stack(xs), np.stack(ys)


def class_balance(y):
    """Share of pixels per class -- tells you what the model will ignore."""
    if y.size == 0:
        return {}
    return {["Sea", "Oil", "Look-alike", "Ship", "Land"][c]:
            round(float((y == c).mean()), 5) for c in range(N_CLASSES)}
