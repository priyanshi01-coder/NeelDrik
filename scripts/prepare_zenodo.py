"""Turn the Trujillo-Acatitla Sentinel-1 dataset into patches NEELDRIK can train on.

Dataset: "Sentinel-1 SAR Oil spill image dataset for train, validate, and test
deep learning models", Trujillo-Acatitla et al. (2024).

    Part I    zenodo.org/records/8346860    1200 oil spill images       40.7 GB
    Part II   zenodo.org/records/8253899     685 no-oil + 685 look-alike 45.9 GB
    Part III  zenodo.org/records/13761290    150 of each class (test)     9.9 GB

READ THIS BEFORE DOWNLOADING 46 GB:
Part II contains NO oil spill images at all. Its masks are entirely zeros --
that is why they compress to 420 kB against 23 GB of imagery. Training on
Part II alone teaches a model that oil never exists. You need Part I (or
Part III) for positive examples.

Expected layout (Part III, as extracted)
----------------------------------------
    <src>/Images/Oil/00000.tif          2048 x 2048 x 2  float32, Sigma0 dB
    <src>/Images/Lookalike/00000.tif
    <src>/Images/No oil/00000.tif
    <src>/Mask/Oil/00000_segmentation.tif    2048 x 2048  uint8, {0,1}
    <src>/Mask/Lookalike/...
    <src>/Mask/No oil/...

Only paths BELOW --src are inspected when deciding image-vs-mask and class.
The extracted folder is itself called "02_Test_images_and_ground_truth", so
matching against the absolute path files every image as a mask.

Four properties of the real data this script handles
----------------------------------------------------
1. Polarisation order is detected, not assumed. Over ocean VV backscatter
   sits well above VH, so the band with the higher median is VV. Override
   with --band if you know better.

2. Scenes are zero-padded. Where the Sentinel-1 swath does not cover the
   tile the value is exactly 0.0 dB -- which is not a dark pixel but the
   brightest possible one, and physically impossible for sea (Sigma0 = 1).
   Some scenes are more than half padding. Padding is excluded from the
   contrast stretch and patches containing much of it are dropped; letting
   it through would train the model that a hard bright edge means "not oil".

3. The stretch is per scene, on percentiles of the valid pixels only.
   That is deliberate: oil scenes in this dataset have visibly different
   global brightness from clean ones, and a fixed dB window would hand the
   model that difference as a shortcut. Normalising it away forces the
   decision onto shape and texture, which is what has to generalise.

4. Only the Oil class has pixel labels. Look-alike and No-oil masks are
   entirely zero -- verified, not assumed. The dataset says a look-alike
   scene contains one, never where it is.

Class mapping
-------------
    oil scene,       mask 1  ->  1  Oil
    oil scene,       mask 0  ->  0  Sea
    no-oil scene             ->  0  Sea
    look-alike scene         ->  0  Sea      (hard negative)

Look-alike scenes are the most valuable negatives here -- they are the dark
patches that are NOT oil, which is exactly the discrimination that produces
false alarms.

--weak-lookalike labels dark pixels in look-alike scenes as class 2 instead.
That gives the Look-alike class a training signal, but the labels are inferred
by a threshold rather than annotated. It is off by default, and if you use it,
say so when you report results.

Ship and Land receive no supervision from this dataset. Their IoU will be ~0
and should not be quoted.

Usage
-----
    python scripts/prepare_zenodo.py --src "G:/02_Test_images_and_ground_truth" --dry-run
    python scripts/prepare_zenodo.py --src "G:/02_Test_images_and_ground_truth" --limit 5
    python scripts/prepare_zenodo.py --src "G:/02_Test_images_and_ground_truth"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEA, OIL, LOOKALIKE = 0, 1, 2
TIF = (".tif", ".tiff")

NODATA = 0.0          # exact 0 dB marks swath padding, never real sea

# Checked in this order: "look-alike" and "no oil" both contain the word
# "oil", so a plain oil test has to come last.
CLASS_PATTERNS = [
    ("lookalike", re.compile(r"look[_\- ]?alike", re.I)),
    ("nooil",     re.compile(r"no[_\- ]?oil|oil[_\- ]?free|clean", re.I)),
    ("oil",       re.compile(r"(^|[/_\- ])oil([/_\- ]|$)|oil[_\- ]?spill", re.I)),
]
MASK_RE = re.compile(r"mask|ground[_\- ]?truth|label|segmentation|_gt\b", re.I)
_NUM = re.compile(r"(\d{3,})")


def read_tiff(path):
    """-> float32 array, shape (H, W) or (H, W, C)."""
    try:
        import tifffile
        return np.asarray(tifffile.imread(path), dtype=np.float32)
    except Exception:
        return np.asarray(Image.open(path), dtype=np.float32)


def rel(path, src):
    """Path below --src, forward-slashed.

    Everything downstream matches on this rather than the absolute path. The
    extracted folder is named '..._and_ground_truth' and a drive might be
    'G:/oil', so absolute matching misfiles every single file.
    """
    try:
        r = os.path.relpath(path, src)
    except ValueError:                       # different drive on Windows
        r = path
    return r.replace("\\", "/")


def classify(relpath):
    for name, pat in CLASS_PATTERNS:
        if pat.search(relpath):
            return name
    return None


def is_mask(relpath):
    return bool(MASK_RE.search(relpath))


def scene_key(path):
    """The trailing number that pairs an image with its mask."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _NUM.findall(stem)
    return m[-1] if m else stem


def choose_band(img, forced=None):
    """Which channel is VV? Returns (index, why).

    Over the ocean VV sits roughly 5-10 dB above VH, so the channel with the
    higher median of valid pixels is VV. Measuring it beats hardcoding an
    index that silently flips if the archive is ever rebuilt.
    """
    if img.ndim == 2:
        return None, "single band"
    if img.shape[-1] > 4 and img.ndim == 3:            # channel-first
        img = np.moveaxis(img, 0, -1)
    n = img.shape[-1]
    if forced is not None:
        return min(forced, n - 1), f"forced --band {forced}"
    meds = []
    for c in range(n):
        ch = img[..., c]
        v = ch[np.isfinite(ch) & (ch != NODATA)]
        meds.append(float(np.median(v)) if v.size else -np.inf)
    idx = int(np.argmax(meds))
    return idx, ("VV by median " +
                 " vs ".join(f"b{c}={m:.1f}dB" for c, m in enumerate(meds)))


def take_band(img, idx):
    if img.ndim == 2:
        return img
    if img.shape[-1] > 4 and img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    return img[..., idx]


def db_to_u8(a, valid, lo_pct=2.0, hi_pct=98.0):
    """Sigma0 dB -> 8-bit, stretched on percentiles of the VALID pixels.

    Padding is filled with the scene median before quantising so that a
    dropped-but-not-dropped-enough patch carries no bright artificial edge.
    """
    v = a[valid]
    if v.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    filled = np.where(valid, a, np.median(v))
    out = np.clip((filled - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def tiles(h, w, size):
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            yield y, x


def weak_lookalike_mask(u8, valid, pct=12.0, min_area_frac=0.0008):
    """Guess WHERE the look-alike is in a scene labelled as containing one.

    Thresholding raw SAR picks out speckle: single dark pixels scattered over
    the whole scene, which would teach the model that a dark speck is a
    look-alike. A look-alike is a *region* -- a wind shadow, an algal slick,
    a rain cell -- so the guess is only credible after speckle is suppressed
    and small blobs are discarded.

    These labels are inferred, not annotated. Disclose them if you train on
    them.
    """
    try:
        import cv2
    except Exception:
        return np.zeros(u8.shape, bool)

    # Median first kills speckle spikes without smearing a real edge; the
    # blur then merges what survives into regions.
    sm = cv2.medianBlur(u8, 7)
    sm = cv2.GaussianBlur(sm, (0, 0), 6.0)

    v = sm[valid]
    if v.size == 0:
        return np.zeros(u8.shape, bool)
    dark = (sm < np.percentile(v, pct)) & valid

    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = np.zeros(u8.shape, bool)
    min_area = max(64, int(min_area_frac * u8.size))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out |= (lab == i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="folder containing the extracted .7z contents")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "real"))
    ap.add_argument("--patch", type=int, default=256,
                    help="patch size cut from the 2048px scene (resized to 128)")
    ap.add_argument("--resize", type=int, default=128)
    ap.add_argument("--band", type=int, default=None,
                    help="force a channel index instead of detecting VV")
    ap.add_argument("--limit", type=int, default=0,
                    help="max scenes PER CLASS (0 = all). Start small.")
    ap.add_argument("--min-oil", type=float, default=0.02,
                    help="a patch counts as oil if this share of it is oil")
    ap.add_argument("--max-pos-per-scene", type=int, default=16,
                    help="cap so one big slick cannot dominate the set")
    ap.add_argument("--neg-per-scene", type=int, default=4,
                    help="negative patches kept from each scene")
    ap.add_argument("--min-valid", type=float, default=0.98,
                    help="drop a patch unless this share of it is real data")
    ap.add_argument("--weak-lookalike", action="store_true",
                    help="label dark pixels in look-alike scenes as class 2 "
                         "(inferred labels -- disclose if you use this)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what was found and write nothing")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    if not os.path.isdir(src):
        print(f"\n  --src is not a folder: {src}\n")
        return 1

    # ---- find and pair files -------------------------------------------- #
    images, masks, unclassified = {}, {}, []
    for dirpath, _, files in os.walk(src):
        for fn in files:
            if not fn.lower().endswith(TIF):
                continue
            full = os.path.join(dirpath, fn)
            r = rel(full, src)
            cls = classify(r)
            if cls is None:
                unclassified.append(r)
                continue
            (masks if is_mask(r) else images).setdefault(cls, {})[
                scene_key(full)] = full

    print(f"\nScanning {src}")
    total_pairs = 0
    for cls in ("oil", "lookalike", "nooil"):
        ni, nm = len(images.get(cls, {})), len(masks.get(cls, {}))
        paired = len(set(images.get(cls, {})) & set(masks.get(cls, {})))
        total_pairs += paired
        print(f"  {cls:10s} images {ni:5d}   masks {nm:5d}   paired {paired:5d}")

    if unclassified:
        print(f"\n  {len(unclassified)} TIFFs matched no class, e.g.:")
        for u in unclassified[:3]:
            print(f"     {u}")

    if not total_pairs:
        print("\n  Nothing paired. --src should be the folder that directly "
              "contains Images/ and Mask/.")
        here = sorted(os.listdir(src))[:8]
        print(f"  {src} currently contains: {', '.join(here) or '(empty)'}")
        return 1

    if not images.get("oil"):
        print("\n  WARNING: no oil spill scenes found.")
        print("  Part II contains only no-oil and look-alike images -- its")
        print("  masks are all zeros. Training on it alone produces a model")
        print("  that never predicts oil. Add Part I (zenodo.org/records/")
        print("  8346860) or Part III (zenodo.org/records/13761290).")

    if a.dry_run:
        print("\n  --dry-run: nothing written.\n")
        return 0

    idir = os.path.join(a.out, "images")
    mdir = os.path.join(a.out, "masks")
    os.makedirs(idir, exist_ok=True)
    os.makedirs(mdir, exist_ok=True)

    rng = np.random.default_rng(0)
    written = {"oil": 0, "lookalike": 0, "nooil": 0}
    oil_patches = neg_patches = 0
    scenes_done = 0
    dropped_padding = 0
    band_note = ""

    for cls in ("oil", "lookalike", "nooil"):
        keys = sorted(set(images.get(cls, {})) & set(masks.get(cls, {})))
        if a.limit:
            keys = keys[:a.limit]
        for k in keys:
            try:
                raw = read_tiff(images[cls][k])
                msk = read_tiff(masks[cls][k])
            except Exception as exc:
                print(f"  skip {cls}/{k}: {exc}")
                continue

            bidx, why = choose_band(raw, a.band)
            if not band_note:
                band_note = why
                print(f"  polarisation: channel {bidx} ({why})")
            img = take_band(raw, bidx) if bidx is not None else raw

            if msk.ndim == 3:
                msk = msk[..., 0]
            if img.shape[:2] != msk.shape[:2]:
                print(f"  skip {cls}/{k}: image {img.shape[:2]} vs "
                      f"mask {msk.shape[:2]} -- shapes disagree")
                continue

            valid = np.isfinite(img) & (img != NODATA)
            if valid.mean() < 0.02:
                print(f"  skip {cls}/{k}: {(1-valid.mean())*100:.0f}% padding")
                continue

            u8 = db_to_u8(img, valid)
            lab = np.zeros(msk.shape, np.uint8)
            lab[msk > 0.5] = OIL
            if cls == "lookalike" and a.weak_lookalike:
                lab[weak_lookalike_mask(u8, valid)] = LOOKALIKE

            h, w = u8.shape
            pos, neg = [], []
            for (y, x) in tiles(h, w, a.patch):
                vsub = valid[y:y + a.patch, x:x + a.patch]
                if vsub.mean() < a.min_valid:
                    dropped_padding += 1
                    continue
                sub = lab[y:y + a.patch, x:x + a.patch]
                frac = float((sub == OIL).mean())
                dark = float(u8[y:y + a.patch, x:x + a.patch].mean())
                (pos if frac >= a.min_oil else neg).append((y, x, frac, dark))

            # Positives: prefer tiles that straddle the slick edge. A tile
            # that is 99% oil teaches almost nothing about where oil stops,
            # and this scene's slick covers half the frame, so unsorted
            # sampling fills the set with them.
            pos.sort(key=lambda t: abs(t[2] - 0.5))
            pos = pos[:a.max_pos_per_scene]

            if cls == "oil":
                rng.shuffle(neg)
                keep = pos + neg[:max(a.neg_per_scene, len(pos))]
            else:
                # Hard negatives: the darkest tiles are where the look-alike
                # actually is, and where a detector is tempted to cry oil.
                # Random tiles from these scenes are mostly plain sea, which
                # the model already has plenty of.
                neg.sort(key=lambda t: t[3])
                hard = neg[:a.neg_per_scene]
                rest = neg[a.neg_per_scene:]
                rng.shuffle(rest)
                keep = hard + rest[:max(0, a.neg_per_scene // 2)]
            keep = [(y, x) for (y, x, *_ ) in keep]

            for n, (y, x) in enumerate(keep):
                pi = u8[y:y + a.patch, x:x + a.patch]
                pm = lab[y:y + a.patch, x:x + a.patch]
                if a.resize and a.resize != a.patch:
                    pi = np.asarray(Image.fromarray(pi).resize(
                        (a.resize, a.resize), Image.BILINEAR))
                    pm = np.asarray(Image.fromarray(pm).resize(
                        (a.resize, a.resize), Image.NEAREST))
                stem = f"{cls}_{k}_{n:03d}"
                Image.fromarray(pi).save(os.path.join(idir, stem + ".png"))
                Image.fromarray(pm).save(os.path.join(mdir, stem + ".png"))
                written[cls] += 1
                if (pm == OIL).any():
                    oil_patches += 1
                else:
                    neg_patches += 1

            scenes_done += 1
            if scenes_done % 25 == 0:
                print(f"  ... {scenes_done} scenes, "
                      f"{sum(written.values())} patches")

    tot = sum(written.values())
    print(f"\nWrote {tot} patches from {scenes_done} scenes -> {a.out}")
    for c, n in written.items():
        print(f"  {c:10s} {n:6d} patches")
    print(f"  containing oil   {oil_patches:6d}")
    print(f"  pure negative    {neg_patches:6d}")
    print(f"  dropped (padding){dropped_padding:6d}")
    if tot:
        print(f"  positive share   {oil_patches/tot*100:.1f}%")
    if oil_patches == 0:
        print("\n  No positive patches were written. A detector cannot be "
              "trained from this -- get Part I or Part III.")

    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump({"source": "Trujillo-Acatitla et al. 2024, Sentinel-1 SAR",
                   "band_choice": band_note, "patch": a.patch,
                   "resized_to": a.resize, "scenes": scenes_done,
                   "patches": written, "oil_patches": oil_patches,
                   "dropped_padding_patches": dropped_padding,
                   "min_valid": a.min_valid,
                   "weak_lookalike": bool(a.weak_lookalike),
                   "note": "second polarisation dropped; Ship and Land have "
                           "no supervision in this dataset"}, fh, indent=2)

    print("\nNext:  python train.py --torch --epochs 40")
    print("       python scripts/evaluate.py --wind\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
