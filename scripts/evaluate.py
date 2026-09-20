"""Honest evaluation: precision, recall, F1 -- not accuracy.

Accuracy is the wrong metric here and quoting it is how student projects end
up claiming 95% on a system that does not work.  Most of the ocean is clean,
so a detector that answers "no spill" to everything scores ~90% accuracy and
is useless.  What matters is:

    precision = of the scenes we flagged, how many really were spills
                (1 - precision IS the false-positive rate)
    recall    = of the real spills, how many did we catch
    F1        = their harmonic mean

These trade off against each other.  You cannot maximise both, and any claim
of "100% with no false positives" is a claim to have beaten that trade-off.

The three-way verdict is scored two ways, because REVIEW is not a wrong
answer -- it is a deferral:

    strict   SPILL counts as an alert; REVIEW counts as no alert.
             This is the honest false-positive rate of automatic alerts.
    flagged  SPILL or REVIEW counts as "a human should look".
             This is the honest miss rate of the system as deployed.

Run:  python scripts/evaluate.py [--n 300] [--wind]
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.datagen import make_scene            # noqa: E402
from backend.ml.predict import detect                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def png_bytes(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def counts(pairs):
    """pairs of (truth_is_spill, predicted_alert) -> confusion + metrics."""
    tp = sum(1 for t, p in pairs if t and p)
    fp = sum(1 for t, p in pairs if not t and p)
    fn = sum(1 for t, p in pairs if t and not p)
    tn = sum(1 for t, p in pairs if not t and not p)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


def show(name, m, n):
    print(f"\n  {name}")
    print(f"    confusion   TP {m['tp']:4d}   FP {m['fp']:4d}   "
          f"FN {m['fn']:4d}   TN {m['tn']:4d}")
    print(f"    precision   {m['precision']*100:5.1f}%   "
          f"(false-positive rate {100-m['precision']*100:4.1f}%)")
    print(f"    recall      {m['recall']*100:5.1f}%   "
          f"(missed {m['fn']} of {m['tp']+m['fn']} real spills)")
    print(f"    F1          {m['f1']*100:5.1f}%        on {n} scenes")


def load_real():
    """Real labelled images, if the user has supplied any. -> (rows, source)

    Two layouts are understood:

        data/real/spill/*.png  +  data/real/nospill/*.png     hand-sorted
        data/real/images/*.png +  data/real/masks/*.png       prepare_zenodo

    For the second, only the HELD-OUT scenes are scored, reusing the exact
    scene split the trainer used (val_frac 0.25, seed 0). Scoring the whole
    folder would grade the model on patches it was fitted to and produce a
    number that collapses the moment it meets new imagery.
    """
    out = []
    for kind, truth in (("spill", True), ("nospill", False)):
        d = os.path.join(ROOT, "data", "real", kind)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                out.append((os.path.join(d, fn), truth))
    if out:
        return out, "hand-sorted data/real/{spill,nospill}"

    try:
        from backend.ml import dataset as ds
        from backend.ml.unet import OIL
    except Exception:
        return [], ""
    pairs = ds.discover(ROOT, verbose=False)
    if not pairs:
        return [], ""
    _, val = ds.scene_split(pairs, val_frac=0.25, seed=0, verbose=False)
    if not val:
        return [], ""
    for p in val:
        mk, _how = ds.decode_mask(p["mask"], 128)
        if mk is None:
            continue
        out.append((p["image"], bool((mk == OIL).mean() > 0.005)))
    n_scene = len({p["scene"] for p in val})
    return out, (f"{n_scene} held-out scenes, same split as training "
                 f"(val_frac 0.25, seed 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300,
                    help="synthetic scenes to evaluate (ignored if real data exists)")
    ap.add_argument("--wind", action="store_true",
                    help="supply a plausible wind speed with each scene")
    args = ap.parse_args()

    real, source = load_real()
    rng = np.random.default_rng(20260918)
    rows = []

    if real:
        n_pos = sum(1 for _, t in real if t)
        print(f"\nEvaluating on {len(real)} REAL images from data/real/")
        print(f"  source: {source}")
        print(f"  {n_pos} with oil, {len(real)-n_pos} without")
        if n_pos == 0 or n_pos == len(real):
            print("  WARNING: this held-out set is all one class, so "
                  "precision/recall are undefined (they print as nan).")
            print("  Re-run prepare_zenodo.py over more scenes, or check "
                  "that the oil masks loaded.")
        print("These numbers describe real-world performance.\n")
        for path, truth in real:
            # a clean scene is only a look-alike risk at low wind; give the
            # detector no free information it would not have operationally
            wind = float(rng.uniform(2.0, 14.0)) if args.wind else None
            r = detect(open(path, "rb").read(),
                       filename=os.path.basename(path), wind_ms=wind)
            rows.append((truth, r))
    else:
        print(f"\nEvaluating on {args.n} SYNTHETIC scenes "
              f"(data/real/ is empty).")
        print("WARNING: these numbers describe performance on generated data")
        print("only. They are NOT a claim about real Sentinel-1 imagery.\n")
        for i in range(args.n):
            kind = ["spill", "lookalike", "clean"][i % 3]
            img, _ = make_scene(rng, 128, kind)
            wind = float(rng.uniform(2.0, 14.0)) if args.wind else None
            r = detect(png_bytes(img), filename=f"gen{i}.png", wind_ms=wind)
            rows.append((kind == "spill", r))

    n = len(rows)
    strict = counts([(t, r["verdict"] == "SPILL") for t, r in rows])
    flagged = counts([(t, r.get("flagged", r["verdict"] == "SPILL"))
                      for t, r in rows])

    show("STRICT   (automatic alert = SPILL only)", strict, n)
    show("FLAGGED  (SPILL or REVIEW reaches a human)", flagged, n)

    tally = {}
    for _, r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("\n  verdict spread  " +
          "   ".join(f"{k} {v}" for k, v in sorted(tally.items())))

    deferred = tally.get("REVIEW", 0)
    print(f"\n  {deferred} of {n} scenes ({deferred/n*100:.1f}%) were deferred "
          "for human review")
    print("  rather than forced into a verdict. That is the cost of the "
          "lower false-positive rate.\n")

    if not real:
        print("  To get numbers that mean something, put real labelled SAR")
        print("  images in data/real/spill/ and data/real/nospill/ and re-run.\n")


if __name__ == "__main__":
    main()
