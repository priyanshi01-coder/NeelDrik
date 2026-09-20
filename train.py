"""Train the NEELDRIK 5-class U-Net.

    python train.py                    # train on synthetic + any real images
    python train.py --epochs 40        # longer run
    python train.py --torch            # train the PyTorch twin instead

Real training images are picked up automatically from:
    data/train/images/<name>.png   grayscale SAR patch
    data/train/masks/<name>.png    indexed mask, pixel value = class id
                                   0=Sea 1=Oil 2=Look-alike 3=Ship 4=Land
Drop in as few as 5 of your own labelled images and they are oversampled
alongside the synthetic set.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
from PIL import Image

from backend.ml.datagen import build_dataset
from backend.ml.decide import DEFAULTS, analyse, save_config
from backend.ml.nn import Adam, softmax_ce
from backend.ml.unet import CLASS_NAMES, N_CLASSES, OIL, UNet

ROOT = os.path.dirname(os.path.abspath(__file__))
SIZE = 128


# --------------------------------------------------------------------------- #
def load_real(size=SIZE):
    idir = os.path.join(ROOT, "data", "train", "images")
    mdir = os.path.join(ROOT, "data", "train", "masks")
    if not os.path.isdir(idir):
        return np.zeros((0, 2, size, size), np.float32), np.zeros((0, size, size), np.int8)
    xs, ys = [], []
    for fn in sorted(os.listdir(idir)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue
        stem = os.path.splitext(fn)[0]
        mp = None
        for ext in (".png", ".tif", ".tiff"):
            cand = os.path.join(mdir, stem + ext)
            if os.path.exists(cand):
                mp = cand
                break
        if mp is None:
            continue
        from backend.ml.preprocess import prepare_array
        raw = np.asarray(Image.open(os.path.join(idir, fn)).convert("L"), np.float32)
        im, _, _ = prepare_array(raw, size)
        mk = Image.open(mp).convert("L").resize((size, size), Image.NEAREST)
        m = np.asarray(mk).astype(np.int16)
        if m.max() > N_CLASSES - 1:          # tolerate 0/255 binary oil masks
            m = (m > 127).astype(np.int16) * OIL
        xs.append(im.astype(np.float32))
        ys.append(m.astype(np.int8))
    if not xs:
        return np.zeros((0, 2, size, size), np.float32), np.zeros((0, size, size), np.int8)
    return np.stack(xs), np.stack(ys)


def augment(x, y, rng):
    """x is (N,1,H,W), y is (N,H,W) -- spatial axes differ, so flip both explicitly."""
    if rng.random() < 0.5:                       # vertical flip
        x, y = x[:, :, ::-1, :], y[:, ::-1, :]
    if rng.random() < 0.5:                       # horizontal flip
        x, y = x[:, :, :, ::-1], y[:, :, ::-1]
    k = int(rng.integers(0, 4))                  # 90 deg rotations
    if k:
        x, y = np.rot90(x, k, (2, 3)), np.rot90(y, k, (1, 2))
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def batches(X, Y, bs, rng, train=True):
    idx = rng.permutation(len(X)) if train else np.arange(len(X))
    for i in range(0, len(idx) - (bs - 1 if train else 0), bs):
        j = idx[i:i + bs]
        x = X[j].astype(np.float32)
        y = Y[j][:, ::2, ::2].astype(np.int64)
        if train:
            x, y = augment(x, y, rng)
        yield x, y


# --------------------------------------------------------------------------- #
def evaluate(net, X, Y, kinds, bs=8):
    """Per-class IoU plus end-to-end spill/no-spill accuracy."""
    inter = np.zeros(N_CLASSES); union = np.zeros(N_CLASSES)
    correct = 0
    confusion = {}
    for i in range(0, len(X), bs):
        x = X[i:i + bs].astype(np.float32)
        y = Y[i:i + bs][:, ::2, ::2].astype(np.int64)
        proba = net.predict_proba(x)
        pred = proba.argmax(1)
        for c in range(N_CLASSES):
            pc, yc = pred == c, y == c
            inter[c] += np.logical_and(pc, yc).sum()
            union[c] += np.logical_or(pc, yc).sum()
        for b in range(len(x)):
            res = analyse(proba[b])
            truth = "SPILL" if kinds[i + b] == "spill" else "NOT"
            got = "SPILL" if res["is_spill"] else "NOT"
            correct += int(truth == got)
            confusion[(truth, got)] = confusion.get((truth, got), 0) + 1
    iou = {CLASS_NAMES[c]: (float(inter[c] / union[c]) if union[c] else float("nan"))
           for c in range(N_CLASSES)}
    tp = confusion.get(("SPILL", "SPILL"), 0)
    fp = confusion.get(("NOT", "SPILL"), 0)
    fn = confusion.get(("SPILL", "NOT"), 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return iou, correct / len(X), confusion, {"precision": prec, "recall": rec, "f1": f1}


def prf(truth, got):
    """precision, recall, F1 -- the metrics that describe a detector."""
    tp = int((truth & got).sum())
    fp = int((~truth & got).sum())
    fn = int((truth & ~got).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def calibrate(net, X, Y, kinds, bs=8):
    """Grid-search the decision thresholds on validation data.

    Optimises F1, NOT accuracy.  Accuracy on an imbalanced problem rewards a
    detector that simply answers "no spill" more often, which is the opposite
    of what this system is for.
    """
    probas = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].astype(np.float32)
        probas.append(net.predict_proba(x))
    P = np.concatenate(probas)
    truth = np.array([k == "spill" for k in kinds])
    best, best_cfg, best_pr = -1.0, dict(DEFAULTS), (0.0, 0.0)
    fallback, fb_cfg, fb_pr = -1.0, dict(DEFAULTS), (0.0, 0.0)

    # A configuration is only acceptable if the REVIEW band it implies stays
    # small.  Optimising strict F1 alone happily picks thresholds that shove a
    # quarter of all scenes into "needs review", which looks good on paper and
    # is useless operationally -- an analyst cannot hand-check 25% of the ocean.
    MAX_REVIEW = 0.15

    for oil_thr in (0.40, 0.45, 0.50, 0.55, 0.60):
        for saf in (0.003, 0.005, 0.008, 0.012, 0.018):
            for dom in (1.0, 1.05, 1.15, 1.3):
                cfg = dict(DEFAULTS, oil_thr=oil_thr, spill_area_frac=saf,
                           min_area_frac=max(0.002, saf * 0.7), dominance=dom)
                res = [analyse(P[i], cfg) for i in range(len(P))]
                got = np.array([x["is_spill"] for x in res])
                review = float(np.mean([x["verdict"] == "REVIEW" for x in res]))
                p, r, f = prf(truth, got)
                if f > fallback:                    # best ignoring the constraint
                    fallback, fb_cfg, fb_pr = f, cfg, (p, r)
                if review <= MAX_REVIEW and f > best:
                    best, best_cfg, best_pr = f, cfg, (p, r)

    if best < 0:                                    # nothing met the constraint
        print(f"  WARNING: no threshold set kept the review rate under "
              f"{MAX_REVIEW*100:.0f}%. Using the best unconstrained set; "
              f"expect a high deferral rate.")
        return fb_cfg, fallback, fb_pr
    return best_cfg, best, best_pr


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--train-n", type=int, default=520)
    ap.add_argument("--val-n", type=int, default=140)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--base", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-real", type=int, default=0,
                    help="cap real TRAINING patches (0 = all). Lower this if "
                         "the run dies with an allocator/out-of-memory error.")
    ap.add_argument("--torch", action="store_true", help="train the PyTorch twin")
    a = ap.parse_args()

    if a.torch:
        from backend.ml.torch_unet import train_torch
        return train_torch(a)

    t0 = time.time()
    print(f"[data] generating {a.train_n} train / {a.val_n} val synthetic scenes ...")
    Xtr, Ytr, Ktr = build_dataset(a.train_n, SIZE, seed=a.seed)
    Xva, Yva, Kva = build_dataset(a.val_n, SIZE, seed=a.seed + 9999)

    # Real labelled data, SPLIT BY SCENE so no acquisition is in both halves.
    from backend.ml import dataset as ds
    real_hold = None           # scored separately at the end; see below
    pairs = ds.discover(ROOT)
    if pairs:
        rtr, rva = ds.scene_split(pairs, val_frac=0.25, seed=a.seed)
        Xr, Yr = ds.load_pairs(rtr)
        print("[data] real class balance:", ds.class_balance(Yr))
        if len(Xr) and a.max_real and len(Xr) > a.max_real:
            keep = np.random.default_rng(a.seed).choice(
                len(Xr), a.max_real, replace=False)
            Xr, Yr = Xr[keep], Yr[keep]
            print(f"[data] capped real training patches at {a.max_real} "
                  f"(--max-real)")
        if len(Xr):
            rep = max(1, int(0.25 * len(Xtr) / len(Xr)))
            # [Xr]*rep rather than np.tile: concatenate reads the same buffer
            # rep times instead of materialising a second full copy first,
            # which on this dataset is half a gigabyte of peak memory.
            Xtr = np.concatenate([Xtr] + [Xr] * rep)
            Ytr = np.concatenate([Ytr] + [Yr] * rep)
            Ktr += ["real"] * (len(Xr) * rep)
            print(f"[data] + {len(Xr)} real patches, oversampled x{rep}")
            del Xr, Yr
            gc.collect()
        if rva:
            Xrv, Yrv = ds.load_pairs(rva, verbose=False)
            # held-out REAL scenes go into validation, labelled by oil presence
            Krv = ["spill" if (m == OIL).mean() > 0.005 else "clean" for m in Yrv]
            real_hold = (Xrv, Yrv, Krv, len({p["scene"] for p in rva}))
            Xva = np.concatenate([Xva, Xrv])
            Yva = np.concatenate([Yva, Yrv])
            Kva = Kva + Krv
            print(f"[data] + {len(Xrv)} real patches held out for validation "
                  f"from {len({p['scene'] for p in rva})} unseen scene(s)")
    else:
        print("[data] no labelled real data found -- synthetic only.")
        print("       Put images in data/real/images/ and masks in "
              "data/real/masks/ to train on real SAR.")

    # Inputs came from 8-bit PNGs, so float16 holds them exactly as well as
    # float32 does while halving resident memory; every consumer already
    # casts per batch. This is what keeps a 5k-patch run inside 8 GB.
    Xtr = Xtr.astype(np.float16, copy=False)
    Xva = Xva.astype(np.float16, copy=False)
    gc.collect()
    print(f"[data] arrays resident: "
          f"{(Xtr.nbytes + Xva.nbytes + Ytr.nbytes + Yva.nbytes)/1e6:.0f} MB "
          f"({len(Xtr)} train / {len(Xva)} val patches)")

    # inverse-frequency class weights (Oil/Ship are rare)
    counts = np.bincount(Ytr.reshape(-1).astype(int), minlength=N_CLASSES).astype(float)
    cw = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    cw = cw / cw.mean()
    print("[data] class weights:", {CLASS_NAMES[c]: round(float(cw[c]), 2)
                                    for c in range(N_CLASSES)})

    net = UNet(base=a.base, seed=a.seed + 1)
    opt = Adam(net.params(), lr=a.lr, wd=1e-5)
    rng = np.random.default_rng(a.seed + 2)
    nsteps = a.epochs * (len(Xtr) // a.batch)
    step, best_acc = 0, -1.0
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    wpath = os.path.join(ROOT, "models", "unet_oilspill.npz")

    print(f"[train] {sum(p.size for p in net.params()):,} params, "
          f"{nsteps} steps, batch {a.batch}")
    for ep in range(1, a.epochs + 1):
        run, nb = 0.0, 0
        for x, y in batches(Xtr, Ytr, a.batch, rng):
            opt.lr = a.lr * 0.5 * (1 + np.cos(np.pi * step / nsteps))   # cosine decay
            logits = net.forward(x, True)
            loss, dz = softmax_ce(logits, y, cw)
            net.backward(dz)
            opt.step(net.grads())
            run += loss; nb += 1; step += 1
        if ep % 2 == 0 or ep == a.epochs:
            iou, acc, _, m = evaluate(net, Xva, Yva, Kva)
            print(f"  epoch {ep:3d}/{a.epochs}  loss {run/nb:.4f}  "
                  f"IoU(Oil) {iou['Oil']:.3f}  IoU(Look) {iou['Look-alike']:.3f}  "
                  f"P {m['precision']*100:4.1f}% R {m['recall']*100:4.1f}% "
                  f"F1 {m['f1']*100:4.1f}%  [{time.time()-t0:5.0f}s]")
            if m["f1"] > best_acc:
                best_acc = m["f1"]
                net.save(wpath, {"base": a.base, "size": SIZE})
        else:
            print(f"  epoch {ep:3d}/{a.epochs}  loss {run/nb:.4f}  "
                  f"[{time.time()-t0:5.0f}s]")

    net.load(wpath)
    print("\n[calibrate] grid-searching decision thresholds ...")
    cfg, cal_f1, (cal_p, cal_r) = calibrate(net, Xva, Yva, Kva)
    save_config(cfg)
    print("  ", {k: round(v, 4) for k, v in cfg.items()})
    print(f"   -> F1 {cal_f1*100:.1f}%  (precision {cal_p*100:.1f}%, "
          f"recall {cal_r*100:.1f}%)")

    print("\n[final] held-out evaluation")
    Xt, Yt, Kt = build_dataset(200, SIZE, seed=4242)
    iou, acc, conf, m = evaluate(net, Xt, Yt, Kt)
    for c in CLASS_NAMES:
        print(f"    IoU {c:11s} {iou[c]:.3f}")
    print(f"    precision {m['precision']*100:5.1f}%  "
          f"(false-positive rate {100-m['precision']*100:.1f}%)")
    print(f"    recall    {m['recall']*100:5.1f}%")
    print(f"    F1        {m['f1']*100:5.1f}%   on 200 unseen scenes")
    print(f"    confusion (truth, predicted): {conf}")
    print("    NOTE: synthetic scenes. Not a claim about real SAR imagery.")

    # The number that is actually quotable: unseen Sentinel-1 acquisitions,
    # scored on their own. Thresholds were calibrated on a set that includes
    # these patches, so this is an optimistic read of a held-out scene split,
    # not a clean test set -- stated here rather than glossed over.
    real_block = None
    if real_hold is not None:
        Xrv, Yrv, Krv, n_scene = real_hold
        riou, racc, rconf, rm = evaluate(net, Xrv, Yrv, Krv)
        print(f"\n[final] held-out REAL Sentinel-1 ({n_scene} unseen scenes, "
              f"{len(Xrv)} patches)")
        for c in ("Sea", "Oil", "Look-alike"):
            print(f"    IoU {c:11s} {riou[c]:.3f}")
        print(f"    precision {rm['precision']*100:5.1f}%  "
              f"(false-positive rate {100-rm['precision']*100:.1f}%)")
        print(f"    recall    {rm['recall']*100:5.1f}%")
        print(f"    F1        {rm['f1']*100:5.1f}%")
        print("    Ship and Land are unsupervised in this dataset -- ignore "
              "their IoU.")
        real_block = {
            "iou": riou, "spill_accuracy": racc,
            "precision": rm["precision"], "recall": rm["recall"],
            "f1": rm["f1"], "scenes": int(n_scene), "patches": int(len(Xrv)),
            "confusion": {f"{k[0]}->{k[1]}": v for k, v in rconf.items()},
            "caveat": "held-out scenes from the same archive; thresholds "
                      "calibrated on a set including these patches",
        }

    with open(os.path.join(ROOT, "models", "metrics.json"), "w") as fh:
        json.dump({"iou": iou, "spill_accuracy": acc,
                   "precision": m["precision"], "recall": m["recall"],
                   "f1": m["f1"], "evaluated_on": "synthetic",
                   "confusion": {f"{k[0]}->{k[1]}": v for k, v in conf.items()},
                   "real_holdout": real_block,
                   "params": int(sum(p.size for p in net.params())),
                   "trained_on": int(len(Xtr)), "epochs": a.epochs,
                   "decision": cfg}, fh, indent=2)
    print(f"\n[done] weights -> models/unet_oilspill.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
