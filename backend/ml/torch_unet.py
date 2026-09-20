"""PyTorch twin of the NumPy U-Net -- the project's stated AI stack.

Same architecture, same tensor layout, same weight file.  If torch is
installed, `predict.py` runs this automatically; if not, the NumPy
implementation runs and results are identical (verified by
tests/test_parity.py when torch is present).

    python train.py --torch        # train this one instead
"""
from __future__ import annotations

import gc
import os

import numpy as np

try:
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:                       # pragma: no cover
    HAVE_TORCH = False
    tnn = object                        # type: ignore

N_CLASSES = 5
IN_CH = 2


if HAVE_TORCH:

    class ConvBlock(tnn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.c1 = tnn.Conv2d(cin, cout, 3, 1, 1)
            self.n1 = tnn.BatchNorm2d(cout, eps=1e-5, momentum=0.1)
            self.c2 = tnn.Conv2d(cout, cout, 3, 1, 1)
            self.n2 = tnn.BatchNorm2d(cout, eps=1e-5, momentum=0.1)

        def forward(self, x):
            x = F.relu(self.n1(self.c1(x)))
            return F.relu(self.n2(self.c2(x)))

    class TorchUNet(tnn.Module):
        """Mirror of backend.ml.unet.UNet."""

        def __init__(self, base=8):
            super().__init__()
            b = base
            self.stem_c = tnn.Conv2d(IN_CH, b, 3, 2, 1)
            self.stem_n = tnn.BatchNorm2d(b, eps=1e-5, momentum=0.1)
            self.enc1 = ConvBlock(b, b * 2)
            self.enc2 = ConvBlock(b * 2, b * 4)
            self.bott = ConvBlock(b * 4, b * 6)
            self.dec2 = ConvBlock(b * 6 + b * 4, b * 4)
            self.dec1 = ConvBlock(b * 4 + b * 2, b * 2)
            self.head = tnn.Conv2d(b * 2, N_CLASSES, 1, 1, 0)

        def forward(self, x):
            x = F.relu(self.stem_n(self.stem_c(x)))
            s1 = self.enc1(x)
            s2 = self.enc2(F.max_pool2d(s1, 2))
            x = self.bott(F.max_pool2d(s2, 2))
            x = self.dec2(torch.cat([F.interpolate(x, scale_factor=2,
                                                   mode="nearest"), s2], 1))
            x = self.dec1(torch.cat([F.interpolate(x, scale_factor=2,
                                                   mode="nearest"), s1], 1))
            return self.head(x)

    # ------------------------------------------------------------------ #
    # weight interchange with the NumPy model
    # ------------------------------------------------------------------ #
    def _ordered_modules(m: "TorchUNet"):
        """Same order as UNet._all_layers(): leaves first, then blocks."""
        out = [("conv", m.stem_c), ("bn", m.stem_n), ("conv", m.head)]
        for blk in (m.enc1, m.enc2, m.bott, m.dec2, m.dec1):
            out += [("conv", blk.c1), ("bn", blk.n1), ("relu", None),
                    ("conv", blk.c2), ("bn", blk.n2), ("relu", None)]
        return out

    def load_npz_into_torch(model: "TorchUNet", path: str):
        d = np.load(path, allow_pickle=True)
        mods = _ordered_modules(model)
        pi = 0
        with torch.no_grad():
            for li, (kind, mod) in enumerate(mods):
                if kind == "relu":
                    continue
                if kind == "conv":
                    mod.weight.copy_(torch.from_numpy(d[f"p{pi}"])); pi += 1
                    mod.bias.copy_(torch.from_numpy(d[f"p{pi}"])); pi += 1
                else:
                    mod.weight.copy_(torch.from_numpy(d[f"p{pi}"])); pi += 1
                    mod.bias.copy_(torch.from_numpy(d[f"p{pi}"])); pi += 1
                    mod.running_mean.copy_(torch.from_numpy(d[f"rm{li}"]))
                    mod.running_var.copy_(torch.from_numpy(d[f"rv{li}"]))
        return model

    def save_torch_as_npz(model: "TorchUNet", path: str):
        mods = _ordered_modules(model)
        out, pi = {}, 0
        for li, (kind, mod) in enumerate(mods):
            if kind == "relu":
                continue
            out[f"p{pi}"] = mod.weight.detach().cpu().numpy(); pi += 1
            out[f"p{pi}"] = mod.bias.detach().cpu().numpy(); pi += 1
            if kind == "bn":
                out[f"rm{li}"] = mod.running_mean.detach().cpu().numpy()
                out[f"rv{li}"] = mod.running_var.detach().cpu().numpy()
        out["_meta"] = np.array("{'base': 8, 'size': 128, 'src': 'torch'}", dtype=object)
        np.savez_compressed(path, **out)


class TorchUNetRunner:
    """Duck-typed stand-in for the NumPy UNet used by predict.py."""

    def __init__(self, weights_path, base=8):
        if not HAVE_TORCH:
            raise RuntimeError("torch not installed")
        if not os.path.exists(weights_path):
            raise RuntimeError("weights not found")
        self.m = TorchUNet(base)
        load_npz_into_torch(self.m, weights_path)
        self.m.eval()

    def predict_proba(self, x):
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(x, np.float32))
            return torch.softmax(self.m(t), dim=1).numpy()


# --------------------------------------------------------------------------- #
def train_torch(args):                  # pragma: no cover - needs torch
    """Train the PyTorch twin.

    This runs the SAME pipeline as the NumPy trainer -- identical augmentation,
    the same scene-split real data, the same F1-based threshold calibration and
    the same metrics file -- so `--torch` is a change of execution engine, not
    a second, weaker training script.  Weights are written in the shared npz
    format, so either engine can load the other's output.
    """
    if not HAVE_TORCH:
        raise SystemExit("PyTorch is not installed. `pip install torch` first, "
                         "or run `python train.py` to use the NumPy trainer.")
    import json
    import time

    from backend.ml.datagen import build_dataset
    from backend.ml.decide import save_config
    from backend.ml import dataset as ds
    from backend.ml.unet import CLASS_NAMES, OIL
    import train as T                     # shared augment / evaluate / calibrate

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    t0 = time.time()
    print(f"[data] generating {args.train_n} train / {args.val_n} val scenes ...")
    Xtr, Ytr, Ktr = build_dataset(args.train_n, 128, seed=args.seed)
    Xva, Yva, Kva = build_dataset(args.val_n, 128, seed=args.seed + 9999)

    # Kept aside so the final report can score real SAR on its own. Mixed
    # into a synthetic average, a real number is unreportable.
    real_hold = None

    pairs = ds.discover(root)
    if pairs:
        rtr, rva = ds.scene_split(pairs, val_frac=0.25, seed=args.seed)
        Xr, Yr = ds.load_pairs(rtr)
        if len(Xr) and args.max_real and len(Xr) > args.max_real:
            keep = np.random.default_rng(args.seed).choice(
                len(Xr), args.max_real, replace=False)
            Xr, Yr = Xr[keep], Yr[keep]
            print(f"[data] capped real training patches at {args.max_real} "
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
            Krv = ["spill" if (m == OIL).mean() > 0.005 else "clean"
                   for m in Yrv]
            real_hold = (Xrv, Yrv, Krv, len({p["scene"] for p in rva}))
            Kva = Kva + Krv
            Xva = np.concatenate([Xva, Xrv])
            Yva = np.concatenate([Yva, Yrv])
            print(f"[data] + {len(Xrv)} real patches held out for validation "
                  f"from {real_hold[3]} unseen scene(s)")
    else:
        print("[data] no labelled real data found -- synthetic only.")

    # Inputs came from 8-bit PNGs, so float16 holds them exactly as well as
    # float32 does while halving resident memory; every consumer already
    # casts per batch. This is what keeps a 5k-patch run inside 8 GB.
    Xtr = Xtr.astype(np.float16, copy=False)
    Xva = Xva.astype(np.float16, copy=False)
    Ytr = Ytr.astype(np.int8, copy=False)
    Yva = Yva.astype(np.int8, copy=False)
    gc.collect()
    mb = (Xtr.nbytes + Xva.nbytes + Ytr.nbytes + Yva.nbytes) / 1e6
    print(f"[data] arrays resident: {mb:.0f} MB "
          f"({len(Xtr)} train / {len(Xva)} val patches)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] PyTorch {torch.__version__} on {dev}")
    model = TorchUNet(args.base).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = max(1, len(Xtr) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)

    counts = np.bincount(Ytr.reshape(-1).astype(int), minlength=N_CLASSES).astype(float)
    cw = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    cw = torch.tensor(cw / cw.mean(), dtype=torch.float32, device=dev)
    lossf = tnn.CrossEntropyLoss(weight=cw)

    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    wpath = os.path.join(root, "models", "unet_oilspill.npz")
    rng = np.random.default_rng(args.seed + 2)
    best_f1 = -1.0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {n_params:,} params, {args.epochs * steps_per_epoch} steps")

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(Xtr))
        run, nb = 0.0, 0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            j = perm[i:i + args.batch]
            # identical augmentation to the NumPy trainer
            xb, yb = T.augment(Xtr[j].astype(np.float32),
                               Ytr[j][:, ::2, ::2].astype(np.int64), rng)
            x = torch.tensor(xb, dtype=torch.float32, device=dev)
            y = torch.tensor(yb, device=dev)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step(); sched.step()
            run += loss.detach().item(); nb += 1   # detach: this is a report,
            #                                        not part of the graph

        if ep % 2 == 0 or ep == args.epochs:
            model.eval()
            runner = _RunnerFor(model)
            iou, acc, _, m = T.evaluate(runner, Xva, Yva, Kva)
            print(f"  epoch {ep:3d}/{args.epochs}  loss {run/max(nb,1):.4f}  "
                  f"IoU(Oil) {iou['Oil']:.3f}  P {m['precision']*100:4.1f}% "
                  f"R {m['recall']*100:4.1f}% F1 {m['f1']*100:4.1f}%  "
                  f"[{time.time()-t0:5.0f}s]")
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                save_torch_as_npz(model, wpath)
        else:
            print(f"  epoch {ep:3d}/{args.epochs}  loss {run/max(nb,1):.4f}  "
                  f"[{time.time()-t0:5.0f}s]")

    load_npz_into_torch(model, wpath)       # best checkpoint
    model.eval()
    runner = _RunnerFor(model)

    print("\n[calibrate] grid-searching decision thresholds (F1) ...")
    cfg, cal_f1, (cal_p, cal_r) = T.calibrate(runner, Xva, Yva, Kva)
    save_config(cfg)
    print("  ", {k: round(v, 4) for k, v in cfg.items()})
    print(f"   -> F1 {cal_f1*100:.1f}%  (precision {cal_p*100:.1f}%, "
          f"recall {cal_r*100:.1f}%)")

    print("\n[final] held-out evaluation")
    Xt, Yt, Kt = build_dataset(200, 128, seed=4242)
    iou, acc, conf, m = T.evaluate(runner, Xt, Yt, Kt)
    for c in CLASS_NAMES:
        print(f"    IoU {c:11s} {iou[c]:.3f}")
    print(f"    precision {m['precision']*100:5.1f}%  "
          f"(false-positive rate {100-m['precision']*100:.1f}%)")
    print(f"    recall    {m['recall']*100:5.1f}%")
    print(f"    F1        {m['f1']*100:5.1f}%   on 200 unseen scenes")
    print("    NOTE: synthetic scenes. Not a claim about real SAR imagery.")

    # The number that is actually quotable: unseen Sentinel-1 acquisitions,
    # scored on their own. Thresholds were calibrated on a set that includes
    # these patches, so this is an optimistic read of a held-out scene split,
    # not a clean test set -- stated here rather than glossed over.
    real_block = None
    if real_hold is not None:
        Xrv, Yrv, Krv, n_scene = real_hold
        riou, racc, rconf, rm = T.evaluate(runner, Xrv, Yrv, Krv)
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

    with open(os.path.join(root, "models", "metrics.json"), "w") as fh:
        json.dump({"iou": iou, "spill_accuracy": acc,
                   "precision": m["precision"], "recall": m["recall"],
                   "f1": m["f1"], "evaluated_on": "synthetic",
                   "confusion": {f"{k[0]}->{k[1]}": v for k, v in conf.items()},
                   "real_holdout": real_block,
                   "params": int(n_params), "trained_on": int(len(Xtr)),
                   "epochs": args.epochs, "runtime": "pytorch",
                   "torch_version": torch.__version__,
                   "decision": cfg}, fh, indent=2)
    print(f"\n[done] weights -> models/unet_oilspill.npz (PyTorch, "
          f"{time.time()-t0:.0f}s)")


class _RunnerFor:
    """Wrap a live torch model in the predict_proba interface the shared
    evaluate/calibrate helpers expect."""

    def __init__(self, model):
        self.m = model

    def predict_proba(self, x):
        with torch.no_grad():
            dev = next(self.m.parameters()).device
            t = torch.tensor(np.asarray(x, np.float32), device=dev)
            return torch.softmax(self.m(t), dim=1).cpu().numpy()
