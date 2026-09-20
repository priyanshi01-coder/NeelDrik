"""NumPy engine vs PyTorch engine: do they give the SAME answer?

The project ships two implementations of one network. That is only defensible
if they agree -- otherwise "we use PyTorch" and "we use NumPy" describe two
different systems with two different accuracies.

This test loads the same weight file into both, runs the same inputs through
both, and requires the probability maps to match to within float tolerance.
It also checks the full end-to-end verdict agrees, because a tiny numerical
difference either side of a decision threshold is exactly where a silent
divergence would hide.

Skips cleanly when PyTorch is not installed.

    python tests/test_parity.py
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "unet_oilspill.npz")

from backend.ml.torch_unet import HAVE_TORCH                    # noqa: E402

if not HAVE_TORCH:
    print("\nPyTorch is not installed here -- parity test SKIPPED.")
    print("Install it (`pip install torch`) and re-run to verify that the")
    print("PyTorch and NumPy engines agree.\n")
    sys.exit(0)

from backend.ml.datagen import make_scene                       # noqa: E402
from backend.ml.decide import analyse, load_config              # noqa: E402
from backend.ml.preprocess import preprocess                    # noqa: E402
from backend.ml.torch_unet import TorchUNetRunner               # noqa: E402
from backend.ml.unet import UNet                                # noqa: E402

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


def png_bytes(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


print("\nEngine parity: NumPy vs PyTorch, same weights")
if not os.path.exists(WEIGHTS):
    print("  models/unet_oilspill.npz missing -- run `python train.py` first")
    sys.exit(1)

np_net = UNet(base=8)
np_net.load(WEIGHTS)
pt_net = TorchUNetRunner(WEIGHTS, base=8)

rng = np.random.default_rng(31337)
cfg = load_config()

max_abs = 0.0
verdict_mismatch = []
for i in range(12):
    kind = ["spill", "lookalike", "clean"][i % 3]
    img, _ = make_scene(rng, 128, kind)
    pre = preprocess(png_bytes(img))
    x = pre["model_input"]

    a = np_net.predict_proba(x)
    b = pt_net.predict_proba(x)

    if a.shape != b.shape:
        check(f"scene {i}: output shapes match", False, f"{a.shape} vs {b.shape}")
        continue
    max_abs = max(max_abs, float(np.abs(a - b).max()))

    va = analyse(a[0], cfg)["verdict"]
    vb = analyse(b[0], cfg)["verdict"]
    if va != vb:
        verdict_mismatch.append((i, kind, va, vb))

check("probability maps agree to 1e-4", max_abs < 1e-4,
      f"max abs difference {max_abs:.2e}")
check("every end-to-end verdict agrees", not verdict_mismatch,
      "" if not verdict_mismatch else f"mismatches: {verdict_mismatch}")

# argmax class maps must be identical, not merely close
img, _ = make_scene(rng, 128, "spill")
x = preprocess(png_bytes(img))["model_input"]
same = (np_net.predict_proba(x).argmax(1) == pt_net.predict_proba(x).argmax(1))
check("segmentation masks are pixel-identical", bool(same.all()),
      f"{same.mean()*100:.3f}% of pixels agree")

print(f"\n{'='*58}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  failed:", ", ".join(FAIL))
    print("\n  The two engines disagree. Do NOT quote one engine's accuracy")
    print("  while running the other -- fix the divergence first.")
sys.exit(1 if FAIL else 0)
