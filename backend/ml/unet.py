"""5-class U-Net for SAR oil-spill segmentation (Sea / Oil / Look-alike / Ship / Land).

Encoder-decoder with skip connections, exactly as described in the NEELDRIK
flow document.  A stride-2 stem keeps the internal resolution at H/2 so the
network trains on CPU in minutes; probability maps are resized back to the
input resolution at inference.

Input   (N, 2, 128, 128)  float32 -- [despeckled intensity, local CoV]
Output  (N, 5,  64,  64)  logits
"""
from __future__ import annotations

import numpy as np

from .nn import (Adam, BatchNorm2D, Conv2D, MaxPool2, ReLU, Upsample2,
                 softmax_ce)

N_CLASSES = 5
IN_CH = 2
CLASS_NAMES = ["Sea", "Oil", "Look-alike", "Ship", "Land"]
SEA, OIL, LOOKALIKE, SHIP, LAND = range(5)


class ConvBlock:
    """(Conv -> BN -> ReLU) x 2"""

    def __init__(self, cin, cout, rng):
        self.c1, self.n1, self.r1 = Conv2D(cin, cout, 3, 1, rng=rng), BatchNorm2D(cout), ReLU()
        self.c2, self.n2, self.r2 = Conv2D(cout, cout, 3, 1, rng=rng), BatchNorm2D(cout), ReLU()
        self.layers = [self.c1, self.n1, self.r1, self.c2, self.n2, self.r2]

    def forward(self, x, training=True):
        for l in self.layers:
            x = l.forward(x, training)
        return x

    def backward(self, g):
        for l in reversed(self.layers):
            g = l.backward(g)
        return g


class UNet:
    def __init__(self, base=8, seed=0):
        rng = np.random.default_rng(seed)
        b = base
        self.stem_c = Conv2D(IN_CH, b, 3, 2, rng=rng)
        self.stem_n, self.stem_r = BatchNorm2D(b), ReLU()

        self.enc1 = ConvBlock(b, b * 2, rng)          # 64x64, 16ch
        self.pool1 = MaxPool2()
        self.enc2 = ConvBlock(b * 2, b * 4, rng)      # 32x32, 32ch
        self.pool2 = MaxPool2()
        self.bott = ConvBlock(b * 4, b * 6, rng)      # 16x16, 48ch

        self.up2 = Upsample2()
        self.dec2 = ConvBlock(b * 6 + b * 4, b * 4, rng)
        self.up1 = Upsample2()
        self.dec1 = ConvBlock(b * 4 + b * 2, b * 2, rng)
        self.head = Conv2D(b * 2, N_CLASSES, 1, 1, pad=0, rng=rng)

        self._blocks = [self.enc1, self.enc2, self.bott, self.dec2, self.dec1]
        self._leaves = [self.stem_c, self.stem_n, self.head]

    # ---------------------------------------------------------------- #
    def _all_layers(self):
        out = list(self._leaves)
        for bl in self._blocks:
            out += [l for l in bl.layers]
        return out

    def params(self):
        p = []
        for l in self._all_layers():
            p += l.params()
        return p

    def grads(self):
        g = []
        for l in self._all_layers():
            g += l.grads()
        return g

    # ---------------------------------------------------------------- #
    def forward(self, x, training=True):
        x = self.stem_r.forward(
            self.stem_n.forward(self.stem_c.forward(x, training), training), training)
        s1 = self.enc1.forward(x, training)                 # 64
        x = self.pool1.forward(s1, training)                # 32
        s2 = self.enc2.forward(x, training)                 # 32
        x = self.pool2.forward(s2, training)                # 16
        x = self.bott.forward(x, training)                  # 16

        x = self.up2.forward(x, training)                   # 32
        self._c2 = x.shape[1]
        x = self.dec2.forward(np.concatenate([x, s2], 1), training)
        x = self.up1.forward(x, training)                   # 64
        self._c1 = x.shape[1]
        x = self.dec1.forward(np.concatenate([x, s1], 1), training)
        return self.head.forward(x, training)               # (N,5,64,64)

    def backward(self, g):
        g = self.head.backward(g)
        g = self.dec1.backward(g)
        gup1, gs1 = g[:, :self._c1], g[:, self._c1:]
        g = self.up1.backward(gup1)
        g = self.dec2.backward(g)
        gup2, gs2 = g[:, :self._c2], g[:, self._c2:]
        g = self.up2.backward(gup2)
        g = self.bott.backward(g)
        g = self.pool2.backward(g) + 0.0
        g = self.enc2.backward(g + gs2)
        g = self.pool1.backward(g)
        g = self.enc1.backward(g + gs1)
        g = self.stem_c.backward(self.stem_n.backward(self.stem_r.backward(g)))
        return g

    # ---------------------------------------------------------------- #
    def predict_proba(self, x):
        """x (N,1,H,W) float32 0..1 -> softmax probabilities (N,5,H/2,W/2)."""
        z = self.forward(x, training=False)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---------------------------------------------------------------- #
    def save(self, path, meta=None):
        d = {f"p{i}": p for i, p in enumerate(self.params())}
        for i, l in enumerate(self._all_layers()):
            if isinstance(l, BatchNorm2D):
                d[f"rm{i}"], d[f"rv{i}"] = l.rm, l.rv
        d["_meta"] = np.array(str(meta or {}), dtype=object)
        np.savez_compressed(path, **d)

    def load(self, path):
        d = np.load(path, allow_pickle=True)
        for i, p in enumerate(self.params()):
            p[...] = d[f"p{i}"]
        for i, l in enumerate(self._all_layers()):
            if isinstance(l, BatchNorm2D):
                l.rm[...], l.rv[...] = d[f"rm{i}"], d[f"rv{i}"]
        return self


def overfit_check(steps=60, verbose=True):
    """Sanity: the net must be able to drive loss down on one tiny batch."""
    from .datagen import build_dataset
    X, Y, _ = build_dataset(4, seed=11)
    x = X.astype(np.float32)
    y = Y[:, ::2, ::2].astype(np.int64)
    net = UNet(base=8, seed=1)
    opt = Adam(net.params(), lr=4e-3)
    first = last = None
    for s in range(steps):
        logits = net.forward(x, True)
        loss, dz = softmax_ce(logits, y)
        net.backward(dz)
        opt.step(net.grads())
        if s == 0:
            first = loss
        last = loss
    if verbose:
        print(f"  overfit check: loss {first:.4f} -> {last:.4f} "
              f"({'PASS' if last < first * 0.5 else 'FAIL'})")
    return first, last


if __name__ == "__main__":
    net = UNet()
    import numpy as _np
    x = _np.zeros((1, IN_CH, 128, 128), _np.float32)
    print("  output shape:", net.forward(x, False).shape)
    print("  parameters:  ", sum(p.size for p in net.params()))
    overfit_check()
