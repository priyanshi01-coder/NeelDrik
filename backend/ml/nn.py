"""Minimal CNN layers with hand-written backprop, NumPy only.

The project's stated stack is PyTorch.  PyTorch is unavailable in some
environments (no wheel index), so the identical U-Net is provided twice:

    nn.py + unet.py        NumPy implementation -- always runs, ships trained
                           weights, and is what the API uses by default.
    torch_unet.py          The same architecture in PyTorch; `train.py --torch`
                           trains it and weights convert between the two.

Every backward pass here is verified against numerical gradients by
`python -m backend.ml.nn` (and by tests/test_gradients.py).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# im2col helpers
# --------------------------------------------------------------------------- #
def im2col(x, k, stride, pad):
    """(N,C,H,W) -> (N, C*k*k, OH*OW) patch matrix."""
    n, c, h, w = x.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    s = x.strides
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, c, k, k, oh, ow),
        strides=(s[0], s[1], s[2], s[3], s[2] * stride, s[3] * stride),
        writeable=False)
    return np.ascontiguousarray(view).reshape(n, c * k * k, oh * ow), oh, ow


def col2im(cols, xshape, k, stride, pad, oh, ow):
    """Adjoint of im2col: scatter-add patch gradients back to image shape."""
    n, c, h, w = xshape
    hp, wp = h + 2 * pad, w + 2 * pad
    out = np.zeros((n, c, hp, wp), np.float32)
    cols = cols.reshape(n, c, k, k, oh, ow)
    for i in range(k):
        for j in range(k):
            out[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride] += \
                cols[:, :, i, j]
    return out[:, :, pad:pad + h, pad:pad + w] if pad else out


# --------------------------------------------------------------------------- #
# layers
# --------------------------------------------------------------------------- #
class Layer:
    def params(self):
        return []

    def grads(self):
        return []


class Conv2D(Layer):
    def __init__(self, cin, cout, k=3, stride=1, pad=None, rng=None, bias=True):
        rng = rng or np.random.default_rng(0)
        self.k, self.stride = k, stride
        self.pad = (k // 2) if pad is None else pad
        fan_in = cin * k * k
        self.W = (rng.standard_normal((cout, cin, k, k)) *
                  np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.b = np.zeros(cout, np.float32) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None

    def forward(self, x, training=True):
        self.xshape = x.shape
        cols, oh, ow = im2col(x, self.k, self.stride, self.pad)
        self.cols, self.oh, self.ow = cols, oh, ow
        n = x.shape[0]
        Wr = self.W.reshape(self.W.shape[0], -1)                 # (F, C*k*k)
        out = np.einsum("fp,npq->nfq", Wr, cols, optimize=True)
        if self.b is not None:
            out += self.b[None, :, None]
        return out.reshape(n, self.W.shape[0], oh, ow)

    def backward(self, g):
        n, f, oh, ow = g.shape
        gf = g.reshape(n, f, oh * ow)
        self.dW[...] = np.einsum("nfq,npq->fp", gf, self.cols,
                                 optimize=True).reshape(self.W.shape)
        if self.b is not None:
            self.db[...] = gf.sum(axis=(0, 2))
        Wr = self.W.reshape(f, -1)
        gcols = np.einsum("fp,nfq->npq", Wr, gf, optimize=True)
        return col2im(gcols, self.xshape, self.k, self.stride, self.pad, oh, ow)

    def params(self):
        return [self.W] + ([self.b] if self.b is not None else [])

    def grads(self):
        return [self.dW] + ([self.db] if self.b is not None else [])


class BatchNorm2D(Layer):
    def __init__(self, c, mom=0.9, eps=1e-5):
        self.g = np.ones(c, np.float32)
        self.b = np.zeros(c, np.float32)
        self.dg = np.zeros_like(self.g)
        self.db = np.zeros_like(self.b)
        self.rm = np.zeros(c, np.float32)
        self.rv = np.ones(c, np.float32)
        self.mom, self.eps = mom, eps

    def forward(self, x, training=True):
        if training:
            mu = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            self.rm = self.mom * self.rm + (1 - self.mom) * mu
            self.rv = self.mom * self.rv + (1 - self.mom) * var
        else:
            mu, var = self.rm, self.rv
        self.istd = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu[None, :, None, None]) * self.istd[None, :, None, None]
        self.training = training
        return self.g[None, :, None, None] * self.xhat + self.b[None, :, None, None]

    def backward(self, gy):
        n = gy.shape[0] * gy.shape[2] * gy.shape[3]
        self.dg[...] = (gy * self.xhat).sum(axis=(0, 2, 3))
        self.db[...] = gy.sum(axis=(0, 2, 3))
        gxhat = gy * self.g[None, :, None, None]
        gx = (self.istd[None, :, None, None] / n) * (
            n * gxhat
            - gxhat.sum(axis=(0, 2, 3))[None, :, None, None]
            - self.xhat * (gxhat * self.xhat).sum(axis=(0, 2, 3))[None, :, None, None])
        return gx

    def params(self):
        return [self.g, self.b]

    def grads(self):
        return [self.dg, self.db]


class ReLU(Layer):
    def forward(self, x, training=True):
        self.m = x > 0
        return x * self.m

    def backward(self, g):
        return g * self.m


class MaxPool2(Layer):
    def forward(self, x, training=True):
        n, c, h, w = x.shape
        self.xshape = x.shape
        self.ph, self.pw = h // 2, w // 2
        v = x[:, :, :self.ph * 2, :self.pw * 2].reshape(n, c, self.ph, 2, self.pw, 2)
        v = v.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, self.ph, self.pw, 4)
        self.arg = v.argmax(-1)
        return v.max(-1)

    def backward(self, g):
        n, c, ph, pw = g.shape
        flat = np.zeros((n, c, ph, pw, 4), np.float32)
        idx = np.indices((n, c, ph, pw))
        flat[idx[0], idx[1], idx[2], idx[3], self.arg] = g
        out = flat.reshape(n, c, ph, pw, 2, 2).transpose(0, 1, 2, 4, 3, 5)
        out = out.reshape(n, c, ph * 2, pw * 2)
        full = np.zeros(self.xshape, np.float32)
        full[:, :, :ph * 2, :pw * 2] = out
        return full


class Upsample2(Layer):
    """Nearest-neighbour x2."""

    def forward(self, x, training=True):
        self.xshape = x.shape
        return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)

    def backward(self, g):
        n, c, h, w = g.shape
        return g.reshape(n, c, h // 2, 2, w // 2, 2).sum(axis=(3, 5))


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def softmax_ce(logits, target, class_w=None):
    """logits (N,K,H,W), target (N,H,W) int -> (loss, dlogits)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n, k, h, w = logits.shape
    oh = np.zeros_like(p)
    nn_, hh, ww = np.indices((n, h, w))
    oh[nn_, target.astype(int), hh, ww] = 1.0
    wmap = np.ones((n, h, w), np.float32) if class_w is None \
        else np.asarray(class_w, np.float32)[target.astype(int)]
    loss = -(wmap * np.log(np.clip((p * oh).sum(axis=1), 1e-9, None))).sum() / wmap.sum()
    dz = (p - oh) * wmap[:, None] / wmap.sum()
    return float(loss), dz.astype(np.float32)


# --------------------------------------------------------------------------- #
# optimiser
# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(self.p, grads)):
            if self.wd:
                g = g + self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            p -= self.lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + self.eps)


# --------------------------------------------------------------------------- #
# gradient check
# --------------------------------------------------------------------------- #
def _numeric_grad(f, x, eps=1e-4):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; a = f()
        x[i] = old - eps; b = f()
        x[i] = old
        g[i] = (a - b) / (2 * eps)
        it.iternext()
    return g


def gradient_check(verbose=True):
    """Analytic vs numerical gradients for every layer. Returns max rel error."""
    rng = np.random.default_rng(3)
    worst = 0.0
    tests = [
        ("Conv2D s1", Conv2D(2, 3, 3, 1, rng=rng), (2, 2, 7, 7)),
        ("Conv2D s2", Conv2D(2, 3, 3, 2, rng=rng), (2, 2, 8, 8)),
        ("BatchNorm", BatchNorm2D(3), (2, 3, 5, 5)),
        ("ReLU", ReLU(), (2, 3, 5, 5)),
        ("MaxPool2", MaxPool2(), (2, 3, 6, 6)),
        ("Upsample2", Upsample2(), (2, 3, 4, 4)),
    ]
    for name, layer, shape in tests:
        x = rng.standard_normal(shape).astype(np.float64) * 1.5
        wts = rng.standard_normal(layer.forward(x.copy()).shape)

        def loss_of(inp):
            return float((layer.forward(inp, True) * wts).sum())

        y = layer.forward(x, True)
        gx = layer.backward(wts.astype(np.float32))
        gn = _numeric_grad(lambda: loss_of(x), x)
        err = np.abs(gx - gn).max() / (np.abs(gn).max() + 1e-8)
        worst = max(worst, err)
        if verbose:
            print(f"  {name:12s} d/dx  rel-err {err:.2e}")

        for pi, (p, dp) in enumerate(zip(layer.params(), layer.grads())):
            p64 = p.astype(np.float64)

            def loss_p():
                return float((layer.forward(x, True) * wts).sum())

            layer.forward(x, True)
            layer.backward(wts.astype(np.float32))
            analytic = dp.copy()
            num = _numeric_grad(loss_p, p)
            e2 = np.abs(analytic - num).max() / (np.abs(num).max() + 1e-8)
            worst = max(worst, e2)
            if verbose:
                print(f"  {name:12s} dparam{pi} rel-err {e2:.2e}")

    # loss gradient
    lg = rng.standard_normal((2, 5, 4, 4))
    tg = rng.integers(0, 5, (2, 4, 4))

    def lf():
        return softmax_ce(lg, tg)[0]

    _, dl = softmax_ce(lg, tg)
    dn = _numeric_grad(lf, lg)
    e3 = np.abs(dl - dn).max() / (np.abs(dn).max() + 1e-8)
    worst = max(worst, e3)
    if verbose:
        print(f"  {'softmax_ce':12s} dlogits rel-err {e3:.2e}")
        print(f"\n  worst relative error: {worst:.2e} "
              f"-> {'PASS' if worst < 2e-4 else 'FAIL'}")
    return worst


if __name__ == "__main__":
    gradient_check()
