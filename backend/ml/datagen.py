"""Physics-based synthetic Sentinel-1 SAR patch generator for NEELDRIK.

Why synthetic: the discriminating physics of SAR oil-spill detection is well
defined, so it can be simulated faithfully and used to teach the network the
right cue instead of a dataset artefact.

    Sea        Bragg scattering off wind-driven capillary waves -> moderate
               backscatter carrying strong multiplicative speckle.
    Oil        A surfactant film damps those capillary waves, so backscatter
               collapses.  The signature is a DARK patch that is also SMOOTH
               (damped speckle -> low internal variance) with a COHERENT,
               fairly sharp boundary, usually elongated along drift.
    Look-alike Low-wind cells, biogenic slicks and rain cells are also dark,
               but they keep more internal texture and their boundary is
               DIFFUSE.  Boundary gradient + internal variance is what
               separates them from oil -- this is the hard, decisive case.
    Ship       Corner reflection -> very bright, very small, often a wake.
    Land       Bright and strongly textured, attached to an image edge.

Speckle is modelled as Gamma(L, 1/L) multiplicative noise, L = number of looks.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

SEA, OIL, LOOKALIKE, SHIP, LAND = 0, 1, 2, 3, 4
N_CLASSES = 5
CLASS_NAMES = ["Sea", "Oil", "Look-alike", "Ship", "Land"]


# --------------------------------------------------------------------------- #
# shape helpers
# --------------------------------------------------------------------------- #
def _blob(rng, size, scale, coverage, elong=1.0, angle=None):
    """Organic binary blob from low-pass filtered noise."""
    n = rng.standard_normal((size, size))
    sy = max(scale / max(elong, 1e-3), 1.0)
    sx = max(scale * max(elong, 1e-3), 1.0)
    n = ndimage.gaussian_filter(n, (sy, sx))
    if angle is not None:
        n = ndimage.rotate(n, angle, reshape=False, order=1, mode="nearest")
    n = (n - n.mean()) / (n.std() + 1e-8)
    thr = np.quantile(n, 1.0 - coverage)
    return n >= thr


def _largest_component(mask):
    lab, k = ndimage.label(mask)
    if k == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, k + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _soft(mask, sigma):
    """Binary mask -> soft 0..1 alpha with a controllable edge width."""
    if sigma <= 0.05:
        return mask.astype(np.float32)
    return np.clip(ndimage.gaussian_filter(mask.astype(np.float32), sigma), 0, 1)


# --------------------------------------------------------------------------- #
# scene synthesis
# --------------------------------------------------------------------------- #
def _speckle(rng, shape, looks):
    """Multiplicative Gamma speckle, unit mean."""
    return rng.gamma(shape=looks, scale=1.0 / looks, size=shape).astype(np.float32)


def make_scene(rng, size=128, kind=None):
    """Return (image uint8 HxW, mask int8 HxW).

    kind: 'spill' | 'lookalike' | 'clean' | None (random)
    """
    if kind is None:
        kind = rng.choice(["spill", "lookalike", "clean"], p=[0.45, 0.30, 0.25])

    mask = np.full((size, size), SEA, np.int8)
    S = size / 128.0          # all length scales are defined at 128 px

    # --- sea: wind-dependent mean backscatter + large-scale wind streaks ---
    wind = rng.uniform(0.35, 1.0)                       # normalised wind speed
    sea_level = 0.22 + 0.55 * wind
    streaks = ndimage.gaussian_filter(
        rng.standard_normal((size, size)), rng.uniform(8, 22) * S)
    streaks /= (np.abs(streaks).max() + 1e-8)
    sigma0 = sea_level * (1.0 + rng.uniform(0.05, 0.22) * streaks)

    # --- land (sometimes), anchored to one edge ---
    if rng.random() < 0.22:
        land = _blob(rng, size, rng.uniform(10, 20) * S, rng.uniform(0.10, 0.30))
        edge = np.zeros((size, size), bool)
        w = rng.integers(size // 5, size // 2)
        side = rng.integers(4)
        if side == 0:   edge[:w, :] = True
        elif side == 1: edge[-w:, :] = True
        elif side == 2: edge[:, :w] = True
        else:           edge[:, -w:] = True
        land = land & edge
        if land.sum() > 40:
            tex = ndimage.gaussian_filter(rng.standard_normal((size, size)), 1.2 * S)
            sigma0 = np.where(land, rng.uniform(1.05, 1.55) * (1 + 0.40 * tex), sigma0)
            mask[land] = LAND

    water = mask == SEA

    # --- look-alikes: dark, DIFFUSE edge, texture only partly damped ---
    if kind in ("lookalike", "spill") or rng.random() < 0.25:
        n_la = rng.integers(1, 3) if kind == "lookalike" else rng.integers(0, 2)
        for _ in range(int(n_la)):
            b = _blob(rng, size, rng.uniform(12, 26) * S,
                      rng.uniform(0.06, 0.20), elong=rng.uniform(0.8, 1.3))
            b = _largest_component(b) & water
            if b.sum() < 60 * S * S:
                continue
            alpha = _soft(b, rng.uniform(3.5, 7.0) * S)      # diffuse boundary
            damp = rng.uniform(0.35, 0.62)               # partial damping
            sigma0 = sigma0 * (1.0 - alpha * damp)
            mask[(alpha > 0.55) & water] = LOOKALIKE

    # --- oil: dark, SMOOTH, COHERENT, elongated along drift ---
    if kind == "spill":
        for _ in range(int(rng.integers(1, 3))):
            b = _blob(rng, size, rng.uniform(9, 20) * S, rng.uniform(0.05, 0.16),
                      elong=rng.uniform(1.6, 3.0), angle=rng.uniform(0, 180))
            b = _largest_component(b) & (mask == SEA)
            if b.sum() < 70 * S * S:
                continue
            alpha = _soft(b, rng.uniform(0.6, 1.8) * S)      # sharp boundary
            damp = rng.uniform(0.72, 0.93)               # strong damping
            sigma0 = sigma0 * (1.0 - alpha * damp)
            mask[(alpha > 0.5) & (mask == SEA)] = OIL

    # --- speckle (damped inside slicks: smooth surface -> fewer scatterers) ---
    looks = rng.uniform(1.6, 6.0)
    sp = _speckle(rng, (size, size), looks)
    calm = np.isin(mask, (OIL, LOOKALIKE))
    if calm.any():
        # oil damps speckle far more than a look-alike does
        pull = np.where(mask == OIL, rng.uniform(0.55, 0.80),
                        rng.uniform(0.15, 0.35)).astype(np.float32)
        sp = np.where(calm, 1.0 + (sp - 1.0) * (1.0 - pull), sp)
    img = sigma0 * sp

    # --- ships: bright point targets, optional wake ---
    for _ in range(int(rng.integers(0, 4))):
        yy, xx = rng.integers(int(4*S), size - int(4*S)), rng.integers(int(4*S), size - int(4*S))
        if mask[yy, xx] == LAND:
            continue
        r = max(1, int(rng.integers(1, 3) * S))
        ys, xs = np.ogrid[:size, :size]
        disc = (ys - yy) ** 2 + (xs - xx) ** 2 <= r * r
        img[disc] = rng.uniform(2.2, 4.5)
        mask[disc] = SHIP

    # --- sensor blur + thermal noise floor ---
    img = ndimage.gaussian_filter(img, rng.uniform(0.5, 1.1) * S)
    img = img + rng.uniform(0.0, 0.03) * rng.standard_normal(img.shape)

    return to_uint8(img, rng), mask


def to_uint8(img, rng=None):
    """Percentile stretch -> 8-bit, mirroring the inference preprocessing."""
    lo_p = 1.0 if rng is None else float(rng.uniform(0.5, 3.0))
    hi_p = 99.0 if rng is None else float(rng.uniform(97.0, 99.7))
    lo, hi = np.percentile(img, [lo_p, hi_p])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    if rng is not None:                                   # mild photometric jitter
        out = np.clip(out * rng.uniform(0.88, 1.12) + rng.uniform(-0.06, 0.06), 0, 1)
    return (out * 255).astype(np.uint8)


def build_dataset(n, size=128, seed=0, balance=(0.45, 0.30, 0.25),
                  vary_resolution=True):
    """n scenes -> (X float32 [n,2,size,size], Y int8 [n,size,size], kinds).

    Scenes are synthesised at a RANDOM native resolution and then pushed through
    the very same preparation the API applies to an upload (percentile stretch ->
    Lee despeckle -> resize).  Without this the network would be trained on raw
    full-speckle arrays and then shown smoothed, downsampled ones at inference --
    a train/test mismatch that makes calm sea look like oil.
    """
    from .preprocess import prepare_array

    rng = np.random.default_rng(seed)
    kinds = rng.choice(["spill", "lookalike", "clean"], size=n, p=list(balance))
    X = np.zeros((n, 2, size, size), np.float32)
    Y = np.zeros((n, size, size), np.int8)
    for i, k in enumerate(kinds):
        native = int(rng.choice([size, size, int(size * 1.5), size * 2, size * 3])) \
            if vary_resolution else size
        img, mask = make_scene(rng, native, str(k))
        X[i], _, _ = prepare_array(img, size)
        if native != size:
            step = native // size
            mask = mask[::step, ::step][:size, :size]
            if mask.shape != (size, size):       # non-integer ratio -> nearest
                yi = (np.arange(size) * (mask.shape[0] / size)).astype(int)
                xi = (np.arange(size) * (mask.shape[1] / size)).astype(int)
                mask = mask[np.ix_(yi, xi)]
        Y[i] = mask
    return X, Y, list(kinds)


if __name__ == "__main__":
    X, Y, k = build_dataset(12, seed=1)
    for i in range(12):
        present = {CLASS_NAMES[c]: int((Y[i] == c).sum()) for c in range(N_CLASSES)
                   if (Y[i] == c).any()}
        print(f"{k[i]:10s} mean={X[i].mean():6.1f}  {present}")
