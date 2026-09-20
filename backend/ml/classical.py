"""Physics-based SAR dark-spot detector -- no training distribution required.

Why this exists.  The U-Net is trained on simulated SAR, so it is only as good
as the simulator; on a real Sentinel-1 scene it can be confidently wrong.  This
detector instead measures the quantities the SAR oil-spill literature
(Solberg et al.; Topouzelis) actually uses, all of them RELATIVE to the sea
around the candidate:

    contrast_db   how far below the local sea background the patch sits.
                  Oil damps capillary waves, so backscatter drops sharply --
                  typically 3-10 dB.  A low-wind or biogenic look-alike is
                  shallower, ~1-3 dB.
    damping       std inside the patch divided by std of the sea ring around
                  it.  Oil flattens the surface, so its speckle collapses and
                  the ratio falls well below 1; a look-alike keeps most of the
                  sea's texture.  (Using std/MEAN instead is unstable: in a very
                  dark patch the mean approaches zero and the ratio explodes.)
    edge_sharp    gradient across the boundary, normalised by contrast.  An oil
                  slick has a coherent, fairly abrupt edge; a wind shadow fades.
    shape         area, elongation, and how ragged the outline is.

Because every quantity is a RATIO against the surrounding sea, the result does
not depend on the image's exposure, bit depth, scale or speckle level -- which
is exactly what makes it survive contact with real imagery.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# Thresholds are physical, not fitted to a dataset.
CFG = {
    "z_dark": 1.40,          # how many robust sigmas below background to seed on
    "min_area_frac": 0.0025,  # ignore specks below this share of the scene
    "max_area_frac": 0.60,   # a "spill" covering most of the frame is the frame
    "oil_contrast_db": 2.6,  # minimum drop below sea to call it oil
    "strong_contrast_db": 4.5,
    "max_damping": 1.15,     # slick texture must be at most the sea's own
    "min_edge_sharp": 0.18,
}


def _robust_bg(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, max(mad, 1e-3)


def _region_features(img, comp, bg_med, bg_sig):
    """Descriptors for one candidate region, all relative to the sea."""
    area = int(comp.sum())
    inside = img[comp]
    r_mean = float(inside.mean())
    r_std = float(inside.std())

    # local background: a ring around the region, excluding other dark pixels
    ring = ndimage.binary_dilation(comp, np.ones((9, 9))) & ~comp
    local = img[ring]
    local = local[local > bg_med - bg_sig] if local.size else np.array([bg_med])
    l_mean = float(local.mean()) if local.size else bg_med
    l_std = float(local.std()) if local.size > 4 else bg_sig

    eps = 1e-4
    contrast_db = float(10.0 * np.log10(max(l_mean, eps) / max(r_mean, eps)))
    damping = float(r_std / max(l_std, eps))       # <1 means texture was damped

    # boundary gradient, normalised by the contrast it should produce
    gy, gx = np.gradient(img)
    gmag = np.hypot(gx, gy)
    edge = ndimage.binary_dilation(comp, np.ones((3, 3))) ^ \
        ndimage.binary_erosion(comp, np.ones((3, 3)))
    edge_grad = float(gmag[edge].mean()) if edge.any() else 0.0
    drop = max(l_mean - r_mean, eps)
    edge_sharp = float(edge_grad / drop)

    ys, xs = np.nonzero(comp)
    if area > 8:
        cov = np.cov(np.stack([ys, xs]).astype(float))
        ev = np.linalg.eigvalsh(cov)
        elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
    else:
        elong = 1.0
    per = int(ndimage.binary_dilation(comp).sum() - area)
    complexity = float(per / max(np.sqrt(area), 1e-6))

    h, w = img.shape
    touches = bool(ys.min() == 0 or xs.min() == 0 or
                   ys.max() == h - 1 or xs.max() == w - 1)

    return {
        "area_px": area,
        "area_frac": area / float(img.size),
        "contrast_db": round(contrast_db, 2),
        "damping": round(damping, 3),
        "edge_sharp": round(edge_sharp, 3),
        "elongation": round(elong, 2),
        "complexity": round(complexity, 2),
        "touches_border": touches,
        "centroid": [round(float(xs.mean()) / w, 4), round(float(ys.mean()) / h, 4)],
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def _score(f, cfg):
    """0..1 oil likelihood for one region, plus the reasons behind it."""
    reasons = []
    if f["contrast_db"] < cfg["oil_contrast_db"]:
        reasons.append(f"only {f['contrast_db']:.1f} dB below the sea "
                       f"(oil is normally >{cfg['oil_contrast_db']:.0f} dB)")
        return 0.0, reasons
    if f["damping"] > cfg["max_damping"]:
        reasons.append(f"texture inside is not damped "
                       f"(speckle {f['damping']:.2f}x the surrounding sea)")
        return 0.15, reasons
    if f["area_frac"] > cfg["max_area_frac"]:
        reasons.append("covers most of the scene - this is the background, "
                       "not a slick")
        return 0.0, reasons

    # contrast is the dominant term, saturating around 8 dB
    s = min(1.0, (f["contrast_db"] - cfg["oil_contrast_db"]) /
            (cfg["strong_contrast_db"] - cfg["oil_contrast_db"]) * 0.55 + 0.25)
    reasons.append(f"{f['contrast_db']:.1f} dB below the surrounding sea")

    if f["damping"] < 0.70:
        s += 0.18
        reasons.append(f"speckle damped to {f['damping']:.2f}x the sea")
    if f["edge_sharp"] >= cfg["min_edge_sharp"]:
        s += 0.12
        reasons.append("coherent boundary")
    else:
        s -= 0.10
        reasons.append("boundary is diffuse - closer to a wind shadow")
    if f["elongation"] >= 1.8:
        s += 0.08
        reasons.append(f"elongated ({f['elongation']:.1f}:1), consistent with drift")
    if f["touches_border"] and f["area_frac"] > 0.25:
        s -= 0.15
        reasons.append("runs off the edge of the frame")

    return float(np.clip(s, 0.0, 0.99)), reasons


def detect_classical(raw, cfg=None):
    """raw: image in values PROPORTIONAL TO BACKSCATTER (no offset applied).

    Feed this the raster as read, not a normalised copy.  contrast_db is a
    RATIO of means, so any additive offset -- such as the one robust_normalise
    introduces to centre the sea at 0.5 -- silently corrupts it.  That bug made
    simulated look-alikes measure 12 dB when no real look-alike exceeds ~3 dB.
    """
    cfg = dict(CFG, **(cfg or {}))
    img = np.asarray(raw, np.float32)
    img = img - min(0.0, float(img.min()))         # keep it non-negative only
    bg_med, bg_sig = _robust_bg(img)

    z = (img - bg_med) / bg_sig
    dark = z < -cfg["z_dark"]
    dark = ndimage.binary_opening(dark, np.ones((3, 3)))
    dark = ndimage.binary_closing(dark, np.ones((3, 3)))

    lab, n = ndimage.label(dark)
    min_area = cfg["min_area_frac"] * img.size
    regions, mask = [], np.zeros_like(dark)
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < min_area:
            continue
        f = _region_features(img, comp, bg_med, bg_sig)
        f["score"], f["reasons"] = _score(f, cfg)
        regions.append(f)
        if f["score"] >= 0.5:
            mask |= comp

    regions.sort(key=lambda r: -r["score"])
    best = regions[0]["score"] if regions else 0.0
    return {
        "is_spill": bool(best >= 0.5),
        "score": round(float(best), 4),
        "regions": regions,
        "mask": mask,
        "background": {"median": round(bg_med, 4), "sigma": round(bg_sig, 4)},
        "candidates_examined": len(regions),
    }
