"""Decide what KIND of image was uploaded, before trying to detect anything.

This module exists because of a real failure: the segmentation model is trained
on SAR (radar), where oil is DARK because the film damps capillary waves.  In an
optical or aerial photograph oil is usually the opposite -- bright silver or
rainbow sheen, or brown crude.  Feeding an aerial photo to the SAR model makes
it hunt for a dark patch, miss the bright slick, and call a real spill clean;
on an ordinary sea photo it finds some dark patch and cries spill.

So every upload is first classified into one of:

    sar          grayscale, speckled, radar-like  -> analyse with the U-Net
    unsupported  everything else                  -> refuse, do not guess

Only SAR is analysed, and that is a deliberate decision rather than a gap.
The problem statement (SIH26143) is Sentinel-1 SAR based, the model is trained
on SAR physics, and a colour photograph carries the opposite signature.  A
detector forced to answer every image will invent spills in images with no sea
in them at all -- a photograph of a bird against a blue sky was reported as a
spill, because blue sky scores high on any naive "is it water?" test.  Refusing
out-of-domain input is the single largest reduction in false positives
available, and it is honest: the system says what it cannot judge.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

SAR, UNSUPPORTED = "sar", "unsupported"
OPTICAL = "optical"          # recognised, but deliberately not analysed


def _features(rgb: np.ndarray) -> dict:
    """rgb uint8 HxWx3 -> measurable descriptors used by the rules below."""
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx, mn = f.max(2), f.min(2)
    val = mx
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # how colourful is it really (a grayscale PNG saved as RGB has sat == 0)
    sat_mean = float(sat.mean())
    sat_p90 = float(np.percentile(sat, 90))
    chan_spread = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)))

    # speckle: local coefficient of variation in the mid-tone areas
    m = ndimage.uniform_filter(gray, 7)
    sq = ndimage.uniform_filter(gray ** 2, 7)
    cov = np.sqrt(np.maximum(sq - m ** 2, 0)) / np.maximum(m, 1e-3)
    mid = (gray > 0.12) & (gray < 0.88)
    cov_mid = float(np.median(cov[mid])) if mid.sum() > 200 else 0.0

    # flatness: share of pixels whose 3x3 neighbourhood is essentially constant
    lap = np.abs(ndimage.laplace(gray))
    flat_frac = float((lap < 0.004).mean())

    # how many distinct tones (documents/diagrams use very few)
    hist = np.histogram(gray, bins=64, range=(0, 1))[0].astype(float)
    p = hist / max(hist.sum(), 1)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    # near-white / near-black dominance (screenshots, scans, documents)
    white_frac = float((gray > 0.93).mean())
    black_frac = float((gray < 0.06).mean())

    # water-ish hue share: blue/green/cyan dominant pixels
    watery = float(((b >= r) & (b > 0.06) & (sat > 0.06)).mean())
    green = float(((g > r) & (g > b) & (sat > 0.10)).mean())

    return {
        "sat_mean": sat_mean, "sat_p90": sat_p90, "chan_spread": chan_spread,
        "cov_mid": cov_mid, "flat_frac": flat_frac, "entropy": entropy,
        "white_frac": white_frac, "black_frac": black_frac,
        "watery": watery, "green": green,
        "mean": float(gray.mean()), "std": float(gray.std()),
    }


def classify(rgb: np.ndarray) -> dict:
    """rgb uint8 HxWx3 -> {'domain', 'confidence', 'reason', 'features'}."""
    f = _features(rgb)
    is_gray = f["chan_spread"] < 0.02 and f["sat_mean"] < 0.05

    # A speckled image is a photograph of something, never a document, so the
    # synthetic-graphic rules below must not fire on it.
    speckled = f["cov_mid"] >= 0.045

    # ---- documents, screenshots, diagrams, charts -------------------- #
    if not speckled:
        if f["white_frac"] > 0.35 and f["entropy"] < 4.6:
            return _out(UNSUPPORTED, 0.95, "looks like a document, slide or "
                        "screenshot (mostly white with few distinct tones)", f)
        if f["flat_frac"] > 0.55 and f["entropy"] < 4.2:
            return _out(UNSUPPORTED, 0.90, "large flat areas and very few "
                        "tones - looks like a diagram, not a photograph", f)
    if f["std"] < 0.04:
        return _out(UNSUPPORTED, 0.92, "almost no variation in the image", f)

    # ---- SAR: grayscale with genuine speckle ------------------------- #
    if is_gray:
        if speckled and f["entropy"] > 3.6:
            conf = min(0.99, 0.62 + 2.0 * f["cov_mid"])
            return _out(SAR, conf, "grayscale with radar speckle", f)
        if f["entropy"] > 4.6:
            # grayscale, textured, but speckle was smoothed away (re-saved
            # JPEG, screenshot of a SAR view). Still analysable, less certain.
            return _out(SAR, 0.55, "grayscale scene; speckle looks smoothed "
                        "(re-compressed image) - analysing as SAR", f)
        return _out(UNSUPPORTED, 0.75,
                    "grayscale but with neither radar speckle nor photographic "
                    "detail", f)

    # ---- colour photograph ------------------------------------------- #
    # The detector is trained on Sentinel-1 SAR, where oil is DARK because the
    # film damps capillary waves.  In a colour photograph oil is usually the
    # opposite - bright sheen or brown crude - so the SAR model cannot read one
    # and must not try.  Blue sky in particular scores high on any naive
    # "is it blue?" water test, which is how a photograph of a bird against sky
    # was reported as an oil spill.
    return _out(UNSUPPORTED, 0.88,
                "this is a colour photograph, not a SAR (radar) scene. "
                "NEELDRIK detects oil in Sentinel-1 SAR imagery, where oil "
                "appears dark; in an optical photo oil looks completely "
                "different, so analysing it would give an unreliable answer",
                f)


def _out(domain, conf, reason, f):
    return {"domain": domain, "confidence": round(float(conf), 3),
            "reason": reason,
            "features": {k: round(v, 4) for k, v in f.items()}}
