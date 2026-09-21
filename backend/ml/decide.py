"""Turn a 5-class probability map into a verdict.

Post-processing chain:
    probabilities -> oil mask -> connected components -> tiny-noise removal
    -> per-region characterisation -> oil vs look-alike arbitration
    -> wind plausibility -> verdict.

Two things here exist specifically to cut false positives, and both are
physics rather than tuning.

WIND.  Oil is dark in SAR because the film damps capillary waves.  Below
roughly 3 m/s the sea is glassy on its own and produces dark patches that are
indistinguishable from oil by shape or contrast -- this is the single largest
source of false alarms in every operational service.  Above roughly 12 m/s
wind breaks a real slick up and mixes it down, so a confident call is not
supportable either.  Wind is therefore evidence about whether a dark patch CAN
be oil, independent of how the patch looks.  When wind is unknown the system
says so rather than assuming a convenient value.

THE REVIEW BAND.  Oil and look-alikes genuinely overlap in SAR; two trained
analysts disagree on the same patch.  Forcing every scene into spill/clean
manufactures errors at the boundary.  A third verdict -- NEEDS REVIEW -- sends
ambiguous scenes to a human, which is how CleanSeaNet actually operates: it
issues alerts for verification, not verdicts.

Thresholds live in DEFAULTS and are calibrated by train.py on held-out data;
calibrated values are written to models/decision.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import ndimage

from .unet import LOOKALIKE, OIL

DEFAULTS = {
    "oil_thr": 0.50,         # P(Oil) above which a pixel is a candidate
    "min_area_frac": 0.004,  # drop components smaller than this share of the scene
    "spill_area_frac": 0.006,  # total oil share needed to call a spill
    # A separate, much higher bar for calling a scene a LOOK-ALIKE.  It used to
    # reuse spill_area_frac (0.8%), which is a threshold tuned for *detecting
    # oil*, not for naming a feature.  At that level a scene with 1% scattered
    # look-alike pixels -- speckle, not a feature -- was labelled LOOKALIKE
    # instead of CLEAN.  A real look-alike (wind shadow, algal slick) covers a
    # coherent 10-30% of the frame.
    "lookalike_area_frac": 0.05,
    "dominance": 1.05,       # oil evidence must beat look-alike evidence by this
    # --- review band -------------------------------------------------- #
    "review_area_ratio": 0.45,   # >= this share of the area threshold -> review
    "review_dominance": 1.15,    # oil/look-alike ratio below this -> review
    # --- wind plausibility (m/s); None disables ------------------------ #
    "wind_low": 3.0,         # below: dark patches are usually low-wind, not oil
    "wind_high": 12.0,       # above: a real slick is broken up; calls unreliable
}

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                         "models", "decision.json")

SPILL, REVIEW, LOOKALIKE_V, CLEAN = "SPILL", "REVIEW", "LOOKALIKE", "CLEAN"

LABELS = {
    SPILL: "Oil spill detected",
    REVIEW: "Needs human review",
    LOOKALIKE_V: "Look-alike dominant - no confirmed spill",
    CLEAN: "No oil spill detected",
}


def load_config(path=None):
    p = path or _CFG_PATH
    cfg = dict(DEFAULTS)
    try:
        with open(p) as fh:
            cfg.update(json.load(fh))
    except Exception:
        pass
    return cfg


def save_config(cfg, path=None):
    p = path or _CFG_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(cfg, fh, indent=2)


def wind_plausibility(wind_ms, cfg):
    """Can oil produce this dark patch at this wind speed?

    Returns (state, note).  state is one of 'unknown', 'too_low', 'ok',
    'too_high'.  This is deliberately not a probability: it is a statement
    about whether the physics permits the observation, and it reads clearly
    in an evidence trail.
    """
    if wind_ms is None:
        return "unknown", ("wind speed not supplied - cannot rule out a "
                           "low-wind look-alike")
    w = float(wind_ms)
    if w < cfg["wind_low"]:
        return "too_low", (f"wind {w:.1f} m/s is below {cfg['wind_low']:.0f} m/s; "
                           "a glassy sea produces dark patches that look like oil")
    if w > cfg["wind_high"]:
        return "too_high", (f"wind {w:.1f} m/s is above {cfg['wind_high']:.0f} m/s; "
                            "a real slick would be broken up and mixed down")
    return "ok", (f"wind {w:.1f} m/s is in the range where oil damping is "
                  "the most likely cause of a dark patch")


def analyse(proba, cfg=None, px_area_km2=None, wind_ms=None):
    """proba (5,H,W) softmax map -> dict with verdict, confidence, regions, mask."""
    cfg = cfg or load_config()
    h, w = proba.shape[1:]
    total = float(h * w)

    p_oil = proba[OIL]
    p_look = proba[LOOKALIKE]

    raw = p_oil > cfg["oil_thr"]
    raw = ndimage.binary_opening(raw, np.ones((3, 3)))
    lab, n = ndimage.label(raw)
    keep = np.zeros_like(raw)
    regions = []
    min_area = cfg["min_area_frac"] * total
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < min_area:
            continue
        keep |= comp
        ys, xs = np.nonzero(comp)
        per = int(ndimage.binary_dilation(comp).sum() - area)
        compact = float(4 * np.pi * area / max(per * per, 1))
        eig = np.linalg.eigvalsh(np.cov(np.stack([ys, xs]).astype(float))) \
            if area > 4 else np.array([1.0, 1.0])
        elong = float(np.sqrt(max(eig[1], 1e-6) / max(eig[0], 1e-6)))
        regions.append({
            "area_px": area,
            "area_frac": round(area / total, 5),
            "area_km2": (round(area * px_area_km2, 4) if px_area_km2 else None),
            "centroid": [round(float(xs.mean()) / w, 4), round(float(ys.mean()) / h, 4)],
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "mean_oil_p": round(float(p_oil[comp].mean()), 4),
            "mean_lookalike_p": round(float(p_look[comp].mean()), 4),
            "compactness": round(compact, 4),
            "elongation": round(elong, 3),
        })

    regions.sort(key=lambda r: -r["area_px"])
    oil_frac = float(keep.sum() / total)
    oil_ev = float(p_oil[keep].mean()) if keep.any() else float(p_oil.max())
    look_ev = float(p_look[keep].mean()) if keep.any() else float(p_look.mean())
    # Look-alike area gets the SAME component filter as oil: scattered single
    # pixels above 0.5 are speckle, and counting them was what turned clean
    # scenes into LOOKALIKE verdicts.
    look_raw = ndimage.binary_opening(p_look > 0.5, np.ones((3, 3)))
    look_lab, look_n = ndimage.label(look_raw)
    look_keep = np.zeros_like(look_raw)
    for i in range(1, look_n + 1):
        comp = look_lab == i
        if int(comp.sum()) >= min_area:
            look_keep |= comp
    look_frac = float(look_keep.sum() / total)

    # ---- how far past each threshold are we, as ratios ---------------- #
    area_ratio = oil_frac / max(cfg["spill_area_frac"], 1e-9)
    dom_ratio = oil_ev / max(cfg["dominance"] * look_ev, 1e-9)

    wind_state, wind_note = wind_plausibility(wind_ms, cfg)

    # ---- verdict ------------------------------------------------------ #
    reasons = []
    if not regions:
        verdict = (LOOKALIKE_V if look_frac >= cfg.get("lookalike_area_frac", 0.05)
                   else CLEAN)
        reasons.append("no oil region survived the minimum-area filter")
    else:
        strong = area_ratio >= 1.0 and dom_ratio >= 1.0
        near = area_ratio >= cfg["review_area_ratio"]

        if strong and dom_ratio < cfg["review_dominance"]:
            verdict = REVIEW
            reasons.append(
                f"oil evidence only {dom_ratio:.2f}x the look-alike evidence - "
                f"too close to call (needs {cfg['review_dominance']:.2f}x)")
        elif strong:
            verdict = SPILL
            reasons.append(
                f"oil covers {oil_frac*100:.2f}% of the scene "
                f"({area_ratio:.1f}x the threshold) and outweighs look-alike "
                f"evidence by {dom_ratio:.2f}x")
        elif near:
            verdict = REVIEW
            reasons.append(
                f"oil covers {oil_frac*100:.2f}% of the scene, below the "
                f"{cfg['spill_area_frac']*100:.2f}% needed to call a spill but "
                "too much to dismiss")
        elif look_frac >= cfg.get("lookalike_area_frac", 0.05):
            verdict = LOOKALIKE_V
            reasons.append(f"look-alike features cover {look_frac*100:.1f}% of "
                           "the scene and no oil region survived the filters")
        else:
            verdict = CLEAN
            reasons.append("oil area below the detection threshold")

    # ---- wind can only ever make a call WEAKER, never stronger -------- #
    if verdict == SPILL and wind_state == "too_low":
        verdict = REVIEW
        reasons.append(wind_note)
    elif verdict == SPILL and wind_state == "too_high":
        verdict = REVIEW
        reasons.append(wind_note)
    elif wind_state == "ok" and verdict in (SPILL, REVIEW):
        reasons.append(wind_note)
    elif wind_state == "unknown" and verdict == SPILL:
        reasons.append(wind_note)

    is_spill = verdict == SPILL
    flagged = verdict in (SPILL, REVIEW)      # anything a human should look at

    # ---- confidence: margin over the thresholds that decided it ------- #
    # How certain is the network itself, averaged over the scene?  The old
    # CLEAN branch was `1 - area_ratio*0.5`, which is exactly 1.0 whenever
    # there is no oil at all -- so every clean scene reported 99.9%.  That is
    # not a measurement, it is an artefact of subtracting zero.
    mean_certainty = float(proba.max(axis=0).mean())

    if verdict == SPILL:
        conf = oil_ev * min(1.0, area_ratio / 3.0) ** 0.35 * min(1.0, dom_ratio / 1.5) ** 0.35
    elif verdict == REVIEW:
        conf = 0.5                      # by definition: the system is not sure
    elif verdict == LOOKALIKE_V:
        look_ratio = look_frac / max(cfg.get("lookalike_area_frac", 0.05), 1e-9)
        conf = mean_certainty * min(1.0, look_ratio / 2.0) ** 0.35
    else:                                # CLEAN
        conf = mean_certainty * (1.0 - min(1.0, area_ratio) * 0.5)

    return {
        "verdict": verdict,
        "label": LABELS[verdict],
        "is_spill": is_spill,
        "flagged": flagged,
        "confidence": round(float(np.clip(conf, 0.0, 0.99)), 4),
        "oil_area_frac": round(oil_frac, 5),
        "oil_evidence": round(oil_ev, 4),
        "lookalike_evidence": round(look_ev, 4),
        "area_ratio": round(area_ratio, 3),
        "dominance_ratio": round(dom_ratio, 3),
        "wind_ms": (None if wind_ms is None else round(float(wind_ms), 2)),
        "wind_state": wind_state,
        "reasons": reasons,
        "candidate_count": len(regions),
        "regions": regions,
        "mask": keep,
    }
