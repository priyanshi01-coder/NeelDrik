"""Inference orchestrator: bytes in -> full detection result out.

Backend selection: the PyTorch twin is used when torch is installed (the
project's stated stack), otherwise the NumPy U-Net runs the identical
architecture with the same trained weights.  Either way the weights are real
and were produced by train.py.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from .decide import analyse, load_config
from .domain import SAR, classify
from .postprocess import (mask_to_polygons, polygons_to_geojson, render_overlay,
                          to_png_b64, to_thumb_b64)
from .preprocess import MODEL_SIZE, preprocess, pixel_area_km2, read_rgb
from .unet import CLASS_NAMES, LOOKALIKE, OIL, UNet

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.path.join(_ROOT, "models", "unet_oilspill.npz")

_lock = threading.Lock()
_state = {"net": None, "backend": None, "loaded": False}


def _load():
    if _state["loaded"]:
        return _state
    with _lock:
        if _state["loaded"]:
            return _state
        backend = "numpy"
        net = None
        if os.environ.get("NEELDRIK_BACKEND", "auto") in ("auto", "torch"):
            try:
                from .torch_unet import TorchUNetRunner
                net = TorchUNetRunner(WEIGHTS)
                backend = "pytorch"
            except Exception:
                net = None
        if net is None:
            net = UNet(base=8)
            if os.path.exists(WEIGHTS):
                net.load(WEIGHTS)
            else:                        # pragma: no cover
                raise RuntimeError(
                    "models/unet_oilspill.npz not found - run `python train.py` first")
        _state.update(net=net, backend=backend, loaded=True)
    return _state


def _json_safe(o):
    """Replace NaN/Infinity with None, recursively.

    IoU is nan for any class with an empty union -- a class the model never
    predicted and the labels never contained. That is a legitimate result,
    but JSONResponse refuses non-finite floats, so leaving it in returns 500
    for the whole payload and takes the model card, the dashboard and the
    landing-page stats down together.
    """
    import math
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def model_info():
    s = _load()
    import json
    meta = {}
    mp = os.path.join(_ROOT, "models", "metrics.json")
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp))
        except Exception:
            meta = {}
    return _json_safe({
        "architecture": "U-Net, 5-class (Sea / Oil / Look-alike / Ship / Land)",
        "runtime": s["backend"],
        "input": f"{MODEL_SIZE}x{MODEL_SIZE} grayscale",
        "parameters": meta.get("params"),
        "trained_on": meta.get("trained_on"),
        "epochs": meta.get("epochs"),
        "iou": meta.get("iou"),
        "spill_accuracy": meta.get("spill_accuracy"),
        "precision": meta.get("precision"),
        "recall": meta.get("recall"),
        "f1": meta.get("f1"),
        # What the headline figures above were measured on. Synthetic scenes
        # score far higher than real SAR, so a UI that shows one number
        # without this field is quoting the flattering one by accident.
        "evaluated_on": meta.get("evaluated_on", "synthetic"),
        # Present once the model has been trained against real imagery. These
        # are the only numbers worth quoting outside the project.
        "real_holdout": meta.get("real_holdout") or None,
        "decision": meta.get("decision") or load_config(),
        "weights_present": os.path.exists(WEIGHTS),
    })


def detect(data: bytes, assumed_scene_km: float | None = None,
           bounds=None, filename: str = "", force: bool = False,
           wind_ms: float | None = None) -> dict:
    """Run the full pipeline on raw image bytes.

    Out-of-domain input is REFUSED rather than guessed at (see domain.py).
    Pass force=True to analyse anyway; the result is then flagged as forced.
    """
    t0 = time.time()
    s = _load()

    # ---- domain gate: is this even a SAR scene? --------------------- #
    rgb = read_rgb(data)
    if rgb is None:
        raise ValueError("could not decode this file as an image")
    dom = classify(rgb)
    if dom["domain"] != SAR and not force:
        return {
            "filename": filename,
            "verdict": "UNSUPPORTED",
            "label": "Not a SAR scene - not analysed",
            "is_spill": False,
            "flagged": False,
            "reasons": [dom["reason"]],
            "wind_ms": (None if wind_ms is None else float(wind_ms)),
            "wind_state": "unknown",
            "area_ratio": 0.0,
            "dominance_ratio": 0.0,
            "analysed": False,
            "confidence": dom["confidence"],
            "domain": dom["domain"],
            "domain_reason": dom["reason"],
            "candidate_count": 0,
            "estimated_area_km2": 0.0,
            "oil_area_frac": 0.0,
            "oil_evidence": 0.0,
            "lookalike_evidence": 0.0,
            "regions": [],
            "geojson": {"type": "FeatureCollection", "features": []},
            "scene_png": to_png_b64(rgb),
            "overlay_png": to_png_b64(rgb),
            "runtime": s["backend"],
            "elapsed_ms": int((time.time() - t0) * 1000),
            "evidence": [
                {"step": "Input type check", "ok": False,
                 "detail": dom["reason"]},
                {"step": "Detection skipped", "ok": False,
                 "detail": "No verdict is given for input the model was not "
                           "trained on - guessing here is what produces false "
                           "alarms"},
            ],
        }

    pre = preprocess(data)

    proba = s["net"].predict_proba(pre["model_input"])[0]       # (5, S/2, S/2)
    cfg = load_config()

    px_km2, georef = pixel_area_km2(pre["geo"], pre["source_shape"],
                                    proba.shape[-1], assumed_scene_km)
    res = analyse(proba, cfg, px_area_km2=px_km2, wind_ms=wind_ms)

    oil_mask = res.pop("mask")
    look_mask = proba[LOOKALIKE] > 0.5

    polys = mask_to_polygons(oil_mask)
    geojson = polygons_to_geojson(
        polys, bounds,
        {"class": "oil", "verdict": res["verdict"],
         "confidence": res["confidence"]})

    overlay = render_overlay(pre["display"], oil_mask, look_mask)
    total_km2 = round(float(oil_mask.sum() * px_km2), 4)

    class_share = {CLASS_NAMES[c]: round(float((proba.argmax(0) == c).mean()), 4)
                   for c in range(len(CLASS_NAMES))}

    return {
        "filename": filename,
        "analysed": True,
        "domain": "sar",
        "domain_reason": dom["reason"],
        "domain_confidence": dom["confidence"],
        "forced": bool(force and dom["domain"] != SAR),
        "verdict": res["verdict"],
        "label": res["label"],
        "is_spill": res["is_spill"],
        "flagged": res["flagged"],
        "confidence": res["confidence"],
        "reasons": res["reasons"],
        "wind_ms": res["wind_ms"],
        "wind_state": res["wind_state"],
        "area_ratio": res["area_ratio"],
        "dominance_ratio": res["dominance_ratio"],
        "candidate_count": res["candidate_count"],
        "oil_area_frac": res["oil_area_frac"],
        "oil_evidence": res["oil_evidence"],
        "lookalike_evidence": res["lookalike_evidence"],
        "estimated_area_km2": total_km2,
        "area_is_estimate": not georef,
        "georeferenced": bool(pre["geo"]["georeferenced"]),
        "crs": pre["geo"]["crs"],
        "source_shape": pre["source_shape"],
        "scene_stats": pre["stats"],
        "class_share": class_share,
        "regions": res["regions"],
        "geojson": geojson,
        "overlay_png": to_png_b64(overlay),
        "scene_png": to_png_b64(pre["display"]),
        "scene_thumb": to_thumb_b64(pre["display"]),
        "overlay_thumb": to_thumb_b64(overlay),
        "runtime": s["backend"],
        "elapsed_ms": int((time.time() - t0) * 1000),
        "evidence": _evidence(res, pre, georef),
    }


def _evidence(res, pre, georef):
    """The explainability trail the flow document calls the Evidence Engine."""
    return [
        {"step": "Candidate identified",
         "ok": res["candidate_count"] > 0,
         "detail": f"{res['candidate_count']} dark region(s) survived noise removal"},
        {"step": "AI segmented",
         "ok": True,
         "detail": "5-class U-Net assigned every pixel"},
        {"step": "Oil dominant over look-alike",
         "ok": res["oil_evidence"] >= res["lookalike_evidence"],
         "detail": f"P(oil)={res['oil_evidence']:.3f} vs "
                   f"P(look-alike)={res['lookalike_evidence']:.3f}"},
        {"step": "Spatially coherent",
         "ok": bool(res["regions"]) and res["regions"][0]["area_frac"] >= 0.004,
         "detail": (f"largest region {res['regions'][0]['area_frac']*100:.2f}% "
                    f"of scene, elongation {res['regions'][0]['elongation']}")
                   if res["regions"] else "no region above minimum area"},
        {"step": "Boundary extracted",
         "ok": bool(res["regions"]),
         "detail": "polygon boundary vectorised to GeoJSON"},
        {"step": "Georeferencing",
         "ok": georef,
         "detail": "from raster CRS" if georef
                   else "not georeferenced - area is a prototype estimate"},
    ]
