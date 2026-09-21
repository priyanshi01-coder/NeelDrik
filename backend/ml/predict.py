"""Inference orchestrator: bytes in -> full detection result out.

Backend selection: the PyTorch twin is used when torch is installed (the
project's stated stack), otherwise the NumPy U-Net runs the identical
architecture with the same trained weights.  Either way the weights are real
and were produced by train.py.
"""
from __future__ import annotations

import math
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


# --------------------------------------------------------------------------- #
# Where on Earth is this scene?
# --------------------------------------------------------------------------- #
# A SAR chip carries no position unless something tells us one. Until this was
# fixed the pipeline shipped the slick outline in NORMALISED image coordinates
# (0..1) and the map drew them as if they were degrees -- which is longitude
# 0, latitude 0: the Gulf of Guinea, off West Africa. Every scene landed on
# the same spot there because every scene has the same 0..1 coordinate box.
#
# There are only three honest sources for a position, in this order of trust:
#   raster    - the GeoTIFF's own CRS and affine transform
#   operator  - the analyst typed the scene centre on the upload form
#   (none)    - then the scene is NOT placed on a world map at all
# The third case is why `scene_center` may be None: a missing position is
# reported as missing rather than invented.
EARTH_KM_PER_DEG = 111.32


def _bounds_from_centre(lat, lon, width_km, aspect=1.0):
    """(west, south, east, north) box of `width_km` centred on lat/lon."""
    lat = max(min(float(lat), 89.0), -89.0)
    shrink = max(math.cos(math.radians(lat)), 0.05)   # metres per degree of
    half_w = (float(width_km) / 2.0) / (EARTH_KM_PER_DEG * shrink)  # longitude
    half_h = (float(width_km) * float(aspect) / 2.0) / EARTH_KM_PER_DEG
    return (lon - half_w, lat - half_h, lon + half_w, lat + half_h)


def _bounds_from_raster(geo, source_shape):
    """(west, south, east, north) from a GeoTIFF's transform, or None."""
    t = geo.get("transform")
    if not geo.get("georeferenced") or not t or len(t) < 6:
        return None
    h, w = source_shape
    a, _b, c, _d, e, f = t[:6]
    xs = (c, c + a * w)
    ys = (f, f + e * h)
    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)
    crs = str(geo.get("crs") or "").upper()
    if "4326" in crs or "CRS84" in crs:
        out = (west, south, east, north)
    else:
        try:                            # projected CRS -> degrees
            from rasterio.warp import transform_bounds
            out = tuple(float(v) for v in transform_bounds(
                geo["crs"], "EPSG:4326", west, south, east, north))
        except Exception:
            return None
    if not all(math.isfinite(v) for v in out):
        return None
    if not (-180.5 <= out[0] <= 180.5 and -90.5 <= out[1] <= 90.5):
        return None
    return out


def _locate(pre, bounds, lat, lon, assumed_scene_km):
    """Resolve the scene's footprint. Returns (bounds, scene_center|None)."""
    src = None
    if bounds and len(bounds) == 4:
        src = "client"
    else:
        bounds = _bounds_from_raster(pre["geo"], pre["source_shape"])
        src = "raster" if bounds else None
    if bounds is None and lat is not None and lon is not None:
        h, w = pre["source_shape"]
        bounds = _bounds_from_centre(lat, lon, assumed_scene_km or 40.0,
                                     h / float(max(w, 1)))
        src = "operator"
    if bounds is None:
        return None, None
    centre = {"lat": round((bounds[1] + bounds[3]) / 2.0, 6),
              "lon": round((bounds[0] + bounds[2]) / 2.0, 6),
              "source": src}
    return tuple(round(float(v), 6) for v in bounds), centre


def detect(data: bytes, assumed_scene_km: float | None = None,
           bounds=None, filename: str = "", force: bool = False,
           wind_ms: float | None = None,
           lat: float | None = None, lon: float | None = None) -> dict:
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
            "scene_center": None,
            "scene_bounds": None,
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
    scene_bounds, scene_center = _locate(pre, bounds, lat, lon, assumed_scene_km)
    geojson = polygons_to_geojson(
        polys, scene_bounds,
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
        # None when nothing told us where the scene is. The UI must then refuse
        # to place it on a world map rather than default to somewhere.
        "scene_center": scene_center,
        "scene_bounds": list(scene_bounds) if scene_bounds else None,
        "overlay_png": to_png_b64(overlay),
        "scene_png": to_png_b64(pre["display"]),
        "scene_thumb": to_thumb_b64(pre["display"]),
        "overlay_thumb": to_thumb_b64(overlay),
        "runtime": s["backend"],
        "elapsed_ms": int((time.time() - t0) * 1000),
        "evidence": _evidence(res, pre, georef, scene_center),
    }


def _evidence(res, pre, georef, centre=None):
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
        {"step": "Scene located",
         "ok": bool(centre),
         "detail": (f"centre {centre['lat']:.4f}, {centre['lon']:.4f} "
                    f"({'raster CRS' if centre['source'] == 'raster' else 'entered by operator'})")
                   if centre else
                   "no position - the scene carries no CRS and none was entered, "
                   "so the slick is not placed on a map"},
    ]
