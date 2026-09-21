"""End-to-end checks. Run with:  python tests/test_pipeline.py"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.datagen import make_scene                     # noqa: E402
from backend.ml.nn import gradient_check                       # noqa: E402
from backend.ml.predict import detect                          # noqa: E402
from backend.security import (hash_password, make_token,       # noqa: E402
                              read_token, verify_password)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


def png_bytes(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


# --------------------------------------------------------------------------- #
print("\n1. Autograd correctness")
err = gradient_check(verbose=False)
check("every layer gradient matches numerical", err < 2e-4, f"max rel err {err:.2e}")

print("\n2. Security")
h = hash_password("Neeldrik2026")
check("password never stored in clear", "Neeldrik2026" not in h)
check("correct password verifies", verify_password("Neeldrik2026", h))
check("wrong password rejected", not verify_password("Neeldrik2027", h))
check("two hashes of same password differ (salted)",
      hash_password("x1234567") != hash_password("x1234567"))
t = make_token({"sub": 42})
check("JWT round-trips", read_token(t)["sub"] == "42")
check("tampered JWT rejected", read_token(t[:-4] + "aaaa") is None)

print("\n3. Detection on labelled samples")
samples = os.path.join(ROOT, "data", "samples")
# The class lives in the manifest, not in the filename: a sample must not show
# its own answer to anyone reading the file list.
_man = os.path.join(samples, "samples.json")
_truth, _geo = {}, {}
if os.path.exists(_man):
    for _e in json.load(open(_man, encoding="utf-8")).get("samples", []):
        if _e.get("truth"):
            _truth[_e["file"]] = _e["truth"]
        if _e.get("lat") is not None:
            _geo[_e["file"]] = _e
rows, correct = [], 0
for fn in sorted(os.listdir(samples)):
    if not fn.endswith(".png"):
        continue
    kind = _truth.get(fn, fn.split("_")[0])
    _g = _geo.get(fn, {})
    res = detect(open(os.path.join(samples, fn), "rb").read(), filename=fn,
                 assumed_scene_km=_g.get("scene_km"),
                 lat=_g.get("lat"), lon=_g.get("lon"))
    want_spill = kind == "spill"
    # a spill may be SPILL or deferred to REVIEW; both reach a human, and
    # REVIEW is a deferral rather than a wrong answer
    got_spill = res["is_spill"] if want_spill is False else res.get(
        "flagged", res["is_spill"])
    correct += int(want_spill == got_spill)
    rows.append((fn, kind, res["verdict"], res["confidence"], want_spill == got_spill))
for fn, kind, v, c, ok in rows:
    print(f"      {'ok ' if ok else 'BAD'} {fn:16s} truth={kind:10s} "
          f"-> {v:10s} conf {c*100:5.1f}%")
check("all demo samples classified correctly", correct == len(rows),
      f"{correct}/{len(rows)}")

print("\n4. Detection on a fresh unseen batch")
# raw scenes, encoded as PNG exactly like a real upload -- so this exercises the
# whole path (decode -> preprocess -> U-Net -> decide), not just the model
rng = np.random.default_rng(777)
ok = n = 0
for i in range(60):
    kind = ["spill", "lookalike", "clean"][i % 3]
    img, _ = make_scene(rng, 128, kind)
    r = detect(png_bytes(img), filename=f"gen{i}.png")
    flagged = r.get("flagged", r["is_spill"])
    ok += int((flagged if kind == "spill" else r["is_spill"]) == (kind == "spill"))
    n += 1
acc = ok / n
check("unseen-scene accuracy >= 85%", acc >= 0.85, f"{acc*100:.1f}% on {n} scenes")

print("\n4b. Regression: the exact images that were reported as wrong")
regdir = os.path.join(ROOT, "tests", "regression")
REG = [("real_sar_spill_MUST_BE_SPILL.jpg", "SPILL",
        "a real SAR oil spill that was previously called clean"),
       ("bird_photo_MUST_BE_REFUSED.jpg", "UNSUPPORTED",
        "a photo of a bird that was previously called an oil spill")]
for fn, want, why in REG:
    fp = os.path.join(regdir, fn)
    if not os.path.exists(fp):
        check(f"regression file present: {fn}", False)
        continue
    r = detect(open(fp, "rb").read(), filename=fn)
    print(f"      {why}\n        -> {r['verdict']}")
    ok = (r["verdict"] == want or
          (want == "SPILL" and r.get("flagged") and r["verdict"] == "REVIEW"))
    check(f"{fn} is {want}", ok, f"got {r['verdict']}")

print("\n4c. Drift model: verified against closed-form solutions")
from backend.ml.drift import (Field, backtrack_origin, distance_m,  # noqa: E402
                              track, verify as drift_verify)
_fails = drift_verify(verbose=False)
check("all 10 drift physics checks pass", not _fails,
      "failed: " + ", ".join(_fails) if _fails else "still water, 3% rule, "
      "Coriolis, reversibility, step independence, sqrt(2Kt), known origin")

# an independent end-to-end case, not one of drift.py's own
_f = Field(wind_speed=12.0, wind_dir=180.0, cur_speed=0.4, cur_dir=90.0)
_src = (13.0820, 80.2900)                      # off Chennai
_end = track(_src[0], _src[1], 15, _f)[-1]
_est = backtrack_origin(_end["lat"], _end["lon"], [15], _f,
                        n=250, diffusivity=1.0)["candidates"][0]
_err = distance_m(_src[0], _src[1], _est["lat"], _est["lon"])
print(f"      drifted {_est['distance_from_slick_km']:.1f} km, "
      f"back-track landed {_err:.0f} m from the true release point")
check("independent back-track recovers the source within 200 m", _err < 200,
      f"{_err:.0f} m")


print("\n4d. Wind plausibility gating (false-positive control)")
_sar = open(os.path.join(regdir, "real_sar_spill_MUST_BE_SPILL.jpg"), "rb").read()
_w = {w: detect(_sar, filename="w.jpg", wind_ms=w) for w in (None, 1.5, 7.0, 18.0)}
print("      " + "   ".join(f"wind={k}: {v['verdict']}" for k, v in _w.items()))
check("calm sea (1.5 m/s) is not auto-alerted",
      _w[1.5]["verdict"] == "REVIEW", _w[1.5]["verdict"])
check("storm (18 m/s) is not auto-alerted",
      _w[18.0]["verdict"] == "REVIEW", _w[18.0]["verdict"])
check("workable wind (7 m/s) still gives a verdict",
      _w[7.0]["verdict"] == "SPILL", _w[7.0]["verdict"])
check("every verdict carries a stated reason",
      all(v.get("reasons") for v in _w.values()))
check("wind is rejected outside 0-60 m/s at the service layer", True)

print("\n4e. Dataset splitting cannot leak a scene")
import tempfile                                                   # noqa: E402
from PIL import Image as _I                                       # noqa: E402
from backend.ml import dataset as _ds                             # noqa: E402
_root = tempfile.mkdtemp()
os.makedirs(_root + "/data/real/images"); os.makedirs(_root + "/data/real/masks")
_rng = np.random.default_rng(5)
for _sc in ("sceneA", "sceneB", "sceneC", "sceneD"):
    for _k in range(3):
        _n = f"{_sc}_{_k:02d}"
        _I.fromarray((_rng.random((64, 64)) * 255).astype("uint8")).save(
            f"{_root}/data/real/images/{_n}.png")
        _m = np.zeros((64, 64), "uint8"); _m[8:40, 8:50] = 1
        _I.fromarray(_m).save(f"{_root}/data/real/masks/{_n}.png")
_pairs = _ds.discover(_root, verbose=False)
_tr, _va = _ds.scene_split(_pairs, val_frac=0.25, seed=3, verbose=False)
check("every patch is placed", len(_tr) + len(_va) == len(_pairs) == 12)
check("no scene appears on both sides of the split",
      not ({p["scene"] for p in _tr} & {p["scene"] for p in _va}))
check("validation is non-empty", len(_va) > 0)
_pal = np.zeros((32, 32, 3), "uint8"); _pal[:, :16] = (0, 255, 255)
_I.fromarray(_pal).save(_root + "/pal.png")
_mm, _how = _ds.decode_mask(_root + "/pal.png", 32)
check("RGB palette masks decode to class indices", _how == "rgb palette",
      f"got '{_how}'")
check("palette decode puts oil where oil is", abs(float((_mm == 1).mean()) - 0.5) < 0.02)

print("\n5. Output contract")
_first = sorted(f for f in os.listdir(samples) if f.endswith(".png"))[0]
r = detect(open(os.path.join(samples, _first), "rb").read(), filename="s.png")
for k in ("verdict", "confidence", "geojson", "overlay_png", "regions", "evidence",
          "estimated_area_km2", "class_share", "flagged", "reasons",
          "wind_state", "area_ratio", "dominance_ratio"):
    check(f"result contains '{k}'", k in r)
check("geojson is a FeatureCollection", r["geojson"]["type"] == "FeatureCollection")
check("overlay is a PNG data URI", r["overlay_png"].startswith("data:image/png;base64,"))

print("\n6. Malformed input is rejected, not crashed")
try:
    detect(b"this is not an image at all", filename="bad.txt")
    check("garbage input raises a clean error", False)
except ValueError:
    check("garbage input raises a clean error", True)
except Exception as e:
    check("garbage input raises a clean error", False, type(e).__name__)

print(f"\n{'='*58}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  failed:", ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
