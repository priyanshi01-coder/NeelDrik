#!/usr/bin/env python3
"""NEELDRIK prototype launcher.

    python run.py              # http://127.0.0.1:8000
    python run.py --port 9000
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(os.path.join(HERE, "models", "unet_oilspill.npz")):
        print("No trained weights found.\n"
              "Run:  python train.py\n", file=sys.stderr)
        sys.exit(2)

    import uvicorn
    from backend.app import HAVE_FASTAPI
    print(f"  NEELDRIK  ·  SIH26143  ·  Team HackSmiths")
    print(f"  framework : {'FastAPI' if HAVE_FASTAPI else 'Starlette (FastAPI core)'}")
    try:
        from backend.ml.predict import model_info
        mi = model_info()
        print(f"  model     : {mi['architecture']}  [{mi['runtime']}]")
        rh = mi.get("real_holdout")
        if rh:
            print(f"  real SAR  : Oil IoU {rh['iou']['Oil']:.3f}  "
                  f"precision {rh['precision']*100:.1f}%  "
                  f"recall {rh['recall']*100:.1f}%   "
                  f"({rh['scenes']} held-out Sentinel-1 scenes)")
        elif mi.get("spill_accuracy") is not None:
            print(f"  accuracy  : {mi['spill_accuracy']*100:.1f}% spill / no-spill"
                  f"   (SYNTHETIC scenes -- not a real-SAR claim)")
    except Exception as e:
        print(f"  model     : NOT LOADED ({e})")
    print(f"\n  open  ->  http://{a.host}:{a.port}\n")
    uvicorn.run("backend.app:app", host=a.host, port=a.port,
                reload=a.reload, log_level="warning")


if __name__ == "__main__":
    main()
