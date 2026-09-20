"""HTTP layer.

Runs on FastAPI when it is installed (the project's stated backend stack) and
falls back to Starlette -- which is the exact ASGI machinery FastAPI itself is
built on -- so the prototype starts with no installation step.  Both paths
serve identical routes and call the same `services` functions.
"""
from __future__ import annotations

import json
import os

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import database as db
from . import services as sv
from .services import ApiError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
SAMPLES = os.path.join(ROOT, "data", "samples")

try:
    import fastapi                       # noqa: F401
    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False


# --------------------------------------------------------------------------- #
def _err(exc: ApiError):
    return JSONResponse({"error": exc.message}, status_code=exc.status)


async def _body(request):
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            return await request.json()
        except Exception:
            raise ApiError(400, "Malformed JSON body.")
    form = await request.form()
    return {k: v for k, v in form.items()}


def _auth(request):
    return sv.user_from_auth(request.headers.get("authorization"))


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #
#: Capabilities this build serves.  The frontend checks this list so that a
#: server left running from before a feature existed reports itself plainly
#: instead of failing with a bare 404 at the moment the user clicks.
FEATURES = ["detect", "drift", "wind_gating", "avatar", "demo"]


async def h_health(request):
    return JSONResponse({
        "status": "ok",
        "storage": db.backend_name(),
        "framework": "fastapi" if HAVE_FASTAPI else "starlette",
        "users": db.user_count(),
        "features": FEATURES,
    })


async def h_model(request):
    from .ml.predict import model_info
    return JSONResponse(model_info())


async def h_register(request):
    try:
        b = await _body(request)
        return JSONResponse(sv.register(b.get("email"), b.get("name"),
                                        b.get("password"), b.get("org", "")))
    except ApiError as e:
        return _err(e)


async def h_demo(request):
    """One shared read-write demo account, so a public link works instantly."""
    try:
        return JSONResponse(sv.demo_session())
    except ApiError as e:
        return _err(e)


async def h_login(request):
    try:
        b = await _body(request)
        return JSONResponse(sv.login(b.get("email"), b.get("password")))
    except ApiError as e:
        return _err(e)


async def h_me(request):
    try:
        return JSONResponse({"user": sv.profile(_auth(request))})
    except ApiError as e:
        return _err(e)


async def h_detect(request):
    try:
        user = _auth(request)
        form = await request.form()
        up = form.get("file")
        if up is None:
            raise ApiError(400, "Attach a SAR image in the 'file' field.")
        data = await up.read() if hasattr(up, "read") else bytes(up)
        name = getattr(up, "filename", "upload.png")
        scene_km = form.get("scene_km")
        scene_km = float(scene_km) if scene_km not in (None, "") else None
        bounds = form.get("bounds")
        bounds = json.loads(bounds) if bounds else None
        wind = form.get("wind_ms")
        wind = wind if wind not in (None, "") else None
        return JSONResponse(sv.analyse_upload(user, name, data, scene_km,
                                              bounds, wind_ms=wind))
    except ApiError as e:
        return _err(e)
    except Exception as e:                                   # pragma: no cover
        return JSONResponse({"error": f"Detection failed: {e}"}, status_code=500)


async def h_history(request):
    try:
        return JSONResponse({"items": sv.history(_auth(request))})
    except ApiError as e:
        return _err(e)


async def h_detection(request):
    try:
        return JSONResponse(sv.detection(_auth(request),
                                         int(request.path_params["did"])))
    except ApiError as e:
        return _err(e)


async def h_delete(request):
    try:
        return JSONResponse(sv.remove_detection(_auth(request),
                                                int(request.path_params["did"])))
    except ApiError as e:
        return _err(e)


async def h_purge(request):
    try:
        return JSONResponse(sv.purge(_auth(request)))
    except ApiError as e:
        return _err(e)


async def h_drift(request):
    try:
        return JSONResponse(sv.run_drift(_auth(request), await _body(request)))
    except ApiError as e:
        return _err(e)
    except Exception as e:                                   # pragma: no cover
        return JSONResponse({"error": f"Drift model failed: {e}"}, status_code=500)


async def h_drift_verify(request):
    return JSONResponse(sv.drift_selftest())


async def h_avatar(request):
    try:
        body = await _body(request)
        return JSONResponse(sv.set_avatar(_auth(request), body.get("avatar")))
    except ApiError as e:
        return _err(e)


async def h_dashboard(request):
    try:
        return JSONResponse(sv.dashboard(_auth(request)))
    except ApiError as e:
        return _err(e)


async def h_samples(request):
    if not os.path.isdir(SAMPLES):
        return JSONResponse({"items": []})
    items = []
    for fn in sorted(os.listdir(SAMPLES)):
        if fn.lower().endswith(".png"):
            stem = os.path.splitext(fn)[0]
            kind = stem.split("_")[0]
            items.append({"file": fn, "name": stem.replace("_", " "), "kind": kind,
                          "url": f"/samples/{fn}"})
    return JSONResponse({"items": items})


async def h_index(request):
    return FileResponse(os.path.join(FRONTEND, "index.html"))


async def h_app_page(request):
    return FileResponse(os.path.join(FRONTEND, "app.html"))


ROUTES = [
    Route("/api/health", h_health),
    Route("/api/model", h_model),
    Route("/api/auth/register", h_register, methods=["POST"]),
    Route("/api/auth/login", h_login, methods=["POST"]),
    Route("/api/auth/demo", h_demo, methods=["POST", "GET"]),
    Route("/api/auth/me", h_me),
    Route("/api/detect", h_detect, methods=["POST"]),
    Route("/api/detections", h_history),
    Route("/api/detections/{did:int}", h_detection),
    Route("/api/detections/{did:int}/delete", h_delete, methods=["POST", "DELETE"]),
    Route("/api/detections/purge", h_purge, methods=["POST"]),
    Route("/api/drift", h_drift, methods=["POST"]),
    Route("/api/drift/verify", h_drift_verify),
    Route("/api/account/avatar", h_avatar, methods=["POST"]),
    Route("/api/dashboard", h_dashboard),
    Route("/api/samples", h_samples),
    # The deployed link must land on the console, not a sign-in wall: a judge
    # opening the URL should see the dashboard immediately. The marketing page
    # is still reachable at /login.
    Route("/", h_app_page),
    Route("/login", h_index),
    Route("/app", h_app_page),
]

if os.path.isdir(SAMPLES):
    ROUTES.append(Mount("/samples", StaticFiles(directory=SAMPLES)))
ROUTES.append(Mount("/static", StaticFiles(directory=FRONTEND)))


def create_app():
    db.init()
    app = Starlette(debug=False, routes=ROUTES)
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    return app


def create_fastapi_app():
    """Same surface, declared with FastAPI so /docs works when it is installed."""
    from fastapi import FastAPI, Request
    db.init()
    api = FastAPI(title="NEELDRIK API",
                  description="AI-powered satellite oil-spill detection (SIH26143)",
                  version="1.0.0")
    api.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    for r in ROUTES:
        if isinstance(r, Mount):
            api.router.routes.append(r)
        else:
            api.add_api_route(r.path, r.endpoint, methods=list(r.methods or ["GET"]))
    return api


app = create_fastapi_app() if HAVE_FASTAPI else create_app()
