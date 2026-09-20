"""Framework-agnostic business logic.

Kept separate from the web layer so the identical code path runs under FastAPI
(the stated stack) and under Starlette (FastAPI's own foundation, used when
FastAPI is not installed).
"""
from __future__ import annotations

import os
import re
import time

from . import database as db
from .ml.drift import Field, backtrack_origin, to_geojson, verify as drift_verify
from .ml.predict import detect as run_detect, model_info
from .security import (hash_password, make_token, password_problem, read_token,
                       verify_password)

MAX_UPLOAD = 25 * 1024 * 1024
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------- #
def register(email, name, password, org=""):
    email = (email or "").strip().lower()
    name = (name or "").strip()
    if not EMAIL_RE.match(email):
        raise ApiError(400, "Enter a valid email address.")
    if len(name) < 2:
        raise ApiError(400, "Enter your name.")
    problem = password_problem(password or "")
    if problem:
        raise ApiError(400, problem)
    if db.get_user(email):
        raise ApiError(409, "An account with that email already exists.")
    uid = db.create_user(email, name, org.strip(), hash_password(password))
    return _session(uid, email, name, org)


def login(email, password):
    u = db.get_user((email or "").strip().lower())
    # constant-ish work whether or not the user exists
    ok = bool(u) and verify_password(password or "", u["pw_hash"])
    if not ok:
        raise ApiError(401, "Email or password is incorrect.")
    return _session(u["id"], u["email"], u["name"], u["org"])


def _session(uid, email, name, org):
    return {
        "token": make_token({"sub": uid, "email": email, "name": name}),
        "user": {"id": uid, "email": email, "name": name, "org": org or ""},
    }


DEMO_EMAIL = "demo@neeldrik.in"


def demo_session():
    """Sign in (creating on first use) the shared demo account.

    A public demo URL that opens on a login form gets closed. This keeps the
    deployed link usable by anyone while real accounts still work normally.
    """
    u = db.get_user(DEMO_EMAIL)
    if not u:
        uid = db.create_user(DEMO_EMAIL, "Demo Analyst", "NEELDRIK demo",
                             hash_password("demo-" + make_token({"sub": 0})[:24]))
        return _session(uid, DEMO_EMAIL, "Demo Analyst", "NEELDRIK demo")
    return _session(u["id"], u["email"], u["name"], u["org"])


def user_from_auth(header: str | None):
    if not header or not header.lower().startswith("bearer "):
        raise ApiError(401, "Sign in to continue.")
    claims = read_token(header.split(" ", 1)[1].strip())
    if not claims:
        raise ApiError(401, "Session expired. Sign in again.")
    return {"id": int(claims["sub"]), "email": claims.get("email"),
            "name": claims.get("name")}


def profile(user):
    """The signed-in user plus anything held server-side rather than in the token.

    The JWT is issued once and cannot carry a picture chosen later, so the
    avatar has to be read from the row on each /me -- otherwise signing in on a
    second browser shows the initials again even though a picture was saved.
    """
    row = db.get_user(user["email"]) if user.get("email") else None
    return {**user,
            "org": (row or {}).get("org", ""),
            "avatar": db.get_avatar(user["id"])}


# --------------------------------------------------------------------------- #
def analyse_upload(user, filename, data, scene_km=None, bounds=None,
                   wind_ms=None):
    if not data:
        raise ApiError(400, "No file received.")
    if wind_ms is not None:
        try:
            wind_ms = float(wind_ms)
        except (TypeError, ValueError):
            raise ApiError(400, "Wind speed must be a number in m/s.")
        if not 0.0 <= wind_ms <= 60.0:
            raise ApiError(400, "Wind speed must be between 0 and 60 m/s.")
    if len(data) > MAX_UPLOAD:
        raise ApiError(413, "File is larger than 25 MB.")
    ext = os.path.splitext(filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXT:
        raise ApiError(415, f"Unsupported file type '{ext}'. "
                            f"Use {', '.join(sorted(ALLOWED_EXT))}.")
    try:
        result = run_detect(data, assumed_scene_km=scene_km, bounds=bounds,
                            filename=filename or "upload", wind_ms=wind_ms)
    except ValueError as exc:
        raise ApiError(400, f"Could not read that image: {exc}") from exc
    result["id"] = db.add_detection(user["id"], result)
    result["created"] = time.time()
    return result


MAX_AVATAR = 400_000          # ~300 KB of image once base64 is decoded


def set_avatar(user, data_uri):
    """Store a profile picture as a small data URI, or clear it with None.

    Kept in the users row rather than on disk so the prototype stays a single
    portable folder. The size cap is what stops a 12 MP phone photo being
    pasted into every API response that carries the user.
    """
    if data_uri in (None, "", "null"):
        db.set_avatar(user["id"], None)
        return {"avatar": None}
    if not isinstance(data_uri, str) or not data_uri.startswith("data:image/"):
        raise ApiError(400, "Profile picture must be an image.")
    if len(data_uri) > MAX_AVATAR:
        raise ApiError(413, "Profile picture is too large - use an image "
                            "under about 300 KB, or let the page resize it.")
    db.set_avatar(user["id"], data_uri)
    return {"avatar": data_uri}


def history(user, limit=100):
    return db.list_detections(user["id"], limit)


def detection(user, det_id):
    d = db.get_detection(user["id"], det_id)
    if d is None:
        raise ApiError(404, "Detection not found.")
    return d


def remove_detection(user, det_id):
    if not db.delete_detection(user["id"], det_id):
        raise ApiError(404, "Detection not found.")
    return {"deleted": det_id}


def purge(user):
    return {"deleted": db.delete_all_for_user(user["id"])}


def run_drift(user, body):
    """Back-track a slick to its likely origin."""
    def num(key, default=None, lo=None, hi=None):
        v = body.get(key, default)
        if v in (None, ""):
            if default is None:
                raise ApiError(400, f"'{key}' is required.")
            v = default
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ApiError(400, f"'{key}' must be a number.")
        if lo is not None and v < lo or hi is not None and v > hi:
            raise ApiError(400, f"'{key}' must be between {lo} and {hi}.")
        return v

    lat = num("lat", lo=-90, hi=90)
    lon = num("lon", lo=-180, hi=180)
    hours = body.get("hours") or [6, 12, 24]
    if isinstance(hours, (int, float, str)):
        hours = [hours]
    try:
        hours = [float(h) for h in hours]
    except (TypeError, ValueError):
        raise ApiError(400, "'hours' must be a list of numbers.")
    hours = [h for h in hours if 0 < h <= 240]
    if not hours:
        raise ApiError(400, "Give at least one elapsed time between 0 and 240 hours.")

    field = Field(wind_speed=num("wind_speed", 0, lo=0, hi=80),
                  wind_dir=num("wind_dir", 0, lo=0, hi=360),
                  cur_speed=num("cur_speed", 0, lo=0, hi=5),
                  cur_dir=num("cur_dir", 0, lo=0, hi=360))
    res = backtrack_origin(
        lat, lon, hours, field,
        n=int(num("particles", 200, lo=10, hi=2000)),
        diffusivity=num("diffusivity", 5.0, lo=0, hi=200),
        wind_factor=num("wind_factor", 0.03, lo=0, hi=0.1),
        deflection=num("deflection", 15.0, lo=0, hi=45))
    res["geojson"] = to_geojson(res)
    res["settings"] = {
        "wind_speed": num("wind_speed", 0), "wind_dir": num("wind_dir", 0),
        "cur_speed": num("cur_speed", 0), "cur_dir": num("cur_dir", 0),
        "wind_factor": num("wind_factor", 0.03),
        "deflection": num("deflection", 15.0),
        "diffusivity": num("diffusivity", 5.0),
    }
    return res


def drift_selftest(user=None):
    """Run the closed-form verification and report it to the UI."""
    fails = drift_verify(verbose=False)
    return {"passed": not fails, "failed_checks": fails, "checks": 10}


def dashboard(user):
    s = db.stats(user["id"])
    recent = db.list_detections(user["id"], 8)
    return {"stats": s, "recent": recent, "model": model_info(),
            "storage": db.backend_name()}
