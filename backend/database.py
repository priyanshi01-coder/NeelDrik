"""Persistence layer.

SQLite by default so the prototype runs with zero setup.  Set DATABASE_URL to a
PostgreSQL DSN and, if psycopg is installed, the same schema is used there with
PostGIS geometry for the slick polygons (the stated production stack):

    export DATABASE_URL=postgresql://user:pass@localhost/neeldrik
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "models", "neeldrik.db")

_local = threading.local()
_PG = None

DSN = os.environ.get("DATABASE_URL", "")
if DSN.startswith("postgres"):
    try:
        import psycopg2
        import psycopg2.extras
        _PG = psycopg2
    except Exception:
        _PG = None


def _conn():
    if _PG:
        if not hasattr(_local, "pg"):
            _local.pg = _PG.connect(DSN)
            _local.pg.autocommit = True
        return _local.pg
    if not hasattr(_local, "sq"):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        c = sqlite3.connect(DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        _local.sq = c
    return _local.sq


def _q(sql):
    """SQLite uses ?, psycopg uses %s."""
    return sql.replace("?", "%s") if _PG else sql


def backend_name():
    return "postgresql+postgis" if _PG else "sqlite"


# --------------------------------------------------------------------------- #
SCHEMA_SQLITE = [
    """CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        org TEXT,
        pw_hash TEXT NOT NULL,
        avatar TEXT,
        created REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        created REAL NOT NULL,
        filename TEXT,
        verdict TEXT,
        confidence REAL,
        area_km2 REAL,
        candidates INTEGER,
        payload TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_det_user ON detections(user_id, created DESC)",
]

SCHEMA_PG = [
    """CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        org TEXT,
        pw_hash TEXT NOT NULL,
        avatar TEXT,
        created DOUBLE PRECISION NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS detections (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        created DOUBLE PRECISION NOT NULL,
        filename TEXT,
        verdict TEXT,
        confidence DOUBLE PRECISION,
        area_km2 DOUBLE PRECISION,
        candidates INTEGER,
        payload TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_det_user ON detections(user_id, created DESC)",
]


def init():
    c = _conn()
    cur = c.cursor()
    for stmt in (SCHEMA_PG if _PG else SCHEMA_SQLITE):
        cur.execute(stmt)
    if _PG:
        try:                              # PostGIS geometry column, when available
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
            cur.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS "
                        "geom geometry(MultiPolygon, 4326)")
        except Exception:
            pass
    else:
        c.commit()
    cur.close()
    migrate()          # bring an older database up to the current columns


# --------------------------------------------------------------------------- #
def create_user(email, name, org, pw_hash):
    c = _conn(); cur = c.cursor()
    cur.execute(_q("INSERT INTO users(email,name,org,pw_hash,created) "
                   "VALUES(?,?,?,?,?)"),
                (email.lower(), name, org, pw_hash, time.time()))
    if not _PG:
        c.commit()
        uid = cur.lastrowid
    else:
        cur.execute("SELECT currval(pg_get_serial_sequence('users','id'))")
        uid = cur.fetchone()[0]
    cur.close()
    return uid


def get_user(email):
    cur = _conn().cursor()
    cur.execute(_q("SELECT id,email,name,org,pw_hash,avatar FROM users WHERE email=?"),
                (email.lower(),))
    r = cur.fetchone(); cur.close()
    if not r:
        return None
    return {"id": r[0], "email": r[1], "name": r[2], "org": r[3], "pw_hash": r[4]}


def user_count():
    cur = _conn().cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    n = cur.fetchone()[0]; cur.close()
    return int(n)


def add_detection(user_id, result):
    slim = {k: v for k, v in result.items() if k != "scene_png"}
    c = _conn(); cur = c.cursor()
    cur.execute(_q("INSERT INTO detections"
                   "(user_id,created,filename,verdict,confidence,area_km2,"
                   "candidates,payload) VALUES(?,?,?,?,?,?,?,?)"),
                (user_id, time.time(), result.get("filename"), result.get("verdict"),
                 result.get("confidence"), result.get("estimated_area_km2"),
                 result.get("candidate_count"), json.dumps(slim)))
    if not _PG:
        c.commit()
        did = cur.lastrowid
    else:
        cur.execute("SELECT currval(pg_get_serial_sequence('detections','id'))")
        did = cur.fetchone()[0]
    cur.close()
    return did


def list_detections(user_id, limit=100):
    cur = _conn().cursor()
    cur.execute(_q("SELECT id,created,filename,verdict,confidence,area_km2,candidates "
                   "FROM detections WHERE user_id=? ORDER BY created DESC LIMIT ?"),
                (user_id, limit))
    rows = cur.fetchall(); cur.close()
    return [{"id": r[0], "created": r[1], "filename": r[2], "verdict": r[3],
             "confidence": r[4], "area_km2": r[5], "candidates": r[6]} for r in rows]


def get_detection(user_id, det_id):
    cur = _conn().cursor()
    cur.execute(_q("SELECT payload FROM detections WHERE id=? AND user_id=?"),
                (det_id, user_id))
    r = cur.fetchone(); cur.close()
    return json.loads(r[0]) if r else None


def delete_detection(user_id, det_id):
    c = _conn(); cur = c.cursor()
    cur.execute(_q("DELETE FROM detections WHERE id=? AND user_id=?"), (det_id, user_id))
    n = cur.rowcount
    if not _PG:
        c.commit()
    cur.close()
    return n


def delete_all_for_user(user_id):
    c = _conn(); cur = c.cursor()
    cur.execute(_q("DELETE FROM detections WHERE user_id=?"), (user_id,))
    n = cur.rowcount
    if not _PG:
        c.commit()
    cur.close()
    return n


def stats(user_id):
    cur = _conn().cursor()
    cur.execute(_q("SELECT verdict, COUNT(*), COALESCE(SUM(area_km2),0), "
                   "COALESCE(AVG(confidence),0) FROM detections WHERE user_id=? "
                   "GROUP BY verdict"), (user_id,))
    rows = cur.fetchall(); cur.close()
    out = {"total": 0, "by_verdict": {}, "area_km2": 0.0, "avg_confidence": 0.0}
    tot_conf, n = 0.0, 0
    for v, c_, a, avg in rows:
        out["by_verdict"][v] = int(c_)
        out["total"] += int(c_)
        out["area_km2"] += float(a or 0)
        tot_conf += float(avg or 0) * int(c_); n += int(c_)
    out["area_km2"] = round(out["area_km2"], 3)
    out["avg_confidence"] = round(tot_conf / n, 4) if n else 0.0
    return out


# --------------------------------------------------------------------------- #
def migrate():
    """Add columns that later versions introduced.

    An existing database predates the avatar column; ALTER TABLE is cheap and
    idempotent here because we check first. Without this, upgrading in place
    breaks every query that selects `avatar`.
    """
    c = _conn(); cur = c.cursor()
    try:
        cur.execute("SELECT avatar FROM users LIMIT 1")
        cur.fetchall()
    except Exception:
        try:
            cur.execute("ALTER TABLE users ADD COLUMN avatar TEXT")
            c.commit()
        except Exception:
            pass
    cur.close()


def get_avatar(user_id):
    cur = _conn().cursor()
    cur.execute(_q("SELECT avatar FROM users WHERE id=?"), (user_id,))
    r = cur.fetchone(); cur.close()
    return (r[0] if r else None) or None


def set_avatar(user_id, data_uri):
    """data_uri None clears the picture."""
    c = _conn(); cur = c.cursor()
    cur.execute(_q("UPDATE users SET avatar=? WHERE id=?"), (data_uri, user_id))
    c.commit(); cur.close()
    return True
