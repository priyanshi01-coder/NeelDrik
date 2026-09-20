"""Authentication and privacy controls.

Passwords are never stored: only a PBKDF2-HMAC-SHA256 hash with a per-user
random salt.  Sessions are stateless JWTs signed with HS256.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

PBKDF2_ROUNDS = 200_000
TOKEN_TTL_S = 8 * 3600

_SECRET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "models", ".jwt_secret")


def _secret() -> bytes:
    env = os.environ.get("NEELDRIK_SECRET")
    if env:
        return env.encode()
    os.makedirs(os.path.dirname(_SECRET_PATH), exist_ok=True)
    if not os.path.exists(_SECRET_PATH):
        with open(_SECRET_PATH, "w") as fh:
            fh.write(secrets.token_urlsafe(48))
        try:
            os.chmod(_SECRET_PATH, 0o600)
        except Exception:
            pass
    return open(_SECRET_PATH).read().strip().encode()


# --------------------------------------------------------------------------- #
def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt, want = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(rounds))
        return hmac.compare_digest(dk.hex(), want)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(payload: dict, ttl: int = TOKEN_TTL_S) -> str:
    """HS256 JWT. Uses PyJWT when installed, otherwise an equivalent local impl."""
    body = dict(payload, iat=int(time.time()), exp=int(time.time()) + ttl)
    if "sub" in body:
        body["sub"] = str(body["sub"])      # RFC 7519: sub must be a string

    try:
        import jwt                                   # PyJWT
        return jwt.encode(body, _secret().decode(), algorithm="HS256")
    except Exception:
        head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"},
                               separators=(",", ":")).encode())
        pl = _b64(json.dumps(body, separators=(",", ":")).encode())
        sig = _b64(hmac.new(_secret(), f"{head}.{pl}".encode(), hashlib.sha256).digest())
        return f"{head}.{pl}.{sig}"


def read_token(token: str) -> dict | None:
    if not token:
        return None
    try:
        import jwt
        return jwt.decode(token, _secret().decode(), algorithms=["HS256"])
    except ImportError:
        pass
    except Exception:
        return None
    try:
        head, pl, sig = token.split(".")
        want = _b64(hmac.new(_secret(), f"{head}.{pl}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, want):
            return None
        body = json.loads(_unb64(pl))
        if body.get("exp", 0) < time.time():
            return None
        return body
    except Exception:
        return None


def password_problem(pw: str) -> str | None:
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if pw.isdigit() or pw.isalpha():
        return "Use a mix of letters and numbers."
    return None
