"""Post-processing: boundary extraction, GeoJSON, and overlay rendering."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import cv2
    HAVE_CV2 = True
except Exception:                       # pragma: no cover
    HAVE_CV2 = False

OIL_RGB = (214, 66, 52)
LOOK_RGB = (222, 178, 62)


# --------------------------------------------------------------------------- #
def mask_to_polygons(mask, simplify=1.5):
    """Binary mask -> list of polygons in normalised (0..1) image coordinates."""
    h, w = mask.shape
    polys = []
    if HAVE_CV2:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) < 3:
                continue
            approx = cv2.approxPolyDP(c, simplify, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            polys.append([[round(float(x) / w, 5), round(float(y) / h, 5)]
                          for x, y in approx])
    else:                               # pragma: no cover - bbox fallback
        lab, n = ndimage.label(mask)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            polys.append([[x0 / w, y0 / h], [x1 / w, y0 / h],
                          [x1 / w, y1 / h], [x0 / w, y1 / h]])
    return polys


def polygons_to_geojson(polys, bounds=None, props=None):
    """Normalised polygons -> GeoJSON FeatureCollection.

    bounds = (west, south, east, north) in degrees when the scene is
    georeferenced; otherwise a local 0..1 grid CRS is declared honestly.
    """
    feats = []
    for i, poly in enumerate(polys):
        if bounds:
            wsen = bounds
            ring = [[wsen[0] + x * (wsen[2] - wsen[0]),
                     wsen[3] - y * (wsen[3] - wsen[1])] for x, y in poly]
        else:
            ring = [[x, y] for x, y in poly]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        feats.append({
            "type": "Feature",
            "id": f"slick-{i + 1}",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": dict(props or {}, index=i + 1),
        })
    fc = {"type": "FeatureCollection", "features": feats}
    if not bounds:
        fc["crs"] = {"type": "name",
                     "properties": {"name": "local-image-normalised"}}
    return fc


# --------------------------------------------------------------------------- #
def _resize_prob(p, hw):
    h, w = hw
    if HAVE_CV2:
        return cv2.resize(p.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR), np.float32) / 255.0


def render_overlay(display_u8, oil_mask_small, look_mask_small=None, alpha=0.45):
    """Grayscale scene + coloured slick overlay + boundary -> RGB uint8."""
    h, w = display_u8.shape
    rgb = np.stack([display_u8] * 3, -1).astype(np.float32)

    oil = _resize_prob(oil_mask_small.astype(np.float32), (h, w)) > 0.5
    if look_mask_small is not None:
        look = _resize_prob(look_mask_small.astype(np.float32), (h, w)) > 0.5
        look &= ~oil
        rgb[look] = (1 - alpha * 0.6) * rgb[look] + alpha * 0.6 * np.array(LOOK_RGB)

    rgb[oil] = (1 - alpha) * rgb[oil] + alpha * np.array(OIL_RGB)

    edge = ndimage.binary_dilation(oil, np.ones((3, 3))) & ~oil
    rgb[edge] = np.array([255, 235, 230], np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def to_png_b64(arr):
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_thumb_b64(arr, size=320, quality=78):
    """A small JPEG preview, cheap enough to keep in the database.

    The full-resolution scene PNG is stripped before storage (see
    database.add_detection) because base64 PNGs bloat the row. A 320px JPEG is
    a few kilobytes, so the operations console can show the actual scene that
    was analysed rather than a placeholder.
    """
    im = Image.fromarray(arr)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
