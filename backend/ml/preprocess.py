"""SAR preprocessing: format/data validation -> radiometric preparation ->
speckle handling -> normalisation -> resize, per the NEELDRIK flow document.

Uses rasterio for GeoTIFF (georeferencing) when installed and falls back to
Pillow/OpenCV for ordinary rasters, so the prototype accepts both a real
Sentinel-1 GeoTIFF and a plain PNG/JPEG screenshot.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image
from scipy import ndimage

try:                                   # optional, only needed for GeoTIFF
    import rasterio
    from rasterio.io import MemoryFile
    HAVE_RASTERIO = True
except Exception:                      # pragma: no cover
    HAVE_RASTERIO = False

try:
    import cv2
    HAVE_CV2 = True
except Exception:                      # pragma: no cover
    HAVE_CV2 = False

MODEL_SIZE = 128
MAX_PIXELS = 40_000_000


class BadImage(ValueError):
    pass


# --------------------------------------------------------------------------- #
def lee_filter(img, size=5, cu=0.523):
    """Classic Lee speckle filter.

    SAR speckle is multiplicative; Lee shrinks the local mean toward the pixel
    value in proportion to how much local variance exceeds the speckle floor,
    so flat sea is smoothed while slick edges are preserved.
    """
    mean = ndimage.uniform_filter(img, size)
    sq = ndimage.uniform_filter(img ** 2, size)
    var = np.maximum(sq - mean ** 2, 0.0)
    ci2 = var / np.maximum(mean ** 2, 1e-8)
    w = np.clip(1.0 - (cu * cu) / np.maximum(ci2, 1e-8), 0.0, 1.0)
    return mean + w * (img - mean)


def percentile_stretch(img, lo=1.0, hi=99.0):
    a, b = np.percentile(img, [lo, hi])
    if b - a < 1e-9:
        b = a + 1e-9
    return np.clip((img - a) / (b - a), 0.0, 1.0)


def robust_normalise(img, spread=0.16):
    """Centre on the median, scale by the MAD.

    A plain percentile stretch is set by the brightest pixels in the scene, so
    one bright coastline compresses the whole sea toward black -- and dark sea
    is exactly what the detector is trained to call oil.  The median and MAD are
    driven by the DOMINANT surface instead (nearly always water), so sea lands
    near 0.5 whether or not land is in frame, oil falls well below it and land
    rises above.  This is the single most important normalisation choice in the
    pipeline: without it, any scene containing a coastline reads as a spill.
    """
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    scale = max(1.4826 * mad, 1e-3)
    return np.clip(0.5 + spread * (img - med) / scale, 0.0, 1.0)


# --------------------------------------------------------------------------- #
def read_rgb(data: bytes, cap: int = 1400):
    """bytes -> uint8 HxWx3 for the domain check.

    Never downscales: resizing averages radar speckle away, and speckle is the
    signal that tells a SAR scene apart from a photograph.  Large images are
    centre-cropped instead.
    """
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        im = im.convert("RGB")
    except Exception:
        im = _geotiff_preview(data)     # a float32 GeoTIFF PIL cannot open
        if im is None:
            return None
    w, h = im.size
    if w > cap or h > cap:
        l, t = max(0, (w - cap) // 2), max(0, (h - cap) // 2)
        im = im.crop((l, t, min(w, l + cap), min(h, t + cap)))
    return np.asarray(im)


# --------------------------------------------------------------------------- #
# A GeoTIFF reader that needs no GDAL
def _geotiff_preview(data: bytes):
    """An 8-bit RGB view of a plain GeoTIFF, for the domain check only.

    PIL cannot open a two-band float32 Sigma0 export, and without this the
    domain gate refuses a real Sentinel-1 product as "not an image".
    """
    got = _plain_geotiff(data)
    if got is None:
        return None
    a = got[0]
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    valid = np.isfinite(a) & (a != 0.0)
    if not valid.any():
        return None
    lo, hi = np.percentile(a[valid], [2.0, 98.0])
    hi = hi if hi - lo > 1e-6 else lo + 1e-6
    filled = np.where(valid, a, np.median(a[valid]))
    u8 = (np.clip((filled - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([u8] * 3, -1))


# --------------------------------------------------------------------------- #
_TIFF_FMT = {1: 'B', 2: 's', 3: 'H', 4: 'I', 6: 'b', 8: 'h', 9: 'i', 11: 'f', 12: 'd'}
_TIFF_SZ = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def _plain_geotiff(data: bytes):
    """Pixels + WGS84 footprint from an uncompressed, strip-based GeoTIFF.

    rasterio is deliberately left out of the deployed build -- GDAL is far too
    big for a free instance -- but a Sentinel-1 export still carries its own
    coordinates, and throwing them away would force the analyst to type a
    position the file already knows. This reads the handful of tags needed for
    that: the tie point and the pixel scale. Anything unusual returns None and
    the caller falls back to the ordinary image path.
    """
    import struct
    if data[:2] not in (b'II', b'MM'):
        return None
    bo = '<' if data[:2] == b'II' else '>'
    try:
        if struct.unpack_from(bo + 'H', data, 2)[0] != 42:
            return None                                  # BigTIFF, not ours
        ifd = struct.unpack_from(bo + 'I', data, 4)[0]
        n = struct.unpack_from(bo + 'H', data, ifd)[0]
        t = {}
        for i in range(n):
            e = ifd + 2 + i * 12
            tag, typ, cnt = struct.unpack_from(bo + 'HHI', data, e)
            size = _TIFF_SZ.get(typ, 1) * cnt
            vo = e + 8 if size <= 4 else struct.unpack_from(bo + 'I', data, e + 8)[0]
            if typ == 2:
                t[tag] = data[vo:vo + cnt].split(b'\0')[0].decode('latin1', 'replace')
            else:
                t[tag] = list(struct.unpack_from(bo + _TIFF_FMT.get(typ, 'B') * cnt,
                                                 data, vo))
        if 33550 not in t or 33922 not in t:
            return None                                  # no georeferencing: not ours
        if t.get(259, [1])[0] != 1 or 273 not in t:
            return None                                  # compressed or tiled
        w, h = t[256][0], t[257][0]
        if w * h > MAX_PIXELS:
            raise BadImage("raster too large")
        spp, bits, sfmt = t.get(277, [1])[0], t[258][0], t.get(339, [1])[0]
        dt = {(32, 3): np.float32, (32, 1): np.uint32, (16, 1): np.uint16,
              (8, 1): np.uint8, (8, 2): np.int8, (16, 2): np.int16}.get((bits, sfmt))
        if dt is None:
            return None
        buf = b''.join(data[o:o + c] for o, c in zip(t[273], t[279]))
        arr = np.frombuffer(buf, dtype=np.dtype(dt).newbyteorder(bo))
        if arr.size < w * h * spp:
            return None
        arr = arr[:w * h * spp].reshape(h, w, spp) if spp > 1 else arr[:w * h].reshape(h, w)

        sx, sy = float(t[33550][0]), float(t[33550][1])
        i, j, _k, x, y, _z = (float(v) for v in t[33922][:6])
        west, north = x - i * sx, y + j * sy
        crs = (t.get(34737, '') or '').strip('|').strip()
        # A geographic CRS gives degrees; convert to a ground size so the area
        # estimate stays in km^2. Everything else is treated as metres.
        if abs(sx) < 1.0 and abs(west) <= 180.0 and abs(north) <= 90.0:
            lat = north - h * sy / 2.0
            per_px = float(np.sqrt(abs(sx) * np.cos(np.radians(lat)) * abs(sy))) * 111_320.0
            crs = crs or 'WGS 84'
            if '4326' not in crs:
                crs = f'EPSG:4326 ({crs})' if crs else 'EPSG:4326'
        else:
            per_px = abs(sx)
        geo = {"georeferenced": True, "crs": crs,
               "transform": [sx, 0.0, west, 0.0, -sy, north],
               "pixel_size_m": per_px}
        return arr, geo
    except BadImage:
        raise
    except Exception:
        return None


def read_raster(data: bytes):
    """bytes -> (float array HxW, geo dict). Raises BadImage on unreadable input."""
    geo = {"georeferenced": False, "crs": None, "transform": None,
           "pixel_size_m": None}

    if HAVE_RASTERIO:
        try:
            with MemoryFile(data) as mf, mf.open() as ds:
                if ds.width * ds.height > MAX_PIXELS:
                    raise BadImage("raster too large")
                band = ds.read(1).astype(np.float32)
                if ds.crs is not None and ds.transform is not None:
                    geo.update(georeferenced=True, crs=str(ds.crs),
                               transform=[float(v) for v in ds.transform[:6]],
                               pixel_size_m=abs(float(ds.transform[0])))
                return band, geo
        except BadImage:
            raise
        except Exception:
            pass                        # not a GeoTIFF -> ordinary image path

    plain = _plain_geotiff(data)        # no GDAL needed for the common case
    if plain is not None:
        band, geo = plain
        if band.ndim == 3:              # pick VV: over ocean it sits above VH
            meds = []
            for c in range(band.shape[-1]):
                ch = band[..., c]
                v = ch[np.isfinite(ch) & (ch != 0.0)]
                meds.append(float(np.median(v)) if v.size else -np.inf)
            band = band[..., int(np.argmax(meds))]
        return band.astype(np.float32), geo

    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        raise BadImage(f"unreadable image: {exc}") from exc
    if im.width * im.height > MAX_PIXELS:
        raise BadImage("image too large")
    if im.mode not in ("L", "I;16", "I", "F"):
        im = im.convert("L")
    return np.asarray(im).astype(np.float32), geo


def cov_map(img, size=7, clip=1.1):
    """Local coefficient of variation = std/mean over a small window.

    The texture channel.  Because CoV is normalised by the local mean it
    responds jointly to how far backscatter has collapsed and to how much
    residual speckle survives, which is precisely the pair of effects that
    distinguishes a damped oil film from a low-wind or biogenic look-alike.
    Measured on 120 generated scenes it separates the two classes about twice
    as strongly as intensity alone (Cohen's d 0.51 vs 0.24).

    It is computed on the RAW stretched image, before despeckling: the Lee
    filter is what would erase this signal, so the network gets it as its own
    channel rather than losing it to preprocessing.
    """
    m = ndimage.uniform_filter(img, size)
    sq = ndimage.uniform_filter(img ** 2, size)
    var = np.maximum(sq - m ** 2, 0.0)
    return np.clip(np.sqrt(var) / np.maximum(m, 1e-3), 0, clip) / clip


def _resize(a, size):
    if HAVE_CV2:
        return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
                      .resize((size, size), Image.BILINEAR), np.float32) / 255.0


def prepare_array(raw, size: int = MODEL_SIZE, despeckle: bool = True):
    """float/uint8 array -> (model_input 2xSxS float32, display uint8, work).

    THE single preparation path.  train.py pushes its training scenes through
    this same function, so the network is trained on exactly the distribution it
    is given at inference.

    Two channels:
        0  despeckled intensity  -- where is it dark
        1  local CoV of the raw  -- how textured is it
    Both physical cues reach the network; neither is filtered away.
    """
    raw = np.asarray(raw, np.float32)
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    if raw.size == 0 or not np.isfinite(raw).any():
        raise BadImage("empty or non-finite raster")
    raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))

    h, w = raw.shape
    stretched = percentile_stretch(raw)                # bring any input to 0..1
    texture = cov_map(stretched)                       # measured BEFORE filtering

    work = robust_normalise(stretched)                 # land-insensitive contrast
    if despeckle:
        work = robust_normalise(lee_filter(work, size=5))

    small = np.stack([_resize(work, size), _resize(texture, size)]).astype(np.float32)

    if HAVE_CV2 and w > 768:
        disp = cv2.resize(work, (768, max(1, int(768 * h / w))),
                          interpolation=cv2.INTER_AREA)
    else:
        disp = work
    return small, (np.clip(disp, 0, 1) * 255).astype(np.uint8), work


def preprocess(data: bytes, size: int = MODEL_SIZE, despeckle: bool = True):
    """bytes -> dict(model_input, display, geo, stats)."""
    raw, geo = read_raster(data)
    h, w = (raw.shape[0], raw.shape[1])
    small, disp, work = prepare_array(raw, size, despeckle)

    return {
        "model_input": small[None],                            # (1,2,S,S)
        "display": disp,
        "geo": geo,
        "source_shape": (int(h), int(w)),
        "stats": {
            "mean": round(float(work.mean()), 4),
            "std": round(float(work.std()), 4),
            "dark_fraction": round(float((work < 0.25).mean()), 4),
        },
    }


def pixel_area_km2(geo, source_shape, model_size=MODEL_SIZE,
                   assumed_scene_km=None):
    """Area of one model pixel in km^2.

    Real georeferencing is used when present; otherwise the caller's assumed
    scene width is used and the result must be labelled an estimate, exactly as
    the flow document requires.
    """
    h, w = source_shape
    if geo.get("georeferenced") and geo.get("pixel_size_m"):
        m = geo["pixel_size_m"] * (w / model_size)
        return (m * m) / 1e6, True
    km = float(assumed_scene_km or 40.0)
    per_px_km = km / model_size
    return per_px_km * per_px_km, False
