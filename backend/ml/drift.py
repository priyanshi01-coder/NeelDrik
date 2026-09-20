"""Lagrangian oil-drift model -- forward and backward.

Unlike the detector, this component can be made exactly correct: it is
deterministic physics with known closed-form cases, so it is verified against
analytical solutions rather than against a dataset.  `python -m backend.ml.drift`
runs that verification.

THE PHYSICS
-----------
Surface oil is carried by the current and dragged by the wind:

    V_oil = V_current + a * R(theta) * V_wind

* a  -- the wind drift factor, ~3% of wind speed.  This "3% rule" is the
        standard used operationally (NOAA GNOME, ADIOS) and comes from the
        balance of wind stress against water drag on a thin surface film.
* R(theta) -- Coriolis deflection.  In the northern hemisphere the wind-driven
        component is deflected to the RIGHT of the wind vector (left in the
        southern hemisphere), typically 0-25 deg; 15 deg is a common default.
* The current term has no factor: the slick is embedded in the water.

Integration is RK4 in a local tangent plane, converting metres to degrees with
the local radius of curvature.  Backward tracking is the same integration with
a negative timestep, which is exactly reversible for a time-independent field --
that reversibility is what the round-trip test checks.

Turbulent spreading is added as a random walk with horizontal eddy diffusivity
K (m^2/s); the particle cloud's growth follows sigma = sqrt(2*K*t).

WHAT THIS DOES NOT DO
---------------------
No weathering (evaporation, emulsification, dispersion), no Stokes drift from
waves, no shoreline interaction.  Those change how much oil survives and how it
behaves in detail; they do not change where the surface centroid goes, which is
what origin back-tracking needs.
"""
from __future__ import annotations

import math

import numpy as np

EARTH_R = 6_371_000.0
DEG = math.pi / 180.0

DEFAULT_WIND_FACTOR = 0.03      # the 3% rule
DEFAULT_DEFLECTION = 15.0       # degrees, right in N hemisphere
DEFAULT_DIFFUSIVITY = 5.0       # m^2/s, typical coastal horizontal eddy value


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def metres_per_degree(lat_deg):
    """Local metres per degree of latitude and longitude."""
    lat = lat_deg * DEG
    m_lat = 111_132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_lon = 111_412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_lat, max(m_lon, 1e-6)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance on a SPHERE of radius EARTH_R.

    Kept for reference only.  Do not mix it with the integrator: the integrator
    converts metres to degrees on the WGS84 ELLIPSOID, and a sphere disagrees
    with that by ~0.2% at the equator.  Measuring an integrated displacement
    with this function makes a correct result look 0.12% wrong.
    """
    p1, p2 = lat1 * DEG, lat2 * DEG
    dp, dl = (lat2 - lat1) * DEG, (lon2 - lon1) * DEG
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


def distance_m(lat1, lon1, lat2, lon2):
    """Distance using the SAME ellipsoidal metres-per-degree the integrator uses.

    This is the exact inverse of the integration mapping, so a displacement
    produced by `track` measures back to speed x time to within rounding.
    """
    m_lat, m_lon = metres_per_degree(0.5 * (lat1 + lat2))
    dy = (lat2 - lat1) * m_lat
    dx = (lon2 - lon1) * m_lon
    return math.hypot(dx, dy)


def uv_from_speed_dir(speed, direction_deg, convention="from"):
    """Meteorological vector -> (east, north) components in m/s.

    convention='from' is the meteorological standard: a 90 deg wind blows FROM
    the east, so it pushes things WEST (u negative).  Ocean currents are
    normally quoted as 'to', the direction of travel.
    """
    th = direction_deg * DEG
    u, v = speed * math.sin(th), speed * math.cos(th)
    return (-u, -v) if convention == "from" else (u, v)


def _deflect(u, v, deg, lat):
    """Rotate a vector by `deg`, to the right in the N hemisphere."""
    s = -1.0 if lat >= 0 else 1.0           # clockwise (right) when lat >= 0
    th = s * deg * DEG
    c, si = math.cos(th), math.sin(th)
    return u * c - v * si, u * si + v * c


# --------------------------------------------------------------------------- #
# field
# --------------------------------------------------------------------------- #
class Field:
    """Wind and current. Constant, or interpolated from a time series.

    series entries: {"t_h": hours_from_start, "wind_speed", "wind_dir",
                     "cur_speed", "cur_dir"}
    """

    def __init__(self, wind_speed=0.0, wind_dir=0.0, cur_speed=0.0, cur_dir=0.0,
                 series=None, wind_convention="from", cur_convention="to"):
        self.const = (wind_speed, wind_dir, cur_speed, cur_dir)
        self.series = sorted(series, key=lambda r: r["t_h"]) if series else None
        self.wc, self.cc = wind_convention, cur_convention

    def at(self, t_h):
        if not self.series:
            ws, wd, cs, cd = self.const
        else:
            ts = [r["t_h"] for r in self.series]
            if t_h <= ts[0]:
                r = self.series[0]
                ws, wd, cs, cd = r["wind_speed"], r["wind_dir"], r["cur_speed"], r["cur_dir"]
            elif t_h >= ts[-1]:
                r = self.series[-1]
                ws, wd, cs, cd = r["wind_speed"], r["wind_dir"], r["cur_speed"], r["cur_dir"]
            else:
                i = int(np.searchsorted(ts, t_h))
                a, b = self.series[i - 1], self.series[i]
                f = (t_h - a["t_h"]) / max(b["t_h"] - a["t_h"], 1e-9)
                def lerp(k):
                    return a[k] + f * (b[k] - a[k])
                # interpolate direction through vector components, not degrees
                au, av = uv_from_speed_dir(a["wind_speed"], a["wind_dir"], self.wc)
                bu, bv = uv_from_speed_dir(b["wind_speed"], b["wind_dir"], self.wc)
                cu_, cv_ = uv_from_speed_dir(a["cur_speed"], a["cur_dir"], self.cc)
                du_, dv_ = uv_from_speed_dir(b["cur_speed"], b["cur_dir"], self.cc)
                return ((au + f * (bu - au), av + f * (bv - av)),
                        (cu_ + f * (du_ - cu_), cv_ + f * (dv_ - cv_)))
        return (uv_from_speed_dir(ws, wd, self.wc),
                uv_from_speed_dir(cs, cd, self.cc))

    def velocity(self, lat, t_h, wind_factor=DEFAULT_WIND_FACTOR,
                 deflection=DEFAULT_DEFLECTION):
        """Total oil drift velocity (east, north) in m/s."""
        (wu, wv), (cu, cv) = self.at(t_h)
        du, dv = _deflect(wu, wv, deflection, lat)
        return cu + wind_factor * du, cv + wind_factor * dv


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #
def _step_rk4(lat, lon, t_h, dt_s, field, wf, defl):
    """One RK4 step. dt_s < 0 integrates backwards."""
    dt_h = dt_s / 3600.0

    def deriv(la, lo, t):
        u, v = field.velocity(la, t, wf, defl)
        m_lat, m_lon = metres_per_degree(la)
        return v / m_lat, u / m_lon          # deg/s

    k1a, k1o = deriv(lat, lon, t_h)
    k2a, k2o = deriv(lat + 0.5 * dt_s * k1a, lon + 0.5 * dt_s * k1o, t_h + 0.5 * dt_h)
    k3a, k3o = deriv(lat + 0.5 * dt_s * k2a, lon + 0.5 * dt_s * k2o, t_h + 0.5 * dt_h)
    k4a, k4o = deriv(lat + dt_s * k3a, lon + dt_s * k3o, t_h + dt_h)
    lat2 = lat + dt_s / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
    lon2 = lon + dt_s / 6.0 * (k1o + 2 * k2o + 2 * k3o + k4o)
    return lat2, lon2


def track(lat, lon, hours, field, backward=False, dt_min=10.0,
          wind_factor=DEFAULT_WIND_FACTOR, deflection=DEFAULT_DEFLECTION):
    """Integrate one particle. Returns list of {t_h, lat, lon}.

    backward=True walks the slick back toward where it came from.
    """
    if hours <= 0:
        return [{"t_h": 0.0, "lat": lat, "lon": lon}]
    sign = -1.0 if backward else 1.0
    dt_s = sign * dt_min * 60.0
    n = max(1, int(round(hours * 60.0 / dt_min)))
    out = [{"t_h": 0.0, "lat": float(lat), "lon": float(lon)}]
    t_h = 0.0
    for _ in range(n):
        lat, lon = _step_rk4(lat, lon, t_h, dt_s, field, wind_factor, deflection)
        t_h += sign * dt_min / 60.0
        out.append({"t_h": round(t_h, 4), "lat": float(lat), "lon": float(lon)})
    return out


def cloud(lat, lon, hours, field, n=300, backward=False, dt_min=10.0,
          diffusivity=DEFAULT_DIFFUSIVITY, seed=0,
          wind_factor=DEFAULT_WIND_FACTOR, deflection=DEFAULT_DEFLECTION):
    """Particle cloud with turbulent diffusion -> endpoints + spread statistics."""
    rng = np.random.default_rng(seed)
    sign = -1.0 if backward else 1.0
    dt_s = sign * dt_min * 60.0
    steps = max(1, int(round(hours * 60.0 / dt_min)))
    sigma_step = math.sqrt(2.0 * diffusivity * abs(dt_s))      # metres per step

    lats = np.full(n, float(lat))
    lons = np.full(n, float(lon))
    t_h = 0.0
    for _ in range(steps):
        for i in range(n):
            lats[i], lons[i] = _step_rk4(lats[i], lons[i], t_h, dt_s,
                                         field, wind_factor, deflection)
        m_lat, m_lon = metres_per_degree(float(lats.mean()))
        lats += rng.normal(0.0, sigma_step, n) / m_lat
        lons += rng.normal(0.0, sigma_step, n) / m_lon
        t_h += sign * dt_min / 60.0

    clat, clon = float(lats.mean()), float(lons.mean())
    d = [distance_m(clat, clon, la, lo) for la, lo in zip(lats, lons)]
    return {
        "centroid": {"lat": clat, "lon": clon},
        "points": [{"lat": float(a), "lon": float(b)} for a, b in zip(lats, lons)],
        "radius_m": {"p50": float(np.percentile(d, 50)),
                     "p90": float(np.percentile(d, 90)),
                     "max": float(np.max(d))},
        "hours": hours, "backward": backward, "particles": n,
    }


# --------------------------------------------------------------------------- #
# origin estimation
# --------------------------------------------------------------------------- #
def backtrack_origin(lat, lon, hours_list, field, **kw):
    """Back-track a slick to candidate origins over several elapsed times.

    Returns the trajectory plus one origin estimate per elapsed time -- the
    "origin/time window" a vessel search is run against.
    """
    hours_list = sorted(set(float(h) for h in hours_list if h > 0))
    if not hours_list:
        raise ValueError("hours_list must contain at least one positive value")
    longest = max(hours_list)
    TRACK_KW = ("dt_min", "wind_factor", "deflection")
    CLOUD_KW = ("dt_min", "diffusivity", "wind_factor", "deflection")
    traj = track(lat, lon, longest, field, backward=True,
                 **{k: v for k, v in kw.items() if k in TRACK_KW})

    candidates = []
    for h in hours_list:
        c = cloud(lat, lon, h, field, backward=True, n=int(kw.get("n", 200)),
                  seed=int(h * 7) % 9973,
                  **{k: v for k, v in kw.items() if k in CLOUD_KW})
        candidates.append({
            "hours_before": h,
            "lat": c["centroid"]["lat"], "lon": c["centroid"]["lon"],
            "uncertainty_m": round(c["radius_m"]["p90"], 1),
            "distance_from_slick_km": round(
                distance_m(lat, lon, c["centroid"]["lat"],
                           c["centroid"]["lon"]) / 1000.0, 3),
        })
    return {"slick": {"lat": lat, "lon": lon},
            "trajectory": traj, "candidates": candidates}


def to_geojson(result):
    """Trajectory + origin candidates -> GeoJSON for the map."""
    feats = [{
        "type": "Feature",
        "properties": {"kind": "backtrack", "hours": abs(result["trajectory"][-1]["t_h"])},
        "geometry": {"type": "LineString",
                     "coordinates": [[p["lon"], p["lat"]] for p in result["trajectory"]]},
    }, {
        "type": "Feature",
        "properties": {"kind": "slick"},
        "geometry": {"type": "Point",
                     "coordinates": [result["slick"]["lon"], result["slick"]["lat"]]},
    }]
    for c in result["candidates"]:
        feats.append({
            "type": "Feature",
            "properties": {"kind": "origin", "hours_before": c["hours_before"],
                           "uncertainty_m": c["uncertainty_m"],
                           "distance_km": c["distance_from_slick_km"]},
            "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]},
        })
    return {"type": "FeatureCollection", "features": feats}


# --------------------------------------------------------------------------- #
# verification against closed-form solutions
# --------------------------------------------------------------------------- #
def verify(verbose=True):
    """Check the integrator against cases with known exact answers."""
    fails = []

    def chk(name, cond, detail=""):
        (fails.append(name) if not cond else None)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
                  f"{('  ' + detail) if detail else ''}")

    # 1. no forcing -> no motion
    f0 = Field()
    t = track(10.0, 72.0, 12, f0)
    chk("still water: particle does not move",
        distance_m(10.0, 72.0, t[-1]["lat"], t[-1]["lon"]) < 1e-6)

    # 2. pure current: displacement must equal speed x time exactly
    f1 = Field(cur_speed=0.5, cur_dir=90.0)          # 0.5 m/s due EAST
    t = track(10.0, 72.0, 10, f1)
    want = 0.5 * 10 * 3600.0
    got = distance_m(10.0, 72.0, t[-1]["lat"], t[-1]["lon"])
    chk("pure current: distance = v*t", abs(got - want) / want < 2e-4,
        f"{got:.1f} m vs {want:.1f} m exact")
    chk("pure current: heading is due east",
        abs(t[-1]["lat"] - 10.0) < 1e-6 and t[-1]["lon"] > 72.0,
        f"dlat={t[-1]['lat']-10.0:.2e}")

    # 3. the 3% rule: 10 m/s wind -> 0.30 m/s drift
    f2 = Field(wind_speed=10.0, wind_dir=270.0)      # FROM the west -> blows east
    t = track(0.0, 80.0, 6, f2, deflection=0.0)
    want = 0.03 * 10.0 * 6 * 3600.0
    got = distance_m(0.0, 80.0, t[-1]["lat"], t[-1]["lon"])
    chk("3% rule: 10 m/s wind for 6 h = 6.48 km",
        abs(got - want) / want < 2e-4, f"{got/1000:.3f} km vs {want/1000:.3f} km")

    # 4. Coriolis deflects to the RIGHT in the northern hemisphere
    tn = track(20.0, 80.0, 6, f2, deflection=20.0)
    chk("N hemisphere: wind drift deflects right (southward for an eastward wind)",
        tn[-1]["lat"] < 20.0, f"dlat={tn[-1]['lat']-20.0:+.4f} deg")
    ts = track(-20.0, 80.0, 6, f2, deflection=20.0)
    chk("S hemisphere: deflects left (northward)",
        ts[-1]["lat"] > -20.0, f"dlat={ts[-1]['lat']+20.0:+.4f} deg")

    # 5. reversibility -- the property back-tracking depends on
    f3 = Field(wind_speed=8.0, wind_dir=200.0, cur_speed=0.35, cur_dir=60.0)
    fwd = track(15.0, 73.0, 18, f3)
    end = fwd[-1]
    back = track(end["lat"], end["lon"], 18, f3, backward=True)
    err = distance_m(15.0, 73.0, back[-1]["lat"], back[-1]["lon"])
    chk("forward then backward returns to the start", err < 1.0,
        f"round-trip error {err:.3f} m over "
        f"{distance_m(15.0,73.0,end['lat'],end['lon'])/1000:.1f} km")

    # 6. RK4 convergence: halving the step must not change the answer materially
    a = track(15.0, 73.0, 18, f3, dt_min=20.0)[-1]
    b = track(15.0, 73.0, 18, f3, dt_min=2.5)[-1]
    chk("result is step-size independent (RK4 converged)",
        distance_m(a["lat"], a["lon"], b["lat"], b["lon"]) < 1.0,
        f"{distance_m(a['lat'],a['lon'],b['lat'],b['lon']):.3f} m apart")

    # 7. diffusion grows as sqrt(2*K*t)
    c = cloud(12.0, 74.0, 12, Field(), n=600, diffusivity=10.0, seed=3)
    want_sigma = math.sqrt(2 * 10.0 * 12 * 3600.0)
    got_sigma = c["radius_m"]["p50"] / 1.1774          # p50 of Rayleigh -> sigma
    chk("cloud spread follows sqrt(2*K*t)",
        abs(got_sigma - want_sigma) / want_sigma < 0.12,
        f"sigma {got_sigma:.0f} m vs {want_sigma:.0f} m theory")

    # 8. back-tracking finds the true release point
    src_lat, src_lon = 18.9, 72.3
    f4 = Field(wind_speed=7.0, wind_dir=225.0, cur_speed=0.25, cur_dir=45.0)
    drifted = track(src_lat, src_lon, 9, f4)[-1]
    est = backtrack_origin(drifted["lat"], drifted["lon"], [9], f4,
                           n=200, diffusivity=2.0)["candidates"][0]
    err_km = distance_m(src_lat, src_lon, est["lat"], est["lon"]) / 1000.0
    chk("back-track recovers a known release point", err_km < 0.5,
        f"{err_km*1000:.0f} m from truth, "
        f"drifted {est['distance_from_slick_km']:.1f} km")

    if verbose:
        print(f"\n  {'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return fails


if __name__ == "__main__":
    import sys
    print("NEELDRIK drift model - verification against closed-form solutions\n")
    sys.exit(1 if verify() else 0)
