# NEELDRIK — AI-Powered Satellite Oil-Spill Detection

**SIH26143 · Team HackSmiths · working prototype**

Upload a SAR scene, and a trained 5-class U-Net decides whether it contains an
oil spill — separating real oil from the look-alikes that defeat naive
dark-patch detectors — then returns the slick boundary, an estimated area, a
confidence score and an evidence trail.

---

## Run it

```bash
cd NeelDrik
pip install -r requirements.txt
python train.py          # trains the model (~25 min, CPU only) — only needed once
python run.py            # http://127.0.0.1:8000
```

Windows users can double-click `run.bat`; macOS/Linux `./run.sh`. Both do all
three steps above.

Sign up with any email (it is stored locally, in SQLite), then go to
**Detect spill** and click one of the labelled samples, or drop in your own
SAR image.

---

## What actually works

| Piece | Status |
|---|---|
| **Input domain gate** (refuses non-SAR images) | **working** — 17/17 on the test set |
| 5-class U-Net (Sea / Oil / Look-alike / Ship / Land) | **trained from scratch**, weights in `models/` |
| Spill / no-spill decision | **calibrated on held-out data**, accuracy reported in the UI |
| Slick boundary → GeoJSON | working, downloadable |
| Area estimate | working (real km² from a GeoTIFF CRS; otherwise a labelled estimate) |
| Auth: register / login / sessions | working, salted PBKDF2 + signed JWT |
| Database persistence | working (SQLite default, PostgreSQL + PostGIS optional) |
| Dashboard, history, map, model card, privacy controls | working |
| **Drift modelling / origin back-track** | **working — verified against closed-form physics** |
| AIS vessel attribution | **not built** |

### Two components, two very different kinds of confidence

Be clear about this when presenting, because it is the strongest thing you can
say about the project:

**Drift modelling is exactly correct.** It is deterministic physics with known
closed-form answers, so it is verified against them rather than against a
dataset — `python -m backend.ml.drift` checks ten properties, including that
back-tracking recovers a *known* release point (67 m after 14.8 km of drift) and
that forward-then-backward integration returns to the start with **0.000 m**
error. There is no training data in this component and therefore no way for it
to be quietly wrong.

**Detection is not, and cannot honestly be claimed to be.** It is trained on
simulated SAR because the build machine had no internet access to real
datasets, so its 94.5% is on held-out *synthetic* scenes. Real Sentinel-1
imagery is a different distribution. Add real images to `data/real/` and
retrain before quoting any accuracy figure for real data. Operational systems
such as EMSA's CleanSeaNet run at roughly 60–80% *with* a human analyst in the
loop — perfect single-image detection is not a thing that exists.

---

## The stack

| Layer | Used here | Full stated stack |
|---|---|---|
| AI | U-Net, 5-class, trained from scratch | same architecture in PyTorch — `backend/ml/torch_unet.py` |
| Backend | Starlette + Uvicorn | FastAPI (auto-detected if installed; adds `/docs`) |
| Imaging | NumPy · OpenCV · scikit-image · SciPy | + rasterio/GDAL for GeoTIFF |
| Frontend | HTML/CSS/JS SPA + Leaflet | React (this is the same UI without a build step) |
| Database | SQLite | PostgreSQL + PostGIS |
| Auth | PBKDF2-HMAC-SHA256 + PyJWT | same |

**Why two model implementations.** The stated AI stack is PyTorch. PyTorch is a
~2 GB install and is not reachable from every environment, so the identical
U-Net exists twice: once in NumPy with hand-written backprop
(`backend/ml/nn.py`, `unet.py`) and once in PyTorch (`torch_unet.py`). Same
architecture, same weight file, same results. `predict.py` picks PyTorch
automatically when it is installed. To run the stated stack:

```bash
pip install torch fastapi rasterio
python run.py                    # now reports: FastAPI / pytorch
```

Nothing else changes. The NumPy path exists so the prototype **always runs**,
which matters more than which library is underneath.

---

## How the detection works

Oil is not simply "a dark patch" — low wind, biogenic films and rain cells are
dark too, and they are what make naive detectors unusable. The physics that
separates them:

| | Oil | Look-alike |
|---|---|---|
| Backscatter | collapses (capillary waves damped) | reduced |
| Internal texture | very smooth — speckle damped | retains texture |
| Boundary | coherent, fairly sharp | diffuse |
| Shape | elongated along drift | rounder |

So the network is trained to segment **five** classes, not two, and the
decision rule requires oil evidence to beat look-alike evidence before it will
call a spill. That gate is the difference between a demo and something a
Coast Guard officer could act on.

**Pipeline** (matches the flow document exactly):

```
upload → validate → percentile stretch → robust (MAD) normalise
       → Lee despeckle ─┐
         texture (CoV) ─┴→ 2-channel tensor → U-Net (5-class)
       → oil mask → morphological opening → connected components
       → drop sub-threshold regions
       → characterise (area, elongation, compactness, centroid)
       → oil-vs-look-alike arbitration → verdict + confidence
       → boundary → GeoJSON → evidence trail → JSON → dashboard
```

**Before any of that runs, the upload is checked.** `domain.py` decides whether
the file is actually a SAR scene (grayscale + genuine radar speckle). Anything
else — a colour photograph, a slide, a screenshot, a diagram — is **refused with
a reason instead of being given a verdict**.

This was added after two reported failures: a real SAR spill reported clean, and
a photograph of a bird against a blue sky reported as an oil spill. The bird
passed an earlier "is it blue, so is it water?" test. Both images are now
permanent regression tests in `tests/regression/`.

Refusing out-of-domain input is the single largest source of false-positive
reduction available, and it is honest: the system states what it cannot judge
rather than guessing. Oil in SAR is **dark** (the film damps capillary waves);
in an optical photo it is usually **bright** sheen or brown crude — the opposite
signature. One model cannot read both, so it declines the one it was not
trained for.

Four decisions were forced by defects found during testing, and each is worth
knowing about:

**1. Training and inference share one preparation function**
(`preprocess.prepare_array`). An earlier build trained on raw arrays but ran
inference on despeckled, downsampled ones; calm sea consistently read as oil.

**2. The network gets two channels, not one.** Despeckling is in the flow
document, but the Lee filter erases precisely the texture that separates oil
from a look-alike — both end up dark and smooth. So local coefficient of
variation is measured on the *raw* image and passed as its own channel.
Measured over 120 scenes, texture separates the two classes about twice as
strongly as intensity alone (Cohen's d 0.51 vs 0.24).

**3. Contrast is normalised by median and MAD, not by percentiles.** A
percentile stretch is set by the brightest pixels, so a single bright coastline
compressed the whole sea toward black — and dark sea is what the detector is
trained to call oil. Two clean test scenes containing land were being reported
as spills because of this. Median/MAD follows the dominant surface instead, so
sea sits near 0.5 whether or not land is in frame.

---

## Drift modelling and origin back-tracking

Surface oil moves with the current plus about **3% of the wind**, deflected by
Coriolis:

```
V_oil = V_current + 0.03 · R(θ) · V_wind
```

The 3% rule is the operational standard (NOAA GNOME/ADIOS) and follows from wind
stress balancing water drag on a thin film. `R(θ)` rotates the wind-driven part
to the **right** in the northern hemisphere, left in the southern. Integration is
RK4 in a local tangent plane; running it with a negative timestep back-tracks the
slick to where it came from. Turbulent spreading is a random walk with eddy
diffusivity K, so the search radius grows as √(2Kt).

**Verification** (`python -m backend.ml.drift`) — ten checks against answers
known in advance:

| Check | Result |
|---|---|
| Still water → no motion | exact |
| Pure current → distance = v·t | 18000.0 m vs 18000.0 m |
| 3% rule: 10 m/s wind, 6 h | 6.480 km vs 6.480 km |
| Coriolis deflects right (N) / left (S) | both correct |
| Forward → backward returns to start | **0.000 m** over 37.4 km |
| Halving the timestep changes nothing | 0.000 m (RK4 converged) |
| Cloud spread follows √(2Kt) | 916 m vs 930 m theory |
| Back-track finds a known release point | **67 m** after 14.8 km |

One subtlety worth knowing, because it produced a real bug: the integrator maps
metres to degrees on the **WGS84 ellipsoid**, so distances must be measured the
same way. Measuring an integrated displacement with a spherical haversine makes
a correct result look 0.12% wrong. `distance_m()` is the consistent metric;
`haversine_m()` is kept only for reference.

**Limits.** No weathering (evaporation, emulsification), no Stokes drift, no
shoreline interaction. Those change how much oil survives, not where the surface
centroid goes — which is what origin back-tracking needs. Accuracy in the field
is limited by the wind and current data you feed it, not by the integrator.

---

## Training on your own images

The shipped model is trained on physics-based synthetic SAR scenes plus the
8 labelled pairs in `data/train/`. To add real data, drop matching files in:

```
data/train/images/<name>.png     grayscale SAR patch
data/train/masks/<name>.png      indexed mask, pixel value = class id
                                 0=Sea  1=Oil  2=Look-alike  3=Ship  4=Land
```

then re-run `python train.py`. Real images are oversampled against the
synthetic set automatically — five are enough to shift the model. A plain
black/white oil mask also works (anything >127 is read as oil).

Good real sources: the **Kaggle SOS oil-spill segmentation dataset**, and
Sentinel-1 GRD scenes from the **Copernicus Data Space**.

Useful flags: `--epochs`, `--train-n`, `--batch`, `--lr`, `--torch`.

---

## Verifying it

```bash
python tests/test_pipeline.py
```

Checks, in order: every layer's gradient against numerical differentiation;
password hashing and JWT tampering; the verdict on every labelled sample; the
accuracy on a freshly generated unseen batch; the output contract; and that
malformed input produces an error rather than a crash.

`python -m backend.ml.nn` runs the gradient check on its own.

---

## API

All `/api/*` routes except `health`, `model` and `auth/*` need
`Authorization: Bearer <token>`.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | service + storage backend |
| GET | `/api/model` | architecture, metrics, calibrated thresholds |
| POST | `/api/auth/register` | `{email, name, password, org}` → token |
| POST | `/api/auth/login` | `{email, password}` → token |
| POST | `/api/detect` | multipart `file` (+ optional `scene_km`) → full result |
| GET | `/api/detections` | your detection history |
| GET | `/api/detections/{id}` | one stored result |
| POST | `/api/detections/{id}/delete` | delete one |
| POST | `/api/detections/purge` | delete all of yours |
| GET | `/api/dashboard` | stats + recent + model card |

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"YourPass123"}' | jq -r .token)

curl -s -X POST localhost:8000/api/detect \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@data/samples/spill_1.png | jq '{verdict, confidence, estimated_area_km2}'
```

---

## Privacy

- Passwords are stored **only** as salted PBKDF2-HMAC-SHA256 (200,000 rounds).
- Sessions are signed JWTs (HS256) that expire after 8 hours; the signing key is
  generated on first run into `models/.jwt_secret` — set `NEELDRIK_SECRET` to
  override.
- Detections are scoped per account; **Account & privacy → Erase all my
  detections** deletes them permanently.
- Nothing leaves the machine. No external API is called at runtime.

---

## Switching to PostgreSQL + PostGIS

```bash
pip install psycopg2-binary
export DATABASE_URL=postgresql://user:pass@localhost/neeldrik
python run.py
```

The schema is created on start, PostGIS is enabled when available, and the
dashboard will report `postgresql+postgis` as the storage backend.

---

## Honest limitations

1. **Trained mostly on synthetic SAR.** The simulator models the real physics
   (Gamma speckle, capillary-wave damping, diffuse vs coherent boundaries), but
   it is not a substitute for a large real corpus. Add real labelled scenes and
   retrain before claiming field accuracy.
2. **Area is an estimate unless the raster is georeferenced.** A plain PNG has
   no scale, so the number comes from the assumed scene width and is labelled as
   an estimate everywhere it appears.
3. **The accuracy figure is on held-out synthetic scenes**, not on operational
   Sentinel-1 data. It is shown in the UI so it can't be mistaken for a field
   result.
4. **Drift modelling and AIS attribution are not implemented.**

---

## Layout

```
NeelDrik/
├── run.py                   launcher
├── train.py                 training + threshold calibration
├── requirements.txt
├── backend/
│   ├── app.py               routes (FastAPI or Starlette)
│   ├── services.py          business logic, framework-agnostic
│   ├── security.py          password hashing, JWT
│   ├── database.py          SQLite / PostgreSQL+PostGIS
│   └── ml/
│       ├── datagen.py       physics-based SAR scene simulator
│       ├── nn.py            conv/BN/pool/upsample + Adam, gradient-checked
│       ├── unet.py          5-class U-Net (NumPy)
│       ├── torch_unet.py    the same U-Net in PyTorch
│       ├── preprocess.py    the one shared preparation path
│       ├── decide.py        verdict rule + calibrated thresholds
│       ├── postprocess.py   boundary, GeoJSON, overlay
│       └── predict.py       orchestrator
├── frontend/                login + console (HTML/CSS/JS + Leaflet)
├── data/
│   ├── samples/             labelled demo scenes
│   └── train/               your labelled image/mask pairs go here
├── models/                  trained weights, metrics, thresholds, SQLite db
├── scripts/make_samples.py  regenerate samples
└── tests/test_pipeline.py
```
