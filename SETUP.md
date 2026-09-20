# NEELDRIK — running it, and getting real numbers

## Run it

```bash
pip install -r requirements.txt
python train.py                 # NumPy engine
python train.py --torch         # PyTorch engine (same data, same calibration)
python run.py                   # http://127.0.0.1:8000
```

## Verify it

```bash
python tests/test_pipeline.py   # 39 checks: autograd, security, detection,
                                # drift physics, wind gating, split leakage
python tests/test_parity.py     # NumPy vs PyTorch must agree (needs torch)
python scripts/evaluate.py --wind
```

`test_parity.py` matters now that PyTorch is installed. The project ships two
implementations of one network; that is only defensible if they give the same
answer. The test loads one weight file into both and requires the probability
maps to match to 1e-4 and every verdict to agree. Run it before quoting an
accuracy figure, so you are not quoting one engine while running the other.

## Getting numbers that mean something

Everything currently reported is measured on **synthetic** scenes. To replace
that with real performance:

### 1. Request the dataset

Krestenitis et al. (2019), Sentinel-1, ~1,002 train + 110 test images, five
classes identical to ours — sea / oil spill / look-alike / ship / land.

https://m4d.iti.gr/oil-spill-detection-dataset/

Requires an **institutional email**, a project title and abstract. **Students
cannot request it directly — your supervisor must submit it.** Ask your faculty
mentor early; approval is not instant.

### 2. Drop it in

```
data/real/images/<name>.png     SAR patch
data/real/masks/<name>.png      mask
```

Masks are decoded automatically in three forms, and the loader prints which it
used:

| Form | Handling |
|---|---|
| Single-channel 0–4 label masks | used directly |
| RGB colour masks | cyan→oil, red→look-alike, brown→ship, green→land, black→sea |
| Binary 0/255 | treated as an oil mask |

If it prints `AMBIGUOUS`, stop and check the masks before trusting any score.

### 3. Scene grouping

Training patches are cut from a much smaller number of source scenes. If
patches from one scene land in both train and test, the model is graded on sea
it has already memorised and the score is inflated. This is the single most
common cause of implausible accuracy figures.

By default the scene id is the filename with any trailing `_<digits>` removed.
If your filenames do not encode the scene, write `data/real/scenes.json`:

```json
{"img_0001.jpg": "S1A_20170114", "img_0002.jpg": "S1A_20170114"}
```

`scene_split` then guarantees no scene appears on both sides, and asserts it.

### 4. Retrain and measure

```bash
python train.py --torch --epochs 60
python scripts/evaluate.py --wind
```

Report **precision and recall**, never accuracy. Most of the ocean is clean, so
a detector that always answers "no spill" scores ~90% accuracy and is useless.

## What to compare against

Published benchmarks on this exact dataset (Krestenitis et al. 2019, Table 4),
IoU %:

| Model | Sea | Oil | Look-alike | Ship | Land | mIoU |
|---|---|---|---|---|---|---|
| U-Net | 93.90 | **53.79** | 39.55 | **44.93** | 92.68 | 64.97 |
| LinkNet | 94.99 | 51.53 | 43.24 | 40.23 | 93.97 | 64.79 |
| PSPNet | 92.78 | 40.10 | 33.79 | 24.42 | 86.90 | 55.60 |
| DeepLabv2 | 94.09 | 25.57 | 40.30 | 11.41 | 74.99 | 49.27 |
| DeepLabv2 (msc) | 95.39 | 49.53 | 49.28 | 31.26 | 88.65 | 62.83 |
| DeepLabv3+ | **96.43** | 53.38 | **55.40** | 27.63 | 92.44 | **65.06** |

The best published oil IoU is **53.79%**, by U-Net — the architecture used
here. If your model reports far above that, suspect leakage before celebrating.

## Wind

Wind speed is an optional field on the detect page and an optional `wind_ms`
form field on `POST /api/detect`. Below 3 m/s a glassy sea makes dark patches
that mimic oil; above 12 m/s a real slick breaks up. Wind can only ever weaken
a verdict, never strengthen one — it downgrades SPILL to NEEDS REVIEW and never
the reverse. Leave it blank if unknown; the system says so rather than assuming.

## Known gaps

- AIS vessel attribution is not built.
- `backend/ml/classical.py` is dead code — nothing imports it. Do not describe
  it as part of the system.
- Ship IoU is very low, as it is for every published model on this dataset.
