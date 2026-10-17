# Honest calibration — implementation notes (v1.4.0 Phase 1, methodology fix)

## The problem (why this exists)

The shipped v1.3.2 calibration map was fit AND evaluated on the same
Jul–Oct 2026 games, so its 0.000 calibration error is fake by construction:
an isotonic map graded on its own fit games always looks perfect. An honest
test — map fit on Jul–Aug, graded on fresh September games — scored 0.682
log-loss vs 0.649 raw. A stale map can hurt, and one September game was
mapped to exactly 1.000 and lost, contributing ~0.03 to that honest log-loss
all by itself (a single game dominating a 1,142-game average).

## What changed

### 1. Nested-split calibration evaluation — `src/train.py`

- New constants: `CAL_FIT_END = "2026-09-01"` (map fit on
  [2026-07-01, 2026-09-01), graded on the untouched slice
  [2026-09-01, 2026-10-01)) and `CAL_MIN_STEP_COUNT = 100`.
- New `ece()` helper (10 equal-width bins, weighted |mean_pred − observed|).
- New `honest_map_from_bundle(bundle, df, key)` — the single source of truth:
  takes the already-trained base model, fits the isotonic map on the fit
  slice only, coarsens it, grades it on the untouched eval slice, and also
  computes the raw (uncalibrated) and shipped-style in-sample numbers for
  comparison. Returns the exact app-toggle JSON plus a full report dict.
- `main()` now runs the nested evaluation for both `full` and `nodraft`,
  stores it as `report[key]["honest_calibration"]` in `metrics.json`, and
  writes `model/honest_calibration.json`. **Every retrain reports honest
  ECE/log-loss on untouched games**; in-sample numbers are labeled as such
  and must never be quoted as calibration quality.
- `main()` was NOT re-run here: `model.joblib`, `metrics.json`, and
  `phone_artifacts.json` are untouched (checksums verified, see below).
  The honest file was built from the existing bundle via
  `src/build_honest_calibration.py`, which calls the same helper.

### 2. Clipping — `src/model_utils.py`

- New `PRED_MIN = 0.001`, `PRED_MAX = 0.999`. `IsotonicCalibrated` now
  bounds map outputs to [0.001, 0.999] (was [0.0, 1.0]) and clips base
  probabilities to the same range.
- **Justification.** The calibration fit slices hold ~2–3k games, so the
  finest empirical resolution is ~1/3000; 0.001 sits just above that floor
  — we never claim more confidence than ~1-in-1000 from the fit data. It
  also caps the worst-case per-game log-loss at −ln(0.001) ≈ 6.91, so no
  single game can ever again contribute more than ~0.006 to a ~1,100-game
  average (vs ~0.030 for the old 1.000-mapping). Future retrains
  (including `train_live.py`, which shares `IsotonicCalibrated`) inherit
  the bounds automatically.
- Measured effect of clipping alone (honest map, uncoarsened):
  0.6819 → 0.6569 log-loss on the untouched eval slice.

### 3. Fewer, wider calibration steps — `coarsen_isotonic()` in `model_utils.py`

- Keeps a subset of the fitted threshold points such that every retained
  segment covers ≥ 100 fit games (`CAL_MIN_STEP_COUNT`). Kept points are a
  subset of the fitted thresholds, so monotonicity is preserved exactly;
  the app's piecewise-linear interpolation works unchanged.
- **Justification.** The shipped v1.3.2 map has a step 0.3pp wide — a step
  value resting on a handful of games. With ≥ 100 games per step, the
  binomial standard error of any step is capped at sqrt(0.25/100) = 0.05.
- Measured effect of coarsening (honest map, clipped): 0.6569 → 0.6545
  log-loss, 0.0634 → 0.0536 ECE, 46 steps → 12 steps. Coarsening *helped*
  on untouched games — the thin steps were noise, not signal.

### 4. The honest map for the app toggle — `model/honest_calibration.json`

Built from the **full** model (the primary model; the nodraft honest map
can be generated the same way via `honest_map_from_bundle(bundle, df,
"nodraft")` but is not shipped — the toggle schema holds one map). Schema
is exactly as specified:

```json
{"xs": [...], "ys": [...], "fitted_on": "2026-07-01..2026-08-31",
 "n": 2226, "honest_eval": {"logloss": 0.6545, "ece": 0.0536, "n": 1142}}
```

Verified: schema keys/types exact; 12 steps; xs/ys monotone; ys within
[0.116, 0.999] (no exact 0/1); independent re-application of the map via
interpolation reproduces `honest_eval` exactly (log-loss 0.6545, ECE
0.0536, n=1142).

## Honest vs shipped metrics (read this table carefully)

All numbers below are the **full** model. "Honest" = map fit Jul–Aug,
graded on untouched Sep games (n=1,142). "Shipped" = map fit on the full
Jul–Oct tail, graded on that same tail (in-sample).

| Method | Log-loss | ECE | n |
|---|---|---|---|
| Raw model, no calibration (untouched Sep) | 0.6488 | 0.0403 | 1,142 |
| **Honest map, coarsened (untouched Sep)** | **0.6545** | **0.0536** | 1,142 |
| Honest map, uncoarsened (untouched Sep) | 0.6569 | 0.0634 | 1,142 |
| Shipped map, in-sample (full tail) | 0.6253 | 0.0000 | 3,368 |

Nodraft honest map (same protocol): coarsened 0.6524 / 0.0422; raw
0.6468 / 0.0367; shipped in-sample 0.6270 / 0.0000.

**The honest statement:** on this particular eval slice (September 2026),
the raw uncalibrated model *still* beats the honest calibration map
(0.6488 vs 0.6545). September was a different world — blue side won
49.5% of games vs 56.7% in Jul–Aug — so a map learned in July–August
transfers imperfectly. That does NOT mean calibration is useless or that
the shipped map is better: the shipped 0.6253 / 0.0000 is graded on its
own fit games and cannot be compared to the honest numbers. The honest
numbers are the only ones that estimate real-world performance, and they
say: calibration on a 2-month-old map is roughly neutral on fresh games
(slightly negative here), while an unclipped, thin-stepped map is
actively dangerous (0.6819, with a single 1.000-mapping game costing
~0.03). The fix in this change is twofold: (a) never again quote
in-sample calibration numbers, and (b) never again ship a map that can
output 0/1 or rest steps on a handful of games. Whether the honest map
beats raw on *future* games is an empirical question the toggle itself
will answer — that is what the toggle is for.

## How the app toggle should use this

- The app keeps BOTH maps: the shipped map in `phone_artifacts.json`
  (unchanged) and this honest map. The toggle switches which map is
  applied to the raw model probability.
- Apply the honest map with the same math as the shipped map:
  clip raw p to [xs[0], xs[-1]], linearly interpolate over (xs, ys),
  clip the result to [0.001, 0.999].
- The toggle is a calibration choice only — the base model (coefficients,
  Elo, draft features) is identical under both settings.
- Do NOT re-derive the shipped map from this file; do NOT write this file
  into `phone_artifacts.json`.

## Regeneration

- Without retraining: `.venv/bin/python src/build_honest_calibration.py`
  (loads the existing `model.joblib`, rebuilds the honest file).
- On retrain: `train.py main()` calls the same helper and regenerates the
  file plus the honest metrics in `metrics.json`.

## Files changed / added

- `src/model_utils.py` — `PRED_MIN`/`PRED_MAX`, bounded
  `IsotonicCalibrated`, `coarsen_isotonic()`, `apply_isotonic_map()`.
- `src/train.py` — nested-split evaluation (`CAL_FIT_END`,
  `CAL_MIN_STEP_COUNT`, `ece()`, `honest_map_from_bundle()`), honest
  reporting + honest-file write in `main()`.
- `src/build_honest_calibration.py` — NEW standalone builder (no retrain).
- `model/honest_calibration.json` — NEW, schema-verified.
- `reports/honest_calibration_implementation.md` — this file.

## Verification

- `python -m py_compile` passes on all three changed/new Python files.
- `model/phone_artifacts.json`, `model/model.joblib`, `model/metrics.json`
  checksums unchanged after this work (shipped artifacts untouched):
  phone_artifacts `c63e4945…`, model.joblib `7da0afd5…`,
  metrics.json `005d6e2f…` (full hashes in the build log).
- The shipped-style in-sample recomputation reproduces the shipped full
  holdout log-loss exactly (0.6253), confirming the new code path matches
  the shipped pipeline.
