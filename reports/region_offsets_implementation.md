# Region offsets — implementation notes + gate results

v1.4.0 Phase 1, item 1. Spec: `reports/region_strength_analysis.md`.
Implemented 2026-10-17. **Nothing pushed. Shipped `phone_artifacts.json`
untouched** — the new model is validated only against the draft artifact
`model/phone_artifacts_1.4.0_draft.json` (see §6).

## 1. What was built

| File | Change |
|---|---|
| `src/region_offsets.py` (new) | Hierarchical fit + region map + JSON writer (see §2) |
| `src/build_features.py` | Computes `elo_diff_adj`; snapshots carry `team_region` + `region_offsets` |
| `src/train.py` | Trains `full` + `nodraft` on `elo_diff_adj`; requires the column (fails loudly if missing); backtests run baseline + offset variant |
| `src/predict.py` | Applies the region adjustment in `predict_game`; shows it in CLI output |
| `src/refresh.py` | `region_offsets` step between `build_features` and `train` |
| `src/export_artifacts.py` | Artifact carries `region_offsets` table + per-team `region`; `PHONE_ARTIFACTS_OUT` env override for draft exports |
| `src/validate_export.py` | `json_predict` applies the region adjustment; `ARTIFACTS_PATH` env override; checks artifact offsets == canonical JSON; clip bounds aligned to `model_utils` `PRED_MIN/PRED_MAX` (see §7) |

## 2. `src/region_offsets.py`

- Model: `logit p = a + b·logit(p_elo) + off[blue] − off[red]`, ridge λ=1 on the
  offsets only, fit by IRLS on **cross-region games only** (MSI / Worlds /
  First Stand). Fit always uses **raw** `elo_diff` — fitting on
  `elo_diff_adj` would be circular.
- Region map: `league → region` for top-flight + developmental leagues
  (`LEAGUE_TO_REGION`), LCP resolved by recursion (walks back to the team's
  most recent prior mapped non-LCP league), LTA excluded from modal
  computation (cross-conference), KeSPA Cup → LCK, LCKC → LCK, LDL → LPL.
  Manual overrides: MVK Esports → Vietnam, Team Secret Whales → Vietnam.
  Minor regions (Japan, Oceania, Turkiye) get offset 0.
- **Fit-set rule (spec §1, found by reproduction, see §3):** only games where
  BOTH sides have fitted regions enter the fit. Games against minor /
  unknown / unfitted regions are excluded — they cannot identify the fitted
  offsets (their dummy is an arbitrary 0), and including them breaks the
  ridge centering.
- Backtests use **cutoff-scoped offsets AND cutoff-scoped team-year regions**
  (`fit_offsets(..., cutoff=...)` + `team_year_regions(..., cutoff=...)`).
  The main-model offsets fit on ALL cross-region games in current data
  (documented 7-parameter lookahead, same order as the isotonic map being
  fit on the tail).
- Offsets are **constant** (no decay — decay was tested and rejected in the
  spec). Refit on every refresh because LPL's edge shrank ~40% 2024 → 2025–26.
- `model/region_offsets.json` contains: offsets, intercept, slope, fit date,
  n_games, method, lambda, fitted regions, region appearances, unknown-region
  games, the auditable `league_to_region` map, manual overrides,
  `team_year_regions`, and `team_region` (latest region per team).
- `main()` fits on `features.csv`, writes the JSON, and refreshes the
  `elo_diff_adj` column in `features.csv`.

## 3. Fit reproduction — a real bug found and fixed

First fit (all 564 cross-region games) gave offsets ≈ report + ~58 Elo
constant shift (mean +59.3), e.g. LCK +234.9 vs report +178. Debugging:

1. Verified the optimizer: penalized log-likelihood gradient norm 0.0 at my
   beta; `scipy.optimize.minimize` (BFGS) converged to the identical beta.
   The IRLS was **not** buggy.
2. The report's beta had gradient norm 2.77 on my design matrix — not optimal
   there — so the difference was in the **design**, not the optimizer.
3. Root cause: the 4 Japan games (Fukuoka SoftBank HAWKS gaming vs GAM /
   100 Thieves, Worlds 2024). The spec's Protocol B uses **n=560**, and spec
   §1 says Japan is "excluded from fitting" — meaning its *games* are
   excluded, not just its dummy. With minor-side games included, the ridge
   centering argument fails (those rows have only one dummy), so the offsets
   float up by a constant.
4. Fix: exclude games where either side's region is not in the fitted set.
   Refit reproduces the report **exactly**:

| Region | This implementation | Report | Bootstrap 95% CI (500 resamples, seed 7) | Report CI |
|---|---|---|---|---|
| LCK | +177.9 | +178 | [127.2, 235.8] | [133, 233] |
| LPL | +108.5 | +108 | [63.4, 160.1] | [63, 163] |
| EMEA | +24.1 | +24 | [−21.9, 74.4] | [−34, 75] |
| NA | +13.7 | +14 | [−45.2, 69.9] | [−45, 71] |
| Taiwan | −33.5 | −34 | [−106.6, 45.0] | [−109, 36] |
| LATAM | −156.9 | −157 | [−260.6, −78.1] | [−245, −71] |
| Vietnam | −133.8 | −134 | [−216.6, −54.2] | [−220, −56] |

Intercept 0.2541 (report +0.254), slope 0.6497 (report 0.65), n=560.
Bootstrap CIs match the report's within resampling noise (different seed).
**Only the offsets ship** — the fitted intercept/slope are recorded in the
JSON for audit but NOT used (production has `blue_edge`; slope shrinkage
adds zero held-out value once offsets are in, per the spec).

Region appearances match the spec's Table §2 exactly (LPL 309, LCK 301,
EMEA 172, NA 134, Taiwan 77, LATAM 67, Vietnam 64, Japan 4).

## 4. Pipeline order (documented deviation from the spec)

The spec says "region_offsets before build_features". Implemented order is
**build_features → region_offsets → train → export → validate**, because the
fit needs `features.csv` (it reads `elo_diff`, team/league/date per game).
`build_features.py` computes `elo_diff_adj` from the *existing*
`region_offsets.json` when present (else `elo_diff_adj = elo_diff`);
`region_offsets.py`'s post-fit refresh of the column is authoritative and
guarantees the offsets match the current `features.csv`. On a fresh
checkout with no `region_offsets.json`, the first `build_features` run
writes `elo_diff_adj = elo_diff`, then `region_offsets` fits and refreshes
the column, then `train` uses it. `refresh.py` runs the steps in this order.

## 5. GATE — production-config backtest

Protocol: offsets folded into `elo_diff` (`elo_diff_adj`) + **full model
refit** (all 6 features, fresh C tune, recency weights anchored at the event
cutoff), scored on the event's games. Offsets and team-year regions are
cutoff-scoped (fit only on games before the event). Baseline = same
procedure on raw `elo_diff`.

| Event | n | Offset-variant LL | Baseline LL | **ΔLL** | Offset acc |
|---|---|---|---|---|---|
| Worlds 2025 | 96 | 0.6151 | 0.6803 | **−0.0651** | 0.667 |
| MSI 2026 | 71 | 0.5778 | 0.6214 | **−0.0436** | 0.676 |
| **Pooled (n-weighted)** | 167 | — | — | **−0.0560** | — |

Cutoff-scoped offsets used: Worlds 2025 — LCK +178.9, LPL +106.0, EMEA
+2.6, NA +7.9, Taiwan +9.7, LATAM −157.3, Vietnam −147.8. MSI 2026 — LCK
+191.3, LPL +115.3, EMEA +28.5, NA −9.8, Taiwan −7.9, LATAM −154.5,
Vietnam −162.9.

**Gate verdict: PASSED.** The standalone validation spans ΔLL −0.041
(forward-chaining) to −0.051 (Protocol A) across its honest protocols; the
production numbers (−0.065, −0.044, pooled −0.056) confirm the same effect
at the same magnitude — both events improve substantially, and the pooled
production ΔLL is modestly *stronger* than the standalone point estimates.
The per-event values straddle the standalone range, consistent with
event-level sampling noise (n=96, n=71). The failure mode (ΔLL ≈ 0 or
positive — offsets not helping in production config) is clearly absent.

Full-model holdout also improved slightly vs v1.3.2 (full: LL 0.6253 →
0.6235, acc 0.6508 → 0.6511, AUC 0.6947 → 0.697; nodraft LL 0.6262).

## 6. Current state — what is / is not shipped

- `model.joblib`, `metrics.json`, `snapshots.joblib`, `features.csv` now
  include region offsets (regenerated by the full pipeline run 2026-10-17).
  Durable pre-change backups: `model.joblib.bak-pre-region-offsets`,
  `metrics.json.bak-pre-region-offsets`,
  `snapshots.joblib.bak-pre-region-offsets`,
  `data/processed/features.csv.bak-pre-region-offsets`.
- **Shipped `model/phone_artifacts.json` NOT overwritten.** The new artifact
  was exported to `model/phone_artifacts_1.4.0_draft.json` via
  `PHONE_ARTIFACTS_OUT` and validated against it (see §7). The app therefore
  still serves the pre-offset model until the draft is deliberately shipped
  (parent's call — this also interacts with the Models-tab calibration
  toggle work, which landed in `train.py` / `model_utils.py` concurrently;
  this implementation preserves that code untouched).
- `model/region_offsets.json` is the canonical offset file (fit date
  2026-10-16 data snapshot).

## 7. Validation

- `validate_export.py` against the **draft** artifact (`ARTIFACTS_PATH=...`):
  **ALL CHECKS PASSED** — 5 matchups (pre-draft + draft) TS/Python parity
  0.00e+00, roster identity OK, isotonic map vs sklearn 5.55e-17,
  region_offsets in artifact == canonical JSON, live buckets 6.94e-18.
- One validator fix during this work: the live-bucket check initially
  failed at 9.99e-04. Root cause was a validator/model mismatch from the
  concurrent honest-calibration change (`model_utils.IsotonicCalibrated`
  now clips to [0.001, 0.999]; the validator still clipped to 1e-6).
  Fixed the validator to import `PRED_MIN`/`PRED_MAX` from `model_utils`.
  Not a region-offset issue; the app-side clip is parent scope.
- `python -m py_compile` passes on all changed files.
- CLI spot checks: T1 vs BLG — Elo diff −94, region adj +69 (LCK +178 vs
  LPL +108) → adjusted −24, per-map 40.1%, Bo5 31.9%. MVK vs FURIA — Elo
  diff −92, region adj +23 (Vietnam −134 vs LATAM −157) → adjusted −69,
  per-map 41.6%, Bo5 34.6% (v1.3.2: 40.7% / 32.9%).

## 8. Flags for parent

1. **Oct 19 refresh cron will now run the new pipeline** (`refresh.py`
   includes the `region_offsets` step) and regenerate + ship artifacts,
   including the offset model. Decide whether to leave the cron as-is.
2. **Ship decision pending:** `model.joblib` (offsets) vs shipped
   `phone_artifacts.json` (pre-offsets) are now inconsistent by design —
   the draft artifact is validated and ready; shipping is a deliberate step.
3. The gate backtest's Worlds 2025 accuracy (0.667) is higher than the
   v1.3.2 backtest (0.552) partly because the offset variant also uses the
   current full feature set; the LL comparison (baseline vs offset, same
   protocol) is the apples-to-apples number.
