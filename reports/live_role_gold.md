# Live model with role-weighted gold

**Date:** 2026-10-17. **Decision: CONDITIONAL SHIP variant A** (role golds replace
total gold). Variant B (roles + total) adds nothing — do not ship.

**Question:** the lane-gold analysis (`reports/lane_gold_by_lane.md`) found bot-lane
gold is worth ~3× top-lane gold for predicting the win (drop-one log-loss
+0.048 vs +0.016), while support gold barely matters. The shipped live model
pools all gold into one number, so a 1,000-gold ADC lead counts the same as a
1,000-gold support lead. This tests splitting total gold into five per-role
diffs.

## 1. Method

Same walk-forward protocol as the shipped model (`src/train_live.py`):
per-bucket (t = 10/15/20/25) logistic regression, C = 1.0, train on games
with date < 2026-07-01, isotonic calibration on 2026-07-01..2026-10-01.
`killdiff` kept as-is (total). No per-role kill diffs — scope is gold only.

Per-role gold diffs: blue-minus-red `goldat{t}` for top / jungle / mid / ADC
(`bot` position) / support separately, from per-player rows, in units of
1,000 gold. **All four buckets have per-player `goldat{t}` columns** — no
substitution was needed.

### Two data findings that affect interpretation

**a) The shipped `golddiff_k` is 2× the true team gold diff.**
The shipped script sums `goldat{t}` over *all* rows per side, including the
`team` row — and the team row's gold equals the sum of its five players'
gold (verified: exact match on 100% of games). So shipped total = 2× true
total. The model is still valid (a rescaled feature just halves the
coefficient), but any comparison must account for it. This report uses a
clean 1× total (sum of the five role diffs) for `baseline_clean`, and
replicates the shipped definition separately as a check.

**b) Missing gold data.** `goldat{t}` NaN is all-or-nothing per game
(2,382 of 29,942 games lack gold at t=10, mostly 2024). The shipped
script's groupby sum turns those into fake 0-gold games. All variants
here exclude games missing any role gold at that bucket, so the
comparison is apples-to-apples. The replication below shows the
exclusion barely moves the shipped numbers.

### Variants

| Variant | Features |
|---|---|
| `baseline_clean` | `[pre_logit, total_gold (1×), killdiff]` |
| `A_roles` | `[pre_logit, top, jng, mid, adc, sup gold, killdiff]` — total **replaced** |
| `B_roles_plus_total` | A + `total_gold` kept as well |

### Evaluation

Two evaluations, because the shipped protocol grades isotonic on the same
window it was fit on (in-sample, ECE = 0 by construction):

1. **Shipped protocol:** isotonic fit on the cal window, evaluated on the
   same window. Optimistic in absolute terms, but the *relative*
   comparison across variants is fair (identical procedure).
2. **Honest split (nested-style):** isotonic fit on Jul–Aug, evaluated on
   fresh Sep–Oct 1 games (n ≈ 1,128 / 1,128 / 1,121 / 941). This is the
   number to trust for the ship decision — but note it is one small slice,
   and September was an unusual month (blue win rate 49.5% vs 56.7% in
   Jul–Aug), so its CIs are wide.

Paired differences use 2,000 bootstrap resamples of per-game log-loss
differences (negative = variant better).

## 2. Results

### Replication check (validates the pipeline)

| Bucket | Shipped `live_models.json` | Replicated (shipped def) |
|---|---|---|
| 10 | 0.5325 | 0.5324 |
| 15 | 0.4758 | 0.4759 |
| 20 | 0.4141 | 0.4139 |
| 25 | 0.3607 | 0.3609 |

Replication matches to ≤ 0.0003 at every bucket.

### Log-loss: variant A vs baseline

**Shipped protocol** (isotonic in-sample; n_cal ≈ 3,300 / 3,301 / 3,285 / 2,877):

| Bucket | Baseline | A | Δ (A − base) | 95% CI | B | Δ (B − base) |
|---|---|---|---|---|---|---|
| 10 | 0.5298 | 0.5285 | −0.0014 | [−0.0037, +0.0008] | 0.5285 | −0.0014 |
| 15 | 0.4724 | 0.4697 | **−0.0027** | [−0.0053, −0.0001] | 0.4698 | −0.0026 |
| 20 | 0.4096 | 0.4062 | **−0.0033** | [−0.0065, −0.0003] | 0.4062 | −0.0034 |
| 25 | 0.3545 | 0.3528 | −0.0017 | [−0.0049, +0.0015] | 0.3527 | −0.0017 |

**Honest split** (isotonic Jul–Aug → eval Sep–Oct):

| Bucket | Baseline | A | Δ (A − base) | 95% CI | B | Δ (B − base) |
|---|---|---|---|---|---|---|
| 10 | 0.5598 | 0.5602 | +0.0005 | [−0.0038, +0.0049] | 0.5599 | +0.0001 |
| 15 | 0.4895 | 0.4885 | −0.0009 | [−0.0060, +0.0043] | 0.4886 | −0.0008 |
| 20 | 0.4271 | 0.4242 | **−0.0029** | [−0.0088, +0.0032] | 0.4243 | −0.0028 |
| 25 | 0.3806 | 0.3802 | −0.0004 | [−0.0078, +0.0068] | 0.3806 | −0.0000 |

Reading: A improves on baseline in 4/4 buckets under the shipped
protocol (2 significant) and 3/4 under the honest split (0
individually significant — the honest eval set is small, CIs ±0.005–0.008).
Only t=20 clears the project's 0.002 gate on the honest split. The gain
is small (~0.001–0.003) and strongest mid-game (t=15–20), where gold
composition has had time to diverge by role but the game is not yet
decided.

### Accuracy / AUC / ECE (honest split)

| Bucket | Metric | Baseline | A |
|---|---|---|---|
| 10 | acc / AUC / ECE | 0.698 / 0.789 / 0.041 | 0.692 / 0.789 / 0.047 |
| 15 | acc / AUC / ECE | 0.761 / 0.846 / 0.047 | 0.762 / 0.847 / 0.049 |
| 20 | acc / AUC / ECE | 0.798 / 0.886 / 0.031 | **0.807** / 0.889 / **0.027** |
| 25 | acc / AUC / ECE | 0.815 / 0.908 / 0.037 | **0.820** / 0.911 / 0.052 |

A is equal-or-better on accuracy/AUC at t=15/20/25, slightly worse at
t=10. ECE is mixed (better at t=20, worse at t=10/t=25) — the isotonic
maps are refit per variant, so this reflects ranking changes, not a
systematic calibration loss. Nothing here contradicts the log-loss
picture: small gain, concentrated mid-game.

## 3. Coefficients: ADC > support at every bucket — no inversion

Variant A raw coefficients (per 1,000 gold), feature order
`[pre_logit, top, jng, mid, adc, sup, killdiff]`:

| Bucket | pre_logit | top | jng | mid | **adc** | **sup** | killdiff |
|---|---|---|---|---|---|---|---|
| 10 | 0.839 | 0.773 | 0.767 | 0.900 | **0.898** | **0.571** | −0.027 |
| 15 | 0.742 | 0.442 | 0.498 | 0.559 | **0.579** | **0.350** | 0.014 |
| 20 | 0.615 | 0.362 | 0.403 | 0.435 | **0.505** | **0.314** | 0.027 |
| 25 | 0.500 | 0.342 | 0.350 | 0.405 | **0.461** | **0.319** | 0.032 |

Standardized coefficients (raw × SD of the feature on train) — the
fair comparison across roles, since role-gold SDs differ:

| Bucket | top | jng | mid | **adc** | **sup** |
|---|---|---|---|---|---|
| 10 | 0.425 | 0.388 | 0.444 | **0.551** | **0.203** |
| 15 | 0.458 | 0.424 | 0.513 | **0.658** | **0.196** |
| 20 | 0.533 | 0.529 | 0.612 | **0.807** | **0.246** |
| 25 | 0.640 | 0.605 | 0.727 | **0.930** | **0.325** |

**The ordering is exactly as theory predicts, at all four buckets:**
ADC highest, support lowest, mid second, top/jungle in the middle.
This matches the lane-gold report's ordering (bot 2v2 0.888 > mid 0.556 >
jng 0.532 > top 0.462), with ADC-only slightly below bot-2v2 as
expected (support's small contribution is now split out, and it is
small: 0.20–0.33).

**Why this is not a multicollinearity artifact.** Role golds are
correlated (max pairwise r = 0.51, jungle–support at t=25; bot–support
0.36–0.48), but the ADC > support ordering is stable across four
*independently fit* buckets. A collinearity artifact would flip
signs/order across fits; this one never does. The drop-one test below
is the independent corroboration.

### Drop-one-role (honest split): removing ADC gold hurts most

Refit A without each role gold (isotonic Jul–Aug → eval Sep–Oct);
Δ = log-loss increase vs full A (positive = the role carried unique
information):

| Bucket | −top | −jng | −mid | **−adc** | −sup |
|---|---|---|---|---|---|
| 10 | +0.008 | +0.013 | −0.006 | **+0.020** | −0.006 |
| 15 | +0.011 | +0.005 | +0.003 | **+0.027** | +0.002 |
| 20 | +0.011 | +0.021 | −0.001 | **+0.046** | +0.002 |
| 25 | +0.011 | +0.033 | −0.002 | **+0.035** | +0.001 |

Removing ADC gold costs the most at every bucket (up to +0.046 at
t=20); removing support gold costs ~nothing (−0.006…+0.002). Mid gold
is largely redundant given the other roles (drop-one ≈ 0) — its
coefficient reflects correlation with the team state more than unique
signal. Top and jungle carry moderate unique information.

## 4. Variant B adds nothing (as theory says it must)

`total_gold` is *exactly* the sum of the five role diffs, so B is a
reparameterization of A, not a richer model. Ridge splits the weight
arbitrarily between the roles and the total (e.g. at t=10, role coefs
shrink to 0.12/0.12/0.25/0.25/−0.08 with total absorbing 0.65), and
predictions come out within 0.0003 of A at every bucket on both
evaluations. **Do not ship B** — its split coefficients are
uninterpretable (an artifact of how ridge divides weight among
collinear terms), and it buys zero predictive gain over A.

## 5. Decision: CONDITIONAL SHIP variant A

**For:** coefficient ordering exactly as theory predicts at all four
buckets; drop-one corroborates (ADC most valuable, support ~worthless);
3/4 honest buckets improve; 2/4 significant under the shipped protocol;
accuracy/AUC equal-or-better at t=15/20/25.

**Against:** honest gains are small (only t=20 clears the 0.002 gate);
no honest CI excludes zero; t=10 honest is slightly worse; ECE slightly
worse at t=10/t=25.

The evidence favors A, but not strongly enough to size Kelly stakes on
it without live validation. Conditions:

1. **Ship A, not B.**
2. **Shadow-test before sizing:** during Swiss, log both baseline and A
   predictions for every live game; compare realized log-loss on the
   completed games before letting A drive Kelly fractions. If the
   shadow test comes back ≤ 0, revert to the baseline (the shipped
   model stays untouched in `model/live_models.json` — this variant
   lives in a separate file until then).
3. **Watch t=10:** it is the weakest bucket (honest Δ +0.0005, worse
   ECE). If the shadow test confirms t=10 is flat-or-worse, ship A for
   t=15/20/25 only and keep the baseline at t=10.

## 6. App-integration notes (for the conditional ship)

**Artifact schema.** `phone_artifacts.json` → `live_models[t]` currently
holds `{model_type, features: [pre_logit, golddiff_k, killdiff], coef,
intercept, calibration: {xs, ys}}`. For A it becomes (canonical feature name is `bot_gold` — ADC = the `bot`
position in Oracle's Elixir labels):

```json
"live_models": {"15": {"model_type": "LogisticRegression",
  "features": ["pre_logit", "top_gold", "jng_gold", "mid_gold", "bot_gold", "sup_gold", "killdiff"],
  "coef": [0.7422, 0.4424, 0.4984, 0.5594, 0.5785, 0.3501, 0.0143],
  "intercept": -0.0448, "calibration": {"xs": [...], "ys": [...]}}}
```

Note: `model/live_models_role_gold.json` (written by this task) is a
**metrics/coefficients record only** — it does not contain the isotonic
`xs`/`ys` breakpoints. Deploying requires the export step that
serializes the fitted `IsotonicCalibrated` wrappers into
`phone_artifacts.json` (the same step that produces the current
`live_models` entry), run against variant A's fitted models. Do not
hand-construct the artifact.

**`src/model/live.ts` changes (not made here — app files untouched):**

- `LiveGameState` / `LiveFeatures` gain five nullable numbers:
  `topGoldK, jngGoldK, midGoldK, adcGoldK, supGoldK` (blue-minus-red,
  in 1,000s). `goldDiffK` stays for the baseline/OCR path.
- `liveProba` must build `x` from the artifact's `features` list
  rather than the hardcoded `[preLogit, goldDiffK, killDiff]` — or
  branch on a model-version flag. Hardcoding a second 7-length vector
  alongside the first is a bug waiting to happen.
- `snapshot()` must parse per-participant gold from the raw feed
  frames. The app's `asTeam` currently keeps only team totals
  (`FrameTeam.totalGold`) and discards participants — the raw
  lolesports window frames carry a participants array (each with
  `totalGold` and a role). **Verify against a live frame dump before
  coding:** confirm participant `totalGold` and role fields are
  present and that role maps cleanly to top/jng/mid/bot/sup. If the
  feed ever lacks participants, fall back to the baseline model for
  that game (do not zero-fill role golds — a 0 is a claim of equality,
  not missingness).
- **OCR fallback keeps the baseline model.** `ocrToLiveState` only
  has team gold — role diffs are unavailable — so OCR games must use
  the shipped pooled model, not A. The Live tab needs a per-game
  model selector (feed → A, OCR → baseline), not a global switch.

## 7. Limitations

- Honest evaluation is a single small slice (Sep–Oct 1, n ≈ 1,100 per
  bucket, one unusual month). The honest CIs are wide; the case for A
  rests partly on the shipped-protocol comparison and the coefficient
  structure, not on honest significance alone.
- `killdiff` is still a pooled total (per the task scope). Per-role
  kills may carry additional signal — untested.
- Role golds use Oracle's Elixir `position` labels; lane swaps / flex
  picks can mislabel a small share of games.
- t=10 is the weakest bucket; see condition 3 in §5.
- The shipped 2× total scaling (see §1a) means the shipped live
  coefficients in `live_models.json` are on a different feature scale
  than the ones here — do not compare raw coefficient magnitudes
  across the two files without halving the shipped gold coefficient.

## 8. Reproduction

Training script: `/tmp/train_role_gold.py` (scratch — rerun with
`~/workspace/lol-worlds-model/.venv/bin/python`). Inputs:
`data/processed/features.csv`, `data/raw/202{4,5,6}.csv`,
`model/model.joblib` (for `pre_logit`). Outputs:
`model/live_models_role_gold.json` (this task's only model file) and
this report. Shipped `model/live_models.json` untouched. No app files
modified. No pushes, no bets.
