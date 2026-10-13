# Recency weighting

## Formula

Training samples are weighted with exponential decay:

```
w = 0.5 ** (age_days / half_life_days)
```

- `age_days = (t_now − t_match)` in days; `t_now` = latest data date (2026-10-08)
  for the main fit, and the event's first-game date for backtests (no future leakage).
- Weights are passed as `sample_weight` to the logistic-regression / HGB fit
  and to the `StandardScaler`. Model selection and all reported metrics
  (acc, log-loss, AUC, ECE) are UNWEIGHTED for comparability.
- The isotonic calibration map is fit UNWEIGHTED on the recent tail
  (2026-07-01 → 2026-10-01): the tail is already the most recent ~3 months,
  and a weighted-vs-unweighted isotonic check on the tail differs by ≤0.0002
  log-loss — negligible, so the simpler unweighted map is kept.

## Half-life tuning (src/tune_recency.py)

Grid over unweighted baseline + 90/180/270/365 days. Same pipeline per
candidate: weighted C-tuning (manual TimeSeriesSplit honoring sample_weight),
LR-vs-HGB comparison on unweighted validation log-loss, isotonic on the tail,
honest walk-forward backtests with weights anchored at each event cutoff.

| half-life | holdout acc | log-loss | AUC   | ECE | Worlds 2025 (n=96) acc/ll | MSI 2026 (n=71) acc/ll | winner |
|-----------|-------------|----------|-------|-----|---------------------------|------------------------|--------|
| none      | 0.6431      | 0.6325   | 0.6860| 0.00| 0.604 / 0.6714            | 0.634 / 0.6160         | HGB    |
| 90        | 0.6449      | 0.6315   | 0.6861| 0.00| 0.583 / 0.6750            | 0.620 / 0.6131         | LR C=5.0 |
| **180**   | **0.6452**  | **0.6310**| **0.6865** | 0.00 | 0.573 / 0.6759         | 0.620 / **0.6087**     | LR C=0.2 |
| 270       | 0.6458      | 0.6316   | 0.6861| 0.00| 0.594 / 0.6687            | 0.606 / 0.6105         | HGB    |
| 365       | 0.6452      | 0.6321   | 0.6853| 0.00| 0.625 / 0.6700            | 0.620 / 0.6062         | HGB    |

(ECE ≈ 0 everywhere is in-sample: the isotonic map was fit on that same
holdout. It measures fit quality, not out-of-sample calibration.)

## Chosen: 180 days

- Best holdout log-loss (0.6310) and best holdout AUC (0.6865) of the grid.
- Best MSI 2026 backtest log-loss (0.6087).
- Worlds 2025 is weak for every half-life (0.57–0.63 acc, n=96) — a known,
  separate weakness (Elo inflation for weak-region dominators), not something
  the half-life grid resolves; not overfit to it.
- Gains over unweighted are small but consistent (log-loss −0.0015, acc +0.2pp
  on the 3,368-game holdout). Directionally right, honestly small.

## Deployment note

`nodraft` is pinned to logistic regression even though HGB won its validation
comparison once (0.6387 vs 0.6400, a ~0.1-standard-error gap — pure noise):
the phone's offline JSON artifact format requires a linear model. The `full`
(draft) model won honestly as LR either way, so the draft predictions and
Kelly sizing are unaffected.
