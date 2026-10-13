# Confidence tiers for the phone app

## Recommended rule

For each prediction the app has the calibrated blue-win probability `p`
and the exported 2026-holdout calibration table. Compute:

```
margin = |p - 0.5|
bin    = the calibration bin whose pred_range contains p
```

| Tier | Condition |
|------|-----------|
| **High** | margin ≥ 0.20 **and** bin n ≥ 100 |
| **Medium** | margin ≥ 0.10 **and** bin n ≥ 30 |
| **Low** | otherwise |

These thresholds are exported in `phone_artifacts.json` under
`confidence_tiers`, so the app can read them rather than hard-code them.

## Why this shape

- **Margin, not raw p:** |p − 0.5| measures how far the model sticks its
  neck out. A 52/48 call is a coin flip with extra steps regardless of
  calibration.
- **Bin sample size as evidence:** the calibration table's `observed_acc`
  ("the model said ~65% and was right 63.5% of the time, n=888") is only
  trustworthy with enough games behind it. Small-n bins get capped at Low
  even with a wide margin.
- **Show the honest note, not just the tier:** next to the tier, display the
  bin's observed accuracy and n, e.g. *"High confidence — in this range the
  model was right 74.8% of the time (n=743, 2026 holdout)"*. That is the
  calibrated claim; the tier is just the headline.

## Caveats (show in-app where space allows)

1. **In-sample calibration.** The isotonic map was fit on the same 2026
   holdout the table summarizes, so the table flatters calibration slightly
   (ECE ≈ 0 in-sample). Out-of-sample backtests: MSI 2026 67.6% acc (n=71),
   Worlds 2025 56.2% (n=96) — the model is weaker at Worlds, likely due to
   cross-region matchups with no shared history.
2. **Series probabilities** assume independent games with fixed p — fine as a
   rough guide, not a true Bo5 model.
3. **Roster adjustment** changes the team-strength input (see
   `reports/roster_strength.md`); the calibration table was computed with
   raw snapshot Elos, so after heavy roster turnover treat tiers as one notch
   less certain than shown.
