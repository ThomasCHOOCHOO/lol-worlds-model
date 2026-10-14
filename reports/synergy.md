# Champion pairing synergy experiment (2026-10-12) — NOT SHIPPED

Thomas asked whether `d_power` captures duo synergies (Xayah+Rakan,
Kalista+Renata). It didn't (purely additive), so a `d_synergy` feature was
built, evaluated honestly, and **kept out**: neutral result, per the
pre-registered rule (keep only if it doesn't hurt log-loss/ECE; neutral →
keep out).

## Formula (as tested)

For each unordered champion pair fielded together on the same team:

```
synergy(c1, c2, P, t) = (w - E*n) / (n + a),   a = 10
```

- `w, n`: pair wins / games on the SAME patch P, games strictly before t
  (walk-forward, no leakage; patch itself is the window)
- `E`: expected pair win% = mean of the two champions' individual
  patch-aware `champ_power` at time t (same shrunk convention:
  `(w_P + 15*prior)/(n_P + 15)`)
- Pairs with `n < 5` same-patch games → 0
- Equals raw excess win-rate `(w/n - E)` shrunk toward 0 by `n/(n+a)`

Feature: `d_synergy = Σ synergy over blue's C(5,2)=10 pairs − Σ over red's 10 pairs`
(0 when picks unavailable). Added to the full model's feature list only;
nodraft unchanged. Same 180-day recency weighting, same isotonic calibration.

## With vs without (identical pipeline, only the feature differs)

Holdout = 2026 tail (fit_end ≤ date < cal_end), n=3,368:

| metric   | with d_synergy | without | delta   |
|----------|---------------|---------|---------|
| accuracy | 0.6461        | 0.6449  | +0.0012 |
| log_loss | 0.6309        | 0.6310  | −0.0001 |
| roc_auc  | 0.6864        | 0.6864  | −0.0001 |
| ECE      | 0.0001        | 0.0001  | 0.0000  |

Walk-forward backtests:

| event            | with                  | without               |
|------------------|-----------------------|-----------------------|
| Worlds 2025 (n=96) | acc 0.583, ll 0.6756 | acc 0.573, ll 0.6759  |
| MSI 2026 (n=71)    | acc 0.620, ll 0.6085 | acc 0.620, ll 0.6087  |

Learned coefficient: `coef[d_synergy] = −0.0097` (vs `elo_diff` +0.57,
`d_comfort` +0.27) — negative and tiny.

## Why it's neutral (diagnostics)

1. **Sign flip = collinearity, not signal.** Univariately `corr(d_synergy, y) =
   +0.027`, but `corr(d_synergy, d_power) = 0.356`. After conditioning on
   `d_power`, the residual flips slightly negative — a suppression artifact.
   The regularized model assigns it ~zero weight.
2. **The motivating duos don't qualify.** On the current patch (16.18, 539
   pro games), Xayah+Rakan have <5 games together → synergy = 0. Patch-specific
   pair samples are too thin for exactly the pairs the feature was built for.
3. **Top pairs look like noise.** Highest 16.18 synergies: cassiopeia|vi
   +0.19, lulu|orianna +0.19, nocturne|rakan +0.18 — small-sample artifacts,
   not real duos.
4. **All deltas are within noise.** Accuracy delta +0.0012 = 4 games out of
   3,368 (se ≈ 0.008); log-loss delta −0.0001 << se. Backtest n's (96, 71)
   are far too small to distinguish.

Possible coherent story for the negative sign (speculative): high-synergy
pairs get prioritized/banned/countered in draft, so historical overperformance
mean-reverts — but the effect is negligible either way.

## Decision

**Kept out.** Model stays v1.2.0 (6-feature full model). All experiment code
was reverted; pipeline rebuilt and re-validated clean. If revisited later, the
honest directions are: cross-patch pair windows (patch-global like comfort,
more samples) or a team-strength-adjusted expected value — not the
same-patch formulation tested here.
