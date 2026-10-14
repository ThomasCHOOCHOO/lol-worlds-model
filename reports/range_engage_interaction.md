# Range × engage interaction experiment (2026-10-13) — NOT SHIPPED

Follow-up to the dropped range v2 experiment (reports/range_v2.md), which
found `range_diff` wrong-signed: long-range poke comps lose to hard engage in
the 2026 pro meta. New hypothesis: the punishable comp is long-range **AND**
no-engage.

## Formula (as tested)

Static kit data (no leakage): `eff_range(c)` from reports/champion_ranges.csv,
`engage(c)` (0/1/2) from reports/champion_tags.csv.

Per team, from its 5 picks:
- `mean_range` = mean effective range
- `engage_sum` = sum of engage tags

```
exposed(team) = max(0, mean_range − 450) × max(0, 3 − engage_sum)
punish(team, opp) = exposed(team) × engage_sum(opp)
```

- Thresholds data-driven: 450 ≈ p65 of team mean effective range
  (median 412, p90 550); engage_sum ≤ 3 ≈ bottom decile (median 5).
- Only long-range AND low-engage comps score > 0 (~10.5% of teams;
  5.8% of games have a nonzero differential).
- `exposed_diff = exposed(blue) − exposed(red)` — own-comp vulnerability.
- `punish_diff = punish(blue,red) − punish(red,blue)` — vulnerability scaled
  by the enemy's ability to punish it (the sharper hypothesis).
- 0.0 when picks unavailable, consistent with d_power/d_comfort/ban_edge.

## With vs without (identical v1.2.0 pipeline, 180-day recency weights)

Holdout = 2026 tail (fit_end ≤ date < cal_end), n=3,368:

| variant              | acc    | log-loss | AUC    | dLL vs base |
|----------------------|--------|----------|--------|-------------|
| v1.2.0 baseline      | 0.6452 | 0.6310   | 0.6865 | —           |
| A: + exposed_diff    | 0.6452 | 0.6310   | 0.6864 | ~0.0000     |
| B: + both            | 0.6440 | 0.6312   | 0.6860 | +0.0002     |

Gate was log-loss improvement ≥ 0.002. Variant A is exactly neutral;
variant B goes the wrong direction.

Backtests: Worlds 2025 (n=96): A 0.573/0.6759 (same as base),
B 0.573/0.6762. MSI 2026 (n=71): A 0.620/0.6130, B 0.620/0.6128
(base 0.620/0.6087 — both variants slightly worse).

Learned coefficients (variant B): `exposed_diff` +0.0203 (sign FLIPPED vs
variant A's −0.0296 — collinearity with punish_diff), `punish_diff` −0.0525
(right-signed: enemy punishing your exposed comp hurts, but tiny vs
`elo_diff` +0.57, `d_comfort` +0.27). Univariate corr(exposed_diff, y) =
−0.0076 — the raw signal is negligible before the model even sees it.

## Why it failed (diagnostics)

1. **The raw correlation is ~zero.** corr = −0.0076 univariately. The
   hypothesis is directionally right (negative = exposed loses) but the
   effect is too small to matter at pro level.
2. **Pro drafts rarely create the extreme.** Only ~6% of games have a
   nonzero differential; true Xerath-vs-Xayah-style mismatches with a
   no-engage comp on one side are rare — teams don't draft into it.
3. **Collinearity eats the interaction.** In variant B the two features
   fight (sign flip on exposed_diff); the regularized model can't separate
   them, and punish_diff's −0.05 is noise against Elo/draft power.
4. Consistent with range v2's lesson: comp style is largely priced into
   `d_power` (patch-aware champ strength) and `d_comfort` already.

## Decision

**Kept out.** Model stays v1.2.0 (6-feature full model). All pipeline code
reverted via git (`src/` clean, `model/` restored); features.csv rebuilt
from clean source (6 features only); `validate_export.py` ALL CHECKS PASSED.
No schema additions — the phone app needs no changes.
