# Team-comp archetype features experiment (2026-10-12) — NOT SHIPPED

Thomas proposed training on comp types (1-3-1 split, poke, engage-less comps
penalized) and added a range-matchup feature (short-range ADC into long-range
poke). Comp features were built from the confirmed champion tags
(reports/champion_tags.csv: 173 champs, 7 axes 0/1/2 + damage + DDragon 16.20.1
auto-attack range), evaluated honestly, and **kept out**: no variant met the
pre-registered gate (holdout log-loss improvement ≥ 0.002, no ECE degradation).

## What was tested

Per team per game, from the 5 picks (tags are static kit properties, no
walk-forward needed):
- 7 axis sums → differentials: d_engage, d_poke, d_split, d_teamfight,
  d_pick, d_frontline, d_peel (blue − red)
- 3 flag differentials: d_noengage (team engage sum == 0), d_nofrontline
  (frontline sum == 0), d_dmgmono (all 5 same damage type)
- 10 archetype dummies (blue/red × split/poke/pick/protect/teamfight;
  "standard" = reference)
- 2 range features: range_diff (mean auto range, blue − red),
  range_penalty = max(0, red_max − blue_min) − max(0, blue_max − red_min)
  (outrange-exposure differential; e.g. 525 ADC into 650 Caitlyn)

Archetype rule (priority order): split ≥ 4 → "split"; poke ≥ 4 & engage ≤ 2 →
"poke"; pick ≥ 4 & teamfight ≤ 3 → "pick"; peel ≥ 3 & engage ≤ 2 → "protect";
engage ≥ 3 & teamfight ≥ 4 → "teamfight"; else "standard".

## With vs without (identical pipeline: 180-day recency weights, isotonic cal)

Holdout = 2026 tail, n=3,368:

| variant   | features | acc    | log-loss | AUC    | dLL vs base |
|-----------|----------|--------|----------|--------|-------------|
| none      | 6        | 0.6452 | 0.6310   | 0.6865 | —           |
| axes      | 16       | 0.6449 | 0.6308   | 0.6872 | −0.0002     |
| range     | 8        | 0.6446 | 0.6316   | 0.6854 | +0.0006     |
| axes_arch | 26       | 0.6366 | 0.6318   | 0.6835 | +0.0008     |
| all       | 28       | 0.6357 | 0.6317   | 0.6841 | +0.0008     |

Backtests (indicative, n=96 / n=71): no variant improved both; axes_arch/all
degraded holdout accuracy by ~1pp (overfitting the dummies).

## Why it's neutral/negative (diagnostics)

1. **Best case is noise.** axes improves log-loss by 0.0002 — 1/10th of the
   gate. Nothing here clears the bar.
2. **Archetype dummies overfit.** The rule labels ~80% of pro comps
   "teamfight" (23,924/29,793); pick (40) and protect (144) are too sparse to
   estimate. Accuracy drops ~1pp when dummies are added — classic
   sparse-dummy overfitting despite C regularization.
3. **Range adds nothing.** range_diff/range_penalty slightly hurt (+0.0006).
   Auto-attack range barely separates poke mages (Xerath 525 = Xayah 525);
   the signal Thomas cares about lives in spell ranges, which we deliberately
   did not hand-tune.
4. **Axes are mostly repackaged champion strength.** Corr with d_power is
   material; whatever draft-coherence signal exists is small next to Elo.

## Decision

**Kept out.** Model stays v1.2.0 (6-feature full model). All pipeline code
reverted via git; features.csv rebuilt clean; validate_export.py re-run
green. Champion tags (reports/champion_tags.csv/.md) kept as a reviewed
reference — the tags themselves are Thomas-confirmed and may be useful for
app display or future experiments. If revisited: tighter archetype
definitions, spell-range data (not hand-tuned), or comp-coherence as a
draft-quality descriptive stat rather than a model feature.
