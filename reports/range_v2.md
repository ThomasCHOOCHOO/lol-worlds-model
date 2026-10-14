# Range matchup v2 experiment (2026-10-12) — NOT SHIPPED

Thomas's refinement of the dropped v1 range feature: the v1 failure was
diagnosed as auto-attack range not separating poke mages (Xerath auto = 525
= Xayah). Fix: use BOTH auto range and spell range, weighted by how each
champion deals damage (ADCs fight with autos, mages with spells).

## What was built

**`reports/champion_ranges.csv`** (new, kept as reference): per champion —
`auto_range` (DDragon 16.20.1), `spell_range` (judgment: primary threat-spell
range for ranged champs, rounded to 25; melee champs = auto range by
convention), `damage_mode` (auto | spell | hybrid), `effective_range`.

Damage-mode rule: DDragon Marksman tag → auto; Mage tag → spell; else hybrid.
Overrides: battlemages → hybrid (Azir, Cassiopeia, Ryze, Vladimir, Swain,
Viktor, Taliyah, Malzahar, Anivia, Rumble, Mordekaiser, Sylas, Gragas,
Kennen); Jayce → spell, Nidalee → spell (non-mage poke champs); Kayle → auto.
Ambiguous notes: melee champs' mode is irrelevant (spell_range = auto, so
weights cancel); Yuumi's threat range is inherently weird (usually attached).

Effective range = w_auto·auto + w_spell·spell, weights auto 0.8/0.2,
spell 0.2/0.8, hybrid 0.5/0.5. Sanity: Xerath 1065 vs Xayah 525 (gap 540 —
exactly Thomas's motivating case); Caitlyn 770; Ziggs 910; Cassiopeia 700.

**Features** (same formulas as v1, rebuilt on effective range):
- `range_diff` = mean effective range, blue − red
- `range_penalty` = max(0, red_max − blue_min) − max(0, blue_max − red_min)

## With vs without (identical v1.2.0 pipeline, only the 2 features differ)

Holdout = 2026 tail, n=3,368:

| metric   | with range v2 | v1.2.0  | delta    |
|----------|---------------|---------|----------|
| accuracy | 0.6455        | 0.6452  | +0.0003  |
| log-loss | 0.6310058     | 0.6310  | +0.000006|
| AUC      | 0.6860        | 0.6865  | −0.0005  |

Backtests: Worlds 2025 (n=96): 0.583/0.6777 vs 0.573/0.6759.
MSI 2026 (n=71): 0.634/0.6096 vs 0.620/0.6087.

Learned coefficients: `range_diff` −0.0249, `range_penalty` −0.0274
(vs `elo_diff` +0.57, `d_comfort` +0.26).

Gate was log-loss improvement ≥ 0.002. Actual: +0.000006 (wrong direction,
5 orders of magnitude short). Accuracy delta = 1 game of 3,368.

## Why it failed (diagnostics)

1. **The coefficients point the wrong way.** `range_diff` is negative:
   outranging the enemy is associated with *losing*, slightly. In the 2026
   pro meta, long-range poke comps lose to hard-engage comps — range is a
   proxy for a comp style that is currently unfavored, and what little
   signal exists is already in `d_power`/`d_comfort`.
2. **`range_penalty`'s sign is correct but tiny.** It points the right way
   (being outranged hurts) at −0.027, but the effect is negligible next to
   Elo and draft power.
3. **Range barely varies where it matters.** Most pro drafts sit within a
   ±150 band of mean effective range; the Xerath-vs-Xayah extremes are rare
   in pro play (teams don't draft 5 short-range champs into 5 poke mages).

## Decision

**Kept out.** Model stays v1.2.0. All pipeline code reverted via git;
features rebuilt clean; retrain reproduces v1.2.0 numbers exactly;
`validate_export.py` ALL CHECKS PASSED. `champion_ranges.csv` kept as a
Thomas-reviewed reference (may serve app display later). If revisited: the
interaction is probably range × engage (long-range + no-engage is the real
punish), not range alone — but that is a new hypothesis needing its own
honest test.
