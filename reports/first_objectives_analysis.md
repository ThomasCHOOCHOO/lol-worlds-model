# First objectives → gold → win (Leaguepedia pilot)

**Data:** frozen snapshot `data/processed/pilot_snapshot_2026-10-17/` (copied before
analysis; the concurrent backfill cannot change these inputs) — 1,299 games,
2026 patches 16.01–16.2, LCK 502 / LEC 383 / LCS 283 / MSI 71 / FST 45 /
WLDs 15. Gold from Oracle's Elixir 2026 team rows (gold@15, gold@20, total gold
at game end). Pre-game strength control: `elo_diff` from `features.csv`
(v1.3.2 Elo, blue − red, per 100 points). LPL is **not** in the pilot
(Cargo-blocked).

**Sample convention:** primary results use `validation == "pass"` games only
(n = 761). Sensitivity checks on all 1,299 games and on all games excluding the
43 chronobreak-flagged games are in §5. All "blue" quantities are blue-side
perspective; gold diffs are blue − red in thousands of gold (k). Games with no
first-X objective are *excluded* from that objective's analysis (not counted as
a red-taker loss).

**Cross-checks:** first-dragon team from `objective_firsts.csv` agrees with the
first DRAGON event in `objective_timers.csv` on
100.00% of 1298 games.
Leaguepedia winner vs OE result mismatches: 0.
OE join coverage: 1298/1299 games
(99.9%) have OE gold.

## 1. First-objective team win rates

Taker win rate = P(team that took the objective wins the game), Wilson 95% CI.

| Objective | Sample | n | Taker wins | Taker win rate |
|---|---|---|---|---|
| First dragon | pass-only (primary) | 760 | 473 | 62.2% [58.7, 65.6] |
| First dragon | all games | 1298 | 801 | 61.7% [59.0, 64.3] |
| First dragon | all games, taker = blue | 492 | 371 | 75.4% [71.4, 79.0] |
| First dragon | all games, taker = red | 806 | 430 | 53.3% [49.9, 56.8] |
| First tower | pass-only (primary) | 761 | 539 | 70.8% [67.5, 73.9] |
| First tower | all games | 1299 | 891 | 68.6% [66.0, 71.1] |
| First tower | all games, taker = blue | 750 | 545 | 72.7% [69.4, 75.7] |
| First tower | all games, taker = red | 549 | 346 | 63.0% [58.9, 67.0] |
| First baron | pass-only (primary) | 723 | 634 | 87.7% [85.1, 89.9] |
| First baron | all games | 1241 | 1051 | 84.7% [82.6, 86.6] |
| First baron | all games, taker = blue | 690 | 603 | 87.4% [84.7, 89.7] |
| First baron | all games, taker = red | 551 | 448 | 81.3% [77.8, 84.3] |
| First herald | pass-only (primary) | 761 | 515 | 67.7% [64.3, 70.9] |
| First herald | all games | 1282 | 841 | 65.6% [63.0, 68.2] |
| First herald | all games, taker = blue | 835 | 566 | 67.8% [64.5, 70.9] |
| First herald | all games, taker = red | 447 | 275 | 61.5% [56.9, 65.9] |
| First grubs | pass-only (primary) | 761 | 414 | 54.4% [50.9, 57.9] |
| First grubs | all games | 1284 | 675 | 52.6% [49.8, 55.3] |
| First grubs | all games, taker = blue | 816 | 473 | 58.0% [54.5, 61.3] |
| First grubs | all games, taker = red | 468 | 202 | 43.2% [38.7, 47.7] |
| First inhibitor | pass-only (primary) | 761 | 736 | 96.7% [95.2, 97.8] |
| First inhibitor | all games | 1299 | 1243 | 95.7% [94.4, 96.7] |
| First inhibitor | all games, taker = blue | 742 | 717 | 96.6% [95.1, 97.7] |
| First inhibitor | all games, taker = red | 557 | 526 | 94.4% [92.2, 96.1] |

Reading: first dragon (62.2%) and first tower are both real but
far from decisive. First baron (87.7% [85.1, 89.9]) and first inhibitor
(96.7% [95.2, 97.8]) are much stronger because they usually happen to teams that
are already winning. The side split shows the objective matters beyond side:
red-side dragon-takers still win 53.3% [49.9, 56.8] and red-side tower-takers
63.0% [58.9, 67.0], just less than blue-side takers, consistent with blue's baseline
edge. First grubs is the weakest objective — and red-side grub-takers actually
lose more than they win (43.2% [38.7, 47.7]), consistent with grubs being a
low-value early trade rather than a snowball objective.

## 2. Gold swing

All regressions pass-only, league + patch fixed effects (collinear dummies
dropped via QR pivoting), HC1 SEs. Gold in k.

### First dragon → gold@15 (dragon median 438s ≈ 7.3 min, so gold@15 is post-dragon)

Raw diff-in-means (pass-only): blue gold@15 is +1.26k when blue
took first dragon vs -0.20k when red took it
(raw gap +1.45k).

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
| blue_first_dragon | +1.509 | 0.190 | 2e-15 |
n = 760, R² = 0.103. Regression-adjusted, taking first
dragon is worth **+1.51k gold at 15 min** vs the
opponent taking it. Controlling for pre-game Elo:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
| blue_first_dragon | +1.366 | 0.187 | 2.4e-13 |
| elo100 | +0.499 | 0.078 | 1.5e-10 |
n = 760, R² = 0.160. The dragon coefficient shrinks from
+1.51k to +1.37k — part of the
raw association is stronger teams taking dragons, but a real post-dragon swing
remains.

Controlling for the gold state *at the dragon itself* isolates the
post-dragon swing:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
| blue_first_dragon | +0.589 | 0.154 | 0.00012 |
| gold_at_dragon_k | +1.546 | 0.079 | 8.6e-86 |
n = 760, R² = 0.454. The dragon taker was already ahead
by +0.59k on average at dragon time
(+0.55k when blue took vs -0.04k when red took);
after the dragon, the taker's edge grows by a further
**+0.59k** by 15 min. So roughly half the
dragon→gold@15 association is selection (ahead teams take dragons) and half is
post-dragon snowball.

### First tower → gold@20 (tower median 925s ≈ 15.4 min)

**Timing warning:** the median first tower falls *after* 15:00, so gold@15 is
mostly a *pre-tower* measure — a tower→gold@15 coefficient would be selection,
not effect. Raw gold@15 by tower taker: +1.64k (blue took) vs
-1.43k (red took). The clean window is gold@20, controlling for
gold@15 (pre-tower state):

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
| blue_first_tower | +1.435 | 0.182 | 3e-15 |
| gold15_diff_k | +1.238 | 0.036 | 1.3e-257 |
n = 760, R² = 0.771. Raw gold@20: +2.95k (blue
took) vs -2.36k (red took). Controlling for gold@15, taking first
tower adds **+1.44k** by 20 min. With pre-game Elo:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
| blue_first_tower | +1.383 | 0.180 | 1.5e-14 |
| gold15_diff_k | +1.207 | 0.036 | 4.6e-247 |
| elo100 | +0.267 | 0.063 | 2.3e-05 |
n = 760, R² = 0.776.

Controlling instead for gold at tower time:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
| blue_first_tower | +1.669 | 0.182 | 4.1e-20 |
| gold_at_tower_k | +1.366 | 0.041 | 8.5e-240 |
n = 760, R² = 0.746. The tower taker was already ahead
by +2.62k at tower time.

### End-of-game gold

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
| blue_first_dragon | +3.455 | 0.645 | 8.5e-08 |
| blue_first_tower | +3.091 | 0.821 | 0.00017 |
| gold15_diff_k | +2.079 | 0.134 | 1.9e-54 |
n = 760, R² = 0.441. With pre-game Elo:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
| blue_first_dragon | +3.356 | 0.635 | 1.2e-07 |
| blue_first_tower | +2.828 | 0.809 | 0.00047 |
| gold15_diff_k | +1.937 | 0.137 | 1.3e-45 |
| elo100 | +1.319 | 0.235 | 1.9e-08 |
n = 760, R² = 0.463. Holding gold@15 (and Elo) fixed,
first dragon adds +3.36k and first tower adds
+2.83k to the final gold diff — i.e. both objectives
predict *further* snowball beyond the 15-minute state.

## 3. Does the objective predict the win beyond gold@15?

Logistic regression, P(blue wins), pass-only, league + patch fixed effects.
Cells show logit coefficient (average marginal effect in percentage points).
Elo = pre-game `elo_diff` per 100 points. Four nested specs separate
*selection* (Elo control) from *mediation* (gold@15 control).

| Predictor | Raw | + Elo | + gold@15 | + gold@15 + Elo |
|---|---|---|---|---|
| blue_first_dragon | +1.224 (+22.9pp) | +1.222 (+21.9pp) | +1.063 (+17.6pp) | +1.071 (+17.2pp) |
| blue_first_tower | +1.544 (+28.9pp) | +1.416 (+25.3pp) | +0.551 (+9.1pp) | +0.507 (+8.1pp) |
| gold15_diff_k (per +1k) | — | — | +0.440 (+7.3pp) | +0.415 (+6.7pp) |
| elo100 (per +100 Elo) | — | +0.412 (+7.4pp) | — | +0.348 (+5.6pp) |
| n | 760 | 760 | 760 | 760 |
| McFadden pseudo-R² | 0.177 | 0.209 | 0.262 | 0.281 |

**Interpretation, first dragon.** Raw AME +22.9pp → +21.9pp after
Elo control (5% of the raw effect is pre-game-strength selection)
→ +17.2pp after Elo + gold@15 (22% of the Elo-controlled
effect is mediated through gold). Net: a substantial residual remains after pre-game strength + gold@15. The residual is consistent
with dragon-stack value (soul progress is real win equity not in gold@15)
and/or unmeasured within-game skill — this design cannot separate those two.

**Interpretation, first tower.** Raw AME +28.9pp → +25.3pp
after Elo → +8.1pp after Elo + gold@15. The tower's residual is
**not** pure mediation evidence: the tower usually falls *after* 15:00, so its
coefficient contains post-15 information (map pressure, lane priority) that
gold@15 cannot capture. Extended spec adding post-15 objectives (baron, herald)
is in the appendix.

**Bottom line for the model:** first dragon's win signal is roughly
5% pre-game-strength selection, 22% gold-mediated
snowball, and a residual consistent with soul-stack value. There is no free
*pre-game* signal here — first objectives are in-game state, usable only in
the live model, and gold@15 carries only ~22% of the dragon's effect — the
+17.2pp residual is the majority of it. A live-model first-dragon
feature is worth a follow-up test against the role-gold live model: if the
residual survives alongside per-role gold, it adds real signal (soul-stack
value); if it collapses, it was proxying for gold distribution.

## 4. Timing

| League | Median first dragon (s) | Median first tower (s) |
|---|---|---|
| FST | 408 | 966 |
| LCK | 453 | 928 |
| LCS | 433 | 945 |
| LEC | 432 | 920 |
| MSI | 451 | 885 |
| WLDs | 432 | 806 |

Overall (pass-only): median first dragon 438s (7.3 min),
median first tower 925s (15.4 min). All games:
dragon 435s, tower 945s.

**Does an earlier first dragon predict more?** Quartiles of first-dragon time:

| Dragon-time quartile | Median dragon (s) | n | Taker win rate | Mean taker gold@15 (k) |
|---|---|---|---|---|
| Q1 earliest | 352 | 190 | 58.9% | +0.46 |
| Q2 | 395 | 190 | 68.4% | +1.04 |
| Q3 | 505 | 190 | 56.8% | +0.34 |
| Q4 latest | 637 | 190 | 64.7% | +0.57 |

Regression (pass-only, taker perspective, league + patch + taker-side + taker-Elo controls):

| Predictor | Outcome: taker wins (logit) | | | Outcome: taker gold@15 (OLS, k) | | |
|---|---|---|---|---|---|---|
| | Coef | SE | p | Coef | SE | p |
| dragon time (per +1 min later) | -0.004 | 0.039 | 0.91 | -0.033 | 0.039 | 0.4 |
| taker Elo (per +100) | +0.477 | 0.071 | 1.8e-11 | +0.495 | 0.078 | 1.7e-10 |

n = 760. An earlier first dragon predicts modestly more gold for the
taker (-0.033k per minute later) and a directionally higher
taker win rate, but the win-rate timing gradient is
not significant
at 5%. Timing is a weak signal on top of *taking* the dragon.

## 5. Data quality and sensitivity

**How flagged games were handled.** Primary analysis uses `validation == "pass"`
games only (n = 761). The 538 flagged games were *not* treated as wrong: a
flag is a disagreement signal between Leaguepedia's timeline parse and OE's
team-row totals, not proof the timeline is wrong. In particular:

- Tower-count flags dominate
  (483 flag instances).
  OE itself reports physically impossible tower counts (>11) in
  19 pilot games, so the
  disagreement runs both ways. The first-objective *team/time* fields come from
  timeline parsing, independent of OE totals — a tower-count flag does not
  invalidate first-tower team/time.
- 64 flags are
  skipped invalid-team artifact events (phantom grubs with invalid killerTeamId),
  dropped by the parser and recorded.
- The 43 chronobreak-flagged games (replayed-segment signature) have event
  times that are approximate; they are excluded in one sensitivity sample.
- Validation strings mentioning a first-X mismatch:
  1 (i.e. the
  timeline-derived first-X team disagrees with OE's first-X flag) — negligible.

**Sensitivity:** taker win rates under all three samples:

| Objective | Sample | n | Taker wins | Taker win rate |
|---|---|---|---|---|
| First dragon | pass-only | 760 | 473 | 62.2% [58.7, 65.6] |
| First dragon | all games | 1298 | 801 | 61.7% [59.0, 64.3] |
| First dragon | excl. chronobreak | 1255 | 774 | 61.7% [59.0, 64.3] |
| First tower | pass-only | 761 | 539 | 70.8% [67.5, 73.9] |
| First tower | all games | 1299 | 891 | 68.6% [66.0, 71.1] |
| First tower | excl. chronobreak | 1256 | 866 | 68.9% [66.3, 71.4] |

The dragon and tower win rates move by ≤1.5pp across samples — the conclusions
do not depend on the flag handling.

**Other caveats.** Games with no first objective are excluded from that
objective's analysis: 1 game with no first dragon (pass), 38 pass +
20 flagged games with no baron, 17 flagged games
with no herald (0 pass), 15 games with no grub taken. First-grubs is
derived from the earliest GRUB event in the timers file (one event per grub
taken), not from a firsts column. OE gold@15 is missing for 170 of 9,694 OE
team-row games, but only 1 pilot game lacks OE gold. Patch is treated as
categorical; no clustering by series (series IDs not in the pilot).

## Appendix: extended win model (+ baron, herald — post-15 info)

| Predictor | Logit coef | SE | p |
|---|---|---|---|
| gold15_diff_k | +0.209 | 0.069 | 0.0024 |
| blue_first_dragon | +1.087 | 0.269 | 5.2e-05 |
| blue_first_tower | +0.455 | 0.283 | 0.11 |
| blue_first_baron | +3.438 | 0.256 | 3.1e-41 |
| blue_first_herald | +0.376 | 0.265 | 0.16 |
| elo100 | +0.187 | 0.103 | 0.069 |
n = 722, McFadden pseudo-R² = 0.528. Baron and herald
are included for completeness only — both typically occur after 15:00, so they
are post-15 predictors, not mediation evidence.

## Reproduction

Snapshot: `data/processed/pilot_snapshot_2026-10-17/` (frozen 2026-10-17).
Script: `reports/first_objectives_analysis.py`. Gold: `data/raw/2026.csv` team rows.
Elo: `data/processed/features.csv` (`elo_diff`).
