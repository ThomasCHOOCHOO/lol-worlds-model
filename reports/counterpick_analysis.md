# Counterpick and lane-strength analysis

**Date:** 2026-10-17 · **Model:** v1.3.2 baseline · **Data:** Oracle's Elixir 2024–2026, 29,942 games
**Status:** analysis only — no model files changed.

## TL;DR

Counterpick effects are real but small, noisy, and mostly **not** mediated through lane gold.
The tested feature improves holdout log-loss by ~0.001 (paired z = −1.8, not significant),
below the ~0.002 gate this project has used for shipping. **Recommendation: NO-SHIP**
for v1.3.3 as a plain additive feature. The most promising follow-up is a
patch-aware, solo-lane-only variant with hierarchical shrinkage (see §8) — and
even that should have to clear the gate on a fresh evaluation.

| Test | Baseline (6 feats) | + counterpick (best variant) | Δ |
|---|---|---|---|
| Holdout tail, raw LR log-loss (n=3,368) | 0.6332 | 0.6318 (+ cp_win_solo) | −0.0014 (z = −1.8) |
| Holdout tail, accuracy | 0.6485 | 0.6476 | −0.001 |
| Holdout tail, AUC | 0.6901 | 0.6929 | +0.003 |
| Worlds 2025 backtest, log-loss (n=96) | 0.6803 | 0.6771 (+ cp_win + cp_gold) | −0.003 (noisy) |
| Worlds 2025 backtest, accuracy | 0.552 | 0.615 | +0.06 (noisy) |

## 1. Method

**Unit.** One observation per game-lane: (lane, champion A, champion B), pooled across
both side orientations. A matchup is stored canonically — alphabetically first
champion's perspective — with the sign flipped when the orientation reverses,
so (Renekton vs Gnar) and (Gnar vs Renekton) games accumulate into one estimate.

**The confound.** Raw matchup win rates are polluted by team strength: strong teams
get last pick and take counterpicks, so a naive "counterpick win rate" partly
measures "good teams win". Every matchup stat below is **residualized**:

1. Fit a baseline `P(win) ~ elo_diff` logistic regression on the fit set only
   (< 2026-07-01). Coefficient 0.0058/logit per Elo point, intercept +0.14 (blue side).
2. Per player-row: `win_resid = won − P(side wins | Elo)`. A matchup's win edge is
   the mean residual for that champion in that matchup.
3. Gold: per-lane OLS `golddiffat15 ~ elo_signed` on the fit set (slope ≈ +0.3 to
   +0.9 gold per Elo point depending on lane); `gold_resid = actual − predicted`.
   `golddiffat15` in Oracle's Elixir is the player's diff vs their lane opponent,
   own-side perspective.

**Shrinkage.** `edge = Σ resid / (n + k)`, k = 20 pseudo-observations toward 0
(no matchup edge). At n = 30 the estimate is 60% data / 40% prior; at n = 100 it
is 83% data. 95% CIs use SE = 0.5/√n on the win edge (binomial, p ≈ 0.5).

**Walk-forward (no leakage).** Matchup aggregates for a game use **only games with
earlier dates** — the feature for game *t* never sees game *t* or later. The
residualization baselines are 2-parameter models fit on the fit set; applying them
inside the fit set is negligible leakage, and the holdout tail is fully clean.
(One caveat: the Worlds 2025 backtest below reuses the fit-set baselines, which
include Worlds 2025 games — a 2-parameter leak. Treat backtest deltas as
directional only; the honest comparison is baseline-vs-variant in the same setup.)

**Feature tested.** `cp_win` = sum over 5 lanes of the shrunk win edge, blue
perspective. Variants: `cp_gold` (sum of shrunk gold edges / 1000),
`cp_win_solo` (top + mid + bot only), `cp_win_nosup` (all but support),
trailing-365-day-window versions of each.

## 2. How big are counterpick effects?

For matchups with n ≥ 30 (912 pairs), the median |win edge| is **~0.03** (3 points of
residual win probability). The 90th percentile is ~0.08; the largest observed is
~0.15 — and with n = 30–100, the 95% CI on a single matchup is ±0.08–0.18, so
almost no individual "top counterpick" is statistically significant on its own.
Counterpicks are a real but second-order effect, roughly the size of the draft
features already in the model (standardized LR coefficient: `cp_win` 0.116 vs
`d_power` 0.104; adding `cp_win` pulls `d_power`'s coefficient 0.128 → 0.104,
i.e. partial overlap, corr(cp_win, d_power) ≈ 0.15–0.24).

### Top counterpicks by win edge (n ≥ 30, Elo-residualized, shrunk k=20)

**Top** (champion in left column beats the right column):

| Lane | Matchup | n | Win edge | 95% CI | Gold edge |
|---|---|---|---|---|---|
| top | Jax vs Ornn | 63 | +0.146 | ±0.123 | +427 |
| top | Dr. Mundo vs Renekton | 87 | +0.122 | ±0.105 | **−473** |
| top | Camille vs Ornn | 50 | +0.121 | ±0.139 | +296 |
| top | Rumble vs Udyr | 163 | +0.104 | ±0.077 | +318 |
| top | Fiora vs K'Sante | 76 | +0.100 | ±0.112 | +322 |
| jng | Ivern vs Xin Zhao | 68 | +0.128 | ±0.119 | +15 |
| jng | Morgana vs Vi | 36 | +0.118 | ±0.163 | +125 |
| jng | Naafiri vs Skarner | 81 | +0.098 | ±0.109 | +318 |
| jng | Maokai vs Viego | 127 | +0.096 | ±0.087 | **−562** |
| mid | Aurelion Sol vs Orianna | 61 | +0.139 | ±0.125 | −222 |
| mid | Neeko vs Orianna | 117 | +0.106 | ±0.091 | +23 |
| mid | Aurora vs Yone | 126 | +0.087 | ±0.087 | +194 |
| bot | Lucian vs Tristana | 31 | +0.134 | ±0.176 | +356 |
| bot | Xayah vs Zeri | 160 | +0.085 | ±0.077 | +70 |
| bot | Kai'Sa vs Tristana | 100 | +0.081 | ±0.098 | +46 |
| sup | Renata Glasc vs Tahm Kench | 33 | +0.110 | ±0.171 | −197 |
| sup | Rakan vs Tahm Kench | 98 | +0.073 | ±0.099 | −228 |
| sup | Braum vs Poppy | 46 | +0.070 | ±0.144 | +0 |

**Bottom** (left column loses to the right column):

| Lane | Matchup | n | Win edge | Gold edge |
|---|---|---|---|---|
| top | Gwen vs Yorick | 30 | −0.130 | +26 |
| top | K'Sante vs Twisted Fate | 155 | −0.126 | **−989** |
| jng | Sejuani vs Taliyah | 52 | −0.123 | −593 |
| mid | Taliyah vs Zed | 30 | −0.121 | −317 |
| bot | Jinx vs Lucian | 64 | −0.137 | −226 |
| sup | Rell vs Senna | 42 | −0.141 | −423 |

Famous-pair spot checks: Renekton vs Gnar (n=572): win edge −0.029, gold +58 —
the textbook "counterpick" is worth almost nothing in pro play. Fiora vs K'Sante
is a genuine one (+0.100 / +322). Azir vs Ahri (n=294): +0.004 — a coin flip.
Rammus vs Master Yi has **never been played** in this dataset (pro junglers don't
pick Master Yi); the folk example has no pro evidence.

## 3. Does winning a lane mean winning gold? Mostly no.

Across matchups with n ≥ 30, the correlation between win edge and gold edge is
only **0.16** overall (top 0.24, bot 0.23, jng 0.11, sup 0.12, **mid 0.05**).
Winning the matchup and winning lane gold are different things:

- **Dr. Mundo vs Renekton**: +0.122 win edge but **−473 gold** — Mundo loses lane
  and wins games (scaling / teamfight).
- **Maokai vs Viego**: +0.096 win edge, **−562 gold** — utility jungler loses farm,
  wins through setup.
- **Kalista vs Senna** (bot): **+1,033 gold** edge but **−0.029 win** edge —
  Kalista crushes lane gold and Senna outscales it.
- **Senna vs Zeri**: −1,076 gold edge but **+0.045 win** edge — same story reversed.
- **K'Sante vs Twisted Fate**: −0.126 win edge and −989 gold — the one case where
  both agree (TF roams and takes over the map).

Interpretation: a large share of counterpick value is **not** lane dominance. It is
scaling curves, teamfight utility, and map pressure — things that show up in the
win column without showing up in gold@15. A feature built only on lane-gold
edges would miss most of the effect, and a feature built only on win edges should
not be interpreted as "wins lane → wins gold".

### Do 3 winning lanes mean a positive gold differential? Yes, on average — weakly.

Per game, count lanes where blue's matchup edge (prior games only) exceeds +0.02:

| # winning lanes | Games | Blue win rate | Mean team gold@15 |
|---|---|---|---|
| 0 | 4,433 | 49.0% | +171 |
| 1 | 10,412 | 51.9% | +247 |
| 2 | 9,429 | 54.0% | +276 |
| 3 | 4,521 | 57.1% | +473 |
| 4 | 1,026 | 60.9% | +526 |
| 5 | 121 | 54.5% | +565 |

And symmetrically, 0 losing lanes → 57.4% win / +460 gold; 4 losing lanes →
47.2% / +117. A single strong counterpick (any lane edge > +0.10, 2,403 games)
→ 57.3% win / +463 gold vs 49.0% with no lane edge > +0.02.

Caveats: (a) this is descriptive, not causal — better teams get counterpicks via
last pick, and residualizing on Elo removes only part of that; (b) the increments
are modest — going from 0 to 3 winning lanes buys ~+300 gold and ~+8pp win rate,
not a blowout; (c) the 5-lane bucket (n=121) is noise.

## 4. Jungle and support: the diffuse lanes

**Jungle.** Junglers almost never lane against each other, so "jungle matchup" is
really "which jungler's game plan wins" — farm vs gank vs objective control.
The jng-vs-jng numbers behave like a real but weak signal (median |edge| 0.028,
similar to other lanes; e.g. Maokai vs Viego +0.096, Sejuani vs Taliyah −0.123).
What jng-vs-jng does **not** capture is the actual mechanism: a jungler's effect
runs through lanes (gank pressure, counter-ganking). A cleaner formulation is
**jungler + laner vs jungler + laner** (e.g. top+jungle 2v2), but that explodes
the pair space exactly like bot 2v2 below and is too sparse to estimate directly.
Recommendation: keep jng-vs-jng as a heavily-shrunk term if used at all, and
treat it as "jungler game-plan edge", not lane dominance. Empirically it adds
little: `cp_win_nosup` (drops only support) ≈ `cp_win`; the solo-lane variant
that also drops jungle did best on the tail (see §6), consistent with jungle
matchup stats being the noisiest of the five.

**Support.** Sup-vs-sup 1v1 is nearly meaningless — supports don't CS and rarely
fight alone. The real unit is the **bot 2v2** (ADC + support vs ADC + support).
But the pair space is huge: 16,062 distinct 2v2 pairs, only **117 with n ≥ 20**,
and none of those 117 has a significant edge (largest |edge| 0.148, CI ±0.173;
e.g. Ezreal+Rell vs Kai'Sa+Leona n=32, +0.148 ± 0.173). The 2v2 table is
unusable as a feature at current sample sizes — every entry is noise-dominated.
Recommendation: **do not** build a 2v2-pair feature. Use ADC matchup + support
matchup as separate additive terms (both shrunk), which is what `cp_win` does,
and accept that support matchup signal is weak (sup has the smallest median
|edge|, 0.027, and the solo-lane variant that drops both jng and sup performed
best).

## 5. Data sparsity and coverage

Full-history distinct matchup pairs (canonical, both orientations pooled):

| Lane | Distinct pairs | n ≥ 20 | n ≥ 30 | n ≥ 50 |
|---|---|---|---|---|
| top | 1,415 | 230 | 185 | ~130 |
| jng | 1,211 | 255 | 195 | ~150 |
| mid | 1,327 | 250 | 203 | ~165 |
| bot | 744 | 203 | 176 | ~180 |
| sup | 840 | 196 | 153 | ~140 |
| bot 2v2 pairs | 16,062 | 70–117 | — | — |

Coverage at game time (prior games only, walk-forward): a typical game has
**3.68 of 5 lanes** with a prior n ≥ 20 matchup estimate; 29.6% of games have all
5 lanes covered, 1.7% have none, and on average 4.82 of 5 lanes have at least
one prior meeting. So sparsity is **not** the binding constraint for 1v1
matchups — shrinkage handles the thin lanes — but it is fatal for 2v2 pairs.

Data-quality notes: gold@15 coverage is 92% overall (98% in 2026; lower in 2024).
One odd champion label ("Locke", 43 games vs Syndra) appears to be a data
artifact and was not interpreted. Champion pool: 173 champions; per-lane pools
88 (jng) to 131 (top).

## 6. Model test (train.py protocol)

Splits: LR fit < 2026-07-01 (C tuned by 3-fold TimeSeriesSplit, StandardScaler,
180-day recency half-life, same as v1.3.2). Tail = 2026-07-01..2026-10-01
(n=3,368) is the honest evaluation for the raw LR (isotonic is fit on the tail,
so isotonic-tail numbers are parity-only, as in train.py). Baseline replication
check: isotonic-tail log-loss 0.6253 exactly reproduces v1.3.2's reported number.

| Variant | Raw-tail LL | Raw-tail acc | Raw-tail AUC | Isotonic-tail LL |
|---|---|---|---|---|
| Baseline (6 feats) | 0.6332 | 0.6485 | 0.6901 | 0.6253 |
| + cp_win (all 5 lanes) | 0.6323 | 0.6473 | 0.6918 | 0.6257 |
| + cp_gold | 0.6328 | 0.6467 | 0.6909 | 0.6250 |
| + cp_win + cp_gold | 0.6320 | 0.6470 | 0.6921 | 0.6255 |
| **+ cp_win_solo (top/mid/bot)** | **0.6318** | 0.6476 | 0.6929 | 0.6247 |
| + cp_win_nosup | 0.6319 | 0.6473 | 0.6925 | 0.6253 |
| + cp_win_365 (trailing-365d) | 0.6336 | 0.6452 | 0.6896 | 0.6273 |
| + cp_win_solo_365 | 0.6328 | 0.6485 | 0.6912 | 0.6251 |

Paired per-game log-loss differences on the tail (the right significance test —
it removes game-to-game variance shared by both models):

- `cp_win_solo` vs baseline: ΔLL = **−0.00137 ± 0.00076 (z = −1.80, p ≈ 0.07)**,
  better on 52.5% of games.
- `cp_win` vs baseline: ΔLL = −0.00097 ± 0.00088 (z = −1.09).

The trailing-365d window **does not help** — all-history pooling beats it. Fresher
data is not the fix; the signal is just weak.

Worlds 2025 backtest (refit pre-event, n=96 — noisy, directional only):
baseline LL 0.6803 / acc 0.552; +cp_win LL 0.6791 / acc 0.604;
+cp_win+cp_gold LL 0.6771 / acc 0.615. Same direction as the tail, same
non-significance (n=96 moves ±0.03 LL on a handful of games).

## 7. Why the signal is weak: matchup edges don't persist

The sharpest negative result. Split matchup edges into pre-2026 vs 2026 eras
(matchups with n ≥ 30 in **both** eras, n=268 pairs):

| Lane | Pairs | Corr(edge_pre-2026, edge_2026) |
|---|---|---|
| top | 51 | 0.35 |
| bot | 64 | 0.43 |
| sup | 52 | 0.12 |
| jng | 45 | 0.06 |
| mid | 56 | **−0.11** |
| **all** | **268** | **0.17** |

Sign agreement is 53.7% — barely above a coin flip. A matchup that "countered"
in 2024–25 usually does **not** counter in 2026. Patches rework champions,
items, and the meta faster than matchup history accumulates. This is the core
reason the feature underwhelms: pooling 2.75 years of history averages together
several different games, and any patch-restricted slice is too small to estimate
(a per-patch matchup typically has n = 2–5). Top and bot persist modestly;
mid matchups anti-persist — the worst case for a matchup table.

## 8. Recommendation: NO-SHIP, with a path

**Do not ship** a plain additive counterpick feature in v1.3.3. The gains
(−0.001 LL, accuracy flat-to-down, z = −1.8) do not clear the project's ~0.002
gate, and §7 says the underlying table is mostly non-persistent. Shipping it
would add a 5-table lookup pipeline and a new failure mode (stale matchup tables
after every patch) for a sub-noise gain.

If this is revisited, the variant to test is:

1. **Solo lanes only** (top/mid/bot) — the only variant that beat the baseline on
   both LL and AUC, and the only lanes with a real 1v1.
2. **Patch-aware with prior-patch fallback**, mirroring `d_power`: same-patch
   matchup edge shrunk toward the previous patch's shrunk value, with a larger
   k (e.g. 30–40) because matchup samples are thinner than champion samples.
   The 365d-window test shows naive recency doesn't work; patch structure might,
   because it respects the actual discontinuities (reworks) instead of a smooth
   decay.
3. **Hierarchical shrinkage**: shrink each matchup toward its (champion, lane)
   average and toward the matchup's class (e.g. tank-vs-bruiser), not just toward
   0 — this borrows strength across the many thin pairs instead of discarding
   them.
4. **Re-test on a fresh evaluation** (e.g. post-Worlds games as a true holdout),
   and require ΔLL ≤ −0.002 with a paired z beyond ±2 before shipping. Also
   re-test `cp_win_solo` **without** `d_power` to check whether it subsumes
   champion power or genuinely adds matchup-specific information.

## 9. Implementation sketch (if §8 is pursued)

- `src/build_features.py`: add a `MatchupBook` that walks games in date order,
  keyed `(lane, patch, champA, champB)` canonical, storing Elo-residualized win
  sums and gold sums. Per game, emit `cp_win_solo` from the current-patch table
  with prior-patch fallback (same helper shape as `champ_power`), shrunk with
  k=30. Cost: one extra pass over player rows (~300k), same order as the
  existing feature build.
- `src/train.py`: add `cp_win_solo` to `FULL_FEATURES` behind a flag; no other
  changes (scaler/LR/isotonic all generic).
- `src/predict.py` / `export_artifacts.py`: the shipped artifact needs the
  matchup table snapshot (final shrunk edges per pair per patch). Estimated size:
  ~5,500 pairs × small floats — tens of KB, fine for the phone artifact. The
  app-side Predict tab already passes 5 picks/side, so the lookup is
  `sum(edge[pick_blue_lane, pick_red_lane])` per lane.
- Maintenance: the matchup table must be rebuilt on every refresh (same as
  `d_power`); a patch with no prior meetings falls back to the previous patch,
  then to 0.

## 10. Limitations

- Residualization uses Elo only; residual confounding remains (draft order —
  counterpick side usually has last pick; team-specific champion pools; side
  selection). A cleaner adjustment would residualize on Elo + draft order +
  champion power, but draft order is not in the current feature set.
- `golddiffat15` is missing for 8% of rows (mostly 2024); gold aggregates skip
  those rows, which is fine but slightly biases early-era gold edges toward
  better-covered leagues.
- Lane assignments are Oracle's Elixir `position` labels; lane swaps (e.g. a top
  laner roaming bot for an early fight) are not modeled, and flex picks can
  mislabel a matchup.
- All matchup stats pool across leagues and tournament stages. A Worlds-only
  matchup table would be the most decision-relevant slice and also the smallest;
  not estimated here.
- The era-persistence test (§7) uses the same Elo residualization in both eras;
  patch-level persistence (the quantity §8.2 needs) was not directly estimated
  because per-patch samples are too small — that is itself the point.
