# Momentum: within-series and within-tournament tests

**Date:** 2026-10-17. **Decision: NO SHIP for series-score and tournament
momentum. CONDITIONAL SHIP for a single "won previous map" indicator inside
the stateful series model (task 6), with a shrunk coefficient and
pre-registered validation on Worlds 2026.**

## Question

1. Within a series: after controlling for pre-series strength, does the
   series score (up 1-0, down 0-2, …) or the previous map's winner change
   P(win next map)?
2. Is any apparent "momentum" really fearless-draft pool depletion or side
   selection?
3. Within a tournament: does winning the previous series, or a win streak,
   predict the next series win beyond Elo?

## Discipline

Elo already absorbs past wins — it updates map-by-map (K=24) inside
`features.csv`, verified: game-1 `elo_diff` 0 → game-2 −24 after a game-1
loss. So the control is **pre-series** strength: the shipped v1.3.2 nodraft
model applied to game-1 walk-forward features, oriented to each later
game's blue side. Momentum must beat that baseline, and the gate is
**out-of-sample log-loss** (fit on 2024–25, test on 2026; plus a reverse
split fit-2024 → test-2025), not in-sample p-values.

A second, sharper control is used in §2: `logit_now`, the same nodraft
model re-run on the *current* game's features (post-update Elo). That is
what the production model would predict if re-run between maps. If a
momentum variable survives `logit_now`, it adds something beyond simply
re-running the model with updated Elo.

## Data and series construction

29,942 games (2024-01 → 2026-10-16) from `data/raw/202{4,5,6}.csv` +
`data/processed/features.csv`. Oracle's Elixir has no `matchid` column
and `gameid` does not reliably group series games, so series =
(league, calendar day, unordered team pair) with contiguous game numbers
starting at 1 — the construction validated in `reports/fearless_draft.md`.

- 14,931 series; 8,653 multi-game, 6,278 Bo1. Dropped: 263
  non-contiguous groups (e.g. series split across midnight), 14 duplicate
  game numbers.
- Within-series dataset: 14,603 games 2+; blue win rate 0.526.
- Raw blue win rate by blue-team lead entering the game: −2: 0.427
  (n=206), −1: 0.507 (n=2,724), tied: 0.519 (n=9,465), +1: 0.584
  (n=2,036), +2: 0.616 (n=172).

## 1. Within-series: series score

Regression: `y(blue wins game k) ~ logit_pre + lead1 + lead2p + trail1 +
trail2p` (tied = reference), n=14,603.

| Term | coef | SE | p | Effect at p=0.50 |
|---|---|---|---|---|
| logit_pre (control) | +0.888 | 0.028 | <1e-16 | — (slope < 1: model slightly overconfident on games 2+) |
| lead1 (blue up 1) | +0.166 | 0.051 | 0.0012 | +4.2 pp |
| lead2p (blue up ≥2) | +0.305 | 0.162 | 0.060 | +7.6 pp |
| trail1 (blue down 1) | −0.072 | 0.045 | 0.11 | −1.8 pp |
| trail2p (blue down ≥2) | −0.374 | 0.146 | 0.011 | −9.3 pp |

In-sample, leading helps and trailing by 2 hurts. **But the state dummies
fail the out-of-sample gate:**

| Split | Log-loss lift (baseline − full) | Paired t | 95% bootstrap CI |
|---|---|---|---|
| Fit 2024–25 → test 2026 (n=5,132) | **−0.0022** (hurts) | −2.06 | [−0.0044, −0.0001] |
| Fit 2024 → test 2025 | +0.0007 | +0.84 | [−0.0008, +0.0022] |

Four parameters, two of them on n≈380 games (lead2p/trail2p): classic
overfitting. **Series-score dummies: NO SHIP.**

## 2. Within-series: previous-map winner (the one signal that survives)

Regression: `y ~ logit_pre + prev_win_blue` (1 if game k's blue team won
game k−1).

| Spec | prev_win_blue coef | SE | p |
|---|---|---|---|
| vs pre-series control (`logit_pre`), full sample | **+0.514** | 0.036 | <1e-16 |
| vs **current-Elo** control (`logit_now`), full sample | **+0.284** | 0.038 | 6e-14 |
| playoffs only, vs `logit_pre` (n=4,358) | +0.495 | 0.065 | 4e-14 |
| by era: 2024 / 2025 / 2026 | +0.636 / +0.572 / +0.515 | ~0.07 | all <1e-15 |

Two facts matter here:

1. **~45% of the prev-win effect is Elo updating.** `corr(Elo-update,
   prev_win_blue) = 0.70`; moving from `logit_pre` to `logit_now` cuts
   the coefficient from +0.51 to +0.28. And with `logit_now` as the
   control, **all four series-score dummies become non-significant**
   (lead1 p=0.30, trail2p p=0.10). The series score adds nothing once
   Elo is current; the last map's winner still does.
2. **Out-of-sample, the gain is small but replicates in both splits:**

| Split | Log-loss lift | Paired t | 95% bootstrap CI |
|---|---|---|---|
| Fit 2024–25 → test 2026 | +0.0038 | +1.62 | [−0.0009, +0.0084] |
| Fit 2024 → test 2025 | **+0.0063** | **+3.38** | [+0.0026, +0.0098] |
| Playoffs only, fit 2024–25 → test 2026 | +0.0029 | +0.66 | [−0.0055, +0.0115] |

Interpretation — read this carefully, because "momentum" is the wrong
word for what the data can show. Observational series data **cannot**
distinguish three mechanisms: (a) psychological momentum (winning causes
winning); (b) a persistent latent component (a team having a good day, a
matchup edge, or a meta read not in Elo — the previous win is *evidence*
of it, not the *cause*); (c) draft adaptation (winner keeps what works).
The prev-win variable captures whichever of these persist to the next
map. For prediction the distinction is irrelevant; for the narrative,
do not claim the model found psychological momentum.

Also note: this test used the **nodraft** model. The full model (with
draft features) may absorb part of the signal, since draft adaptation
shows up in `d_power`/`d_comfort`. Task 6 must re-test with the full
model before using the coefficient for sizing.

## 3. Fearless-draft confound: rejected

No-repeat rates by league-season (share of multi-game series with no
champion repeated by either team; picks from the per-player `champion`
column — the `pick1`–`pick5` columns are all-NaN in this export):

| League-season | 2024 | 2025 | 2026 |
|---|---|---|---|
| LCK | 0.10 | 1.00 | 1.00 |
| LPL | 0.25 | 1.00 | 1.00 |
| LEC | 0.00 | 1.00 | 1.00 |
| WLDs | 0.03 | 1.00 | — |
| MSI | 0.00 | 1.00 | 1.00 |

Fearless = 2025+ seasons with rate ≥ 0.95; control = 2024 seasons with
rate ≤ 0.50. This reproduces the adoption history in
`reports/fearless_draft.md`.

- Fearless subset (n=9,409): lead1 +0.12 (p=0.050), lead2p +0.36
  (p=0.058), trail1 −0.05 (p=0.41), trail2p −0.11 (p=0.59);
  out-of-sample lift **−0.0046** (hurts).
- Control subset (n=2,765): lead1 **+0.33** (p=0.018), lead2p −0.24
  (p=0.60), trail1 −0.08 (p=0.44), trail2p −0.45 (p=0.12).
- Interaction (state × fearless, pooled): all four interaction terms
  non-significant (p = 0.14, 0.38, 0.81, 0.27).

The lead effect is *larger* without fearless than with it. Pool
depletion does not explain the within-series pattern — consistent with
the prior report, which rejected the fearless mechanism twice.

**Side selection is real but cannot explain the pattern away.** Among
non-tied games 2+, the trailing team is blue 57% of the time (loser
picks side): LCK/LPL 64%, LEC 60%, WLDs 76%, MSI 57%. Trailing teams
get the ~54%-win-rate blue side *more* often, yet still lose more —
so raw win rates *understate* the leader's edge. The regressions already
control for side via `blue_edge`, so this is an interpretation note,
not a bias correction.

## 4. Within-tournament: previous-series win / streak

Series-level dataset: 14,931 series. Tournament block = (league, split,
year); per team, previous series outcome and current win streak entering
each series. 13,378 series (89.6%) have ≥1 team with a previous
in-block series; 12,905 have both.

Regression: `series win (team1) ~ logit_pre1 + prev_win1 + prev_win2 +
has_prev1 + has_prev2` (and a `streak_diff` variant, capped ±3). The
control here already includes the previous series' Elo updates, so any
effect is beyond Elo.

| Subset | n | prev_win1 | p | streak_diff | p | Out-of-sample lift |
|---|---|---|---|---|---|---|
| **All series** | 13,378 | **+0.175** | 1e-5 | **+0.080** | 2e-9 | +0.0012 / +0.0008 |
| Regular season only | 11,500 | **+0.232** | 5e-8 | **+0.103** | 1e-11 | +0.0019 / +0.0007 |
| Playoffs + internationals | 2,071 | −0.169 | 0.097 | −0.008 | 0.79 | **−0.0019 / −0.0017** |
| Playoffs only | 1,878 | −0.158 | 0.14 | −0.004 | 0.89 | −0.0009 / −0.0013 |

The headline "significant" result in ALL series is **entirely driven by
regular-season round-robins**, where series are days/weeks apart — so it
cannot be psychological momentum. It is Elo lag / form: with ~1
series/week and K=24, Elo absorbs a team's current form slowly, and the
previous series' result carries form information Elo hasn't priced in
yet. In playoffs and internationals — the only setting Thomas bets —
the effect is zero to slightly negative with **negative** out-of-sample
lift. (A plausible contributor to the sign flip: in a bracket, winning
the previous series means facing another winner next, and Elo does not
fully capture bracket difficulty.)

**Tournament momentum: NO SHIP.** Do not add prev-series-win or streak
features. The regular-season signal (+0.0019 lift) is tiny and does not
transfer to brackets.

## Decision summary

| Feature | Verdict | Evidence |
|---|---|---|
| Series-score dummies (up 1-0, down 0-2, …) | **NO SHIP** | Fail out-of-sample gate (−0.0022 in one split); ns with current-Elo control |
| Previous-map winner (within series) | **CONDITIONAL SHIP** (see sketch) | Survives current-Elo control (+0.28, p=6e-14); stable across eras; replicates in both time splits; small gain |
| Tournament prev-series win / streak | **NO SHIP** | Null/negative in playoffs + internationals, negative lift there |
| Fearless-specific adjustment | **NO SHIP** | Interaction null; effect larger in non-fearless control |

## Implementation sketch (for task 6, stateful series model)

1. **Feature:** `prev_win` — 1 if the team won the immediately
   preceding map of the same series, 0 if it lost, neutral (0.5 or
   omitted) for game 1. In the current per-map orientation: one
   covariate `prev_win_blue`.
2. **Coefficient:** fitted +0.28 with the current-Elo control (+0.51
   with the pre-series control). Ship **shrunk: +0.14** (half the
   current-Elo estimate) — the out-of-sample gains (+0.004–0.006) are
   small and the in-sample coefficient is likely optimistic
   (winner's-curse + unidentified mechanism). Do not use the full
   fitted value for sizing.
3. **Where it applies:** games 2+ only, inside the series-with-state
   model that re-runs the per-map model between games with updated Elo.
   It must **not** be applied to the pre-series (game-1) prediction,
   and it must **not** be combined with the series-score dummies
   (dropped).
4. **Open check before sizing:** re-test with the **full** model
   (draft features included) — draft adaptation may absorb part of
   the +0.28. If the coefficient collapses under the full model, drop
   the feature.
5. **Pre-registered validation:** evaluate on Worlds 2026 Play-In +
   Swiss games 2+ (fit on pre-Worlds data, score Worlds games). Ship
   criterion: log-loss lift > 0 on that set. If ≤ 0, drop — the
   playoffs-only split in this analysis was already inconclusive
   (t=0.66).
6. **Do not** carry this into the tournament-level model: the
   within-tournament tests say the analogous feature fails exactly
   where it would be used.

## Reproducibility

Analysis scripts: `/tmp/momentum.py` (main + fearless + tournament),
`/tmp/momentum_followup.py` (current-Elo control, paired bootstrap CIs,
reverse split, playoffs-only). Inputs: `data/raw/202{4,5,6}.csv`,
`data/processed/features.csv`, `model/phone_artifacts.json` (v1.3.2
nodraft coefficients). P-values from Hessian-based standard errors
(no statsmodels in the venv). Paired log-loss CIs: 2,000 bootstrap
resamples of per-game log-loss differences.

Caveats: series grouping by (league, day, team pair) drops 277
non-contiguous/duplicate groups; the `pick1`–`pick5` columns are NaN
in this Oracle's Elixir export, so fearless no-repeat uses the
per-player `champion` column; the Q3 block (league, split, year)
treats each split as its own tournament, which is right for form but
lumps play-in + main event where a league reuses a split label.
