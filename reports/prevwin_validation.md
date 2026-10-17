# Previous-map-winner validation with the FULL model

**Date:** 2026-10-17 · **Model version:** v1.3.2 · **Status:** validation only,
no model files changed

## (a) Re-test with the FULL model (draft features included)

Within-series games 2+ (momentum grouping: league + calendar day + team
pair, non-contiguous groups dropped), `y ~ logit_full + prev_win_blue`,
shipped v1.3.2 full-model coefficients/scaler from
`model/phone_artifacts.json`. This mirrors momentum §2, which used the
nodraft model. n = 14603.

| Control | prev-win coef | SE | p | In-sample lift |
|---|---|---|---|---|
| logit_now (current-game features) | +0.284 | 0.038 | 6.8e-14 | +0.0019 |
| logit_pre (game-1 features, oriented) | +0.514 | 0.036 | < 1e-40 | +0.0070 |

Momentum §2 (nodraft) for comparison: +0.284 (current-Elo control),
+0.514 (pre-series control). Independent 4-decimal check: nodraft
now +0.2845 / pre +0.5137 vs full now +0.2836 / pre +0.5135 — the draft
features move the coefficient by ~0.001, i.e. they do not absorb it.

**Verdict on (a):** The coefficient SURVIVES the full model — draft features do not absorb it. The feature stands (subject to part (b)).

## (b) Pre-registered-style validation: refit on pre-Worlds, score Play-In

- Base full LR refit on pre-Worlds data only (date < 2026-10-15,
  n = 29922): C = 0.05 (re-tuned via TimeSeriesSplit grid
  [0.05, 0.2, 1.0, 5.0]), 180-day recency weights, t_now = 2026-10-14.
- Second stage on pre-Worlds games 2+ (n = 14590):
  prev-win coefficient = +0.285 (SE 0.038, p = 4.8e-14).
- Play-In games 2+ (pair-grouping, so the MVK–FURIA series split across
  midnight stays together): n = 11.
  Log-loss with prev-win = 0.6762, without = 0.6890,
  **lift = +0.0128**. Criterion was lift > 0:
  **PASS**.
- Sensitivity (momentum day-grouping, which drops MVK–FURIA games 2–3 as
  non-contiguous): n = 9,
  lift = -0.0140 (directional only — the dropped games are
  exactly the ones the pair-grouping keeps).
  **Fragility note:** the two groupings disagree in sign, and the entire
  difference is the two MVK–FURIA games (both won by MVK after its game-1
  upset). The PASS rests on those two games; treat it as encouraging, not
  conclusive.

Per-game Play-In scoring (p_with vs p_without prev-win):

| Date | Match | Game | prev-win blue | y | p_with | p_wo |
|---|---|---|---|---|---|---|
| 2026-10-15 | Cloud9 vs Karmine Corp | game 2 | prev-win blue=1 | y=0 | p_with=0.564 | p_wo=0.522 |
| 2026-10-15 | Karmine Corp vs Cloud9 | game 3 | prev-win blue=1 | y=0 | p_with=0.651 | p_wo=0.620 |
| 2026-10-15 | Cloud9 vs Karmine Corp | game 4 | prev-win blue=1 | y=0 | p_with=0.513 | p_wo=0.467 |
| 2026-10-15 | Karmine Corp vs Cloud9 | game 5 | prev-win blue=1 | y=1 | p_with=0.688 | p_wo=0.661 |
| 2026-10-16 | MVK Esports vs FURIA | game 2 | prev-win blue=1 | y=1 | p_with=0.389 | p_wo=0.336 |
| 2026-10-16 | MVK Esports vs FURIA | game 3 | prev-win blue=1 | y=1 | p_with=0.446 | p_wo=0.395 |
| 2026-10-16 | Karmine Corp vs MVK Esports | game 2 | prev-win blue=1 | y=1 | p_with=0.771 | p_wo=0.756 |
| 2026-10-16 | MVK Esports vs Karmine Corp | game 3 | prev-win blue=0 | y=0 | p_with=0.276 | p_wo=0.284 |
| 2026-10-16 | Cloud9 vs FURIA | game 2 | prev-win blue=1 | y=0 | p_with=0.607 | p_wo=0.570 |
| 2026-10-16 | FURIA vs Cloud9 | game 3 | prev-win blue=1 | y=1 | p_with=0.517 | p_wo=0.471 |
| 2026-10-16 | FURIA vs Cloud9 | game 4 | prev-win blue=1 | y=1 | p_with=0.588 | p_wo=0.549 |

**Caveat:** n = 11 is small — one game can swing the lift. This is a
pre-registered check, not a proof. **Swiss validation is still pending:**
no Swiss games exist in the data yet (snapshot through 2026-10-16).

## Recommendation

**Validated coefficient: +0.285** (pre-Worlds second-stage fit). Play-In lift = +0.0128 (PASS on n = 11). Ship decision stays with the v1.4.0 plan's conditions: use the shrunk +0.14 in the series-state model (not the full fitted value), games 2+ only, never alongside series-score dummies, and re-check on Swiss games 2+ when they exist.

## Reproducibility

Protocol: base LR = train.py (C grid, 180-day weights, scaler fit with
sample weights); second stage = unpenalized logistic regression
`y ~ logit + prev_win` with Hessian SEs, mirroring momentum §2.
Series = league + day + pair for (a) and the pre-Worlds second stage;
league + pair for Play-In scoring in (b). Draft features in the control
are the games' own post-hoc draft features — legitimate for the diagnostic
question (does prev-win survive when draft is controlled?).
