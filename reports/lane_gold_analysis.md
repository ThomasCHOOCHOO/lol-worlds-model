# Do winning lanes produce a gold differential?

**Question (from the v1.3.2 white-paper follow-ups):** if a team drafts winning
matchups, does that turn into a gold lead? And specifically: does a strong
counterpick (e.g. Rammus into Master Yi) show up as a gold edge in pro data?

**Short answer:** yes, but weakly. Each draft-predicted winning lane is worth
**+249 gold at 15 minutes** (95% CI +196…+303) after controlling for Elo.
Three winning lanes average **+423 gold** — but only a **54.2%** chance of a
positive team gold diff. Winning lanes tilt the early game; they do not decide
it. And the Rammus–Yi counter has **never happened in a pro game in 2024–2026**,
so pro data cannot confirm or deny it — it is ladder folklore as far as this
dataset is concerned.

**Recommendation: NO-SHIP** for a matchup-edge feature in its tested form
(pooled cross-patch, summed or per-lane). It is real and well-calibrated but
too weak to move log-loss: paired improvement **+0.0004** (95% CI
−0.0006…+0.0013) vs the 0.002 ship gate, and accuracy *drops*. See
"Ship / no-ship" below for the refinements worth testing before killing the
idea outright.

---

## 1. Method

### Data

Oracle's Elixir per-player rows, `data/raw/2024.csv + 2025.csv + 2026.csv`.

- **137,795** (game, position) rows with `goldat15` present; **27,559**
  complete games (all 5 positions present).
- **Matchup table** built from **18,035** games in 2024–2025.
- **Evaluation set:** **9,524** complete games in 2026.

### Definitions

**Lane** = one of the 5 positions: top, jungle, mid, bot (ADC), sup (support).
Two caveats: (a) the bot "lane" is really a 2v2 — ADC and support edges are
computed independently and summed, so duo synergy is ignored; (b) jungle
"lane gold" compares two junglers' farm, but junglers differ hugely in how
much they farm vs gank (see Rammus below), so cross-class jungle gold
comparisons are noisy by construction.

**Draft-predicted matchup edge (pre-game, legitimate feature).** For each
ordered pair (position, champion A, champion B), the matchup table stores the
mean lane gold@15 diff of A vs B across 2024–2025 games, shrunk toward 0:

```
edge(A vs B) = sum_gold_diff / (n + 3)        # 3 pseudo-games toward 0
```

For a 2026 game, each position's edge is looked up from that table; unseen
pairs get 0. **Summed matchup edge** = sum of the 5 position edges
(blue perspective). **Winning lane (draft)** = position edge > 0.

A second, win-rate-based version was also computed:
`wr_edge = (wins + 10×0.5)/(n + 10) − 0.5`. It failed (Section 3).

**In-game lane state (live-model only).** **Winning lane (in-game)** = that
position's actual `goldat15` diff > 0. This is observable only once the game
is underway.

**No-leakage discipline.** The matchup table uses *only* 2024–2025 games and
is evaluated on 2026 games, so no eval-set game contributes to its own
prediction. The incremental test (Section 6) additionally fits on 2026 H1 and
tests on 2026 H2.

**Matchup-table thinness.** 8,944 ordered pairs; **median pair n = 3**;
only 39.9% of pairs have n ≥ 5, 26.8% n ≥ 10. Eval-set lane coverage:
93.6% of lanes had a matchup seen in 2024–2025, 85.2% with n ≥ 5. Most
individual matchup edges are therefore shrinkage-dominated — a key reason
the feature underperforms (Section 6).

---

## 2. Winning lanes → gold differential (draft-predicted)

### Regression: team gold@15 diff on number of draft-winning lanes

```
team_gold_diff = −381 + 283 × (# draft-winning lanes),  R² = 0.010
                 95% CI for slope: +230 … +340
```

Controlling for pre-game Elo diff (stronger teams might draft better
matchups — they barely do: corr(#winning lanes, Elo diff) = +0.034):

```
team_gold_diff = −375 + 249 × (# draft-winning lanes) + 9.48 × elo_diff
                 95% CI for slope: +196 … +303
```

So **+249 gold per additional draft-winning lane**, holding team strength
fixed. The unconditional +283 was only mildly confounded.

### Mean team gold diff and P(positive) by number of draft-winning lanes

| # draft-winning lanes | n | mean team gold@15 | median | P(team gold > 0) | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 0 | 348 | −453 | −530 | 42.8% | [37.7%, 48.1%] |
| 1 | 1,775 | −67 | −133 | 47.6% | [45.3%, 49.9%] |
| 2 | 3,157 | +202 | +71 | 51.6% | [49.9%, 53.4%] |
| **3** | **2,827** | **+423** | **+345** | **54.2%** | **[52.4%, 56.0%]** |
| 4 | 1,189 | +756 | +741 | 61.0% | [58.2%, 63.7%] |
| 5 | 228 | +1,215 | +1,257 | 67.5% | [61.2%, 73.3%] |

**Direct answer to "do 3 winning lanes mean a positive gold differential?":
No — not reliably.** Three draft-winning lanes average +423 gold, but the
chance the team is actually ahead on gold is only **54.2%**, barely above a
coin flip. Even winning all 5 draft matchups leaves a ~1-in-3 chance of
being *behind* on gold at 15. Draft matchups tilt the lane phase; jungle
pressure, roams, early fights, and execution dominate the rest. (Game-to-game
SD of team gold@15 is ~3,146 gold; the draft edge explains ~1% of its
variance, R² = 0.010.)

### Regression on summed matchup edge (magnitude, not just count)

```
team_gold_diff = 286 + 1.097 × (summed matchup edge),  R² = 0.029, corr = 0.171
                 95% CI for slope: 0.968 … 1.229
                 controlling for Elo: slope 0.963 (95% CI 0.843 … 1.083)
```

The matchup edge is **well-calibrated in gold units**: 1 gold of predicted
edge ≈ 1 gold of realized edge (slope CI covers 1.0, with or without the Elo
control). The intercept (+286) is just the blue-side gold@15 advantage —
blue teams average +287 gold at 15 across the eval set, consistent with the
known ~54% blue-side win rate. The magnitude carries more signal than the
bare count (R² 0.029 vs 0.010), but both are small.

### Win-rate-based matchup edges do NOT predict gold

Using the WR-edge count instead (lanes where the historical matchup win rate
> 50%):

```
slope = +16 gold per extra WR-winning lane, 95% CI −45 … +72, R² = 0.000
corr(gold-edge count, WR-edge count) = 0.194
```

Matchup win rate reflects the *whole game* (scaling, teamfights, team
strength) and is mostly noise at these sample sizes — it says nothing about
the lane phase. **If matchup edges are ever used, use gold-based edges,
never win-rate-based ones.**

---

## 3. In-game lane state (descriptive only — NOT a pre-game signal)

```
team_gold_diff = f(# in-game winning lanes): slope +1,825 gold, R² = 0.722
```

| # in-game winning lanes | n | mean team gold@15 | P(team gold > 0) | 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 805 | −4,709 | 0.0% | [0.0%, 0.5%] |
| 1 | 1,599 | −2,421 | 3.1% | [2.3%, 4.0%] |
| 2 | 2,132 | −746 | 27.8% | [25.9%, 29.7%] |
| 3 | 2,075 | +862 | 73.4% | [71.5%, 75.3%] |
| 4 | 1,831 | +2,653 | 97.7% | [96.9%, 98.3%] |
| 5 | 1,082 | +4,935 | 100.0% | [99.6%, 100.0%] |

This looks much stronger than the draft version — because it is nearly
**arithmetic, not prediction**. Team gold diff is literally the sum of the
five lane gold diffs. If all 5 lanes lose, the sum *must* be negative
(P = 0 exactly); if all 5 win, it *must* be positive (P = 1 exactly).
The 3-lane row (73.4%) is the only non-trivial one: with 3 lanes ahead and
2 behind, the sign depends on magnitudes. **Do not read this as "winning
3 lanes causes a gold lead"** — lane state and gold diff are two views of
the same in-game outcome. Its legitimate use is the live model, where lane
gold state is already observed (Section 5).

---

## 4. Rammus vs Master Yi: ladder folklore, untestable in pro data

| Matchup | n | Win rate (Rammus side) | 95% CI | Mean gold@15 diff (Rammus) |
|---|---:|---:|---:|---:|
| Rammus vs Master Yi, jungle head-to-head | **0** | — | — | — |
| Rammus in a game AND Master Yi in a game (any position) | **0** | — | — | — |
| Rammus jungle overall | 13 | 30.8% | [12.7%, 57.6%] | −518 |
| Master Yi jungle overall | 10 | 40.0% | [16.8%, 68.7%] | +410 |

**The famous counter never happened.** Across all 27,559 pro games in
2024–2026, Rammus appeared in 14 games (13 jungle, 1 support) and Master Yi
in 12 (10 jungle, 2 top) — and they were **never in the same game**. Pro
teams essentially do not play either champion (26 combined appearances),
so the dataset contains zero evidence about the matchup either way.

Two further notes:

- Rammus's −518 mean jungle gold diff is a **champion-class effect**, not a
  matchup effect: Rammus is a gank-heavy tank who barely farms, so his
  jungle gold is low against almost anyone. Comparing his gold to a
  farm-heavy jungler like Master Yi (+410) measures playstyle, not lane
  dominance. This is a general caveat for jungle "lane" gold comparisons.
- The question "does the counter produce a gold edge?" can only be answered
  with **ladder / solo-queue data**, where both champions are actually
  played — with the caveat that ladder matchups mix all skill levels and
  don't transfer cleanly to pro play.

---

## 5. How much does gold@15 matter for winning? (controlling for Elo)

Logistic regression on the 9,524 eval games, blue win as outcome:

| Model | Log-loss | AUC |
|---|---:|---:|
| Elo only | 0.6350 | 0.689 |
| Elo + team gold@15 diff | 0.4895 | 0.840 |

(Caveat: this comparison is fit and evaluated on the same 2026 set, so the
absolute numbers are optimistic — but the *gap* between the two models under
identical in-sample conditions shows gold@15's incremental signal.)

Predicted blue win probability at even Elo (`elo_diff = 0`):

| Team gold@15 diff | P(blue wins) |
|---:|---:|
| −2,000 | 26.8% |
| −1,000 | 38.3% |
| −500 | 44.7% |
| 0 | 51.3% (blue-side edge) |
| +500 | 57.8% |
| +1,000 | 64.0% |
| +2,000 | 75.1% |

**+1,000 gold at 15 multiplies the odds of winning by 1.69×.** Gold@15 is
a dominant *live* signal — consistent with the shipped live model, where
accuracy rises from 65.1% pre-game to ~82.9% by t=25. This is exactly why
gold@15 belongs in the live model and not (directly) in the pre-game model:
pre-game, gold@15 is unobserved; the only pre-game lever is the weak
draft-matchup proxy of Section 2.

---

## 6. Ship / no-ship: does a matchup-edge feature help the pre-game model?

Honest test: fit logistic regression on v1.3.2 features with and without the
summed matchup edge on 2026 H1 (5,888 games), evaluate on 2026 H2
(3,636 games). Same protocol as the model's own gates (0.002 log-loss
improvement required to ship).

| Model | Test log-loss | Accuracy | AUC | Paired improvement vs base | 95% CI |
|---|---:|---:|---:|---:|---:|
| v1.3.2 features | 0.6338 | 0.6460 | 0.6919 | — | — |
| \+ summed matchup edge | 0.6334 | 0.6425 | 0.6920 | +0.0004 | [−0.0006, +0.0013] |
| \+ 5 per-lane edges | 0.6338 | 0.6449 | 0.6915 | +0.0001 | [−0.0013, +0.0015] |
| \+ summed + # winning lanes | 0.6333 | 0.6444 | 0.6922 | +0.0005 | [−0.0006, +0.0015] |

**Verdict: NO-SHIP.** Every variant's CI includes zero, every improvement is
~4× below the 0.002 gate, and accuracy *falls* in all three variants. The
per-lane breakdown shows why a finer cut doesn't rescue it: per-lane edge
correlations with team gold are top +0.140, jungle +0.062, mid +0.053,
bot +0.043, support +0.046 — top is the only lane with a non-trivial
relationship (the most isolated 1v1 lane), and even it is weak.

Notably, the failure is **not** redundancy: the summed matchup edge correlates
only +0.06…+0.09 with the existing features (`d_power` +0.085,
`eg_gold_diff` +0.085, `d_comfort` +0.064, `elo_diff` +0.061). It is
genuinely new information — just too weak and noisy, because the matchup
table is thin (median pair n = 3) and pooled across three years of balance
patches, so most edges are shrinkage-dominated.

### Refinements worth testing before killing the idea

1. **Patch-aware matchup table.** The tested table pools all of 2024–2025.
   A counter that is real on the current patch may wash out when averaged
   over years of balance changes. Test: trailing-90-day table, or
   same-patch-only with a prior-patch fallback (mirroring `champ_power`).
2. **Extreme matchups only.** The effect may be nonlinear — only *hard*
   counters matter. Test: summed edge restricted to lanes with
   |edge| > 300, or a count of "hard counter" lanes.
3. **Matchup × player comfort.** A counterpick in the hands of a player who
   doesn't play the champion may be worthless; interact the matchup edge
   with the player's comfort on the pick.
4. **Bot-lane 2v2 pairs.** Test ADC+support *pair* matchup edges instead of
   independent ADC/support edges (duo synergy is currently ignored).

Any refinement must pass the same gate: paired log-loss improvement ≥ 0.002
with a CI excluding zero on a time-split holdout.

---

## 7. Causality caveat (read before using any of these numbers)

- **Draft matchup edge → gold diff** is the closest thing here to a
  causal-ish statement: the matchup is fixed at draft time, before the gold
  is earned. But it is still confounded: draft skill correlates with team
  quality, jungle pathing and roams intervene, and a "winning matchup" on
  paper can lose to the better player. Controlling for Elo moved the slope
  from +283 to +249 — small, but the residual confounders (player skill,
  coaching, patch) are unmeasured.
- **In-game lane state ↔ gold diff** is *not* causal in either direction
  for modeling purposes: both are outcomes of the same game. Lane state is
  a *mediator* (draft → lane state → gold → win), and conditioning on a
  mediator is exactly what the live model does — legitimately, because by
  then the mediator is observed fact, not a prediction.
- **Gold diff → win** is likewise not "gold causes wins": gold is a
  summary of who's winning. For the live model that doesn't matter —
  it is a superb *predictor* (Section 5). For the pre-game model, gold is
  unobserved and only the weak draft proxy is available (Section 6).

**Usable pre-game:** draft-based matchup edge (weak; currently no-ship).
**Usable in-game only:** lane gold state, team gold diff (strong; live model).

---

## 8. Implementation sketch (for a future version that passes the gate)

In `src/build_features.py`, mirror the `champ_patch` pattern:

1. Maintain a walk-forward dict keyed `(position, champA, champB)` →
   `[sum_gold_diff, n]`, updated *after* each game's features are computed.
2. Per game, before updating: for each of the 5 positions, look up the
   shrunk edge `(sum + 3×0)/(n + 3)`; emit `d_matchup` = sum of the 5 edges
   (and optionally `edge_top … edge_sup`).
3. For a patch-aware variant, key on `(patch, position, champA, champB)`
   with a prior-patch fallback, mirroring `champ_power`.
4. Prediction path: store the final matchup table in `snapshots.joblib`;
   `predict.py` looks up edges the same way; unseen pairs → 0.
5. Export: add `d_matchup` to `phone_artifacts.json` + `src/model/predict.ts`;
   re-run `validate_export.py` for the TS/Python parity check.

Cost is trivial (~9k pairs). The expensive part is not the code — it is
proving the refined feature clears the 0.002 gate.

---

## 9. Limitations

- **Matchup table thinness** (median pair n = 3) and cross-patch pooling are
  the biggest weaknesses; see refinements in Section 6.
- **Bot lane treated as two independent 1v1s**, ignoring ADC/support synergy.
- **Jungle gold comparisons mix champion classes** (farmers vs gankers);
  jungle matchup "gold edge" partly measures playstyle.
- **2026 eval set includes all leagues**, from major regions to minor
  leagues with different data completeness; no league split was run.
- **Elo+gold comparison in Section 5 is in-sample**; the gap is meaningful,
  the absolute numbers are optimistic.
- **Rammus–Yi is unanswerable from pro data**; ladder data would be needed,
  with its own transfer caveats.

---

## Reproduction

Analysis scripts (scratch, not part of the model):
`/tmp/lane_gold2.py` (main analysis), `/tmp/lane_gold3.py` (per-lane edges,
redundancy, Rammus/Yi same-game check). Eval frame: `/tmp/lane_gold_eval.csv`.
Run with `~/workspace/lol-worlds-model/.venv/bin/python`. Read-only w.r.t.
model files; no model files were modified.
