# Tier-conditional blend: player Elo × team Elo, weighted by roster instability

**Date:** 2026-10-17 · **Model version evaluated:** v1.3.3 (+ Phase 1 offsets) · **Status:** validation only, NOT shipped (default OFF)
**Verdict: DO NOT SHIP as default.** The blend fails the pre-registered gate: it makes tier-1 + internationals *significantly worse* (+0.0059 log-loss, CI entirely positive) even though the pooled gain survives (−0.0075). The strict-cap variant is the only candidate worth re-validating on fresh data. The shipped model stays team Elo.

## 1. The gate (pre-registered, from `player_ratings_analysis.md` §9)

| Condition | Result |
|---|---|
| Tier-1+intl log-loss NO WORSE than team-Elo baseline — non-inferiority: 95% CI upper bound on (blend − baseline) < **+0.002** | **FAIL** — diff +0.0059, CI [+0.0008, +0.0110] |
| Pooled tail gain survives — 95% CI excludes 0 | **PASS** — diff −0.0075, CI [−0.0110, −0.0040] |

Both must pass. The blend fails the first, so it does not ship — even though the pooled number looks good. (This is exactly the trap the gate exists for: the pooled tail is 76% minors, so a pooled win can hide harm in the decision domain.)

## 2. Design

**Context variable — roster instability.** For each side of a game:
`instability = 1 − |today's 5 players ∩ modal 5| / 5`, where the modal lineup
is the 5 players with the most games for that team in the trailing 30 days
(ties: most recent appearance wins, then player key). Overlap compares
PLAYER keys, not (player, role) keys: a role swap by the same five humans is
not a lineup change for the purpose of "does team Elo still describe these
five?" Fallbacks: if fewer than 5 distinct players appear in the window,
compare against the team's most recent lineup; if the team has no prior
game, instability = 0. This replaces the old `roster_changed` flag (previous
game only), which reads False for KeSPA Cup academy squads whose consecutive
games were all academy squads.

In the full data: mean instability 0.087, 26.7% of games have instability > 0,
1.58% have instability = 1 (a full lineup change). **73% of games have
instability exactly 0 — in both regimes.** That fact drives the failure (§5).

**Blend (per-side).** Each side's rating is blended by its own instability:
`blended_side = w(inst_side) · player_elo_side + (1 − w) · team_elo_side`,
`w = sigmoid(a + b·inst + tier_c·is_tier1)`,
`elo_diff_blend = blended_blue − blended_red + (off_blue − off_red)`.
The region-offset diff is added *after* blending: the region correction is
orthogonal to the player/team question and applies to the game regardless
of w. (Caveat: the offset was fit on team Elo; when w is high, the player
side's own region bias is not separately corrected. The validation says
whether that matters — see §4.)

**Fitting w — constrained, not pooled.** An unconstrained pooled-CV fit
degenerates to w = 1 everywhere (pure player Elo), because the fit set is
~76% minor-league games where player Elo wins. That reproduces the M2
failure the blend exists to fix. So (a, b) are fit by grid search subject
to a **non-inferiority constraint on the fit set's tier-1+intl games**:
fit-set tier-1+intl CV log-loss ≤ M0's + 0.002 (the shipping gate). Among
feasible candidates, minimize pooled CV log-loss. The tier term is adopted
only if it beats instability-only pooled CV by ≥ 0.002 — it did not, so
the shipped parameterization has tier_c = 0.

## 3. Fitted parameters

Primary (constrained, cap = M0 fit-set tier-1 CV + 0.002):

| | a | b | tier_c | Fit-set CV pooled | Fit-set CV tier-1+intl |
|---|---|---|---|---|---|
| Blend | +0.50 | +2.00 | 0 | 0.6316 (M0 0.6366) | 0.6363 (M0 0.6344, cap 0.6364) |
| Unconstrained (reference) | +1.00 | +1.00 | −1.00 | 0.6311 | 0.6364 (at the cap) |

Implied w: inst 0 → **0.62**, 0.2 → 0.71, 0.5 → 0.82, 1.0 → **0.92**.
Note how high w(0) is: even a perfectly stable lineup gets 62% player
weight. §5 explains why that is fatal.

## 4. Validation — full model, RAW tail log-loss (Jul–Oct 2026, n = 3,368)

Paired bootstrap (20k resamples) on per-game log-loss differences vs M0
(team Elo). M1 = pure player Elo, for reference.

| Subset | n | M0 | Blend | Blend − M0 | 95% CI | M1 − M0 |
|---|---|---|---|---|---|---|
| **Tier-1 + internationals** | 805 | 0.6421 | 0.6480 | **+0.0059** | **[+0.0008, +0.0110]** | +0.0068 |
| Pooled tail | 3,368 | 0.6319 | 0.6244 | −0.0075 | [−0.0110, −0.0040] | −0.0083 |
| Minors | 2,563 | 0.6287 | 0.6170 | −0.0117 | [−0.0162, −0.0074] | −0.0131 |
| Academy/dev (NACL + LCKC) | 267 | 0.6847 | 0.6833 | −0.0014 | [−0.0088, +0.0060] | +0.0005 |
| KeSPA Cup | 61 | 0.8644 | 0.7265 | −0.1379 | [−0.2452, −0.0443] | −0.1608 |
| \|player − team\| > 100 | 282 | 0.6117 | 0.5471 | −0.0646 | [−0.0972, −0.0337] | −0.0672 |

Nodraft (the edge-watch model): tier-1+intl +0.0058 (CI [+0.0010, +0.0107]),
pooled −0.0076. Same pattern.

**Backtests** (honest: (a, b) + LR refit on pre-event data only): Worlds 2025
(n=96) −0.0079, CI [−0.0236, +0.0086]; MSI 2026 (n=71) +0.0060, CI
[−0.0125, +0.0258]. Mixed, small-n, consistent with the tier-1 story.

**Sensitivities** (exploratory — not the pre-registered primary):
- Tier-term variants (best feasible tier_c = −1, −2 from the fit-set grid):
  tier-1+intl +0.0049 / +0.0042, CIs entirely positive. **The tier term does
  NOT rescue the blend.** (Why: §5 — in tier-1, even unstable lineups are
  better described by team Elo.)
- Strict-cap variant (fit-set tier-1 CV ≤ M0's, i.e. zero tolerance;
  a = −4, b = 8, at the grid corner): tier-1+intl **+0.0009**,
  CI [−0.0002, **+0.0020**] — upper bound misses the +0.002 bar by 0.00004,
  i.e. essentially at the boundary; pooled −0.0029, CI [−0.0050, −0.0010],
  gain survives. Its w curve is the shape one would have hand-drawn:
  w(0) = 0.02, w(0.2) = 0.08, w(0.5) = 0.50, w(1) = 0.98 — ~pure team Elo
  for stable lineups, ~pure player Elo for full lineup changes.

## 5. Why the blend fails

The failure is structural, not a fitting accident:

1. **At instability 0 the regimes want opposite weights, and instability
   cannot separate them.** 73% of games have instability 0 in *both*
   regimes. Minors at inst 0 want player Elo; tier-1 at inst 0 wants team
   Elo (the org component). Any tier_c = 0 blend with w(0) > 0 imports
   player-Elo harm into tier-1. Quantitatively: pure player Elo hurts
   tier-1+intl by +0.0068 on this tail, so a blend with w(0) = 0.62 should
   hurt by ≈ 0.62 × 0.0068 ≈ +0.0042 — close to the observed +0.0059.
   The blend's tier-1 harm is almost exactly linear interpolation between
   M0 and M1.
2. **The fit-set CV constraint did not transfer.** On the fit set, the
   primary blend's tier-1 harm was +0.0019 (inside the +0.002 cap); on the
   tail it is +0.0059. Fit-set CV non-inferiority is not a substitute for
   held-out tail validation — the constraint was too loose and the fit-set
   CV too optimistic. (Lesson for future gates: fit the constraint with
   margin, or validate the constraint itself on a held-out slice.)
3. **Instability conflates two different things.** "One sub, same org"
   (a tier-1 team resting a starter) and "academy squad, different tier"
   (KeSPA Cup) both raise instability, but only the second is a case for
   player Elo. In tier-1, even an unstable lineup is usually better
   described by team Elo — the org (coaching, systems, scrim quality)
   still drives performance, and the sub is typically a similar-level
   player from the same org. A better context variable would need to
   distinguish "these five humans play at a different competitive tier
   than the match" from "same org, one change" — e.g. whether the
   lineup's players' recent games were in a much weaker league than the
   match. That is future work, not a parameter tweak.
4. **The tier term cannot fix it** because the harm is concentrated where
   the tier term is weakest: tier-1 games at low-to-moderate instability
   (subs), where the tier term lowers w but not enough, and the remaining
   player weight still hurts.

## 6. The blend does NOT fix NACL / LCK Challengers

The leagues Thomas cares about: academy/dev (NACL + LCKC, n = 267) shows
blend −0.0014 (CI includes 0) and pure player Elo +0.0005 (CI includes 0).
**Neither the blend nor pure player Elo helps academy/dev.** The earlier
"player Elo helps minors" finding is really "player Elo helps KeSPA Cup
(−0.16) and ERLs" — not academy/dev. Academy/dev (M0 log-loss 0.685,
barely better than coin flip 0.693) remains the model's weakest domain,
and player Elo is not the fix. (LDL has 0 games in the tail; no honest
number exists for it.)

## 7. `roster_strength.py` re-validation — NOT validated for Worlds

`roster_strength.py`'s delta_2026 (roster-adjusted prior at the 2026
boundary) feeds the app's displayed ratings *and* its predictions
(`teams[].elo` is roster-adjusted). Tested the same way (LR on fit set,
RAW tail, base `elo_diff_adj` vs adjusted `elo_diff_adj + delta_diff`):

| Window / subset | n | Adjusted − base | 95% CI |
|---|---|---|---|
| Tail, tier-1+intl | 805 | +0.0035 | [−0.0064, +0.0132] (inconclusive) |
| Tail, pooled | 3,368 | +0.0037 | [−0.0005, +0.0077] (nearly significantly harmful) |
| Early-2026 (Jan–Mar), tier-1+intl | 624 | −0.0059 | [−0.0153, +0.0036] (directionally helpful, inconclusive) |
| Early-2026, pooled | 2,430 | +0.0036 | [−0.0010, +0.0081] |

**Verdict: no evidence the delta helps.** On the tail (the decision
domain) the point estimates are harmful; the one positive signal
(early-2026 tier-1) is inconclusive *and* has roster leakage (delta_2026
uses full-year 2026 rosters, including games after the evaluation
window). The delta is material — mean |delta| 25.8 Elo, median 9.9, max
201.2 across the 94 tier-1+intl teams — so the app's displayed ratings
and predictions are shifted by ~26 Elo on average by an adjustment with
no predictive evidence. **Recommendation: the app should use raw
snapshot Elo (`elo_raw`) for predictions until the delta is validated;
at minimum, do not trust it for Worlds.**

## 8. Worked example — KC vs MVK (Oct 18 Play-In final)

| | Team Elo | Player Elo | Instability | w | Blended |
|---|---|---|---|---|---|
| Karmine Corp (blue) | 1700 | 1751 | 0.00 | 0.62 | 1731 |
| MVK Esports (red) | 1562 | 1565 | 0.20 | 0.71 | 1564 |

Primary blend: per-map KC **93.8%**, series **99.8%**. Shipped team-Elo
model: per-map 79.5%, series 93.8%. The blend is *far* more confident on
a tier-1 matchup — exactly the overconfidence the gate rejects. (The
strict-cap variant gives w = 0.02 / 0.08 here, i.e. ≈ the shipped model.)

## 9. Implementation (default OFF)

- `src/player_elo.py` — player-Elo replay + instability (walk-forward).
- `src/build_player_features.py` — writes
  `data/processed/player_features.csv` (per-game) +
  `model/player_blend_snapshot.json` (final ratings, last/modal lineups,
  instability, league). Validated: per-team player Elo matches
  `player_ratings.json` exactly; per-game team Elo diffs match
  `features.csv` exactly; only MVK/FURIA differ from `snapshots.joblib`,
  by exactly the 1.3.3 patch (±32.29).
- `src/train_blend.py` — constrained fit + full validation + bundle writer
  (`model/model_blend.joblib`, BINARY, local only).
- `src/build_features.py --with-player-elo` — merges the sidecar into
  `features.csv`. Default OFF; the Oct 19 refresh runs without it.
- `src/predict.py --elo-blend [--blue-lineup ...] [--blend-a/b/tier-c ...]` —
  blend predictions with last-seen lineups (or overrides). Team Elo comes
  from the live snapshot (carries patches); player ratings/lineups from
  the blend snapshot.
- `src/export_artifacts.py --with-player-blend` — adds a `player_blend`
  section (fitted params + gate verdict + per-team lineups/instability +
  ratings for lineup players only). Default OFF; shipped artifact unchanged.
- `reports/tier_conditional_blend_validation.json` — machine-readable
  numbers behind this report.

**Promotion requirements** (if the strict-cap variant is ever shipped):
1. Re-validate it on fresh data (Worlds 2026 Swiss + knockout) under the
   SAME pre-registered strict-cap rule — the current tail result is at the
   gate boundary and must not be treated as a pass.
2. Resolve the team-Elo quantity: the blend was validated on replay Elo;
   live prediction uses patched snapshot Elo; the app uses roster-adjusted
   Elo. The promoted blend must be validated on exactly the quantity it
   will consume (see §7: the roster adjustment itself is unvalidated).
3. If a tier term is ever adopted, the app needs per-team league plumbing
   (the snapshot already carries each team's last-game league).

## 10. Limitations

- Player Elo uses team results only — no per-player performance (gold
  share, damage, KDA). A per-player performance rating is a different,
  potentially complementary signal, not tested here.
- The (a, b) grid is coarse (a ∈ {−4..1}, b ∈ {0..8}); the strict-cap
  optimum sits at the grid corner (a = −4, b = 8), so an even steeper
  blend might do slightly better on fit-set CV — not chased, because the
  tail is the arbiter and refining on it would be tail-shopping.
- The tier-1+intl tail (n = 805) is the binding constraint; its CI width
  (~±0.005) limits how finely the gate can discriminate. Worlds 2026
  games will add ~100–150 tier-1+intl games for re-validation.
- All comparisons share the other features; interactions between the blend
  and draft features (e.g. comfort is team-champion, not
  player-champion) were not explored.
- The blend snapshot's team Elo is the pure replay; after an Elo patch
  (e.g. 1.3.3) it is stale for the patched teams. `predict.py` works
  around this by taking team Elo from the live snapshot; the snapshot
  itself is regenerated on the next `build_player_features.py` run.

## Appendix: reproduction

```bash
.venv/bin/python src/build_player_features.py   # replay + snapshot + checks
.venv/bin/python src/train_blend.py              # fit + validation + bundle
.venv/bin/python src/predict.py --blue KC --red MVK --best-of 5 --elo-blend
.venv/bin/python src/predict.py --blue KC --red MVK --best-of 5 --elo-blend \
    --blend-a -4 --blend-b 8                      # strict-cap variant
```

Key numbers to reproduce: primary (a=0.5, b=2.0, tier_c=0) tail
tier-1+intl +0.0059 [CI +0.0008, +0.0110], pooled −0.0075; strict-cap
(a=−4, b=8) tail tier-1+intl +0.0009 [CI −0.0002, +0.0020], pooled −0.0029;
roster_strength tail pooled +0.0037.
