# Feature screen 2 — eight pre-game candidates (2026-10-13)

Second screening batch, same honest gate as screen 1. Seven dropped, **one ships**.

## Shared method

Each feature was tested by an independent worker against an exact replica of the
`train.py` nodraft pipeline (imports from `src/`, no production code touched):

- fit: date < 2026-07-01 (n=26,211); holdout: 2026-07-01 ≤ date < 2026-10-01 (n=3,368)
- StandardScaler fit on fit set with recency weights w = 0.5**(age_days/180)
- LogisticRegression via `fit_winner(force_lr=True)`; `IsotonicCalibrated` fit+evaluated
  on the holdout (same caveat as train.py — raw-LR deltas reported as the clean check)
- Baseline reproduced by every worker: nodraft cal ll **0.6330** / acc 0.6431 / AUC 0.6849
  (shipped v1.2.0 full-model baseline for reference: 0.6310 / 0.6452 / 0.6865)
- Features built strictly walk-forward (inputs only from games before the target date);
  leakage statements in each worker's section under `/tmp/feat_screen/`

Gate (all must hold): coefficient p<0.05 **and** holdout log-loss improves (≥0.002
preferred — the draft itself was worth 0.002) **and** no identified confound explains it.

## Results

| # | Feature | Δll cal | Δll raw | p | Persistence | Verdict |
|---|---------|---------|---------|---|-------------|---------|
| 1 | Objective control (trailing-20 dragon/baron/tower diffs) | +0.0014 | +0.0015 | 4.4e-5 (tower) | — | **DROP** |
| 2 | Early-game gold@15 diff | **−0.0063** | **−0.0057** | 1.3e-46 | r=0.531 | **SHIP** |
| 2b | First-blood rate | −0.0009 | −0.0000 | 0.97 (joint) | r=0.168 | **DROP** |
| 3 | Patch-adaptation speed | +0.0000 | — | 0.31 | r=−0.116 | **DROP** |
| 4 | Meta-comfort alignment | ±0.0001 | — | 0.86 (full) | — | **DROP** |
| 5 | Flex-pick rate | +0.0001 (worse) | +0.0001 (worse) | 0.0002 | r=+0.01 | **DROP** |
| 6 | Rest/fatigue | +0.0001 (worse cal) | −0.0003 | 0.020 / 0.181 | — | **DROP** |
| 7 | Knockout experience | −0.0026 | −0.0010 | <0.001 | — | **DROP** |
| 8 | Comeback (win% when behind@15) | +0.0003 (worse) | +0.0002 (worse) | 6.8e-14 | r=0.18 | **DROP** |

### 1. Objective control — DROP
Trailing-20 dragon/baron/tower-rate diffs. In-sample significance (tower diff p=4.4e-5)
evaporates out-of-sample: paired per-game improvement p=0.28, and the tiny gain is all
in H1 (Jul–mid-Aug), collapsing to ~0 in H2. Kill shot: tower diff is **0.875-correlated
with form_diff**; 69% of the composite's variance is explained by elo+form, and adding
the features makes form_diff's coefficient collapse and flip sign (multicollinearity).
Objectives are downstream of being the stronger team — already absorbed.

### 2. Early-game gold@15 — SHIP (first-blood — DROP)
`eg_gold`: trailing-20 mean `golddiffat15` per team (shrunk to 0, pseudo-count 8, same
convention as `form_rate`), blue-minus-red. **Δll −0.0063 calibrated / −0.0057 raw**
(raw is free of the isotonic-on-holdout caveat), acc +0.86pp, AUC +0.010, paired
t=−3.48, 56% of games improved — broad-based, not outlier-driven. Coef +0.289,
p=1.3e-46 controlling for Elo. Persistent team attribute (split-half r=0.531).
Confounds ruled out: side bias (+286 gold@15 for blue) tested via side-demeaned
variant — lift unchanged; corr with elo_diff 0.615 but fully incremental in the joint
model; no leakage. Mechanism: continuous gold margins are a finer-grained
trailing-strength read than binary W/L — `eg_gold` **fully subsumes `form_diff`**
(its coef collapses 0.16→0.009, p 1e-10→0.74). First-blood rate adds nothing
(Δll −0.0009; p=0.97 joint; persistence r=0.168 — coin-flip noise).

**Ship formula:** `eg_gold_diff = eg_gold(blue) − eg_gold(red)` where
`eg_gold(team,t) = Σ golddiffat15 / (n+8)` over the team's last 20 games with data
strictly before t (skip the 284 team-games with null golddiffat15). Add to
NODRAFT_FEATURES; consider dropping `form_diff` (dead weight at p=0.74 once eg_gold
is in — harmless to keep, leaner to drop).

### 3. Patch-adaptation speed — DROP
Early-patch (≤14d) vs mid-patch winrate differential. Primary construction p=0.31,
zero lift. Structural flaw: patches live ~14 days, so the "mid" window is near-empty
(holdout median 7 games/team/year) and the differential collapses to team strength
(corr 0.34 with elo_diff). The trait doesn't persist (split-half r=−0.116). The
secondary construction's nominal p=0.010 has the wrong sign — a mean-reversion artifact.

### 4. Meta-comfort alignment — DROP
Comfort-weighted patch champion strength, reusing the exact `comfort`/`champ_power`
definitions. Primary construction is **0.997-correlated with form_diff** — structural,
not empirical (meta ≈ form rescaled); its nodraft p=0.0012 is a multicollinearity
artifact, collapsing to p=0.86 on full features. The orthogonal top-8 variant is empty
(p=0.08, no lift anywhere).

### 5. Flex picks — DROP
Empirical flex definition (champion played in ≥2 positions; champion_tags.csv has no
role tags). In-sample p=0.0002 with **zero** out-of-sample lift (paired delta 0.6σ in
the wrong direction). Persistence check explains it: team flex-rate year-over-year
r=+0.01 — the rate follows the drifting global flex meta, not a stable team trait.
The mechanism's carrier doesn't exist.

### 6. Rest/fatigue — DROP
Structural null: fit-median rest is 0.05 days because each Bo5 game is a separate
gameid — consecutive series games are hours apart, so a rest *diff* can never capture
intra-series fatigue (identical for both teams within a series). Binned relationship
flat; calibrated ll worse by 0.0001. The nominal rest_diff p=0.020 is a leverage
artifact of the sparse nonzero tail.

### 7. Knockout experience — DROP (instructive)
Nominally passed the gate (−0.0026 ll, p<0.001) but the **signature test reversed it**:
in actual playoff games (n=830) it *hurts* (ll +0.0010, AUC −0.006); all the gain is
in regular-season games. ko×playoffs interaction p=0.32, negative sign. It's a
veteran-roster proxy (r=0.448 with elo_diff), not clutch experience. Shipping would
degrade exactly the playoff-heavy games the app targets. Dropped on mechanism, not metrics.

### 7b. Comeback factor — DROP
Behind-at-15 win% (shrunk). p=6.8e-14 in-sample, holdout ll *worse* (+0.0003, SE
0.0006) — textbook n=26k overfit. Trait barely persists (r=0.18) and correlates
0.67–0.71 with elo/form: strength re-measured, already priced.

## Bottom line

**Ship `eg_gold` only** — trailing-20 gold@15 differential as a nodraft feature
(Δll −0.0063/−0.0057, ~3× the 0.002 bar on the calibration-free metric). Everything
else died on confounds, non-persistence, or reversed mechanism signatures. With
eg_gold in, `form_diff` is redundant (p=0.74) — drop it from NODRAFT_FEATURES when
integrating. No production changes made; integration awaits Thomas's review.
