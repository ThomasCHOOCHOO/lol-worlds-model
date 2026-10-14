# Fearless draft: champion pool depth backtest

**Date:** 2026-10-13. **Decision: DROP — do not ship.** The pre-registered fearless hypothesis failed its gate. An incidental general (non-fearless) pool-depth signal was found but is heterogeneous and post-hoc; reported as a lead, not shipped.

## Hypothesis (from Thomas)

In fearless draft, a player's effective champion pool shrinks as a series goes on: by game 5, up to 4 comfort picks are burned (can't re-pick) and up to 5 more can be targeted by bans — so a player needs ~10 comfort champions, not 5. Teams with narrow pools (especially star-reliant ones, whose best player gets banned out) should underperform their map-strength-implied series expectation in fearless series.

## Method

**Series construction.** 29,793 Oracle's Elixir games (2024-01 → 2026-10-08) grouped into 15,179 series by (league, day, team pair) with contiguous game numbers; 8,687 multi-game series.

**Fearless flag (empirical, validated).** A series is flagged fearless iff no team repeats any champion pick across its games. League-season rates confirm the known adoption history: 2024 leagues 0.00–0.25 (coincidental no-repeat series; LPL 2024 at 0.25 suggests partial/experimental use), 2025+ leagues ≈ 1.00 (LPL 2025: 264/264, LCK 2025: 209/209, LEC/LCP/LCS 2025: 1.00). Clean quasi-experiment: **fearless** = 2025+ league-seasons with rate ≥ 0.95 (n = 5,809 series); **control** = 2024 league-seasons with rate ≤ 0.50 (n = 1,815).

**Pool metrics** (walk-forward, strictly pre-series; trailing-365d per-player champion counts; pool = champs with ≥3 games; game-1 lineup of 5 per side):
- `depth_mean`: mean pool size across starters
- `depth_min`: minimum (weakest link — the targeted player)
- `depth_ba7`: mean(pool − 7) — ban-adjusted per Thomas's math (4 burned + 3 targeted bans)
- `depth_ge10`: fraction of starters with pool ≥ 10 (Thomas's game-5 threshold)
- `concentration`: max pool / sum of pools (star-reliance)

All tested as blue-minus-red differentials (team1 = game-1 blue side).

**Series baseline.** Pre-series per-map p from the production **nodraft** model (elo_diff, form_diff, blue_edge) on game-1's walk-forward features + isotonic calibration, converted to series probability via binomial (independent maps; best-of from playoffs flag: 5 if playoffs/match went ≥4 games else 3). Logistic regression: `series_outcome ~ logit(p_binom) + depth_diff`. Train = fearless 2025 (n = 2,813), test = fearless 2026 (n = 2,996).

## Results

| Variant | coef (train) | p-value | test log-loss Δ vs baseline |
|---|---|---|---|
| depth_mean | +0.151 | <0.0001 | **+0.0268** |
| depth_min | +0.087 | <0.0001 | +0.0102 |
| depth_ba7 | +0.151 | <0.0001 | **+0.0268** |
| depth_ge10 | +0.567 | 0.0001 | +0.0070 |
| concentration | −1.077 | <0.0001 | +0.0034 |

(ll Δ is mean log-likelihood improvement on the 2026 holdout; positive = better. Baseline test ll −0.5921.)

Looks like a massive effect — until the mechanism checks:

| Check | Result | Verdict |
|---|---|---|
| **Placebo: control-2024 (non-fearless)** | d_ba7 coef **+0.204, p < 0.0001** | **FAILS** — effect exists without fearless, and is larger |
| **Bo5 interaction** (effect should grow with series length under fearless) | +0.0115, p = 0.76 | **FAILS** — Bo3 (+0.183) and Bo5 (+0.169) identical |
| Reverse split: fit on control-2024 → test fearless-2025 | coef +0.20, test ll Δ +0.0114 | transfers across eras — not fearless-specific |
| Volume control (trailing games-played differential) | depth survives (+0.152) | not a games-played artifact |
| Full-model baseline (all 6 features incl. d_comfort) | depth survives (+0.155, Δ +0.0275) | not captured by existing features |
| **LCK only (n = 396)** | coef −0.009, p = 0.88 | **null in the best-measured league** |
| **LCP only (n = 207)** | coef −0.140, p = 0.02 | reversed |
| Majors pooled (LCK/LPL/LEC/LCS/LTA) | +0.078, p = 0.002 | weak |
| International (WLDs/MSI/EWC, n = 231, post-hoc) | +0.244, p = 0.0003 | interesting but small-n and post-hoc |

## Interpretation

The fearless-draft mechanism is **rejected**: the effect is not specific to fearless series (placebo fails) and does not scale with series length (no Bo5 interaction).

What remains is an incidental, **non-fearless** pool-depth signal: teams with broader champion pools overperform their Elo-implied series expectation. The most coherent account is **Elo-freshness proxy, not draft mechanism**: pool breadth (trailing-365d, player-level, current lineup) is a fresher measure of current team strength than Elo, so it corrects Elo where Elo is worst — minor leagues and cross-region events — and adds nothing where Elo is well-calibrated (LCK: exactly zero). The negative concentration coefficient (star-reliant teams underperform) is consistent with Thomas's ban-vulnerability intuition, but inherits the same heterogeneity caveats.

## Decision: DROP, do not ship

1. The pre-registered fearless hypothesis failed its gate (placebo + mechanism tests).
2. The incidental depth signal is real-but-heterogeneous: null in LCK, reversed in LCP, strongest where Elo is noisiest. Shipping it as a universal adjustment would be irresponsible — especially since the target use case (Worlds: LCK/LPL/LEC teams) is where the evidence is weakest.
3. It is post-hoc (not what was gated) and would need a dedicated pre-registered validation on cross-region events before any production use.

**No model or app changes made.** If Thomas wants to pursue it, the honest follow-up is a pre-registered experiment on international events only (WLDs/MSI/EWC), testing whether depth corrects cross-region Elo miscalibration — not a fearless adjustment.

## Reproducibility

Analysis scripts were exploratory (`/tmp/fearless1.py` series construction, `/tmp/fearless3b.py` dataset, `/tmp/fearless6.py` + `/tmp/fearless7.py` regressions). Key inputs: `data/raw/202{4,5,6}.csv`, `data/processed/features.csv`, `model/phone_artifacts.json` (nodraft + full models). Series table cached at `/tmp/series.pkl`, dataset at `/tmp/fearless_data2.pkl` (ephemeral).

---

# Follow-up: weak-team stratification (2026-10-13, pre-registered)

**Date:** 2026-10-13 (evening). **Decision: DROP permanently.** Thomas's refined hypothesis: pool depth matters for weaker teams (elite teams all have deep pools → no variance to detect). Pre-registered gates, single metric (`depth_ba7` differential), weak = both teams' walk-forward Elo below the 33rd percentile of the Elo distribution at series game-1s (X = 1490.0).

## Sample

Rebuilt series from `features.csv` (15,158 series with raw-data champion/player match; fearless/control league-seasons re-validated: 9,107 fearless, 5,064 control). Weak gate: 2,166 series; weak ∩ (fearless ∪ control) = **n = 1,984** (fearless 1,188, control 796). Baseline: production nodraft model on game-1 features → calibrated per-map p → binomial series p. Binomial GLM: `t1win ~ logit(p_binom) + ...`.

## Pre-registered tests

| Test | Result | Verdict |
|---|---|---|
| **T1.** Weak series: `depth:fearless` interaction > 0 (one-sided, p<0.05) | coef +0.037, p = 0.26 | **FAIL** — depth effect not larger in fearless series |
| **T2.** Weak fearless series: `depth:Bo5` interaction > 0 (one-sided, p<0.05) | coef +0.110, p = 0.12 | **FAIL** — no significant series-length scaling |
| Generic (pooled weak): `depth_diff` | coef +0.176, p = 3.6e-14; ll Δ +0.0154 | significant, homogeneous across fearless/control (T1 interaction null) |

The fearless-specific hypothesis is dead under Thomas's own refinement: even among weak teams, the depth effect shows no fearless signature.

## Why the generic variant is not shipped

The pooled generic signal is significant, so per the pre-registered rule the question was homogeneity in the dimension that matters — **league tier**:

| Subset (weak teams) | n | depth coef | p |
|---|---|---|---|
| Major leagues (LCK/LPL/LEC/LCS/LTA) | 390 | −0.003 | 0.955 |
| Minor leagues | 1,594 | +0.216 | 1.5e-16 |
| Era 2024 (control) | 820 | +0.160 | 0.003 |
| Era 2025+ (fearless) | 1,164 | +0.182 | 2.3e-12 |

The entire signal comes from minor leagues; among major-league teams — including weak ones — it is exactly zero. Era-homogeneous, league-heterogeneous: this is the **Elo-freshness proxy**, not a draft or adaptation mechanism. Pool breadth is a fresher read on current strength than Elo where Elo is stale (minor leagues), and adds nothing where Elo is well-measured (majors).

Worlds implication: every Worlds team is major-league or the top of its minor region, and the applicable evidence (major-league weak teams, n=390) says the effect is zero there — including for Play-In mismatches like KC/C9/MVK/CBLOL#2. There is no gating rule under which this adjustment earns its keep at Worlds.

## Decision: DROP, do not ship — permanently

1. T1 and T2 both failed: no fearless mechanism, even among weak teams.
2. The generic depth signal is real but is an Elo-freshness proxy confined to minor leagues; it is null exactly where Thomas bets (major-league teams, p=0.955).
3. No further follow-ups recommended on this line — the mechanism has now failed its signature tests twice (pooled round 1, weak-stratified round 2).

**No model or app changes made.** Repro: `/tmp/fearless_weak.py`, dataset `/tmp/fearless_weak_data.pkl` (ephemeral); inputs `data/raw/202{4,5,6}.csv`, `data/processed/features.csv`, `model/phone_artifacts.json`.
