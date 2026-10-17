# Region-strength analysis: quantifying weak-region Elo inflation

**Date:** 2026-10-17 · **Model:** v1.3.2 pipeline (Elo init 1500, K=24, 40% yearly regression)
**Data:** Oracle's Elixir 2024–2026, `features.csv` (29,942 games) · **Status:** analysis only, no model files changed

## TL;DR

Elo systematically **overrates LATAM, Vietnam, and Taiwan** and **underrates LCK** in cross-region
games. A hierarchical fit (`team_skill = Elo + region offset`) finds large, stable, statistically
significant offsets: **LCK +178, LPL +108, LATAM −157, Vietnam −134 Elo points** (EMEA +24, NA +14,
Taiwan −34, all non-significant). Offsets generalize across tournaments (held-out log-loss improves
by ~0.05) and **do not decay** with cross-region experience — any decay rule hurts. Vietnam is
systematically overrated, **not** high-variance: its "wild card" reputation is the bias, not extra
variance. **Recommendation: SHIP** region offsets on `elo_diff`, subject to the refit-backtest gate
in §7.

## 1. Method

**Cross-region games (primary set).** MSI + Worlds (WLDs) + First Stand (FST), 2024–2026:
564 games across 8 events (MSI 2024: 78, WLDs 2024: 132, FST 2025: 35, MSI 2025: 80,
WLDs 2025: 96, FST 2026: 45, MSI 2026: 71, WLDs 2026: 27 — play-in only, event in progress).

**Elo-implied probability.** Taken directly from `features.csv`'s walk-forward pre-game `elo_diff`:
`p_elo = 1 / (1 + 10^(−elo_diff/400))`. The Elo in `features.csv` is updated from *all* games
(domestic + international) in chronological order, so `p_elo` is leakage-free and is exactly the
rating the production model sees.

**Region mapping.** A team's region = its modal top-flight domestic league that year
(LCK, LPL, LEC→EMEA, LCS/LTA N→NA, CBLOL/LLA/LTA S→LATAM, VCS→Vietnam, PCS→Taiwan,
LJL→Japan, LCO→Oceania, TCL→Turkiye). LCP (the merged APAC league, 2025+) has no single region,
so LCP teams resolve via their most recent prior-year domestic league
(e.g. GAM/MGN Vikings→VCS→Vietnam, CFO/DCG/PSG Talon→PCS→Taiwan, DFM/SHG→LJL→Japan,
Ground Zero→LCO→Oceania). Two manual overrides, flagged as assumptions: **MVK Esports→Vietnam**
(MGN Vikings' 2026 slot/rebrand; no pre-2026 record) and **Team Secret Whales→Vietnam**
(ex-Team Whales, VCS 2024). All 564 primary games resolve; zero unknown regions.

**Hierarchical fit.** Logistic regression on cross-region games only:

```
logit(p) = a + b · logit(p_elo) + off[region_blue] − off[region_red]
```

Ridge penalty λ=1 on the offsets (λ chosen by leave-one-event-out CV). Offsets are reported in
Elo points via the standard conversion ×400/ln(10) ≈ ×173.7. Bootstrap (500 resamples) gives CIs.
Japan (4 games) is excluded from fitting — separation, not estimable (see §2) — and pooled
Minor regions (Japan/Oceania/Turkiye) get offset 0.

Two parameters of the fit deserve a plain reading before the results:
- `a` (intercept, fitted +0.254) is a **blue-edge stand-in**: at even Elo it implies blue wins
  ~56%. Production already handles side advantage with the `blue_edge` feature, so the intercept
  is **not** shipped — only the offsets are.
- `b` (slope, fitted 0.65) says Elo-implied probabilities are **overconfident** on cross-region
  games: a 100-point Elo gap behaves like ~65 points of true skill gap. §4 shows slope shrinkage
  adds nothing once offsets are in, so it is **not** shipped either.

## 2. Cross-region residuals: which regions does Elo misrate?

Per-region, from that region's perspective (actual wins vs Elo-expected wins):

| Region | n | Actual | Expected | Resid/game | z |
|---|---:|---:|---:|---:|---:|
| LCK | 301 | 191 | 158.7 | **+0.107** | **+3.83** |
| LPL | 309 | 164 | 157.7 | +0.021 | +0.73 |
| EMEA | 172 | 76 | 77.8 | −0.011 | −0.29 |
| NA | 134 | 58 | 56.8 | +0.009 | +0.22 |
| Taiwan | 77 | 33 | 44.3 | **−0.147** | **−2.69** |
| Vietnam | 64 | 22 | 33.2 | **−0.176** | **−2.87** |
| LATAM | 67 | 20 | 33.0 | **−0.194** | **−3.24** |
| Japan | 4 | 0 | 2.5 | −0.616 | −2.61 (n=4, not estimable) |

Elo overrates LATAM, Vietnam, and Taiwan by 15–19 points of win probability per game, and
underrates LCK by 11 points. EMEA, NA, and LPL look fair in aggregate — but the aggregate hides
opponent-mix effects, which the joint fit in §3 disentangles.

Notable head-to-head pairs (perspective: first-listed region):

| Pair | n | Actual | Expected | Resid/game | z |
|---|---:|---:|---:|---:|---:|
| LCK vs LPL | 103 | 61 | 51.5 | +0.092 | +1.91 |
| LCK vs NA | 36 | 29 | 23.3 | +0.157 | +2.00 |
| LPL vs NA | 25 | 20 | 13.9 | +0.243 | +2.47 |
| LCK vs Taiwan | 16 | 13 | 7.2 | +0.362 | +3.00 |
| LCK vs Vietnam | 7 | 7 | 3.5 | +0.500 | +2.67 |
| NA vs Vietnam | 8 | 7 | 3.7 | +0.415 | +2.40 |
| EMEA vs Vietnam | 24 | 19 | 11.0 | +0.335 | +3.39 |
| EMEA vs NA | 24 | 7 | 12.2 | −0.216 | −2.15 |
| LATAM vs LCK | 6 | 0 | 2.6 | −0.431 | −2.24 |
| LATAM vs LPL | 7 | 0 | 3.0 | −0.425 | −2.28 |
| LATAM vs Vietnam | 16 | 5 | 8.2 | −0.198 | −1.61 |

Three things stand out. First, **LCK beats LPL head-to-head** (61–42 over 103 games) — the
LCK>LPL gap is real, not just both-beating-minors. Second, **LATAM is winless against the top
two** (0–6 vs LCK, 0–7 vs LPL) and even loses to Vietnam head-to-head. Third, **EMEA underperforms
specifically against NA** (7–17) despite a fair overall record — a matchup-specific wrinkle worth
watching, though n=24 is small.

## 3. Region offsets (hierarchical fit)

| Region | Offset (Elo pts) | 95% bootstrap CI | Significant? |
|---|---:|---:|---|
| LCK | **+178** | [+133, +233] | Yes |
| LPL | **+108** | [+63, +163] | Yes |
| EMEA | +24 | [−34, +75] | No |
| NA | +14 | [−45, +71] | No |
| Taiwan | −34 | [−109, +36] | No |
| LATAM | **−157** | [−245, −71] | Yes |
| Vietnam | **−134** | [−220, −56] | Yes |
| Minor (Japan/Oceania/Turkiye) | 0 (no evidence) | — | n/a |

The joint fit matters because it **disentangles "played tough opponents" from "region is
misrated"**. Taiwan's raw deficit (−0.147/game, z=−2.69) mostly vanishes in the joint model
(offset −34, CI includes 0): Taiwan plays LCK/LPL often, and once their opponents' positive
offsets are accounted for, Taiwan itself is roughly fairly rated. Conversely EMEA's raw
residual is ~0 but its joint offset is +24: EMEA faces a brutal schedule, so after opponent
adjustment it is slightly *better* than Elo implies. The same logic lifts NA (+14). Only four
offsets are significant, but the full fitted vector is what was validated (§5), so ship it as-is.

**Robustness.** Offsets are nearly identical under spec changes: no-intercept variant
(LCK +179, LPL +115, LATAM −167, Vietnam −137, EMEA +28, NA +16, Taiwan −35) and
slope-fixed-at-1 variant (LCK +174, LPL +105, LATAM −157, Vietnam −136, EMEA +34,
NA +29, Taiwan −50). The intercept and slope do not drive the offsets.

**Stability over time.** Offsets fit on 2024 events vs 2025–2026 events:

| Region | 2024 (n=206) | 2025–2026 (n=354) |
|---|---:|---:|
| LCK | +179 | +164 |
| LPL | +139 | +83 |
| EMEA | +35 | +10 |
| NA | +2 | +11 |
| Taiwan | −34 | −35 |
| LATAM | −174 | −129 |
| Vietnam | −146 | −105 |

The pattern is stable. LPL's edge shrank ~40% and EMEA's faded — offsets drift, so they must be
**refit on every refresh**, not frozen.

## 4. Slope shrinkage: tested, not needed

The fitted slope (b=0.65) is real in-sample, but on held-out data the offset model with a free
slope and the offset model with slope fixed at 1 perform identically (LOTOCV log-loss 0.6288 vs
0.6290). The offsets already absorb the miscalibration that the slope was capturing. **Do not
ship slope shrinkage** — it adds complexity for zero held-out gain.

## 5. Honest validation: do offsets generalize?

Three protocols. All fit offsets on train events only and score held-out events against the
Elo-only baseline (no offsets, no intercept).

**Protocol A (literal).** Train on 2024 events + MSI 2026 → test Worlds 2025: Δlog-loss **−0.051**,
ΔBrier −0.022. Train on 2024 + Worlds 2025 → test MSI 2026: Δlog-loss **−0.050**, ΔBrier −0.021.
Generalizes in both directions.

**Protocol B (leave-one-event-out, pooled, n=560).** Δlog-loss **−0.049** (0.6288 vs 0.6773),
ΔBrier −0.022. Six of eight events improve. The two failures are small and explicable:
FST 2025 (+0.047, n=35 — blue won only 45.7% there, so the fit's blue-edge intercept hurt, and
the train set was only two events) and WLDs 2026 (+0.021, n=27 partial play-in, MVK
overperformed). The no-intercept variant gives pooled Δlog-loss −0.045 — slightly worse on
average, but the intercept is fragile across events (blue win rate by event ranges 0.457–0.667),
which is why production should rely on the `blue_edge` feature instead.

**Protocol C (forward-chaining: train only on earlier events, pooled n=482).** Δlog-loss **−0.041**.
Same two failures (FST 2025, WLDs 2026); every other event improves by 0.034–0.072.

**Caveat.** These protocols validate the offset model as a *standalone* predictor (with its own
intercept/slope). The production configuration — offsets applied to `elo_diff`, then the full
model refit — needs its own backtest before shipping (see §7, ship gate 1). One validation
artifact to not misread: in Protocol A, the offset model's aggregate residual on Worlds 2025
was −8.2 (it expected 8.2 more blue wins than occurred). That is the blue-edge intercept
overpredicting at an event where blue won only 51.0% (vs 57.1% across all international games),
not a region-offset failure — per-region, the offset model was *closer* than Elo-only for
EMEA, LATAM, LCK, Taiwan, and Vietnam, and farther only for LPL (30.0→32.3 vs actual 30) and
NA (6.2→6.9 vs actual 4). Note LCK still overperformed even vs its offset at that event
(42 actual vs 36.7 expected).

## 6. Vietnam: systematically overrated, not a wild card

- **Bias:** 22 wins vs 33.2 expected over 64 games (−0.176/game, z=−2.87). Negative residual in
  6 of 7 events; the only positive event is WLDs 2026 play-in (+1.0, MVK, n=6).
- **By team:** Team Secret Whales −6.2 (n=22), GAM −3.8 (n=32), MGN Vikings −2.2 (n=4),
  MVK +1.0 (n=6).
- **Era split:** 2024 VCS era −0.188/game (n=23); 2025–2026 LCP era −0.169/game (n=41).
  The bias **persists after the LCP merger**. LCP cross-subregion play did not fix it — the
  likely mechanism is that Vietnamese teams still farm Elo against weaker LCP sub-regions
  (Japan/Oceania) and each other, while their true international level is lower; beating weak
  sub-regions inflates Elo without calibrating against LCK/LPL.
- **Variance (the "wild card" test):** game-level dispersion ratio Var(residual)/mean-binomial-var
  = **0.97**, 95% CI [0.76, 1.12] — no extra variance. Series-level dispersion (25 series,
  bias-corrected) = 1.42, 95% CI [0.74, 2.11] — includes 1.0, not significant.
- **Conclusion:** Vietnam is **not** high-variance. The wild-card reputation is explained by the
  systematic bias: because Elo overrates them by ~134 points, their losses look like upsets and
  their results look surprising. No variance adjustment is needed; the mean offset (−134) is the
  fix. Do not overweight MVK's 6-game play-in overperformance.

## 7. Interaction with Elo: no decay (tested and rejected)

The prior was that offsets should shrink as a team accumulates cross-region games, since its Elo
absorbs that evidence. The data says the opposite. Leave-one-event-out log-loss for
`offset × 1/(1 + n_cross/κ)`:

| κ (decay scale) | LOTOCV log-loss |
|---|---:|
| 2 | 0.6668 |
| 5 | 0.6571 |
| 10 | 0.6493 |
| 20 | 0.6417 |
| 40 | 0.6361 |
| 100 | 0.6316 |
| **None (no decay)** | **0.6288** |

Any decay hurts, monotonically. **Recommendation: constant offsets, no decay.**
Mechanism: a team's Elo is dominated by domestic games (dozens per split vs a handful of
international games), so cross-region evidence accumulates too slowly in the rating to wash out
the region prior. The offset also captures persistent structural factors (scrim quality, coaching
depth, stage experience) that don't decay with a few international games. Revisit only if the
pipeline ever weights international games more heavily in Elo.

## 8. Does the full model already absorb this? No.

Diagnostic: the production nodraft model (elo_diff, eg_gold_diff, blue_edge), scored on the same
560 games — in-sample for pre-Jul-2026 games, so this is a *floor* on the bias:

| Region | Resid/game under full model |
|---|---:|
| LCK | **+0.110** |
| LPL | +0.022 |
| EMEA | −0.017 |
| NA | −0.018 |
| Taiwan | −0.153 |
| LATAM | −0.173 |
| Vietnam | −0.182 |

Region bias **survives the full model** — the features cannot express region. LCK is still
underrated by 11 points/game even after `eg_gold_diff` etc. One second-order lead: LCK's
controlled style produces smaller gold differentials, so `eg_gold_diff` may systematically
underrate LCK — a region×gold-diff interaction is a follow-up experiment, out of scope for this
ship.

## 9. Sensitivity: extended cross-region set

Adding EWC, ASI (2025 Asia Invitational), and LTA cross-conference games (n=904 fit games):

| Region | Resid/game (extended) | Offset (extended) | Offset (primary) |
|---|---:|---:|---:|
| LCK | +0.099 | +184 | +178 |
| LPL | +0.010 | +105 | +108 |
| EMEA | −0.010 | +29 | +24 |
| NA | +0.004 | −1 | +14 |
| Taiwan | −0.140 | −64 | −34 |
| LATAM | −0.118 | −148 | −157 |
| Vietnam | −0.164 | −104 | −134 |

Consistent with the primary set. (Taiwan is more negative in the extended set — EWC/ASI mix;
small difference, not a concern. One team, 'Team Secret', did not resolve to a region and was
pooled into Minor — negligible.)

## 10. Recommendation: SHIP (with gates)

**Ship** region offsets on `elo_diff`, as follows:

1. **Offsets** (Elo points, ridge λ=1): LCK +178, LPL +108, EMEA +24, NA +14, Taiwan −34,
   LATAM −157, Vietnam −134; Minor/Japan/Oceania/Turkiye = 0 (no evidence; revisit at n≥15).
2. **No decay** (κ = None). **No intercept, no slope shrinkage** — production `blue_edge` +
   model refit cover those.
3. **Ship gates:**
   - (a) **Refit backtest.** Rebuild features with adjusted `elo_diff`, retrain, and confirm the
     Worlds 2025 and MSI 2026 backtest log-losses improve (expect ~−0.04 to −0.05, e.g. Worlds
     2025 from ~0.68 toward ~0.63). Judge by the international backtests, not global holdout:
     internationals are 1.9% of games and domestic games are unaffected (same region both sides
     → offsets cancel), so global holdout will barely move.
   - (b) **Refit offsets on every refresh** from cross-region games to date (they drift — LPL's
     edge already shrank ~40%).
   - (c) **Monitor** per-event region residuals each tournament; if an offset's sign flips for
     two consecutive events, investigate before trusting it.

## 11. Implementation sketch

- **New `src/region_offsets.py`.** Builds the team→region map from raw CSVs (modal domestic
  league per team-year; LCP via prior-year league; explicit manual-override dict for MVK/TSW
  etc.); fits the hierarchical logistic model on cross-region games (MSI/WLDs/FST) with ridge
  λ=1 (re-tunable); writes `model/region_offsets.json`:
  `{region: offset_elo_pts, fit_date, n_games, method, region_map}` (map stored for audit).
- **`src/build_features.py`.** Load offsets; per game compute
  `region_adj = off[reg_blue] − off[reg_red]` and `elo_diff_adj = elo_diff + region_adj`.
  Keep raw `elo_diff` too (diagnostics). Region at game time = the team's region that year.
- **`src/train.py`.** Train on `elo_diff_adj` instead of `elo_diff` (both `full` and `nodraft`).
  Backtests must fit offsets on **pre-cutoff** games only — pass a `cutoff` into
  `region_offsets.py` (no future leakage).
- **`src/predict.py` / snapshots.** `snapshots.joblib` carries per-team region + the offset
  table; prediction uses adjusted Elo = Elo + off[region]. The app's `phone_artifacts.json`
  needs the (small) offset table too.
- **`src/refresh.py`.** Run `region_offsets.py` before `build_features.py` on each refresh.
- **Watch out.** (1) Region map must be deterministic and versioned — it lives in the offsets
  JSON. (2) Teams with no domestic history default to offset 0. (3) LCP sub-region resolution
  is the fragile part — the MVK mapping is an assumption; review the manual-override dict each
  split. (4) Do **not** apply offsets to live-model features directly: the live model takes its
  pre-game logit from the adjusted model, so the correction flows through automatically.
- **Alternative considered and rejected.** Region dummies as model features instead of an Elo
  adjustment: first-order equivalent, but the model would have to learn the offsets from ~2% of
  games (drowned out by domestic games). The two-stage fit is more statistically efficient, and
  the Elo-point adjustment flows through the existing pipeline (live model, series conversion,
  calibration) with no new feature plumbing.

## Appendix: reproduction

All numbers from `features.csv` + raw CSVs, read-only. Analysis scripts were run from `/tmp`
(ephemeral, not preserved); the procedure is fully specified in §1–§2 and re-runnable:
region map (§1) → `p_elo` from `elo_diff` → residuals (§2) → IRLS logistic fit with ridge λ=1
on cross-region games (§1) → bootstrap CIs → validation protocols (§5) → decay grid (§7).
Key parameters: λ=1 (LOTOCV-chosen), bootstrap n=500, series grouping heuristic for §6
(same pair + same event + gap < 3 days).
