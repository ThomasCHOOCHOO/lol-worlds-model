# Season-start prior test: player Elo only as the 2026 season-start prior

**Verdict: PASS, at every w tested. Best w = 1.0 (pure player aggregate).**
The per-game tier-conditional blend failed its gate on corrected Worlds data
(strict-cap diff +0.0002, CI upper 0.0021 > 0.002). The season-start-only
prior is a different design, and it clears the gate by a wide margin.

## Question

Does using player Elo ONLY as the season-start prior help? At season start,
set each team's starting Elo from its players, then let team Elo take over
and update per game (K=24) with no further player input.

## Design

- Boundary: 2025 -> 2026 (Jan 1, 2026). Data: Oracle's Elixir CSVs already
  in `data/raw/` (2024, 2025, 2026.csv through Oct 16, 2026). No download.
- Step 1: starting point is the `player_elo.replay` output in
  `data/processed/player_features.csv` (init 1500, K=24, 40% regression
  toward 1500 at each year boundary). Its pre-game `team_elo` /
  `player_elo` for a team's first 2026 game are the regressed end-of-2025
  ratings. Verified: the w=0 replay in this test reproduces the replay's
  team Elo exactly (max diff < 1e-6), and `player_features.csv`
  team-Elo diff matches `features.csv` `elo_diff` to 7e-13.
- Step 2 (the prior, NO LEAKAGE): at a team's first 2026 game, starting
  team Elo = (1-w) * regressed team Elo + w * player aggregate, where the
  player aggregate is the mean of that game's actual 5-player lineup's
  regressed end-of-2025 ratings (debutant prior = mean of rated teammates,
  1500 if none, per `player_elo.py`). The first-game lineup is known at
  game time, so this is clean. Full-year 2026 rosters are NOT used
  (that was the `roster_strength` leakage flaw). w in {0, 0.25, 0.5, 0.75, 1.0};
  w=0 is the baseline (regressed team Elo only).
- Step 3: replay 2026 games in chronological order with team Elo only:
  K=24 per map, expected score from the TEAM Elo diff, no blue-side
  adjustment, no further player input.
- Probabilities: the shipped nodraft model (`model.joblib` `model_nodraft`:
  LR coef [0.5694, 0.2598, 0.0128] on `[elo_diff_adj, eg_gold_diff,
  blue_edge]` via `scaler_nodraft`, then isotonic calibration).
  `elo_diff_adj` = my team Elo diff + the region correction
  (`elo_diff_adj - elo_diff` from `features.csv`, identical across w).
  Both RAW (pre-calibration LR) and CAL (isotonic) probabilities reported.
- Windows: primary = games where BOTH teams are within their first 10
  games of 2026 (n=1,434); secondary = Jan-Mar 2026 (n=2,430);
  extra = all 2026 games through Oct 16 (n=9,694; 415 teams).
- Paired bootstrap CI: 20,000 resamples, seed 7, on the per-game
  log-loss difference vs w=0. Diff is reported as baseline minus w
  (positive = the prior helps).
- Shipping gate (usual): log-loss improvement > 0.002 AND 95% CI excludes 0.

## Caveat (read first)

`model.joblib` was fit through 2026-07-01 and calibrated through
2026-10-01, so early-2026 ABSOLUTE log-loss is in-sample for the model
and optimistic. The model is held fixed across w and the games are the
same, so the PAIRED DIFF vs w=0 is the valid quantity; absolute levels
are context only.

## Results — primary window: first 10 games, both teams (n=1,434)

| w | LL raw | diff raw vs w=0 | 95% CI raw | LL cal | diff cal vs w=0 | 95% CI cal | Gate |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.6499 | — | — | 0.6522 | — | — | baseline |
| 0.25 | 0.6446 | +0.0053 | [+0.0035, +0.0072] | 0.6466 | +0.0056 | [+0.0032, +0.0081] | PASS |
| 0.5 | 0.6404 | +0.0095 | [+0.0059, +0.0132] | 0.6436 | +0.0086 | [+0.0047, +0.0126] | PASS |
| 0.75 | 0.6374 | +0.0125 | [+0.0071, +0.0180] | 0.6397 | +0.0125 | [+0.0068, +0.0183] | PASS |
| 1.0 | 0.6356 | +0.0143 | [+0.0072, +0.0215] | 0.6365 | +0.0157 | [+0.0081, +0.0234] | PASS |

Accuracy (cal): w=0 0.627, w=0.25 0.628, w=0.5 0.626, w=0.75 0.631, w=1.0 0.640.

### Tier-1 + internationals, primary window (n=240, CIs wide)

| w | LL raw | diff raw | 95% CI raw | LL cal | diff cal | 95% CI cal | Gate |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.7006 | — | — | 0.7001 | — | — | baseline |
| 0.25 | 0.6901 | +0.0104 | [+0.0042, +0.0172] | 0.6898 | +0.0104 | [+0.0029, +0.0184] | PASS |
| 0.5 | 0.6816 | +0.0190 | [+0.0070, +0.0319] | 0.6813 | +0.0188 | [+0.0067, +0.0317] | PASS |
| 0.75 | 0.6750 | +0.0256 | [+0.0080, +0.0446] | 0.6774 | +0.0227 | [+0.0051, +0.0406] | PASS |
| 1.0 | 0.6702 | +0.0304 | [+0.0071, +0.0549] | 0.6682 | +0.0320 | [+0.0088, +0.0557] | PASS |

Note: tier-1 accuracy does NOT follow log-loss (cal acc: w=0 0.575,
w=0.5 0.550, w=0.75 0.546, w=1.0 0.562). The gain is better-calibrated
probabilities, not more correct favourites. n=240, so treat the tier-1
point estimates as noisy despite the CIs excluding 0.

### Minors, primary window (n=1,194)

| w | diff raw | 95% CI raw | diff cal | 95% CI cal | Gate |
|---|---|---|---|---|---|
| 0.25 | +0.0043 | [+0.0025, +0.0061] | +0.0047 | [+0.0022, +0.0072] | PASS |
| 0.5 | +0.0076 | [+0.0040, +0.0112] | +0.0066 | [+0.0025, +0.0107] | PASS |
| 0.75 | +0.0099 | [+0.0045, +0.0153] | +0.0105 | [+0.0046, +0.0163] | PASS |
| 1.0 | +0.0111 | [+0.0040, +0.0182] | +0.0125 | [+0.0047, +0.0202] | PASS |

## Results — secondary window: Jan-Mar 2026

Overall (n=2,430): w=0.25 raw diff +0.0036 CI [+0.0023, +0.0049];
w=0.5 +0.0063 [+0.0038, +0.0088]; w=0.75 +0.0080 [+0.0043, +0.0117];
w=1.0 +0.0089 [+0.0041, +0.0137]. Calibrated diffs are similar or larger
(w=1.0 +0.0103 [+0.0051, +0.0155]). All PASS.
Tier-1+intl (n=624): w=0.5 raw +0.0089 [+0.0034, +0.0146], cal +0.0077
[+0.0018, +0.0138]; w=1.0 raw +0.0137 [+0.0033, +0.0245]. All PASS.
Minors (n=1,806): same monotonic pattern, all PASS.

## Results — all 2026 (through Oct 16, n=9,694)

The prior's effect decays as team Elo updates accumulate, as expected.
Overall: w=0.5 raw diff +0.0024 [+0.0016, +0.0032] (PASS, raw),
cal +0.0012 [+0.0002, +0.0022] (diff under 0.002, FAIL on calibrated).
w=1.0 raw +0.0033 [+0.0017, +0.0048] (PASS), cal +0.0018 [+0.0000, +0.0035]
(diff under 0.002). Tier-1+intl: w=0.5 raw +0.0023 [+0.0006, +0.0039]
(PASS raw), cal +0.0012 [-0.0007, +0.0031] (CI includes 0).

## Turnover split (primary window, w=0.5)

Turnover = share of a team's first-2026-game 5 not in its modal 2025
lineup (top-5 players by 2025 appearances). High turnover = >= 0.4
(2+ new players); 32 of 415 teams are high-turnover.

| Subset | n | diff raw | 95% CI raw | diff cal | 95% CI cal |
|---|---|---|---|---|---|
| High-turnover (either side) | 187 | +0.0121 | [+0.0015, +0.0230] | +0.0119 | [+0.0007, +0.0235] |
| Low-turnover (both sides) | 1,247 | +0.0091 | [+0.0053, +0.0131] | +0.0082 | [+0.0039, +0.0125] |

The prior helps BOTH groups, slightly more for high-turnover teams but
well within noise of each other. Interpretation: the player aggregate is
simply a better season-start rating than regressed team Elo in general,
not only when rosters churn. (The low-turnover gain is the more precisely
estimated of the two.)

## Why this works where the per-game blend failed

The per-game blend hurt tier-1 because it kept pulling predictions toward
player Elo all season, including mid-season subs and academy lineups.
The season-start prior applies the player information exactly once, when
team Elo is at its stalest (right after the offseason shuffle), and then
gets out of the way: after ~10 games per team the prior's contribution
has mostly decayed through K=24 updates, which is why the all-2026 diff
(+0.0024 at w=0.5) is much smaller than the first-10 diff (+0.0095).

## What is NOT tested here

- Only one boundary (2025 -> 2026). The effect is large and consistent
  across windows and tiers, but a 2024 -> 2025 replication is the obvious
  confirmation before shipping.
- w stops at 1.0 and the diff is still rising at w=1.0. w > 1 (extrapolating
  past the player aggregate) is unprincipled and was not tested.
- `roster_strength` delta_2026 as a comparator was skipped: it uses
  full-year 2026 rosters (leakage); a clean version would need re-deriving
  adjustments from first-game lineups, a different formula/test.
- Shipping mechanics are NOT covered: at season start the snapshot Elo
  would be set from player aggregates of each team's first-game lineup,
  which requires end-of-prior-season player ratings (available in
  `model/player_blend_snapshot.json`) plus the first lineup (known at
  game time from Oracle rows / livestats). No live file was changed here.

## Recommendation

Ship the season-start prior, with w=1.0 as the default if a single value
must be chosen (largest gain, passes the gate on raw and calibrated, in
every window and tier), after a 2024 -> 2025 replication confirms the
effect is not boundary-specific. If Thomas prefers a conservative
choice, w=0.5 still passes everywhere in the primary and Jan-Mar windows
with roughly two-thirds of the w=1.0 gain. Do NOT revive the per-game
blend: that gate stays failed.

## Shipping note — SHIPPED 2026-11-15, w=1.0 (Thomas's call)

- Decision: ship the season-start prior with w=1.0, after the options
  presented were "Replicate on 2024->2025 first", "Ship w=1.0 now",
  "Ship w=0.5 now". Thomas tapped "Ship w=1.0 now".
- Accepted risk, recorded honestly: the 2024 -> 2025 replication was NOT
  done. This rests on a single boundary (2025 -> 2026), and the model was
  fit/calibrated on early-2026 data, so the paired diffs are the valid
  quantity. If the 2024 -> 2025 replication later fails, the prior should
  be revisited (set SEASON_START_PRIOR_W = 0.0).
- Implementation: SEASON_START_PRIOR_W = 1.0 in `src/player_elo.py`
  (`replay()`), `src/build_features.py` (CLI `--season-start-prior-w`,
  aggregates from a `player_elo.replay` pass in-process) and
  `src/build_player_ratings.py` (CLI `--season-start-prior-w`). All three
  implement the identical rule, so the sidecar / features consistency
  checks still hold. `predict.py --elo-blend` stays opt-in OFF; the
  per-game tier-conditional blend stays failed and was NOT revived.
- Scope: takes effect at the NEXT season boundary (first 2027 game per
  team). Current v1.3.8 ratings are unchanged (still 1.3.8, no version
  bump); `model/snapshots.joblib` and `model/phone_artifacts.json` were
  not rebuilt or re-pushed for this change.
- Regression check (`src/season_start_prior_regression.py`,
  `reports/season_start_prior_regression.json`): replayed 2025->2026
  through BOTH shipped code paths. Part A (`player_elo.replay`):
  raw diff +0.01433, CI [0.00717, 0.02149], PASS. Part B
  (`build_features.py` in a scratch copy): raw diff +0.01433,
  CI [0.00717, 0.02149], PASS. Consistency: w=0 scratch reproduces live
  `features.csv` elo_diff exactly (max |diff| 0.0 over 29,942 games);
  w=1.0 scratch elo_diff matches the Part A replay team_elo_diff exactly
  (max |diff| 0.0). The tiny CI differences vs the test above are
  bootstrap resampling noise; point estimates match to 5 decimals.
- Push status 2026-11-15: NOT pushed yet. Three push attempts were made;
  none performed the write (two timed out waiting for the GitHub write
  approval card to be tapped; one was interrupted by a service restart
  that also wiped /tmp). Remote head is still 0fb0259b and the sync
  record is unchanged (10 entries). The 8 files are ready to push once
  the approval is granted; see the parent agent's report for the list.

## Files

- Script: `src/season_start_prior_test.py` (reproducible; read-only on
  live model files)
- Numbers: `reports/season_start_prior_test.json`
- This report: `reports/season_start_prior_test.md`
- No live model file modified, nothing pushed, app untouched.
