# Live in-game win probability — research + prototype (2026-10-13)

Thomas wants win probability that updates mid-game from game state (gold,
dragons, turrets, kills). This report covers: training-data inventory, live
data source evaluation (verified), prototype model + metrics, and the app
integration spec. The app was NOT touched.

## 1. Training-data inventory (Oracle's Elixir, data/raw/*.csv)

Player-level rows; team-aggregatable per-minute snapshots exist at t = 10/15/20/25:
- `goldat{t}`, `xpat{t}`, `csat{t}` (+ `opp_*`, `golddiffat{t}` etc.)
- `killsat{t}`, `assistsat{t}`, `deathsat{t}` (+ opp variants)
- `gamelength` (seconds) — for filtering games that reached minute t

**Missing:** objectives at time t. `dragons`, `towers`, `barons`, `heralds`,
`inhibitors` are end-of-game totals only — no at-t snapshots. So dragon/tower
coefficients **cannot** be trained from this data. (Gold diff already prices
most objective value: towers/dragons convert to gold. Live dragons/towers can
still be *displayed*; a future refinement is to collect live frames during
Worlds 2026 and train objective coefficients from that.)

Coverage: 29,793/29,793 games have at-t data (2024-01 → 2026-10-08).

## 2. Live data source — VERIFIED: unofficial LoL Esports API

This is the same API lolesports.com uses. Free, no signup, public key.

- Base: `https://esports-api.lolesports.com/persisted/gw/`
  (mirror: `https://prod-relapi.ewp.gg/persisted/gw/`)
- Auth: header `x-api-key: 0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z`
- Required query param: `hl=en-US`
- Docs: https://vickz84259.github.io/lolesports-api-docs

Verified working 2026-10-13 (read-only requests):
- `getLeagues` → Worlds league id `98767975604431411` ✓
- `getSchedule?leagueId=...` → 80 Worlds events ✓
- `getLive` → valid response (0 events at 1am, correct) ✓
- `getEventDetails?id=<matchId>` → game ids + blue/red sides ✓
- `getWindow`: `https://feed.lolesports.com/livestats/v1/window/{gameId}[?startingTime=RFC3339]`
  → frames[]; each frame has `gameState`, per-team `totalGold`, `totalKills`,
  `towers`, `dragons[]`, `barons`, `inhibitors`, and per-participant
  `level/kills/deaths/assists/creepScore/totalGold`. No key needed on the
  feed host. Note: the window only retains recent frames (~10 without
  `startingTime`); historical replay for old games is NOT available —
  **poll live games as they happen**. Frames update roughly every ~10s
  during live games.

Rejected alternatives:
- **PandaScore**: live in-game stats are paid (€1000+/mo per title); free tier
  covers fixtures/schedules/pre-match only — no live game state.
- **GRID / Bayes Esports**: commercial, no free tier.
- **Leaguepedia Cargo API**: volunteer-entered, delayed, no per-minute
  in-game snapshots — unsuitable for live win prob.
- **Riot spectator API**: technically possible but fragile (needs per-game
  encryption keys); the lolesports feed is simpler and purpose-built.

Caveat: unofficial API — Riot can change/rate-limit it. Poll politely
(10–15s), and the app should degrade to pre-game prob if the feed fails.

## 3. Prototype model

**Method.** Per time bucket t ∈ {10, 15, 20, 25} minutes: logistic regression
on `[pre_logit, golddiff_k, killdiff]` where
- `pre_logit` = logit of the v1.2.0 full model's calibrated pre-game prob
  (the prior — computed from model/model.joblib on features.csv),
- `golddiff_k` = (blue team gold − red team gold)/1000 at minute t,
- `killdiff` = blue kills − red kills at minute t.
Games with `gamelength <= t*60` excluded (already over). Walk-forward:
train date < 2026-07-01 (n=26,211), test 2026-07-01 → 2026-10-01 (n=3,368),
same split as v1.2.0. No leakage (all features known at minute t).

XP was tested and dropped: live frames expose `level` but not team total XP,
and dropping `xpdiff` costs ≤0.004 log-loss.

**Results (test set, per-t apples-to-apples vs pre-game-only baseline):**

| t (min) | n_test | in-game ll | pre-game ll | Δll | in-game acc |
|---|---|---|---|---|---|
| 10 | 3,368 | 0.5392 | 0.6310 | −0.092 | 72.0% |
| 15 | 3,368 | 0.4842 | 0.6310 | −0.147 | 76.4% |
| 20 | 3,353 | 0.4240 | 0.6324 | −0.208 | 79.1% |
| 25 | 2,941 | 0.3701 | 0.6521 | −0.282 | 82.6% |

(Live-field variant: pre_logit + gold diff + kill diff. Coefficients, t=15:
[0.785, 0.252, 0.014], intercept −0.03. Gold dominates; kills add ~nothing
beyond gold early. Note the pre-game baseline itself degrades on long games
(0.6521 at t=25): games that go long were coin-flips pre-game — exactly where
live state helps most.)

**Calibration:** prototype evaluated on log-loss only; ship with an isotonic
re-fit on the 2026 tail before app integration (same convention as v1.2.0).

## 4. App integration spec (for ~/workspace/lol-predictor-app/)

New "Live" tab (all client-side, no backend):

1. **Match discovery** (on tab open + manual refresh):
   `GET {base}/getLive?hl=en-US` → events with `state == "inProgress"` →
   for the chosen match, `GET {base}/getEventDetails?hl=en-US&id={matchId}` →
   games list → pick the game with `state == "inProgress"`.
   Map team codes to model team names via existing `team_aliases`.
2. **Poll** `GET https://feed.lolesports.com/livestats/v1/window/{gameId}`
   every 10–15s. Take the last frame with `gameState == "in_game"`.
3. **Features from frame**: `golddiff_k = (blueTeam.totalGold − redTeam.totalGold)/1000`,
   `killdiff = blueTeam.totalKills − redTeam.totalKills`. Game minute =
   from frame timestamps (first in_game frame ≈ minute 0) — use the bucket
   t = largest of {10,15,20,25} ≤ elapsed minutes; below 10 min show the
   pre-game prob.
4. **Prior**: run the existing v1.2.0 pre-game prediction for the matchup
   (draft mode if the user entered picks), take logit.
5. **Inference**: per-bucket LR coefficients + intercept (to be exported into
   phone_artifacts.json as `live_models: {10: {coef, intercept}, ...}` after
   the isotonic calibration re-fit), sigmoid → calibrated live win %.
6. **Display**: live win % big, plus the state that drives it (gold diff,
   kills, dragons[], towers per side from the frame). Stop polling when
   `gameState` leaves `in_game`.
7. **Fallbacks**: feed error/empty → show pre-game prob with "live feed
   unavailable"; no live matches → empty state.

**Open item for the model side:** export `live_models` (4 buckets ×
{coef[3], intercept, isotonic xs/ys}) into phone_artifacts.json (next minor
bump). No retraining of v1.2.0 needed.

Prototype script: /tmp/live_prototype.py (method documented above).
