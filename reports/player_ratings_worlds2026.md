# Player ratings update through Worlds 2026 (v1.3.8 closeout, RE-BASED)

Incremental player-Elo replay through all 98 Worlds 2026 maps (Oct 15–Nov 14),
starting from the TRUE pre-Worlds snapshot (full-history Oracle replay on
rows with date < 2026-10-15). Same rule as `src/player_elo.py`: K=24, every
player on a side updates by K*(team result − expected team score) where the
expected score uses the PLAYER-aggregate diff (mean of the 5 pre-game player
ratings); debutant prior = mean of rated teammates (1500 if none). No
offseason regression inside the window (no year boundary crossed).

## Re-base (2026-11-14, Thomas approved)

The first build of this update started from the Oct 16 snapshot, whose
team/player ratings already contained the Oct 15–16 Play-In maps — so those
15 maps were counted twice for Karmine Corp, Cloud9, MVK Esports and FURIA.
This version re-bases on the true pre-Worlds snapshot (see
`reports/ratings_patch_1.3.8.md` for the baseline table) and replays the
98 maps from there. The first build's player values are superseded (kept in
`model/*.bak-1.3.8-uncorrected`).

## Sources

- Per-game lineups: livestats window endpoint
  (`feed.lolesports.com/livestats/v1/window/<gameId>`, gameMetadata
  participantMetadata: summonerName, role). Game IDs from lolesports
  getEventDetails per match (x-api-key extracted programmatically from
  `reports/live_ingame.md`). Lineups saved to
  `data/processed/worlds2026_lineups.json` (98 games).
- Winners / blue-red: `reports/worlds2026_walkforward_maps.csv` (87 maps)
  plus the 3 knockout series (11 maps), verified against getEventDetails
  gameWins.
- Mapping: (team, role, bare player name) → Oracle player key in the
  snapshot. Ambiguous names resolved by the team's snapshot last lineup
  (MVK Kratos, G2 SkewMond); "C9 Loki " has a trailing space in the feed.
  **Unmapped players: none (0 of 980 participants).** No debutants: every
  Worlds player already had a snapshot rating.

## Team player Elo (player_elo / team_elo / gap), before → after Worlds

| Team | player_elo pre | player_elo post | team_elo pre | team_elo post | gap pre | gap post |
|---|---|---|---|---|---|---|
| Gen.G | 1857.30 | 1878.85 | 1771.44 | 1807.25 | +85.87 | +71.61 |
| Bilibili Gaming | 1779.85 | 1871.37 | 1739.93 | 1816.71 | +39.93 | +54.66 |
| T1 | 1743.72 | 1773.24 | 1646.06 | 1692.48 | +97.66 | +80.76 |
| Invictus Gaming | 1611.57 | 1671.47 | 1594.32 | 1641.94 | +17.26 | +29.54 |
| Karmine Corp | 1729.70 | 1761.32 | 1676.47 | 1707.87 | +53.23 | +53.45 |
| Cloud9 | 1642.54 | 1619.49 | 1591.91 | 1568.18 | +50.62 | +51.31 |
| MVK Esports | 1536.62 | 1581.47 | 1505.25 | 1540.96 | +31.37 | +40.51 |
| FURIA | 1701.56 | 1641.09 | 1646.01 | 1589.79 | +55.55 | +51.30 |

Team Elo here is the pure replay (true pre-Worlds baseline + 98 Worlds maps);
it reproduces the corrected v1.3.8 patch values within 0.51.

## Final-roster player ratings (post-Worlds)

### Bilibili Gaming (champions)

| Player (role) | Rating |
|---|---|
| Bin (top) | 1892.20 |
| Xun (jng) | 1814.94 |
| Knight (mid) | 1906.04 |
| Viper (bot) | 1882.38 |
| ON (sup) | 1861.25 |

### Gen.G

| Player (role) | Rating |
|---|---|
| Kiin (top) | 1902.05 |
| Canyon (jng) | 1912.43 |
| Chovy (mid) | 1924.49 |
| Ruler (bot) | 1843.95 |
| Duro (sup) | 1811.35 |

## Files changed (local only until pushed)

- `model/player_blend_snapshot.json` — as_of 2026-11-14, blend_params
  unchanged
- `model/player_ratings.json` — same schema, updated ratings
- `data/processed/player_features_worlds2026.csv`
- Backups: `model/*.bak-1.3.8-uncorrected` (first build),
  `model/*.bak-1.3.7` (v1.3.7)

## Blend re-validation on fresh Worlds data (98 maps, all tier-1+intl, corrected)

Strict-cap variant (a=-4, b=8), LR refit on the pre-2026-07-01 fit set,
nodraft features, RAW log-loss (see `reports/worlds2026_blend_validation.json`):

| Model | Log-loss | Acc | Diff vs M0 | 95% CI |
|---|---|---|---|---|
| M0 team Elo | 0.6241 | 0.653 | — | — |
| Strict-cap blend | 0.6243 | 0.653 | +0.0002 | [-0.0017, +0.0021] |
| Primary blend (a=0.5, b=2) | 0.6273 | 0.612 | +0.0032 | [-0.0144, +0.0216] |
| Pure player Elo | 0.6287 | 0.602 | +0.0045 | [-0.0175, +0.0273] |

Gate (strict-cap): non-inferiority (CI upper < +0.002):
FAIL; pooled gain (CI excludes 0):
FAIL. n=98, so CIs are wide; do NOT ship on
this alone — Thomas decides.

Raw vs isotonic (shipped `model.joblib`, nodraft, 98 Worlds maps):
RAW log-loss 0.6241
acc 0.653;
calibrated log-loss 0.6232
acc 0.633.
