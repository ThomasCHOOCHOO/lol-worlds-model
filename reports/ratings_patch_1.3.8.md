# Ratings patch v1.3.8 — Worlds 2026 closeout (semifinals + final, Nov 7–14)

Elo-only patch (K=24, per map, chronological, no blue-side adjustment, no
offseason regression), same precedent as v1.3.3–v1.3.7. Coefficients, gold@15,
champion power, form and `model.joblib` are unchanged.

## Results applied (verified vs lolesports getEventDetails gameWins)

- Semifinal Nov 7: Gen.G beat Invictus Gaming 3-0 (GEN, GEN, GEN).
- Semifinal Nov 8: Bilibili Gaming beat T1 3-1 (BLG, T1, BLG, BLG).
- Final Nov 14: Bilibili Gaming beat Gen.G 3-1 (GEN, BLG, BLG, BLG).
  Model favourite (GEN, 68.3–69.7% series) lost.

## Re-base (2026-11-14, Thomas approved): the Oct 15–16 double-count is FIXED

The first v1.3.8 build (local only, never pushed) started from
`snapshots.joblib.bak-1.3.2` (as_of 2026-10-16 23:43:16) as its
"pre-Worlds" baseline — but that snapshot already contains the Oct 15–16
Play-In maps, so it applied those 15 maps a second time for the four teams
that played them. The numbers below are the CORRECTED v1.3.8; the first
build's values are superseded (kept in `model/*.bak-1.3.8-uncorrected`).

True pre-Worlds Elo, verified by an independent full-history replay of
Oracle's Elixir rows with date < 2026-10-15 (`src/player_elo.replay`,
29,922 games before Oct 15; the replay also reproduces bak-1.3.2 exactly
for every team that did not play Oct 15–16):

| Team | bak-1.3.2 (post-Oct-16) | True pre-Worlds |
|---|---|---|
| Karmine Corp | 1699.51 | 1676.47 |
| Cloud9 | 1568.19 | 1591.91 |
| MVK Esports | 1529.87 | 1505.25 |
| FURIA | 1622.08 | 1646.01 |

This build replays ALL 98 Worlds maps (Oct 15–Nov 14) from that corrected
baseline, same K=24 per-map rule. The independent player-Elo replay
(`reports/player_ratings_worlds2026.md`) reproduces these team Elos within
0.51.

## Elo changes (corrected v1.3.8 vs the uncorrected first build)

| Team | Uncorrected v1.3.8 | Corrected v1.3.8 | Δ |
|---|---|---|---|
| Cloud9 | 1550.82 | 1568.18 | +17.37 |
| MVK Esports | 1556.20 | 1540.96 | -15.23 |
| FURIA | 1575.37 | 1589.79 | +14.43 |
| Karmine Corp | 1719.54 | 1707.87 | -11.66 |
| T1 | 1693.97 | 1692.48 | -1.49 |
| CTBC Flying Oyster | 1556.66 | 1555.82 | -0.84 |
| Movistar KOI | 1579.57 | 1578.76 | -0.80 |
| Bilibili Gaming | 1817.21 | 1816.71 | -0.50 |
| Anyone's Legend | 1619.30 | 1618.80 | -0.50 |
| LYON | 1677.51 | 1677.13 | -0.37 |
| Gen.G | 1807.38 | 1807.24 | -0.14 |
| Top Esports | 1619.71 | 1619.65 | -0.06 |
| Invictus Gaming | 1641.99 | 1641.93 | -0.05 |
| LØS | 1730.85 | 1730.80 | -0.04 |
| Dplus Kia | 1657.54 | 1657.49 | -0.04 |
| Hanwha Life Esports | 1686.05 | 1686.01 | -0.03 |
| Team Liquid | 1704.93 | 1704.90 | -0.03 |

BLG stays above GEN: BLG 1816.71 vs GEN 1807.24.
All other teams unchanged (verified by diffing `snapshots.joblib` against
its `.bak-1.3.8-uncorrected` — exactly the 17 teams above).

## Method / validation

- 98 maps replayed from the corrected baseline with K=24 per map,
  chronological. Expected values from the sensitivity analysis of the first
  build: KC 1707.87, C9 1568.18, MVK 1540.96, FURIA 1589.79, GEN 1807.25,
  BLG 1816.71, T1 1692.48, IG 1641.94. This replay matches within 0.05 on
  all eight teams.
- `model/phone_artifacts.json`: `elo_raw` = new raw Elo rounded to 2dp,
  `elo` = `elo_raw` + existing `roster_delta` (deltas unchanged),
  `model_version` 1.3.8, `generated_at` 2026-11-15T02:46:30Z, `snapshot_date` stays
  2026-10-16. Backups: `model/*.bak-1.3.8-uncorrected` (first build),
  `model/*.bak-1.3.7` (v1.3.7).
- predict.py validation on the corrected files (pre-draft, Bo5):
  - GEN blue vs BLG red: per-map 59.5%, series 67.4% (blue-all; realistic
    alternating-sides range is lower — sides alternate, never quote blue-all
    as the series price).
  - GEN blue vs IG red: per-map 74.9%, series 89.5%.
  - T1 blue vs BLG red: per-map 35.6%, series 24.5%; BLG blue vs T1 red:
    BLG per-map 70.8%, series 84.8% (unchanged from the uncorrected build).
  - NOTE: the T1-blue per-map number moved 40.1% → 35.6% (series 31.9% →
    24.5%) on a ~1.5 Elo change in T1. That is the stepwise isotonic
    calibration flipping a bin, not a real strength change — the raw
    probability moves only ~0.3pp for the same Elo change. Verified by
    running predict.py on both the corrected and the
    `.bak-1.3.8-uncorrected` files and restoring the corrected files
    (md5-verified).

## Files changed (local only until pushed)

- `model/snapshots.joblib` — Elo dict, 17 teams
- `model/phone_artifacts.json` — version 1.3.8 (corrected)
- `model/player_blend_snapshot.json`, `model/player_ratings.json` — re-based
- `reports/ratings_patch_1.3.8.md` — this report (re-written)
- `reports/player_ratings_worlds2026.md`, `reports/worlds2026_blend_validation.json`

## Still blocked

- Full retrain: Oracle's Elixir 2026 CSV download blocked by Google Drive
  quota (`REFRESH_STATUS=noop`). No coefficient / gold@15 / champion-power
  refit.
- `snapshots.joblib` binary not in the remote repo (v1.3.4–v1.3.8 binaries
  are VM-local only); needs PC / browser upload.
- App promotion: NOT done — needs Thomas's explicit approval.
