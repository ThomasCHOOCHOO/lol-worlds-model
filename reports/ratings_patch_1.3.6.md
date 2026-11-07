# Ratings patch v1.3.6 — Worlds 2026, Oct 25 – Nov 6

Elo-only patch (K=24, per map, chronological, no blue-side adjustment, no offseason
regression), same precedent as v1.3.3 / v1.3.4 / v1.3.5. Coefficients, gold@15,
champion power, form and `model.joblib` are unchanged.

## Why this is a patch, not a full retrain

`src/refresh.py --check` still returns `REFRESH_STATUS=noop` (checked 2026-11-07):
the Oracle's Elixir 2026 CSV download remains blocked by Google Drive quota exceeded.
A true full retrain (refitting LR coefficients, gold@15, champion power on new Oracle
rows) is therefore still impossible, and no quota workaround was used. The only
legitimate incorporation of Worlds results is the Elo-only patch below.

## Data used

- Source: Riot lolesports API — `getSchedule` + `getEventDetails` (series winners from
  match-level `gameWins`), fetched 2026-11-07 into `/tmp/worlds2026_results.json`.
- Coverage: all 43 Worlds 2026 series played Oct 15 – Nov 6 (Play-In 6 series / 22 maps,
  Swiss 29 series / 57 maps, Quarterfinals 4 series / 14 maps; 93 maps total).
- This patch covers the 22 series / 50 completed maps played Oct 25 – Nov 6 that were
  unpatched since v1.3.5 (which covered through Oct 24). Correction to the earlier
  working estimate of "21 series / 49 maps": the actual fetched data contains 22 series
  / 50 maps; the missed one was the Oct 25 Bo1 TLAW–BLG.
- Per-game winners: for sweeps and Bo1s the series winner won every played game.
  For the 7 non-sweep multi-game series, per-game order was verified against web
  sources (ESTNN tracker, Strafe, GosuGamers, BoostPirate):
  - Oct 25 GEN–LYON 2–1: LYON G1, GEN G2, GEN G3
  - Oct 28 LYON–IG 1–2: IG G1, LYON G2, IG G3
  - Oct 29 BLG–TES 1–2: BLG G1, TES G2, TES G3
  - Oct 30 LOS–TL 2–1: LOS G1, TL G2, LOS G3
  - Oct 31 LYON–LOS 2–1: LOS G1, LYON G2, LYON G3
  - Nov 3 IG–TES 3–1: IG G1, TES G2, IG G3, IG G4
  - Nov 6 T1–KC 3–1: T1 G1, T1 G2, KC G3, T1 G4
- `getEventDetails` games carry side assignment but NOT per-game winners; the
  livestats window endpoint returns only the last 10 frames (zeroed post-game).
  Leaguepedia Cargo API was unreachable from the VM (RemoteDisconnected, 2026-11-07),
  so the web checks above are the per-game-order source for those 7 series.

## Walk-forward validation (out-of-sample, 50 maps, Oct 25 – Nov 6)

Each game's model probability was computed with the Elo state *before* that game
(and each series' series-probability before the series), so these are genuine
out-of-sample numbers for the v1.3.5 model on new Worlds games:

| | n | Accuracy | Log-loss | AUC |
|---|---|---|---|---|
| Per-map, sequential Elo updates | 50 | 0.700 | 0.6121 | 0.7220 |
| Per-map, v1.3.5 Elo frozen | 50 | 0.680 | 0.6296 | 0.7123 |
| Per-series | 22 | 0.727 | 0.5659 | — |
| Constant 0.5 reference | — | — | 0.6931 | — |

The sequential Elo updates improved walk-forward log-loss by 0.0175 vs frozen Elo
(0.6121 vs 0.6296) and accuracy by 2.0pp on these 50 maps. n=50, so treat the
improvement as directionally positive, not precise.

**On the 0.002 log-loss shipping gate:** the gate applies to coefficient refits
evaluated on the 3,368-game holdout. An Elo-only patch does not change coefficients
and cannot move holdout log-loss (holdout metrics in `model/metrics.json` are
unchanged: full 0.6511 acc / 0.6235 log-loss / 0.6969 AUC), so the gate cannot
pass/fail in the usual sense. The walk-forward numbers above are the honest
substitute until the Oracle refresh unblocks a real retrain.

## Elo changes (raw, v1.3.5 → v1.3.6)

| Team | v1.3.5 | v1.3.6 | Δ |
|---|---|---|---|
| Invictus Gaming | 1604.44 | 1679.61 | +75.16 |
| Karmine Corp | 1717.46 | 1664.20 | −53.26 |
| Gen.G | 1787.38 | 1835.47 | +48.09 |
| Bilibili Gaming | 1734.95 | 1780.00 | +45.05 |
| LYON | 1707.48 | 1663.65 | −43.83 |
| G2 Esports | 1745.36 | 1702.38 | −42.98 |
| Movistar KOI | 1635.40 | 1668.77 | +33.38 |
| CTBC Flying Oyster | 1580.11 | 1555.84 | −24.27 |
| Team Secret Whales | 1708.10 | 1685.65 | −22.45 |
| Top Esports | 1601.23 | 1622.94 | +21.70 |
| Anyone's Legend | 1638.32 | 1618.12 | −20.20 |
| LØS | 1718.93 | 1705.20 | −13.73 |
| T1 | 1649.10 | 1660.96 | +11.86 |
| Hanwha Life Esports | 1697.15 | 1686.63 | −10.52 |
| Dplus Kia | 1664.30 | 1658.30 | −6.01 |
| Team Liquid | 1693.92 | 1695.94 | +2.02 |

All other teams unchanged. `model/phone_artifacts.json`: `elo_raw` set to the new raw
values, `elo` = `elo_raw` + existing `roster_delta` (deltas unchanged).

## Effect on the upcoming semifinals

Calibrated pre-draft output is stepwise (isotonic), so the Elo moves did not cross a
bin boundary for either semifinal; both predictions are unchanged from v1.3.5
(verified with `predict.py` on the written files):

- GEN vs IG, Bo5, Nov 7: GEN 89.5% series (74.9% per map)
- T1 vs BLG, Bo5, Nov 8: T1 31.9% series (40.1% per map)

## Files changed

- `model/snapshots.joblib` — Elo dict updated (binary; **not pushed via connector**,
  needs PC / browser upload for remote CLI checkouts; VM and edge watch already see it)
- `model/phone_artifacts.json` — version 1.3.6, `generated_at` 2026-11-07T06:44:47Z,
  `snapshot_date` unchanged 2026-10-16
- `reports/ratings_patch_1.3.6.md` — this report
- Backups: `model/snapshots.joblib.bak-1.3.5`, `model/phone_artifacts.json.bak-1.3.5`

## Still blocked

- Full retrain: Oracle's Elixir 2026 CSV download blocked by Google Drive quota
  (`REFRESH_STATUS=noop`). No new Oracle rows, so no coefficient / gold@15 /
  champion-power refit, no 0.002-gate evaluation on the holdout.
- `snapshots.joblib` binary not in the remote repo (v1.3.4, v1.3.5 and now v1.3.6
  binaries are VM-local only).
- App promotion: NOT done — needs Thomas's explicit approval; the app still serves
  v1.4.0 artifacts until he approves.
