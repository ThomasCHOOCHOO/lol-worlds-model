# Ratings patch v1.3.7 — Worlds 2026, clean replay Oct 15 – Nov 6 (correction of v1.3.6)

Elo-only patch (K=24, per map, chronological, no blue-side adjustment, no offseason
regression), same precedent as v1.3.3 / v1.3.4 / v1.3.5 / v1.3.6. Coefficients, gold@15,
champion power, form and `model.joblib` are unchanged.

## Why v1.3.7 exists: v1.3.6 Elo was wrong

v1.3.6 (commit 0464d4fd) was pushed 2026-11-07, but a clean walk-forward replay
(`reports/worlds2026_walkforward_predictions.md`, 2026-11-07) showed its Elo does
not reproduce under the stated K=24 per-map rule. The main agent verified this
independently by replaying Oct 25–Nov 6 from `snapshots.joblib.bak-1.3.5`:

- Movistar KOI: v1.3.6 1668.77 vs 1578.78 replayed (+90 too high). MKOI lost all
  6 maps in that window, so it cannot have risen +33 from v1.3.5.
- T1: 1660.96 vs 1706.99 replayed (−46 too low)
- Karmine Corp: 1664.20 vs 1708.16 (−44)
- LØS: 1705.20 vs 1729.95 (−25)
- Gen.G: 1835.47 vs 1817.71 (+18), Invictus Gaming: 1679.61 vs 1661.75 (+18),
  LYON: 1663.65 vs 1677.05 (−13), Top Esports: 1622.94 vs 1620.41 (+3)

The eight diffs sum to ~0, i.e. transfers among those eight teams only. The
walk-forward author tested winner-wins-all-games, series-level K=24, frozen
series-start Elo, flipped expected score, roster-adjusted-Elo start, and winner
swaps in the candidate series; none reproduces v1.3.6. How v1.3.6 was produced
could not be determined from the files, so it is not patched over — it is
replaced by a clean replay from scratch.

Two further errors in the v1.3.6 chain, also fixed here:

1. **Missing Play-In series.** The patch chain never applied the Oct 15–16
   Play-In series: KC–C9 3-2, MVK–FUR 3-0, KC–MVK 3-0, FUR–C9 3-1. v1.3.3
   covered only Oct 17 MVK–FUR; v1.3.4 covered Oct 18 + Oct 23. Proof: Cloud9's
   Elo in v1.3.5 was 1568.19, exactly its pre-Worlds baseline, despite playing
   two series.
2. **Wrong totals in the v1.3.6 report.** Worlds through Nov 6 is 43 series /
   **87 maps**, not 93; Swiss is 33 series / 51 maps, not 29 / 57
   (Play-Ins 6 / 22, Swiss 33 / 51, Quarterfinals 4 / 14).

## Fix method

- Start from the **pre-Worlds Elo** in `model/snapshots.joblib.bak-1.3.2`
  (`as_of` 2026-10-16). Do NOT start from v1.3.5 or v1.3.6, both contaminated
  by the errors above.
- Apply **all** Worlds 2026 series Oct 15–Nov 6, 43 series / 87 maps, in
  chronological order, K=24 per map, no blue-side adjustment, no offseason
  regression.
- Per-game winners from `reports/worlds2026_walkforward_maps.csv` (verified
  against lolesports `getEventDetails` gameWins; the 10 non-sweep series'
  per-game orders are listed in `worlds2026_walkforward_predictions.md`,
  including the four missing Play-In series above).
- Backups of the v1.3.6 files: `model/snapshots.joblib.bak-1.3.6`,
  `model/phone_artifacts.json.bak-1.3.6`.
- `model/phone_artifacts.json`: `elo_raw` = new raw Elo rounded to 2dp,
  `elo` = `elo_raw` + existing `roster_delta` (deltas unchanged),
  `model_version` 1.3.7, `generated_at` updated, `snapshot_date` stays
  2026-10-16. All other snapshot fields unchanged (starting from the v1.3.6
  `snapshots.joblib`, Elo dict only replaced).

## Elo changes (raw, v1.3.6 → v1.3.7, and pre-Worlds → v1.3.7)

| Team | Pre-Worlds | v1.3.6 | v1.3.7 | Δ vs v1.3.6 |
|---|---|---|---|---|
| Gen.G | 1771.44 | 1835.47 | 1817.80 | −17.67 |
| Bilibili Gaming | 1739.93 | 1780.00 | 1772.47 | −7.53 |
| Karmine Corp | 1699.51 | 1664.20 | 1719.54 | +55.34 |
| T1 | 1646.06 | 1660.96 | 1708.59 | +47.63 |
| LØS | 1746.21 | 1705.20 | 1730.85 | +25.65 |
| Team Liquid | 1722.12 | 1695.94 | 1704.93 | +8.99 |
| G2 Esports | 1749.23 | 1702.38 | 1702.38 | 0.00 |
| Hanwha Life Esports | 1698.18 | 1686.63 | 1686.05 | −0.58 |
| Team Secret Whales | 1738.59 | 1685.65 | 1685.65 | 0.00 |
| LYON | 1680.22 | 1663.65 | 1677.51 | +13.86 |
| Invictus Gaming | 1594.32 | 1679.61 | 1661.69 | −17.92 |
| Dplus Kia | 1636.21 | 1658.30 | 1657.54 | −0.76 |
| Top Esports | 1595.55 | 1622.94 | 1619.71 | −3.23 |
| Anyone's Legend | 1662.57 | 1618.12 | 1619.30 | +1.18 |
| Movistar KOI | 1608.21 | 1668.77 | 1579.57 | −89.20 |
| FURIA | 1622.08 | 1589.79 | 1575.37 | −14.42 |
| CTBC Flying Oyster | 1574.13 | 1555.84 | 1556.66 | +0.82 |
| MVK Esports | 1529.87 | 1540.96 | 1556.20 | +15.24 |
| Cloud9 | 1568.19 | 1568.19 | 1550.82 | −17.37 |

All other teams unchanged from pre-Worlds (they did not play at Worlds).

## Validation

- **Replay match:** the v1.3.7 Elo above was computed independently from
  `snapshots.joblib.bak-1.3.2` using the walk-forward maps CSV, and matches the
  clean replay table in `worlds2026_walkforward_predictions.md` within 0.05 Elo
  for all 19 teams (max diff 0.05, rounding only).
- **Semifinals, verified with `src/predict.py` on the written files**
  (schedule order as blue, pre-draft, Bo5):
  - GEN vs IG, Nov 7: GEN 74.9% per map, **89.5% series — unchanged from v1.3.6**.
  - T1 vs BLG, Nov 8: T1 41.6% per map, **34.6% series — changed from v1.3.6**
    (was 40.1% per map, 31.9% series; T1 +2.7pp series, +1.5pp per map, because
    its Elo is corrected up +47.6 and BLG down −7.5). The walk-forward report's
    "semifinal probabilities unchanged" claim holds for GEN–IG only; it does
    not hold for T1–BLG.
  - Side note, existing model behaviour, not a v1.3.7 change: swapping sides
    changes the series probability (IG blue vs GEN gives GEN 75.5% series;
    BLG blue vs T1 gives T1 27.6% series), because per-map p depends on
    blue/red via eg_gold and blue edge. The edge-watch check uses schedule
    order as blue, so the figures above are the alert-time figures.
- **Walk-forward context (unchanged by this patch):** the clean replay's
  pre-match predictions for all 43 series were 29/43 = 67.4% acc,
  log-loss 0.6019; per map 59/87 = 67.8%, log-loss 0.6225.

## Files changed

- `model/snapshots.joblib` — Elo dict replaced with the clean replay (binary;
  **not pushed via connector**, needs PC / browser upload for remote CLI
  checkouts; VM and edge watch already see it)
- `model/phone_artifacts.json` — version 1.3.7, `generated_at`
  2026-11-07T16:31:24Z, `snapshot_date` unchanged 2026-10-16
- `reports/ratings_patch_1.3.7.md` — this report
- Backups: `model/snapshots.joblib.bak-1.3.6`,
  `model/phone_artifacts.json.bak-1.3.6`

## Still blocked

- Full retrain: Oracle's Elixir 2026 CSV download blocked by Google Drive quota
  (`REFRESH_STATUS=noop`). No new Oracle rows, so no coefficient / gold@15 /
  champion-power refit, no 0.002-gate evaluation on the holdout.
- `snapshots.joblib` binary not in the remote repo (v1.3.4, v1.3.5, v1.3.6 and
  now v1.3.7 binaries are VM-local only).
- App promotion: NOT done — needs Thomas's explicit approval; the app still
  serves v1.4.0 artifacts until he approves.
