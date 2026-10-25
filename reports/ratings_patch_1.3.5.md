# Ratings patch v1.3.5 (Elo-only, 2026-10-25)

Elo-only patch over the v1.3.4 baseline (snapshot 2026-10-16). **Coefficients
unchanged** — `model.joblib` untouched. Backups of the v1.3.4 files:
`model/snapshots.joblib.bak-1.3.4`, `model/phone_artifacts.json.bak-1.3.4`.

## Why Elo-only, not a full retrain

A full retrain needs fresh Oracle's Elixir data. `src/refresh.py --check`
run 2026-10-25 returned `REFRESH_STATUS=noop`: the 2026 CSV download is
still unavailable (Google Drive quota exceeded for the 2026 file, same block
as 2026-10-19 and 2026-10-24). Oct 24 games therefore have no Oracle rows
(no gold@15, champion power, etc.) to refit on. The Elo patch is the correct
incorporation: Elo only needs winner/loser, and it is the same procedure
as the v1.3.3 and v1.3.4 patches.

## Update rule (same as training, `build_features.py`)

Per map, after the game: `E' = E + 24 * (S - 1/(1 + 10^(-(E_opp_diff)/400)))`,
where S = 1 for a win, 0 for a loss. No blue-side adjustment in Elo, no
offseason regression mid-tournament. Applied map-by-map in chronological
order. All Oct 24 series are Bo1s, so one update each.

## Series included (winners verified against lolesports event details)

| Series (Oct 24, ET order) | Result |
|---|---|
| Top Esports vs Anyone's Legend (Bo1) | TES wins |
| Movistar KOI vs CTBC Flying Oyster (Bo1) | MKOI wins |
| LOS vs G2 Esports (Bo1) | G2 wins |
| Karmine Corp vs LYON (Bo1) | LYON wins |
| Team Secret Whales vs Invictus Gaming (Bo1) | IG wins |
| Hanwha Life Esports vs Gen.G (Bo1) | GEN wins |
| Dplus KIA vs T1 (Bo1) | DK wins |
| Team Liquid vs Bilibili Gaming (Bo1) | BLG wins |

Source: lolesports `getEventDetails` match-level `gameWins` for each
match ID (Oct 24 IDs 115660540726095021/023/025/027/029/031/033/035),
plus the Oct 24 schedule order.

## Elo (raw snapshot Elo; adjusted = raw + roster_delta, deltas unchanged)

| Team | v1.3.4 raw | v1.3.5 raw | Δ | v1.3.5 adjusted |
|---|---|---|---|---|
| Top Esports | 1587.00 | 1601.23 | +14.24 | 1507.83 |
| Anyone's Legend | 1652.56 | 1638.32 | −14.24 | 1638.32 |
| Movistar KOI | 1624.55 | 1635.40 | +10.84 | 1635.40 |
| CTBC Flying Oyster | 1590.95 | 1580.11 | −10.84 | 1471.11 |
| G2 Esports | 1733.45 | 1745.36 | +11.91 | 1745.36 |
| LØS | 1730.84 | 1718.93 | −11.91 | 1741.23 |
| LYON | 1694.22 | 1707.48 | +13.26 | 1739.98 |
| Karmine Corp | 1730.71 | 1717.46 | −13.26 | 1721.66 |
| Invictus Gaming | 1587.95 | 1604.44 | +16.49 | 1549.74 |
| Team Secret Whales | 1724.59 | 1708.10 | −16.49 | 1680.40 |
| Gen.G | 1777.80 | 1787.38 | +9.58 | 1787.38 |
| Hanwha Life Esports | 1706.73 | 1697.15 | −9.58 | 1657.05 |
| Dplus Kia | 1651.98 | 1664.30 | +12.33 | 1645.00 |
| T1 | 1661.43 | 1649.10 | −12.33 | 1606.10 |
| Bilibili Gaming | 1723.58 | 1734.95 | +11.37 | 1735.25 |
| Team Liquid | 1705.29 | 1693.92 | −11.37 | 1703.92 |

Form, eg_gold, champion power, and all other snapshot fields unchanged.

## Validation: Oct 25 matchups, pre vs post patch (predict.py, schedule-order blue)

Calibrated pre-draft output is stepwise (isotonic), so several matchups do
not move despite Elo changes; only G2–TES crosses a calibration step.

| Matchup | Pre-patch | Post-patch |
|---|---|---|
| G2 vs TES (Bo1) | G2 66.7% | G2 64.3% |
| MKOI vs DK (Bo3 series) | MKOI 29.0% | MKOI 29.0% |
| GEN vs LYON (Bo3 series) | GEN 84.2% | GEN 84.2% |
| HLE vs IG (Bo1, already checked/alerted 12:20 ET) | HLE 70.8% | HLE 70.8% |

Note: the HLE–IG alert and approval from 12:20–12:23 ET used v1.3.4
probabilities; the v1.3.5 calibrated probability for that matchup is
unchanged (70.8%), so the alert stands.

## Files changed

- `model/snapshots.joblib` — `elo` dict for the 16 teams above only.
  (Binary: cannot be pushed via the GitHub connector — see Push note.)
- `model/phone_artifacts.json` — `model_version` 1.3.4 → 1.3.5; the 16
  teams' `elo` / `elo_raw` only. `snapshot_date` stays 2026-10-16 (the
  Oracle snapshot); the Elo overlay is as of Oct 24. `generated_at`
  updated to the patch run (2026-10-25).
- This log.

## Push note

`model/snapshots.joblib` is binary and cannot go via the connector — the
VM copy (used by edge-watch/predict.py here) IS updated, but a PC/remote
CLI checkout will not see the new Elo until the binary is uploaded from
a PC or via browser. `model/phone_artifacts.json` + this report are
pushed via the connector.
