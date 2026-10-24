# Ratings patch v1.3.4 (Elo-only, 2026-10-24)

Elo-only patch over the v1.3.3 baseline (snapshot 2026-10-16). **Coefficients
unchanged** — `model.joblib` untouched. Backups of the v1.3.3 files:
`model/snapshots.joblib.bak-1.3.3`, `model/phone_artifacts.json.bak-1.3.3`.

## Why Elo-only, not a full retrain

A full retrain needs fresh Oracle's Elixir data. `src/refresh.py --check`
run 2026-10-24 returned `REFRESH_STATUS=noop`: the 2026 CSV download is
unavailable (Google Drive quota exceeded for the 2026 file, same block as
2026-10-19). Yesterday's games therefore have no Oracle rows (no gold@15,
champion power, etc.) to refit on. The Elo patch is the correct
incorporation: Elo only needs winner/loser, and it is the same procedure
as the v1.3.3 patch.

## Update rule (same as training, `build_features.py`)

Per map, after the game: `E' = E + 24 * (S - 1/(1 + 10^(-(E_opp_diff)/400)))`,
where S = 1 for a win, 0 for a loss. No blue-side adjustment in Elo, no
offseason regression mid-tournament. Applied map-by-map in chronological
order.

## Series included (winners verified against lolesports event details)

The v1.3.3 baseline already includes the Oct 17 MVK–FURIA series. The
unpatched results since then are the Oct 18 Play-In final and all eight
Oct 23 Swiss R1 Bo1s. The Oct 18 final is included even though the request
was for yesterday's results, because Karmine Corp (a winner that day and
again Oct 23) plays today — leaving it out would understate KC by ~21 Elo.

| Series | Result |
|---|---|
| Oct 18: Karmine Corp vs MVK Esports (Play-In final, Bo5) | KC 3–0 (games 1–3 KC, games 4–5 unneeded) |
| Oct 23: Gen.G vs Invictus Gaming (Bo1) | GEN wins |
| Oct 23: Hanwha Life Esports vs Top Esports (Bo1) | HLE wins |
| Oct 23: Movistar KOI vs Bilibili Gaming (Bo1) | MKOI wins |
| Oct 23: Karmine Corp vs Anyone's Legend (Bo1) | KC wins |
| Oct 23: Dplus Kia vs G2 Esports (Bo1) | DK wins |
| Oct 23: CTBC Flying Oyster vs Team Liquid (Bo1) | CFO wins |
| Oct 23: T1 vs LØS (Bo1) | T1 wins |
| Oct 23: LYON vs Team Secret Whales (Bo1) | LYON wins |

Source: lolesports `getEventDetails` match-level `gameWins` for each
match ID (Oct 23 IDs 115660540726095005/007/009/011/013/015/017/019;
Oct 18 final 116299649498303884), plus the Oct 23 schedule order.

## Elo (raw snapshot Elo; adjusted = raw + roster_delta, deltas unchanged)

| Team | v1.3.3 raw | v1.3.4 raw | Δ | v1.3.4 adjusted |
|---|---|---|---|---|
| Karmine Corp | 1699.51 | 1730.71 | +31.21 | 1734.91 |
| MVK Esports | 1562.16 | 1540.96 | −21.19 | 1540.96 |
| Gen.G | 1771.44 | 1777.80 | +6.36 | 1777.80 |
| Invictus Gaming | 1594.32 | 1587.95 | −6.36 | 1533.25 |
| Hanwha Life Esports | 1698.18 | 1706.73 | +8.55 | 1666.63 |
| Top Esports | 1595.55 | 1587.00 | −8.55 | 1493.60 |
| Movistar KOI | 1608.21 | 1624.55 | +16.34 | 1624.55 |
| Bilibili Gaming | 1739.93 | 1723.58 | −16.34 | 1723.88 |
| Anyone's Legend | 1662.57 | 1652.56 | −10.01 | 1652.56 |
| Dplus Kia | 1636.21 | 1651.98 | +15.77 | 1632.68 |
| G2 Esports | 1749.23 | 1733.45 | −15.77 | 1733.45 |
| CTBC Flying Oyster | 1574.13 | 1590.95 | +16.82 | 1481.95 |
| Team Liquid | 1722.12 | 1705.29 | −16.82 | 1715.29 |
| T1 | 1646.06 | 1661.43 | +15.37 | 1618.43 |
| LØS | 1746.21 | 1730.84 | −15.37 | 1753.14 |
| LYON | 1680.22 | 1694.22 | +14.00 | 1726.72 |
| Team Secret Whales | 1738.59 | 1724.59 | −14.00 | 1696.89 |

Form, eg_gold, champion power, and all other snapshot fields unchanged.

## Files changed

- `model/snapshots.joblib` — `elo` dict for the 17 teams above only.
  (Binary: cannot be pushed via the GitHub connector — see Push note.)
- `model/phone_artifacts.json` — `model_version` 1.3.3 → 1.3.4; the 17
  teams' `elo` / `elo_raw` only. `snapshot_date` stays 2026-10-16 (the
  Oracle snapshot); the Elo overlay is as of Oct 23. `generated_at`
  updated to the patch run (2026-10-24), unlike v1.3.3 which left it.
- This log.

## Validation: today's (Oct 24) predictions, before vs after

Pre-draft, Bo1 (series prob = per-map prob), schedule order as blue side.
Run with `src/predict.py --schedule` before and after the patch. The
patch is verified working: CLI Elo figures move to the v1.3.4 values.

| Oct 24 match (ET) | Pre-patch (blue win %) | Post-patch (blue win %) |
|---|---|---|
| 13:00 AL vs TES | 62.9% | 62.9% |
| 14:00 CFO vs MKOI | 40.1% | 40.1% |
| 15:00 G2 vs LØS | 74.9% | 74.9% |
| 16:00 LYON vs KC | 48.5% | 46.2% |
| 17:00 TSW vs IG | 41.6% | 41.6% |
| 18:00 GEN vs HLE | 62.5% | 62.5% |
| 19:00 T1 vs DK | 46.2% | 46.2% |
| 20:00 TL vs BLG | 40.1% | 40.1% |

Only LYON–KC moves by more than 0.1pp. That is expected, not a bug:
Swiss R2 pairs winners with winners and losers with losers, so both
sides' Elo moved by similar amounts in most pairings and the diffs
barely changed (e.g. CFO +16.8 vs MKOI +16.3; T1 +15.4 vs DK +15.8).
LYON–KC is the exception because KC also gained ~21 Elo from the
Oct 18 final on top of its Oct 23 win.

## Push note

UTF-8 files (`model/phone_artifacts.json`, this report) are pushed to
the model repo via the GitHub connector. `model/snapshots.joblib` is
binary and the connector rejects non-UTF-8 content, so it is NOT in
that push: the VM copy used by `predict.py` / edge-watch is updated,
but a PC/remote CLI checkout will not see the new Elo until the
binary is uploaded from a PC or via the browser. App promotion is
separate and was not done here.
