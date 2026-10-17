# Ratings patch v1.3.1 (Elo-only, 2026-10-16)

Elo-only patch over the v1.3.0 baseline (snapshot 2026-10-08). **Coefficients
unchanged** — `model.joblib` untouched. Supersedes the v1.4.0 Elo patch
(Oct-15-only); this patch re-does the full pass from the v1.3.0 baseline so
Oct 15 is not double-counted. Backups of the v1.4.0 files:
`model/snapshots.joblib.bak-1.4.0`, `model/phone_artifacts.json.bak-1.4.0`.

## Update rule (same as training, `build_features.py`)

Per map, after the game: `E' = E + 24 * (S - 1/(1 + 10^(-(E_opp_diff)/400)))`,
where S = 1 for a win, 0 for a loss. No blue-side adjustment in Elo, no
offseason regression mid-tournament. Applied map-by-map in chronological
order; series order within a day is irrelevant (disjoint team pairs).

## Series included (per-map winners verified against lolesports + recaps)

| Series | Maps (winner order) |
|---|---|
| Oct 15: Karmine Corp 3-2 Cloud9 | C9, KC, C9, KC, KC |
| Oct 15: MVK Esports 3-0 FURIA | MVK, MVK, MVK |
| Oct 16: Karmine Corp 3-0 MVK Esports | KC, KC, KC |
| Oct 16: FURIA 3-1 Cloud9 | C9, FURIA, FURIA, FURIA |

Reproduction check: applying only the two Oct 15 series to the v1.3.0
baseline reproduces the v1.4.0 values exactly (all four teams to <0.01),
confirming the baseline, rule, and map sequences.

## Elo (raw snapshot Elo)

| Team | v1.3.0 baseline | v1.4.0 (Oct-15-only) | v1.3.1 | Δ vs baseline | Δ vs v1.4.0 |
|---|---|---|---|---|---|
| Karmine Corp | 1676.47 | 1677.32 | 1699.49 | +23.01 | +22.17 |
| Cloud9 | 1591.91 | 1591.07 | 1567.85 | −24.06 | −23.21 |
| MVK Esports | 1505.25 | 1551.80 | 1529.63 | +24.38 | −22.17 |
| FURIA | 1643.06 | 1596.51 | 1619.73 | −23.33 | +23.21 |

Roster-adjusted Elo in `phone_artifacts.json` (`elo = elo_raw + roster_delta`,
deltas unchanged: KC +4.2, C9 −21.1, MVK 0.0, FURIA −0.0):
KC 1703.69, C9 1546.75, MVK 1529.63, FURIA 1619.73.

## Files changed

- `model/snapshots.joblib` — `elo` dict for the 4 teams only.
- `model/phone_artifacts.json` — `model_version` 1.4.0 → 1.3.1; the 4 teams'
  `elo` / `elo_raw` only. `snapshot_date` (2026-10-08) and `generated_at`
  left as-is per the patch scope.
- This log.

## Notes

- Full retrain still blocked: Oracle's Elixir Drive file over quota
  (last refresh attempt 2026-10-16 12:28 EDT, no new games). This patch
  uses only the 4 Play-In series results above, not a data refresh.
- Not pushed to GitHub; app repo untouched. The web/app bundle still
  carries the older ratings until a push + deploy.
