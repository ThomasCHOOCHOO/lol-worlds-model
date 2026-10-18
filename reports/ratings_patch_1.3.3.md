# Ratings patch v1.3.3 (Elo-only, 2026-10-17)

Elo-only patch over the v1.3.2 baseline (snapshot 2026-10-16). **Coefficients
unchanged** — `model.joblib` untouched. Backups of the v1.3.2 files:
`model/snapshots.joblib.bak-1.3.2`, `model/phone_artifacts.json.bak-1.3.2`.

## Update rule (same as training, `build_features.py`)

Per map, after the game: `E' = E + 24 * (S - 1/(1 + 10^(-(E_opp_diff)/400)))`,
where S = 1 for a win, 0 for a loss. No blue-side adjustment in Elo, no
offseason regression mid-tournament. Applied map-by-map in chronological
order.

## Series included (per-map winners verified against lolesports + recap)

| Series | Maps (winner order) |
|---|---|
| Oct 17: MVK Esports 3-1 FURIA | MVK, MVK, FURIA, MVK |

Source: lolesports event details (MVK 3 game wins, FURIA 1) + RiftDaily
game-by-game recap (Game 1 MVK, Game 2 MVK, Game 3 FURIA, Game 4 MVK).
This is the only series played since the v1.3.2 snapshot (2026-10-16).

## Elo (raw snapshot Elo)

| Team | v1.3.2 baseline | v1.3.3 | Δ |
|---|---|---|---|
| MVK Esports | 1529.87 | 1562.16 | +32.29 |
| FURIA | 1622.08 | 1589.79 | −32.29 |

Roster-adjusted Elo in `phone_artifacts.json` (`elo = elo_raw + roster_delta`,
deltas unchanged: MVK 0.0, FURIA −0.0):
MVK 1562.16, FURIA 1589.79.

## Files changed

- `model/snapshots.joblib` — `elo` dict for the 2 teams only.
- `model/phone_artifacts.json` — `model_version` 1.3.2 → 1.3.3; the 2 teams'
  `elo` / `elo_raw` only. `snapshot_date` (2026-10-16) and `generated_at`
  left as-is per the patch scope.
- This log.

## Notes

- Not pushed to GitHub; app repo untouched. The web/app bundle still
  serves v1.3.2 ratings until a push + deploy.
- Full retrain still pending after Sunday's KC–MVK Play-In final (needs fresh
  Oracle's Elixir data); this patch covers the Oct 17 result only.
- FURIA is eliminated; its rating is updated for completeness and for any
  future reference, not for an upcoming match.
