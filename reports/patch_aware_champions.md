# Patch-aware champion strength

Every LoL patch rebalances champions, so a trailing-90d global win% that spans
patches mixes pre- and post-nerf data. Champion power is now computed
per-patch, strictly walk-forward (Oracle's Elixir `patch` column).

## Formula

For a game on patch P at time t, champion c's power is

```
power(c, P, t) = (w_P + a * prior) / (n_P + a),   a = 15 (pseudo-games)
```

- `w_P`, `n_P`: c's wins / games on patch P, games strictly before t.
- `prior`: the previous patch version's shrunk value,
  `(w_Q + a*0.5) / (n_Q + a)` on games strictly before t, where Q is the
  patch version immediately below P (versions ordered numerically, so
  15.24 < 16.01). If there is no previous patch, prior = 0.5.

Behavior:
- Cold start (`n_P = 0`) → power = previous patch's value. A champion that
  hasn't been played yet on the new patch keeps its old-patch rating instead
  of collapsing to 50%.
- Large `n_P` → power → this patch's win%. Rebalances show up as soon as pro
  games accumulate.
- This is plain add-k empirical-Bayes shrinkage; the only change vs the old
  feature is that the shrinkage *target* is the previous patch's value
  instead of a flat 0.5, and there is no min-games gate (shrinkage handles it).

`ban_edge` uses the same per-patch power (bans are announced on the game's
own patch).

## Team-champion comfort: kept patch-global (deliberate)

Comfort stays a trailing-365d team×champion win% (min 3 games, shrunk toward
the team's trailing-20 form with 10 pseudo-games). A team's (champion, patch)
sample is ~1–5 games — patch-level comfort would be pure noise — and the
365d window is dominated by recent-patch games anyway.

## Current patch designation

`current_patch = "16.18"` — the highest patch version with ≥100 pro games
(539 games, 2026-09-19 → 2026-10-08). This is the presumed Worlds 2026 patch
(Play-In starts Oct 16); if Riot announces otherwise, override with
`predict.py --patch <version>` or the app's patch selector.

Snapshots export the 12 most recent patches (`16.07`–`16.18`, trailing
~6 months) as `champion_power_by_patch` in `model/phone_artifacts.json`
(v1.2.0), plus `current_patch` and `recent_patches`. Each patch table also
includes champions with zero games on that patch (valued at their
prior-patch blend) so live drafts never drop a champion.

## Sanity check

Same-patch values move with rebalances, e.g. Azir 0.432 (16.17) → 0.388
(16.18), Rakan 0.552 → 0.476, Ahri 0.556 → 0.590 — the feature now reacts to
patches instead of averaging across them.
