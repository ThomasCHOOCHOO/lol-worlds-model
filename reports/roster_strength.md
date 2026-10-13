# Roster-aware team strength

**Why:** team rosters reset every offseason, but the base model
(`build_features.py`) applies the same 40% Elo regression toward 1500 to
every team at each year boundary. A team that kept all five starters got the
same prior as a team that replaced four of them.

**What:** `src/roster_strength.py` derives each team's starting roster per
calendar year from Oracle's Elixir player rows, measures turnover at year
boundaries, and computes a roster-adjusted season-start prior.

## Formula

For the boundary between year y−1 and year y:

```
roster(team, y)  = 5 players with the most games for `team` in year y
                   (substitutes beyond the top 5 are ignored)
t(team, y)       = |roster(team,y) \ roster(team,y-1)| / 5
                   = fraction of starting slots filled by new players
                   (rookies with no year-(y-1) games count toward t)

E_prev(team)     = team's Elo after the last game of year y-1
                   (same Elo replay as build_features: init 1500, K=24,
                   40% regression toward 1500 at each year boundary)
U(team, y)       = 1500 + 0.6 * (E_prev(team) - 1500)
                   # the base model's uniform prior at season start

q(p)             = Elo_{y-1} of player p's main team in year y-1
                   (team p played the most games for; 1500 for rookies /
                   players with no year-(y-1) games)
incoming(team,y) = mean(q(p)) over incoming starters, or 1500 if none
target           = (1 - t) * E_prev(team) + t * incoming(team, y)
A(team, y)       = 1500 + 0.6 * (1 - 0.5 * t) * (target - 1500)
                   # roster-adjusted prior at season start

delta(team)      = A(team, 2026) - U(team, 2026)
rating_adjusted  = snapshot_elo(team) + delta(team)
```

Properties:

- **t = 0** → A = U exactly. No roster change ⇒ identical to the base model.
- **t = 1** → A = 1500 + 0.3·(incoming − 1500). A fully rebuilt roster is
  judged by the quality of the incoming players' old teams, heavily regressed
  toward the mean (maximum uncertainty).
- The delta is computed once at the most recent year boundary (2025→2026) and
  carried forward to the current snapshot; games played since then already move
  the Elo itself.
- Teams with no measurable turnover (e.g. new 2026 orgs like FURIA/LØS, or teams
  absent in 2025) get delta = 0.

## Worked example: Karmine Corp Blue (largest |delta| at the 2025→2026 boundary)

| Quantity | Value |
|---|---|
| 2025 year-end Elo, E_prev | 1867.8 |
| 2025 roster | 3XA, Maynter, SlowQ, Piero, Yukino |
| 2026 roster | Kamiloo, Prime, Yukino, Tao, Looki |
| Turnover t | 4/5 = 0.8 (only Yukino retained) |
| Incoming players' prior-team Elos → mean | 1475.6 |
| Uniform prior U = 1500 + 0.6·(1867.8−1500) | 1720.7 |
| Blend target = 0.2·1867.8 + 0.8·1475.6 | 1554.0 |
| Adjusted prior A = 1500 + 0.6·(1−0.4)·(1554.0−1500) | 1519.5 |
| **delta = A − U** | **−201.2** |

Interpretation: KC Blue dominated 2025 (Elo 1868) but replaced 4/5 starters with
players from ~1476-Elo teams. The uniform prior would have started them at
1721; the roster adjustment says their true starting strength was ~1520
(−201). Their current snapshot Elo then evolves from 2026 games as usual.

A familiar-team example, **T1**: 2025 year-end Elo 1810.8; only Gumayusi→Peyz
changed (t = 0.2); Peyz's prior team Elo 1585.6; U = 1686.5, A = 1643.5,
delta = −43.0.

## Coverage

881 teams seen in 2024–2026; 230 have a measurable 2025→2026 turnover.
delta_2026 ranges from −201.2 to +164.6 (mean −0.7) — roughly symmetric, with
the tails belonging to academy/secondary rosters that churned heavily.

## Approximations (documented)

1. Starters = top 5 by games played per calendar year; substitutes ignored.
2. Player identity is by `playername` string — renames across years are treated
   as different players (counts toward turnover).
3. A player's prior-team quality = that team's year-end Elo; individual player
   skill within a team is not separated.
4. Rookies / imports with no prior-year games contribute 1500 (league mean).
5. The delta persists unchanged until the next year boundary (no decay as 2026
   games accumulate) — conservative and simple; the Elo updates themselves
   already reflect the new roster's results.
6. Year = calendar year (Oracle's Elixir `date`), not competitive split.
