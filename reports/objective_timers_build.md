# Objective timers — Leaguepedia pipeline build

**Date:** 2026-10-17. Status: **pilot complete** — 1,299 games, 30,439 events.
Pipeline code: `src/objective_timers/leaguepedia.py`. Outputs:
`data/processed/objective_timers.csv` (event-level) and
`data/processed/objective_firsts.csv` (per-game firsts).

## What was built

1. **Fetch** — Leaguepedia Timeline pages via the MediaWiki API
   (`action=query&prop=revisions`), polite User-Agent, ≥1.2 s between
   requests, exponential backoff on rate limits. No key, no scraping.
2. **Join** — OE `gameid` *is* the Leaguepedia GameId for most games
   (`LOLTMNT02_*` / `ESPORTSTMNT02_*`), so the join is direct. Remaining
   games (LPL, LDL, some MSI/WLDs/DCup) have OE-style numeric IDs and
   need a Cargo lookup (see Gaps).
3. **Parse** — `ELITE_MONSTER_KILL` (dragon / elder / baron / herald /
   grub / atakhan) and `BUILDING_KILL` (tower / inhibitor) events →
   team, side, game-clock seconds, lane, tower type, killer player
   (via OE player rows: participant 1–5 = blue top/jng/mid/bot/sup,
   6–10 = red), and team gold at the event (latest 60 s frame).
4. **Validate** — every game checked against OE team-row totals
   (dragons / elders / barons / heralds / grubs / towers / inhibitors /
   atakhans + first-X flags). Mismatches are kept in the CSVs with
   `validation = "FLAG: <reasons>"`, never silently dropped.

## Verification

5-game test on known Worlds 2026 Play-In games (KC vs C9 series,
`LOLTMNT02_479621` … `479649`): **5/5 validation pass**, including
first-dragon / first-tower / first-baron / first-herald team + time
matching OE flags. One real discovery during verification: **OE
`dragons` counts elemental + elder dragons** (elder is double-counted
in OE's `elders` column) — validation accounts for this.

## Pilot coverage

| League | Games | Events |
|---|---|---|
| LCK 2026 | 502 | |
| LEC 2026 | 383 | |
| LCS 2026 | 283 | |
| MSI 2026 | 71 | |
| First Stand 2026 | 45 | |
| Worlds 2026 (Play-In) | 15 | |
| **Total** | **1,299** | **30,439** |

Event types: TOWER 15,353 · DRAGON 5,894 · GRUB 3,805 · INHIBITOR 2,163 ·
BARON 1,805 · HERALD 1,282 · ELDER 137. Patches 16.01–16.2, dates
2026-01-14 → 2026-10-16.

Sanity checks on the derived firsts: first-dragon team won **61.7%**,
first-tower team won **68.6%** — in line with known LoL analytics.
Median first-dragon 435 s (~7:15), median first-tower 945 s (~15:45).
Firsts present: dragon 1298/1299, tower 1299/1299, baron 1241/1299
(58 games had no baron), herald 1282/1299, inhibitor 1299/1299.

## Validation results — 58.6% pass, 41.4% flagged

Flag-cause instances (a game can have several):

| Cause | Games |
|---|---|
| Tower count mismatch (parsed vs OE) | 578 issue instances |
| Skipped invalid-team artifact events | 64 |
| Inhibitor count mismatch | 60 |
| Replayed segment (chronobreak signature) | 43 |
| Dragon count mismatch | 5 |
| First-X flag mismatch | 4 |
| Grub / herald count mismatch | 2 / 1 |

**Read this correctly:** a flag is a *disagreement signal*, not proof the
timeline is wrong. OE is not infallible ground truth — **21 games have
OE tower counts above the physical maximum of 11** (up to 16), which is
impossible in a single game. The two sources disagree on towers in both
directions, consistent with a mix of (a) chronobreak/rewound segments
whose pre-rewind events survive in the timeline, (b) OE recording quirks,
and (c) a handful of genuinely corrupted timelines.

## Data-quality findings (all handled in the parser)

- **Phantom grub events** — some timelines contain 3 byte-identical
  `HORDE` events with `killerTeamId: 300` (invalid team). OE confirms
  0 grubs in those games. Parser drops invalid-team events and dedupes
  exact duplicates; the skip is recorded in the validation string.
- **Replayed segments (chronobreak)** — 43 games (3.3%) have the same
  tower falling more times than physically possible (e.g. a team's MID
  NEXUS turret falling 3–4×; each team has only 2). Flagged with
  `possible replayed segment (chronobreak)`. Event times in these games
  should be treated with caution; firsts are usually still correct
  (they happen early, before any rewind).
- **killerId 0 towers** — kept. `killerId: 0` (minion/unknown killer)
  tower kills are real in verified games (e.g. Worlds Play-In game 1:
  C9's 11th tower is a killerId-0 nexus turret that OE counts). Dropping
  them would delete real events and corrupt first-tower times.
- **OE dragons includes elder** — see Verification above.
- **One suspected wrong-game join** — `LOLTMNT05_176711` (LEC
  2026-02-16, Movistar KOI): dragons, barons, heralds, grubs, towers,
  and first-X flags all disagree at once. Either OE's row or the
  timeline belongs to a different game. **Exclude this game from
  analysis** until manually checked.

## Known gaps

- **LPL (~674 games in 2026, ~2,196 across 2024–2026) — BLOCKED.**
  Leaguepedia's Cargo API (`cargoquery`, needed to map OE numeric IDs →
  GameIds) returns `ratelimited` for anonymous API use on every attempt
  across ~30 min, while the regular MediaWiki API works fine. The
  pipeline fails fast on Cargo (60 s backoff, then proceeds) instead of
  stalling. Unblock paths, in order: (1) retry Cargo later — may be a
  time bucket, not a hard block; (2) `Special:CargoExport` via browser
  (same backend, may behave differently); (3) gol.gg fallback for LPL.
- **12 WLDs-labeled qualifier games unresolved** — OE IDs
  `13506-…`, `13507-…`, `13508-…` (LPL Worlds-qualifier series:
  IG–TES, WE–JDG, JDG–IG, Sep 17–19 2026). Same Cargo gap as LPL.
- **Other fuzzy leagues for backfill** — LDL, some MSI 2024 / DCup /
  WLDs games (~2,900 total across 2024–2026). Same Cargo gap.
- **/tmp is a 512 MB tmpfs** — the first pilot run died on disk-full
  writing a timeline cache. The pipeline no longer caches timelines to
  disk; outputs are append-safe CSVs, so re-runs re-fetch only missing
  games.

## Schema

`objective_timers.csv` (one row per objective event): `oe_gameid`,
`source` (= `leaguepedia`), `source_game_id`, `league`, `date`,
`patch`, `game_in_series`, `event_seq`, `event_type` (DRAGON / ELDER /
BARON / HERALD / GRUB / TOWER / INHIBITOR / ATAKHAN), `event_subtype`
(dragon element), `team`, `side`, `game_time_s`, `game_time_ms`,
`lane`, `tower_type`, `killer_player`, `blue_gold_k`, `red_gold_k`,
`gold_diff_k`, `winner`.

`objective_firsts.csv` (one row per game): `oe_gameid`, `league`,
`date`, `patch`, `game_in_series`, `winner`, `validation`,
`first_dragon_team/s`, `first_tower_team/s/lane`,
`first_baron_team/s`, `first_herald_team/s`, `first_inhib_team/s`,
`gold_diff_at_first_dragon_k`, `gold_diff_at_first_tower_k`,
`dragons/towers/barons/heralds/grubs/elders_blue/red`.
Note: `dragons_*` includes elder dragons (OE-compatible); `elders_*`
is the elder-only split.

## Full backfill — commands + estimates

Observed pilot speed: ~1.6 s/game (fetch + parse + validate).
Total OE games 2024–2026: 29,942. Already pulled: 1,299.

```bash
cd ~/workspace/lol-worlds-model
# Full backfill (direct-join games; resume-safe — skips games already in the CSVs).
# Run unattended; ~25,700 direct games remain ≈ 11–12 h wall-clock.
.venv/bin/python -m src.objective_timers.leaguepedia --scope backfill --out-dir data/processed

# After the Cargo gap is unblocked, re-run the SAME command: it picks up
# only the ~2,900 fuzzy games still missing (≈ 1.5 h).
# Single-league runs: add --leagues LPL (or any OE league codes).
# Small test: --scope test (5 Worlds games) or --limit N.
```

## Recommended use

- Join to `data/processed/features.csv` on `oe_gameid` (= OE `gameid`).
- For the first-dragon / first-tower → gold-diff → win analysis, use
  `validation == "pass"` games first (n = 761), then sensitivity-check
  including flagged games — tower-flagged games still have correct
  dragon/herald/baron events in nearly all cases (only 5 dragon and
  1 herald count mismatches in the whole pilot).
- Exclude `LOLTMNT05_176711` (suspected wrong-game join) and treat the
  43 chronobreak-flagged games' event times as approximate.
