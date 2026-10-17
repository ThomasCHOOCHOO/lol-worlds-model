# Retroactive objective timers — brainstorm

**Goal:** collect per-game objective event times (dragon / baron / herald / grub / tower / inhibitor kills, with team + game-clock time) for historical pro games, to analyze first-dragon / first-tower effects on gold differentials and game wins.

**Date:** 2026-10-17. Read-only research — nothing was fetched with a key, nothing signed up for, nothing paid.

## TL;DR

| Rank | Source | What you get | Coverage | Cost / effort |
|---|---|---|---|---|
| **1** | **Leaguepedia V5 Timeline pages** | Full per-event timeline (every objective, ms timestamps) | ~All major pro games, 2014→present | Free; ~1–2 days to build parser + join |
| 2 | gol.gg game pages | Timeline w/ timestamps (visually confirmed) | Major leagues, ~2018→present | Free; extraction method unverified |
| 3 | lolesports livestats (prospective) | 10s-cadence frames → event times ±10s | Future games only | Free; poll + archive during live games |
| 4 | VOD + OCR (extend existing worker) | Objective times from broadcast scoreboard | Any league w/ YouTube VODs | Free but heavy; ~5–15 min compute/game; fragile |
| 5 | Manual annotation | Gold-standard labels | Any sample size you pay for | ~$2–4/game first-objectives-only; ~$8–15/game full timeline |
| 6 | Commercial (GRID / PandaScore / OddsMatrix) | Official telemetry, archives | Best | Sales-quote only; likely $1k+/mo; probably out of reach |
| ✗ | Riot Match-V5 API | Perfect timelines — **but only for public-shard games, not pro games** | Ranked/normal only | Dead end for pro |

**Recommendation:** build on **Leaguepedia V5 timelines** (Option 1). Validate against Oracle's Elixir end-of-game totals. Use VOD+OCR or manual labels only to fill Leaguepedia gaps or to QA the parser.

---

## 0. What we already have (verified locally)

Oracle's Elixir `data/raw/*.csv` (165 columns) has **no objective timestamps**:

- End-of-game totals only: `dragons`, `opp_dragons`, `elders`, `heralds`, `barons`, `towers`, `inhibitors`, `void_grubs`, `atakhans`.
- First-X **flags** only (no times): `firstdragon`, `firstherald`, `firstbaron`, `firsttower`, `firstmidtower`, `firsttothreetowers`.
- The only time column is `gamelength` (seconds).
- At-t snapshots exist for gold/xp/cs/kills at t = 10/15/20/25 (`goldat10`, `golddiffat15`, …) — but **not** for objectives (already documented in `reports/live_ingame.md` §1).

Two related notes:

- **Legacy Oracle's Elixir schema** (pre-2020 data dictionary) *did* include first-objective times: `fdtime`, `fttime`, `fbarontime`, `heraldtime`, `fbtime` (minutes, hundredths). Those columns were dropped from the modern CSVs. If pre-2020 analysis is ever wanted, the old OE files may still carry them — but for 2024–2026 they do not exist.
- **No join key to Riot/Leaguepedia.** OE `gameid` is OE's own ID (e.g. `13057-13057_game_1`), **not** a Riot match ID. The `url` column points at league stats pages (e.g. `lpl.qq.com/...?bmid=13057`) or is empty. Any join to another source must be fuzzy: (league, date, team names, game number in series) + validation against end-of-game totals.

## 1. Riot Match-V5 API — perfect data, wrong population ✗

**What it returns.** `GET /lol/match/v5/matches/{matchId}/timeline` returns 1-minute participant frames (gold, xp, position) plus a full event list with millisecond timestamps. The exact event types we want exist:

- `ELITE_MONSTER_KILL` — `monsterType` (DRAGON, BARON_NASHOR, RIFT_HERALD), `monsterSubType` (e.g. FIRE_DRAGON, HEXTECH_DRAGON), `killerTeamId`, `position`, `timestamp`.
- `BUILDING_KILL` — `buildingType` (TOWER_BUILDING, INHIBITOR_BUILDING), `laneType` (TOP_LANE, MID_LANE, BOT_LANE), `towerType` (OUTER_TURRET, INNER_TURRET, BASE_TURRET, NEXUS_TURRET), `teamId`, `timestamp`.
- Plus `CHAMPION_KILL`, `ITEM_PURCHASED`, etc. Every objective, timestamped to the ms, with team + type + lane. Exactly the schema we want.

**Access.** Free Riot developer key from the developer portal (production keys need approval). Community-reported dev-key limits ≈ 20 req/s and ~100 req/2 min per routing cluster. One timeline call per game → ~30k games ≈ 30k calls ≈ ~10 h of polite wall-clock. Cheap in principle.

**Why it's a dead end for pro games.** Match-V5 serves games played on **public shards** (ranked / normal). Pro games are played on the **Tournament Realm** as custom games with IDs like `ESPORTSTMNT02_2787299`. There is no Match-V5 endpoint for tournament-realm games, and no mapping from OE `gameid` to a Match-V5 ID. (Tournament-V5 only covers tournaments *you* create with your own codes.) **Do not build on Match-V5 for pro data.**

Related: Riot's Oct 2026 replay restrictions (patch 26.20 → delayed to 26.21) cut off third-party replay-file downloads. That kills *replay-based* collection going forward, but does not affect API timeline data (which is a separate channel) — and is one more reason to prefer an already-archived source like Leaguepedia.

## 2. Leaguepedia V5 Timeline pages — RECOMMENDED ★

**What it is.** Leaguepedia (`lol.fandom.com`) archives, per pro game, the same Match-V5-shaped data under `V5 data:<GameId>/Timeline` pages — e.g. `V5 data:ESPORTSTMNT02_2787299/Timeline`. The page content is the raw timeline JSON: frames with per-participant `totalGold` / `xp`, and an `events` array including `ELITE_MONSTER_KILL` (with `monsterType: BARON_NASHOR`, `killerTeamId`) and `BUILDING_KILL`, each with a ms `timestamp`. This is the full timeline, not just firsts.

**Coverage.** Leaguepedia covers essentially all major pro leagues (LCK, LPL, LEC, LTA, LCP, VCS, PCS, minor leagues, MSI, Worlds) going back to ~2014. Realistic expectation: near-complete for Tier-1 leagues 2018→present; patchier for minor leagues / older years. Verify per league/year during the build.

**Access.** Free, no key. Two-step fetch:

1. **Enumerate games → GameId.** Leaguepedia's Cargo API (`cargoquery`) exposes schedule tables (e.g. `MatchScheduleGame`) with team names, date/time, game number, and the `GameId` field. There is also a community Python wrapper (`leaguepedia_parser` on PyPI) for the Cargo tables.
2. **Fetch each Timeline page** via the MediaWiki API (`action=raw` / `parse`) and parse the JSON.

**Join to Oracle's Elixir.** No shared key. Join on (league, date, team1, team2, game #), with team-name normalization (OE `teamname` vs Leaguepedia team names differ in edge cases). Then **validate every joined game**: Leaguepedia-derived event counts (dragons / towers / barons / heralds) must equal OE end-of-game totals, and derived first-X flags must equal OE `firstdragon` / `firsttower` / etc. Mismatches → flag for manual review or drop. This validation is what makes the fuzzy join trustworthy.

**Effort.** ~1–2 days for join + parser + validation on a 100-game sample. Full pull: Timeline pages are roughly 0.5–2 MB each; ~30k games → ~15–60 GB raw if you keep everything (store parsed events only → tens of MB). At a polite ~1 req/s, fetching is ~8+ hours wall-clock, unattended.

**Caveats / ToS.** Leaguepedia is volunteer-run on Fandom. Use the MediaWiki/Cargo APIs (not HTML scraping), a descriptive User-Agent, ≤1 req/s, respect `maxlag` / retry-after. Do not redistribute raw dumps; derived aggregates for research are the normal use case. Timeline pages may be missing or incomplete for some games — the OE-total validation catches that. Exact Cargo field names need confirming against the live Cargo API during the build (not verified in this research pass).

## 3. gol.gg — plausible fallback / cross-check

gol.gg game pages (`gol.gg/game/stats/<id>/page-game/`) show a **"Gold graph & Timeline"** section with timestamped events (e.g. entries at 3:39, 7:46, 14:02…). The event labels did not survive text extraction (JS-rendered / icon-based), so the machine-readable structure is **unverified**. Coverage is major leagues ~2018→present. No public API — extraction would mean HTML/JS scraping. Best use: cross-check a sample of Leaguepedia-derived timelines, or fill Leaguepedia gaps. Lower priority than Option 1.

## 4. lolesports livestats feed — prospective only, not retroactive

The unofficial lolesports API (already used by this project — see `reports/live_ingame.md`) serves `window/{gameId}` frames at ~10s cadence with per-team `totalGold`, `towers`, `dragons[]`, `barons`, `inhibitors`. From frames you can derive objective event times to ±10s (first frame where a count changes).

**Retroactive use is NOT available:** the window endpoint retains only recent frames; historical replay for old games is not served. (The `details/{gameId}` endpoint *may* return full-game frames for completed games — unverified; worth a 10-minute check, but do not count on it.)

**Prospective use IS valuable:** poll + archive frames during live games (Worlds 2026 onward) and you build a timeline dataset going forward, for free, with no VOD processing. Recommended as a standing collector alongside the retroactive Leaguepedia pull.

## 5. VOD + OCR — feasible for a sample, not for 30k games

**Current worker.** `~/workspace/lol-predictor-app/ocr/worker.py` (RapidOCR + ffmpeg + streamlink) reads the broadcast scoreboard: game clock, kills, gold per side, every ~20s. It does **not** read objective events.

**Extension path.** Sample frames every ~5s; OCR/track the scoreboard's dragon-count icons and tower-count numbers; a count change = an objective event at that game-clock time (±5s). Dragon *type* is not in the count — kill-feed icon recognition would be needed for that (much harder); for first-dragon / first-tower / first-baron analysis, counts suffice. Herald / grubs are generally not on the scoreboard → not recoverable this way.

**Costs.** Per game: ~35 min of video; download (yt-dlp) + sampling + OCR ≈ 5–15 min CPU compute per game + QA. 500 games ≈ 40–125 compute-hours + storage. Accuracy risks: overlay layout varies by league/year; replay segments / intermissions / hidden scoreboards cause missed or duplicated events; clock resets between games need game segmentation (draft → game-start detection). Mitigation: OE end-of-game totals are ground truth for counts — any VOD-derived timeline that doesn't sum to OE totals gets rejected.

**Availability.** LoL Esports YouTube channel hosts full VODs for major leagues going back years (Twitch VODs expire after ~60 days for partners — YouTube is the archive). **ToS note:** downloading YouTube VODs violates YouTube's ToS even for research; OCR of broadcasts for personal analysis is a gray area. Don't redistribute VOD-derived data.

**Verdict.** Use only for (a) leagues/years where Leaguepedia timelines are missing, or (b) a QA sample to validate the Leaguepedia parser. Not a primary path.

## 6. Manual annotation — gold standard for a sample

- **First-objectives-only** (first dragon team+time, first tower team+time+lane, first baron, first herald): scrub to first events, ~5–10 min/game → at $15–25/hr contractor/RA rates ≈ **$2–4/game**. 500 games ≈ **$1,000–2,000**; 1,000 games ≈ $2,000–4,000.
- **Full timeline** (every objective event): watch at 1.25–1.5× + log ≈ 25–35 min/game ≈ **$8–15/game**. 500 games ≈ **$4,000–7,500**.
- Quality: double-label a 5% sample for inter-annotator agreement; validate counts vs OE totals.
- Cheapest high-value use: hand-label ~50 games as a **QA set** to measure the Leaguepedia parser's accuracy before trusting it at scale. (Thomas could also label a small set himself.)

## 7. Commercial providers — best data, sales-quote pricing

- **GRID** (`grid.gg`): official Riot data partner; direct server telemetry; LoL covered; **multi-year historical archives** advertised alongside the live feed. No public pricing — "talk to sales". This project previously noted "commercial, no free tier." Expected: enterprise pricing, likely $1k+/mo or usage-based. Also note GRID acquired Bayes Esports' IP assets after Bayes' insolvency (2026), so Bayes is no longer a separate option.
- **PandaScore**: previously quoted at €1,000+/mo per title; free tier covers fixtures/schedules/pre-match only, no in-game state.
- **OddsMatrix**: commercial, coverage breadth is its pitch; pricing on request.
- Verdict: only pursue if a budget appears or if Leaguepedia + gol.gg both fail. Ask for a quote with the exact ask: "historical LoL per-game objective event timelines, 2024–2026, all Tier-1 leagues, one-time archive license."

## 8. Community datasets — checked, not sufficient

- **Kaggle "Worlds 2024 stats"** (gol.gg-derived): counts only, no timestamps.
- **Kaggle / GitHub ranked-match datasets**: Match-V5-derived, i.e. solo queue, not pro.
- **HuggingFace `maknee/league-of-legends-decoded-replay-packets`**: decoded replay packets from pro matches (e.g. patch 12.22 batches) — contains literally everything, including positions. Overkill: packets are huge (a game can be 100 MB+ decoded), coverage is limited to collected patches, and Oct 2026 replay restrictions block future collection. Only relevant if micro-level (positioning) data is ever needed.
- **GitHub exporters (lolexport etc.)**: personal match-history tools, not pro data.

## Recommended path

1. **Verify Leaguepedia (half a day).** Pick 3–5 known games (e.g. Worlds 2026 Play-In games). Query Cargo for their `GameId`s; fetch the Timeline pages; confirm `ELITE_MONSTER_KILL` / `BUILDING_KILL` events parse and counts match OE totals.
2. **Build the join + parser (1–2 days).** OE ↔ Leaguepedia on (league, date, teams, game #) + OE-total validation. Parse events → schema below. Run on a 100-game sample; hand-check 5 games against VOD.
3. **Full pull (unattended).** Polite 1 req/s fetch of Timeline pages for 2024–2026 (or 2018→present if coverage holds). Store parsed events only.
4. **QA.** Hand-label ~50 games (Option 6) as a gold set; report parser accuracy (event recall / time error).
5. **Gap-fill.** For games missing Leaguepedia timelines: gol.gg scrape (Option 3) or VOD+OCR (Option 5) or manual labels, in that order.
6. **Prospective.** Stand up a livestats frame archiver for live games (Option 4) so future games get timelines at ±10s with no VOD work.
7. **Analysis (the actual goal).** With events + OE at-t gold snapshots: first-dragon → gold@15/20/25 diff; first-tower → gold diff; objective timing distributions by patch/league; first-objective → win prob (logistic, controlling for pre-game prob and gold diff — first-objective is partly a *consequence* of leading, so report both raw and controlled estimates).

## Proposed dataset schema

One row per objective event (primary table `objective_events.csv` / parquet):

| Column | Type | Notes |
|---|---|---|
| `oe_gameid` | str | Oracle's Elixir `gameid` (join key to features.csv) |
| `source` | str | `leaguepedia` / `golgg` / `ocr` / `manual` / `livestats` |
| `source_game_id` | str | e.g. `ESPORTSTMNT02_2787299` |
| `league` | str | OE league |
| `date` | date | Game date |
| `patch` | str | e.g. `16.18` |
| `game_in_series` | int | OE `game` |
| `event_seq` | int | 1-based order within game |
| `event_type` | str | `DRAGON`, `ELDER`, `BARON`, `HERALD`, `GRUB`, `TOWER`, `INHIBITOR`, `ATAKHAN` |
| `event_subtype` | str | Dragon element: `FIRE`/`WATER`/`EARTH`/`AIR`/`HEXTECH`/`CHEMTECH`; else empty |
| `team` | str | Team name taking the objective |
| `side` | str | `blue` / `red` |
| `game_time_s` | float | Game-clock seconds (Leaguepedia ms / 1000) |
| `game_time_ms` | int | Raw timestamp, when available |
| `lane` | str | `TOP`/`MID`/`BOT` (towers/inhibitors/herald) |
| `tower_type` | str | `OUTER`/`INNER`/`BASE`/`NEXUS` (towers) |
| `killer_player` | str | Player credited, when available |
| `blue_gold_k` / `red_gold_k` | float | Team gold at event (from timeline frames; Leaguepedia has per-participant gold) |
| `gold_diff_k` | float | blue − red gold at event |
| `winner` | str | Game winner (from OE) |

Derived per-game table `game_firsts.csv` (one row per game):

| Column | Notes |
|---|---|
| `oe_gameid`, `league`, `date`, `patch`, `winner` | Keys / outcome |
| `first_dragon_team`, `first_dragon_s` | First `DRAGON` event |
| `first_tower_team`, `first_tower_s`, `first_tower_lane` | First `TOWER` event |
| `first_baron_team`, `first_baron_s` | First `BARON` event |
| `first_herald_team`, `first_herald_s` | First `HERALD` event |
| `first_inhib_team`, `first_inhib_s` | First `INHIBITOR` event |
| `gold_diff_at_first_dragon_k`, `gold_diff_at_first_tower_k` | Gold diff at those events |
| `dragons_blue`, `dragons_red`, `towers_blue`, `towers_red`, `barons_blue`, `barons_red` | Event counts — must match OE totals (validation) |

## Validation plan (applies to every source)

1. **Count check:** per-game event counts vs OE end-of-game totals (dragons / towers / barons / heralds). Mismatch → reject or manual review.
2. **First-flag check:** derived first-dragon / first-tower / first-baron / first-herald team vs OE `firstdragon` / `firsttower` / `firstbaron` / `firstherald` flags.
3. **Monotonicity:** event times increasing, all within `[0, gamelength]`.
4. **Gold sanity:** gold diff at events should be consistent with OE `golddiffat10/15/20/25` at nearby times (loose check, ±2k).
5. **Gold QA set:** ~50 hand-labeled games; report event recall/precision and median |Δtime|.

## Legality / ToS summary

- **Leaguepedia / Fandom:** use the MediaWiki + Cargo APIs, polite rate limits, descriptive User-Agent; don't redistribute raw dumps. Lowest-risk path.
- **Riot Match-V5:** legitimate API use, but no pro coverage — moot.
- **YouTube VODs:** downloading violates YouTube ToS; personal-research use is common but not compliant. Don't redistribute.
- **GRID / PandaScore:** commercial licenses; terms set by contract.
- **Riot replay restrictions (Oct 2026):** replay-file downloads now restricted to participants; this affects replay-packet collection, not the sources above. Archive timelines *now* — access to historical data has already tightened once.

## Sources consulted

- Oracle's Elixir CSV headers (`data/raw/2026.csv`, inspected locally) + legacy data dictionary (oracleselixir.com / timsevenhuysen.com) for the dropped `fdtime`/`fttime`/`fbarontime` columns.
- `reports/live_ingame.md` (this project) for lolesports feed behavior + prior vendor notes.
- `ocr/worker.py` (this project) for the existing VOD+OCR capability.
- Leaguepedia example Timeline page (`V5 data:ESPORTSTMNT02_2787299/Timeline`) via web search.
- GRID (`grid.gg/products/live-data/`), PandaScore (prior project note), Cito API docs (`citoapi.com/docs/api/league-of-legends/`), gol.gg game pages, HuggingFace `maknee/league-of-legends-decoded-replay-packets`.
- Riot replay-restriction coverage (Oct 2026, patch 26.20/26.21) via news search.
