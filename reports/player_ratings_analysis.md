# Player-level ratings instead of team-level ratings

**Date:** 2026-10-17 · **Model version evaluated:** v1.3.2 · **Status:** analysis only, no model files changed
**Verdict: NO-SHIP** — as a replacement for team Elo, or as a second feature, player Elo makes the model *worse* in exactly the domain it is used for (tier-1 leagues + internationals: Worlds, MSI, LCK/LPL/LEC). It is genuinely better in minor leagues and academy tournaments, but that is not where the bets are. Details and the one path that could still work are in §9.

## 1. The question

The white paper (§9, point 8) notes the model treats "a team as its name": a team's Elo follows the org, no matter who actually plays. If the five players change, the rating should change with them. Player-level Elo is the obvious fix: rate each player, and rate a team as the aggregation of its five players on the day.

Two claims needed testing:

1. **Does it predict better?** Headline test on the v1.3.2 splits.
2. **Does it add training data?** No — see §8. Same games, finer granularity. Any gain must come from *roster-change handling*, not data volume.

## 2. What was built

**Identity.** Players are keyed by Oracle's Elixir `playerid`, not name: no playerid maps to more than one name, but 75 handles map to multiple playerids (name reuse), so name-keying would conflate different people. 3,917 rows (1.3%) have a null playerid, concentrated in minor leagues (LJL 1,761; LPLOL 404; LIT 338) — those fall back to name. All 260 `MISSING`-name rows have a playerid. Every game-side has exactly 5 player rows (59,884 = 29,942 × 2).

**Player Elo replay** (walk-forward, one pass per variant, same discipline as `build_features.py`):

- Rating key: (player, role) by default; role-agnostic variant tested separately.
- Init 1500, K = 24, 40% regression toward 1500 at each calendar-year boundary (same constants as team Elo, so scales are comparable).
- Team aggregate = **mean** of the 5 pre-game player ratings. Expected score from the aggregate diff (standard Elo formula). **Every player on a side updates by K × (team result − expected team score)** — players move with their team's result, exactly as the task specified.
- Debutants (no prior games): prior 1500 by default; `team_prior` variant uses the mean of their rated teammates.
- 5,863 (player, role) ratings tracked; 5,333 player-only; 2,795 with ≥20 games. Median games per player: 28.

**Variants tested:** base (mean, 1500 prior, no decay) · inactivity decay (continuous 0.5^(gap/365) toward 1500 after a 30-day gap) · hard reset (1500 after a 365-day gap) · team prior for debutants · role-agnostic (no role split) · K=40 for a player's first 30 games (`k_new`) · fixed role-weighted mean (top 1.0 / jng 1.2 / mid 1.25 / bot 1.0 / sup 0.55, weights sum to 5.0 so the scale matches the plain mean) · 5 per-role diff features (lets the LR learn role weights).

**Comparison protocol.** Identical to `train.py`: LR, C tuned by 3-fold TimeSeriesSplit, 180-day recency sample weights, scaler fit on the fit set. Splits: fit < 2026-07-01, tail 2026-07-01..2026-10-01 (n = 3,368). **Primary metric: RAW tail log-loss** (LR fit only on pre-Jul-1 data — honest out-of-sample). Calibrated numbers (isotonic fit on the tail, evaluated on the tail, i.e. the v1.3.2 protocol) are reported alongside but are in-sample for the calibration map. Backtests refit on pre-event data only (Worlds 2025, MSI 2026). Paired bootstrap (20k resamples) on per-game log-loss differences.

## 3. Headline results — full feature set (tail, n = 3,368)

| Model | RAW log-loss | RAW acc | RAW AUC | CAL log-loss |
|---|---|---|---|---|
| M0 team Elo (v1.3.2 baseline) | 0.6332 | 0.6485 | 0.6901 | 0.6253 |
| **M1 player Elo, mean** | **0.6259** | 0.6381 | 0.6999 | 0.6195 |
| M2 team + player Elo | 0.6257 | 0.6440 | 0.7006 | 0.6198 |
| M3 per-role diffs (5 features) | 0.6262 | 0.6390 | 0.6998 | 0.6202 |
| M1 role-weighted mean | 0.6271 | 0.6387 | 0.6984 | 0.6207 |
| M1 + inactivity decay | 0.6267 | 0.6404 | 0.6993 | 0.6207 |
| M1 + hard reset (365d) | 0.6259 | 0.6395 | 0.6998 | 0.6198 |
| M1 + team prior for debutants | 0.6253 | 0.6378 | 0.7013 | 0.6188 |
| M1 role-agnostic | 0.6262 | 0.6372 | 0.6994 | 0.6196 |
| M1 + K=40 for first 30 games | 0.6256 | 0.6369 | 0.7000 | 0.6197 |

Paired bootstrap vs M0 (RAW tail): M1 **+0.0073** (95% CI [+0.0023, +0.0125], P(M1 better) = 0.998); M2 +0.0075 (CI [+0.0031, +0.0119], P = 1.000); team-prior variant +0.0079 (CI [+0.0027, +0.0132], P = 0.999). Every variant beats team Elo overall.

**Pre-draft (nodraft) model** — the one the edge watch actually uses: N0 team 0.6343 → N1 player 0.6268 → N2 both 0.6269 (RAW). Same pattern.

**Backtests** (honest, refit pre-event): Worlds 2025 (n=96) — M0 0.6803 / 0.5521, M1 0.6743 / 0.5833, M2 0.6739 / 0.5625 (player Elo slightly better). MSI 2026 (n=71) — M0 0.6214 / 0.6197, M1 0.6250 / 0.6056, M2 0.6255 / 0.5915 (player Elo slightly worse). Mixed, small-n — no clear signal either way.

**Accuracy paradox:** M1 has *lower* accuracy than M0 (0.638 vs 0.649) but *better* log-loss and AUC. M1 is less often on the right side of 0.5, but much better calibrated on the games it gets wrong (less overconfident). Log-loss is the betting-relevant metric, so this is a real win — on the pooled tail.

## 4. Where the gain comes from — and where it reverses

The pooled tail mixes two very different regimes. Splitting by league tier flips the story:

| Subset (tail, RAW) | n | M0 − M1 log-loss diff | 95% CI | P(M1 better) |
|---|---|---|---|---|
| Tier-1 regular (LCK/LPL/LEC/LCS/LCP) | 686 | **−0.0065** | [−0.0147, +0.0015] | 0.055 |
| Internationals (MSI/WLDs/EWC) | 119 | **−0.0157** | [−0.0342, +0.0015] | 0.038 |
| **Tier-1 + internationals** | **805** | **−0.0079** | **[−0.0153, −0.0005]** | **0.019** |
| Minor / regional leagues | 2,563 | +0.0121 | [+0.0059, +0.0184] | 1.000 |
| KeSPA Cup only | 61 | +0.1622 | [+0.0485, +0.2900] | 0.998 |
| Non-KeSPA | 3,307 | +0.0045 | [−0.0001, +0.0092] | 0.972 |
| Gap \|player − team\| ≤ 100 | 3,062 | +0.0021 | [−0.0017, +0.0057] | 0.861 |

League-level gain (sum of per-game log-loss improvements, total +24.74): KeSPA Cup +9.89 (**40.0% of the total gain from 1.8% of the games**), NLC +4.72, RL +3.64, CD +2.91, TCL +2.49, HLL +2.11 — and on the other side LCK −1.79, LPL −1.78, LFL −1.80, LEC −1.52, EWC −1.41.

**M2 (both features) does not rescue tier-1 either:** tier-1+intl −0.0068 (CI [−0.0135, −0.0003], P = 0.021). Why: in M2 the LR puts 4.5× more weight on player Elo (standardized coef +0.52) than team Elo (+0.11), because the fit set is dominated by minor-league games where player Elo is better. A single global linear blend cannot learn "use player Elo in minors, team Elo in tier-1."

**Interpretation.** In tier-1, team Elo captures something real that player Elo discards: the *org* component — coaching, infrastructure, scrim quality, team cohesion. When a star moves from a strong org to a weak one, player Elo immediately credits the weak org with the star's rating; team Elo says the org is still the org. In tier-1, where rosters are comparatively stable and orgs are persistent, the org component is signal. In minor leagues, team identities churn constantly (rosters, academy teams, orgs folding, one-split wonders), so the org component is mostly noise and the player component dominates. Player and team Elo correlate 0.95 in tier-1+intl but only 0.85 in minors; mean |player − team| gap is 26.9 Elo in tier-1+intl vs 46.2 in minors.

## 5. Case study: KeSPA Cup 2026 — where "a team is its name" actually fails

The eight largest player-vs-team disagreements in the tail are all KeSPA Cup games. The mechanism is not a rename (DN Freecs never appears in 2024–2026 data; DN SOOPers is the name throughout) — it is **academy squads playing under the main org's name**:

| Game | Team Elo diff | Player Elo diff | Result |
|---|---|---|---|
| DN SOOPers vs HLE, Jul 21 | −521 | +1 | DN SOOPers won |
| Gen.G vs BRO, Jul 21 | +332 | −173 | Gen.G lost |
| DN SOOPers vs HLE, Aug 11 | −333 | +131 | DN SOOPers won |
| T1 vs Gen.G, Jul 21 | +16 | +327 | T1 won |

Lineups (verified from player rows): HLE fielded all five of **HLE Challengers** (Panther/Jackal/Cracker/Pyeonsik/Bluffing); Gen.G fielded all five of **Gen.G Global Academy**; BRO fielded its **main roster**; DN SOOPers fielded a mix (3 main + Sharvel from DK Challengers + Peter from DN SOOPers Challengers); T1 fielded a squad with 0/5 overlap with its previous game.

Team Elo attached HLE's main-roster rating (~sky-high) to five Challengers players, and Gen.G's main-roster rating to five academy players. Player Elo followed the actual humans and got all four games right. This is the paper's point-8 failure mode in its purest form — and it is worth noting the model got these games badly wrong *in the direction that loses money* (fading HLE's academy squad at main-roster prices).

**Caveat on the subset flags:** `roster_changed` compares a lineup to the team's *immediately previous game*. KeSPA games' previous games were also KeSPA games, so the flag reads False even though the lineup is entirely different from the org's usual (LCK) lineup. That is why the "no roster change" subset (n = 2,865) still shows M1 winning (+0.0068, P = 0.994): it contains the KeSPA games. A better flag would compare against the team's trailing-30-day modal lineup. The tier split in §4 is the cleaner cut, and it is what the recommendation rests on.

## 6. The hard parts, each tested

**Roster changes / substitutions (the main potential win).** Roster-changed games are 14.1% of all games, 14.9% of the tail (n = 503). M1 beats M0 there by +0.0103, but the CI includes zero ([−0.0043, +0.0252], P = 0.914) — directionally right, not conclusive alone. The stronger evidence is the disagreement subset: games where |player − team| > 50 (n = 981): +0.0291, CI [+0.0136, +0.0446], P = 1.000; > 100 (n = 306): +0.0603, CI [+0.0194, +0.1026], P = 0.998. When the two ratings disagree materially, the player rating is usually right — but §4 shows that is mostly a minor-league phenomenon.

**New players (prior?).** Debutants appear in 199 tail games (5.9%). 1500 prior vs team-mean prior: team prior is the best variant overall (RAW 0.6253 vs 0.6259) but the difference is within noise, and it does *not* fix the tier-1 reversal (−0.0079 on tier-1+intl). Recommendation: if player Elo is ever used, debutant prior = mean of rated teammates is the slightly better default — it encodes "a rookie on T1 is probably better than a rookie on a bottom team" without pretending to know more.

**Role swaps.** 438 players (9.3%) have played >1 position; 144 have a second role at ≥20% of their games (genuine role changes, not fill-ins). Per-role keying vs role-agnostic: RAW 0.6259 vs 0.6262 — negligible. Role swaps are too rare, and too often lateral (a mid moving to ADC on a weak team is still on a weak team), to matter at this granularity.

**Inactivity decay / staleness ("good 2 years ago may not be good now").** Three separate tests, all negative:
- Continuous inactivity decay (0.5^(gap/365) toward 1500 after 30d): 0.6267 — *worse* than no decay (0.6259).
- Hard reset to 1500 after a 365-day gap: 0.6259 — identical.
- LR recency half-life for the player model: 90d → 0.6265, 180d → 0.6259, 365d → 0.6257, unweighted → 0.6260. Total range 0.0008 — second-order. (Same for team Elo: 0.6333–0.6342.)

So: **old data neither helps nor hurts much.** The existing machinery — 40% year-boundary regression + 180-day LR sample weights — already handles staleness about as well as anything tested. There is no evidence for aggressive forgetting of player history. (One caveat: these tests vary *either* Elo memory *or* LR weights, not a full joint re-tune; a dedicated staleness study could still find something small.)

**K-factor for new players.** K=40 for a player's first 30 games: 0.6256 vs 0.6259 base — no meaningful gain. New-player ratings converge fast enough at K=24.

## 7. Aggregation: mean vs sum vs role-weighted

- **Sum vs mean: identical.** Sum = 5 × mean whenever all 5 players are present (always, in this data). After standardization the LR cannot tell them apart. Anyone proposing "sum" as a distinct option is mistaken.
- **Fixed role-weighted mean: slightly worse** (0.6271 vs 0.6259). The hand-picked weights (mid/jungle up, support down) add nothing.
- **Learned per-role diffs (5 features): 0.6262** — statistically indistinguishable from the plain mean (0.6259), with 4 extra parameters. The LR does not find exploitable lane-by-lane structure in Elo diffs beyond what the mean already carries. (This is a different question from counterpicks/champion matchups, which Elo cannot see at all — see the companion analyses.)
- **Recommendation if ever used: plain mean.** Simplest, best, no tuning.

## 8. Does player-level rating add training data?

**No.** Same 29,942 games, same outcomes, finer granularity. A team-Elo update and a player-Elo update consume the *same* game result; player Elo just splits the credit five ways and lets the pieces travel independently when rosters change. What player Elo adds is not data but **portability**: a player's rating survives a team change, a sub, or an academy call-up, where team Elo has to re-learn from scratch. The gain (where it exists) comes entirely from that portability — plus one more subtle channel: in the replay, a player's rating is an average over *their* career, so a team of five veterans carries more independent history than a young org's team Elo, which only sees games played under that name.

The flip side, visible in tier-1: portability is also a leak. A player's rating carries their *former teams'* strength with it (every update uses team results), so a player who spent a year on a superteam brings inflated credit to their next team. Team Elo never has that problem — it only knows what happened under that name.

## 9. Recommendation and implementation sketch

**Do not ship player Elo in v1.3.x** — neither as a replacement (M1) nor as an added feature (M2). The model's decision domain is tier-1 + internationals (Worlds, MSI, LCK/LPL/LEC), and there both variants are significantly worse (M1 −0.0079, M2 −0.0068, both CIs exclude zero). Shipping the pooled-tail winner would make Worlds predictions worse.

**The one path that could work: a tier-conditional blend.** The evidence supports "player Elo in unstable contexts, team Elo in stable ones," but a global linear model cannot express that (M2 proves it). A conditional version would need:

1. **A context variable** — e.g. roster instability = 1 − (overlap of today's lineup with the team's trailing-30-day modal lineup), and/or league tier (tier-1 vs rest). The KeSPA Cup case is instability ≈ 1.
2. **A blend** — e.g. `elo_diff_blend = w × player_elo_diff + (1 − w) × team_elo_diff` with `w` increasing in instability, or separate LR coefficients by tier. `w` must be fit, not hand-picked.
3. **A validation bar specific to the decision domain:** tier-1+intl log-loss must be *no worse* than M0 (non-inferiority, paired bootstrap CI excluding meaningful harm), *and* the pooled gain must survive. The current analysis is the baseline to beat: M1 tier-1+intl −0.0079.

**Cheaper alternatives worth considering first:**
- **Diagnostic flag, not a feature.** Compute |player − team| gap at prediction time; if it exceeds ~100 (9% of tail games), surface a warning ("lineup differs materially from the one team Elo was built on — treat with caution / check the actual five"). Zero model risk, captures the KeSPA-type blowup. This is the highest value-per-effort item in this analysis.
- **Re-validate `roster_strength.py`.** The existing year-boundary roster adjustment is a coarser version of the same idea and likely has the same tier dependence. It feeds the app's displayed ratings. It should be checked on tier-1+intl before being trusted for Worlds — not assumed safe because it is already shipped.

**If the conditional blend is pursued, the implementation sketch is:**

- `build_features.py`: add a player-Elo replay alongside the team-Elo replay (same loop structure, per-(player, role) dict, K=24, shared expected score, year regression, debut prior = mean of rated teammates). Emit `player_elo_diff` and `roster_instability` per game. Store final player ratings in `snapshots.joblib`.
- `train.py`: add the blend/interaction feature; validate on the tier-1+intl subset explicitly, not just the pooled tail.
- `predict.py` / `export_artifacts.py`: ship the player-rating snapshot (~5,900 ratings, small JSON) plus a roster resolver (team → current 5 from the most recent game). **UX cost:** the app must know *which five* are playing — today it only asks for a team. Subs and academy call-ups would need a lineup picker, or the app silently uses the last-seen five (which is exactly the KeSPA failure mode, automated).
- `train_live.py`: unchanged — the live model consumes the pre-game logit, so any pre-game improvement flows through automatically.

**Play-In sanity check (Oct 15–16):** team-vs-player Elo gaps for the actual Play-In games were 1–24 (KC–C9: 2; FUR–C9: 1; MVK–FUR: 22–24, player slightly more pro-FURIA; KC–MVK: 20–22). Play-In lineups were stable main rosters, so player Elo would not have changed any Play-In prediction materially. The Worlds games that matter are not the ones where this helps.

## 10. Limitations

- Player Elo uses **team results only** — no per-player performance (gold share, damage, KDA). A player who is individually excellent on a bad team is rated as bad. A per-player *performance* rating (e.g. gold-share-above-expectation) is a different, potentially complementary signal — not tested here.
- The `roster_changed` flag compares to the immediately previous game only (§5 caveat); a modal-lineup comparison would be cleaner.
- Tier is a crude proxy for "roster stability / org strength." A continuous instability variable may separate the regimes better than the tier split.
- Backtests are small (n = 96, 71) and the tail's internationals subset is small (n = 119); the tier-1+intl reversal is significant on the pooled 805 games but should be re-checked as Worlds 2026 games accumulate.
- All comparisons share the same other features (eg_gold, draft, etc.); interactions between player Elo and draft features (e.g. comfort is team-champion, not player-champion) were not explored.

## Appendix: reproduction

Scratch pipeline (not shipped): replay in `/tmp/player_elo/` (scripts deleted with the VM's /tmp; rebuild from this report's §2 spec). Inputs: `data/raw/2024.csv`, `2025.csv`, `2026.csv` + `data/processed/features.csv`. Key numbers to reproduce: M0 RAW 0.6332, M1 RAW 0.6259, M1 tier-1+intl −0.0079 (CI [−0.0153, −0.0005]), M1 minor +0.0121 (CI [+0.0059, +0.0184]), KeSPA Cup 40.0% of total gain.
