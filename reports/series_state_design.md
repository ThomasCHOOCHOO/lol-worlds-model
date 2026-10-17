# Stateful series model — design

**Status:** design + prototype. Nothing here is shipped. The momentum/adaptation
parameters are **placeholders** — their fitted values come from the separate
momentum analysis. This doc defines the interface that analysis must fill.

**Motivation.** Today the app converts a fixed per-map `p` into a series prob
with the binomial (`seriesProb` in `src/model/predict.ts`, `series_prob` in
`src/predict.py`): independent maps, same `p` every game. That is wrong in
three ways it is easy to see in a real series:

1. **Information arrives.** After map 1, we know more about which team is
   better *today* than the pre-series estimate said. A 3–0 favorite that
   loses map 1 in 24 minutes is not the same team it was at 0–0.
2. **Score changes incentives and drafts.** A team down 0–2 drafts
   differently, picks sides differently, and (in Fearless) has fewer
   champions left.
3. **Sides are not random.** The loser of a game takes blue 62.2% of the
   time (1,798 transitions, 2024–2026, §5), so the side of the next game is
   partly predictable from the last result — the binomial's side-averaged
   `p_neutral` throws that away.

The replacement is a **series state machine** with a **Bayesian latent-strength
update** between games. The per-map win prob becomes `p(state)`: a function
of the score, the posterior strength, the next game's side, the next game's
draft, and (placeholder) momentum/adaptation terms.

---

## 1. State

For a best-of-N series between teams A and B (`need = N//2 + 1`):

| Symbol | Meaning |
|---|---|
| `(a, b)` | Maps won by A / B so far. Terminal when `a = need` or `b = need`. |
| `Δ ~ N(μ, 1/λ)` | Latent neutral-ground logit advantage of A over B. Posterior mean `μ`, precision `λ`. |
| `s` | Side logit bonus: half the logit gap between A's win prob as blue and as red (from the pre-series model, §2). |
| `γ[a][b]` | Momentum term: extra logit for A when the score is `(a, b)`. **Placeholder — all 0 until the momentum analysis fits it.** |
| `δ` | Within-series streak term (logit per net consecutive win). **Placeholder — 0 until fitted.** Alternative to `γ`; the momentum analysis picks one, not both. |
| `λ_pool` | Fearless pool-depletion coefficient (logit per net burned comfort champion). **Placeholder — 0 until fitted.** |
| `σ₀` | Prior SD of `Δ` before map 1. **Placeholder — 0.5 in the prototype.** Controls how fast the model changes its mind. |
| `π_loser_blue` | P(loser of a game takes blue next game). Empirical 0.622 (§5). |

`A` is fixed for the whole series (the team that was blue in game 1, in the
prototype; any fixed convention works — the app should use the first-listed
team). All probs below are P(A wins …).

## 2. Pre-series initialization

From the existing pre-draft (nodraft) model, compute both orientations:

```
p_A_blue = nodraft(elo_A − elo_B, eg_A − eg_B, blue_edge)
p_A_red  = 1 − nodraft(elo_B − elo_A, eg_B − eg_A, blue_edge)
p_neutral = (p_A_blue + p_A_red) / 2
s = (logit(p_A_blue) − logit(p_A_red)) / 2        # side logit bonus for A
μ₀ = logit(p_neutral)                              # prior mean
λ₀ = 1 / σ₀²                                       # prior precision
```

`s` is the model's own blue-side bonus for this matchup (≈ +0.04 to +0.15
logit in the prototype series). It is held fixed through the series — the
side effect is a property of the game, not of the day's form.

## 3. Per-map win prob

For the next game, given state `(a, b)`, posterior `(μ, λ)`, side, and draft:

```
p_map = σ( μ + side_term + γ[a][b] + δ·streak_A + draft_lift − λ_pool·depl_net )
```

- `side_term = +s` if A is blue, `−s` if A is red, `0` if the side is not
  yet known (used inside the lookahead DP, §4).
- `streak_A` = A's current consecutive-map win streak minus B's (0 if neither
  team has won 2+ in a row — only one of them can be nonzero). Use `γ` **or**
  `δ·streak`, not both; the momentum analysis decides which fits.
- `draft_lift = logit(p_draft) − logit(p_predraft)` for the *upcoming* game,
  both computed from the **current** snapshot features with the existing
  full/nodraft models. This is the key separation: the draft channel reuses
  the v1.3.2 draft model unchanged, while `μ` carries what the series has
  taught us. Before the draft is known, `draft_lift = 0` in expectation.
- `depl_net = depl_A − depl_B`, where `depl_X` = number of X's
  top-comfort champions already played in this series (Fearless: a champion
  played once cannot be picked again by the same team). Comfort comes from
  the existing `team_champ_comfort` snapshot table. `λ_pool = 0` until fitted.

## 4. Series win prob from a state — DP over the state machine

`P(A wins series | a, b, next_blue_is_A, μ, λ)` is computed by backward
induction over `(a, b, side_of_next_game)`:

```
V(a, b, blueA) = 1                          if a = need
               = 0                          if b = need
               = p·V_A + (1−p)·V_B          otherwise
  p  = σ( μ + (±s or 0) + γ[a][b] )          # draft/momentum terms as in §3,
                                              # draft_lift = 0 for future games
  V_A = (1−π)·V(a+1, b, True) + π·V(a+1, b, False)    # A wins; loser (B)
                                                      # takes blue w.p. π
  V_B = (1−π)·V(a, b+1, False) + π·V(a, b+1, True)    # B wins; loser (A)
                                                      # takes blue w.p. π
  π = π_loser_blue = 0.622
```

The state space is tiny (≤ 9 score states × 2 sides for Bo5). Two deliberate
simplifications, both flagged for later work:

1. **`(μ, λ)` are held fixed during the lookahead.** The DP does not
   anticipate future belief updates — it projects "if strength is what we
   think now, how does the score evolve". Anticipatory updating (a full
   POMDP) is overkill for 2–4 remaining games; the update happens for real
   after each map.
2. **Future drafts are neutral in expectation** (`draft_lift = 0`). The
   draft only enters the *next* game's `p_map`, once it is known.

This DP **replaces** `seriesProb`/`series_prob` for any series that has
started. Before map 1 it reduces to (almost) the binomial: with `γ = 0`,
`δ = 0`, `λ_pool = 0` and side averaged, the DP ≈ binomial on `p_neutral`
(the small difference is the side-transition structure: the DP knows the
loser tends to take blue, the binomial does not).

## 5. Between-game update — the core of the design

After a map ends, two things are observed:

- **The binary outcome** `y ∈ {0,1}` (1 = A won the map).
- **The live model's closing estimate** `c` = live-model P(A wins) at the
  last bucket ≤ game end (t ∈ {10,15,20,25}; §7 for the exact definition).

### 5.1 Why the closing estimate is a *weight*, not a second observation

`y` and `c` are not independent evidence: a team that stomps wins, and a
team that barely wins usually had `c` near 0.5. Counting both as separate
observations would double-count the same game. Instead `c` sets **how much
the game counts**: a decisive game is more informative about underlying
strength than a coin-flip game.

```
decisiveness weight:  w = 0.5 + |c − 0.5|        # ∈ [0.5, 1.0]
```

`w = 0.5` for a game that was a toss-up at the last bucket; `w → 1` for a
game that was already decided. The mapping `w(c)` is a **placeholder
function** — the momentum analysis should calibrate it (e.g. by fitting the
update's predictive log-loss on historical series), but any monotone
`w: [0,1] → [0.5,1]` preserves the no-double-counting property.

### 5.2 The update (one-step Laplace / Newton for logistic regression)

Treat the map as one Bernoulli observation with predicted prob `p̂`
(the §3 `p_map` **including** that game's side, momentum, and draft terms —
so the update attributes to `μ` only what those terms don't explain):

```
p̂  = σ( μ + side_term + γ[a][b] + δ·streak + draft_lift − λ_pool·depl_net )
λ' = λ + w · p̂(1 − p̂)          # precision grows by the weighted Fisher info
μ' = μ + w · (y − p̂) / λ'       # mean moves toward the surprise
```

This is the standard online Bayesian logistic update (one Newton step on
the log-posterior). Worked example: `μ = 0.50`, `λ = 4.0` (σ₀ = 0.5),
A loses as blue with `p̂ = 0.65`, closing `c = 0.34` (A was losing at t=25)
→ `w = 0.66`, `λ' = 4 + 0.66·0.65·0.35 = 4.15`,
`μ' = 0.50 + 0.66·(0 − 0.65)/4.15 = 0.50 − 0.103 = 0.397`.
One map moves `μ` by ~0.1 logit — deliberately conservative with σ₀ = 0.5.

### 5.3 What the update does NOT do

- It does **not** touch the global Elo ratings. `μ` is a series-local
  posterior ("how good is A vs B *today*"). Elo updates happen in the
  normal refresh pipeline, after the series, from the final result.
- It does **not** revise `s`, `γ`, `δ`, `λ_pool`, or `σ₀`. Those are
  global parameters, fitted offline.
- It does **not** use the final gold/kill margin directly — only through
  `c`, the live model's summary of the game state. (A future variant could
  add a Gaussian margin observation `gold_diff ~ N(k·Δ, τ²)`; that needs its
  own calibration and is out of scope here.)

## 6. Parameter interface (for the momentum analysis)

The momentum/adaptation analysis must return fitted values — with
uncertainty — for exactly these:

| Parameter | Prototype value | Meaning of a nonzero value |
|---|---|---|
| `γ[a][b]` (9 entries, Bo5) | 0 | Score-state momentum, e.g. `γ[2][0] > 0` = leading 2–0 helps beyond strength. |
| `δ` | 0 | Per-net-streak logit shift (use instead of `γ`). |
| `λ_pool` | 0 | Logit penalty per net burned comfort champion (Fearless). |
| `σ₀` | 0.5 | Prior SD of `Δ`. Smaller = slower to change its mind. |
| `w(c)` | `0.5 + \|c − 0.5\|` | Decisiveness → observation weight mapping. |
| `π_loser_blue` | 0.622 (fitted, §7) | Side-selection rule. |

**Estimation contract.** Fit `γ`/`δ` by logistic regression on historical
maps `k ≥ 2` with predictors: pre-map `p̂` (offset or covariate), score-state
dummies, streak, and side — **never** by raw win rates by score state.
Raw score-state rates are poisoned by selection: conditioning on a match
lasting ≥ 3 games forces the (1,0) cell to be mostly losses (a Bo3 that
reaches game 3 is one where the leader lost game 2). See §7 for the
demonstration. The regression must also be run separately by format
(Bo3 vs Bo5) or include a format term, since the score states mean
different things in each.

## 7. Prototype — 4 completed Bo5s, Worlds 2026 Play-In

Method: pre-series `p_neutral`/`s` from the v1.3.2 nodraft model on each
series' game-1 features; `μ₀ = logit(p_neutral)`, `σ₀ = 0.5`; after each
map, the §5.2 update with `c` = live-model prob at the last bucket ≤ game
end (all four series' games lasted past 25 min, so `c` is the t=25 bucket
in every case — a proxy for the true closing estimate, see §8). `p_map`
columns: `p_map_nodraft`/`p_map_draft` are the *static* v1.3.2 per-map probs
for that game (blue perspective, from that game's own features);
`p_state` is the stateful per-map prob (A perspective). Series probs are
P(A wins series) from the pre-game state.

### 7.1 γ = 0 (pure strength updating — the safe default)

**KC (A) vs C9 (B), Oct 15 — KC wins 3–2.**
Pre-series: `p_A_blue = 0.653`, `p_A_red = 0.593`, `p_neutral = 0.623`,
`s = +0.127`, static Bo5 P(KC) = 0.722.

| Game | Score | Blue | p_map (nodraft/draft) | p_state (KC) | Closing c (KC) | Result | P(KC series): static | stateful |
|---|---|---|---|---|---|---|---|---|
| G1 | 0–0 | KC | 0.653 / 0.669 | 0.653 | 0.339 | C9 | 0.722 | 0.731 |
| G2 | 0–1 | C9 | 0.407 / 0.536 | 0.568 | 0.257 | KC | 0.722 | 0.456 |
| G3 | 1–1 | KC | 0.653 / 0.649 | 0.646 | 0.379 | C9 | 0.722 | 0.685 |
| G4 | 1–2 | C9 | 0.466 / 0.433 | 0.564 | 0.661 | KC | 0.722 | 0.331 |
| G5 | 2–2 | KC | 0.653 / 0.669 | 0.640 | 0.743 | KC | 0.722 | 0.640 |

Read: KC's series prob swings 0.73 → 0.46 → 0.69 → 0.33 → 0.64 as the
series goes back and forth. The static column never moves. Note G2: the
static draft model liked KC (0.536 as red) but the stateful model, having
just watched KC lose G1, said 0.568 — close, because one loss with σ₀ = 0.5
only moves `μ` by ~0.1. Final `μ = +0.50` (KC still rated above C9 after
winning 3–2, barely changed from `μ₀ = +0.50` — the series went as the
prior roughly expected).

**MVK (A) vs FURIA (B), Oct 15 — MVK wins 3–0 (upset).**
Pre-series: `p_neutral = 0.338`, static Bo5 P(MVK) = 0.217.

| Game | Score | Blue | p_map (nodraft/draft) | p_state (MVK) | Closing c (MVK) | Result | P(MVK series): static | stateful |
|---|---|---|---|---|---|---|---|---|
| G1 | 0–0 | MVK | 0.372 / 0.376 | 0.373 | 0.339 | MVK | 0.217 | 0.229 |
| G2 | 1–0 | MVK | 0.401 / 0.376 | 0.396 | 0.685 | MVK | 0.217 | 0.477 |
| G3 | 2–0 | MVK | 0.407 / 0.376 | 0.420 | 0.878 | MVK | 0.217 | 0.786 |

Read: the stateful model adapts to the upset in progress — 0.23 → 0.48 →
0.79 — while the static model still says 0.217 before the clincher. But
note the final `μ = −0.36`: even after a 3–0 sweep, MVK is *still* rated
below FURIA on neutral ground (prior `μ₀ = −0.66`). Three maps with
σ₀ = 0.5 are not enough to flip the strength estimate; the series-prob
rise comes mostly from banking the 2–0 scoreline, not from believing MVK
got better. Whether that conservatism is right is exactly what σ₀
calibration decides.

**KC (A) vs MVK (B), Oct 16 — KC wins 3–0.**
Pre-series: `p_neutral = 0.662`, static Bo5 P(KC) = 0.783.

| Game | Score | Blue | p_map (nodraft/draft) | p_state (KC) | Closing c (KC) | Result | P(KC series): static | stateful |
|---|---|---|---|---|---|---|---|---|
| G1 | 0–0 | KC | 0.695 / 0.672 | 0.695 | 0.867 | KC | 0.783 | 0.792 |
| G2 | 1–0 | KC | 0.753 / 0.712 | 0.708 | 0.685 | KC | 0.783 | 0.905 |
| G3 | 2–0 | MVK | 0.372 / 0.293 | 0.652 | 0.849 | KC | 0.783 | 0.967 |

Read: G3 is the interesting row. The static per-map model says KC-as-red is
a 0.293–0.372 underdog (the side + matchup features dominate). The
stateful model says 0.652 — two convincing KC wins outweigh the red side.
KC won. This is the update doing its job: it trusts what it just watched
over the pre-series features.

**FURIA (A) vs C9 (B), Oct 16 — FURIA wins 3–1.**
Pre-series: `p_neutral = 0.463`, static Bo5 P(FURIA) = 0.430.

| Game | Score | Blue | p_map (nodraft/draft) | p_state (FURIA) | Closing c (FURIA) | Result | P(FURIA series): static | stateful |
|---|---|---|---|---|---|---|---|---|
| G1 | 0–0 | FURIA | 0.474 / 0.536 | 0.474 | 0.224 | C9 | 0.430 | 0.434 |
| G2 | 0–1 | C9 | 0.597 / 0.540 | 0.430 | 0.557 | FURIA | 0.430 | 0.225 |
| G3 | 1–1 | FURIA | 0.474 / 0.433 | 0.470 | 0.685 | FURIA | 0.430 | 0.444 |
| G4 | 2–1 | FURIA | 0.548 / 0.536 | 0.490 | 0.743 | FURIA | 0.430 | 0.736 |

Read: after dropping G1, FURIA's series prob halves (0.43 → 0.23). The
climb back is driven by the score (2–1) more than by `μ` — final
`μ = −0.00`, i.e. the series taught the model the teams were even, not
that FURIA was better.

### 7.2 Illustrative momentum (PLACEHOLDER γ — NOT fitted, do not use)

Same engine with hand-picked `γ[2][0] = +0.15`, `γ[0][2] = −0.15`,
`γ[2][1] = +0.08`, `γ[1][2] = −0.08`, `γ[1][0] = +0.05`,
`γ[0][1] = −0.05`. Effect is small, as it should be for 0.05–0.15 logit
shifts: e.g. MVK–FURIA G3 series prob 0.786 → 0.805; KC–MVK G3 0.967 →
0.972; FUR–C9 G2 0.225 → 0.215. This run exists only to show the
plumbing works end-to-end — the numbers carry no evidence.

### 7.3 Per-map log-loss (15 games — descriptive, NOT evidence)

| Model | Mean per-map log-loss |
|---|---|
| Static per-map (pre-series neutral p + side) | 0.712 |
| Stateful, γ = 0 | 0.703 |
| Stateful, illustrative γ | 0.695 |

n = 15 games from 4 series. The ordering is in the right direction and
the magnitudes are sane, but this sample cannot distinguish the models —
the standard error on a 15-game log-loss difference is far larger than
0.01–0.02. The real test is the backtest in §9.

### 7.4 Empirical side rule

Across all multi-game matches in `features.csv` (2024–2026, grouping by
match-prefix verified: all 1,136 groups have consistent team pairs):

- 1,798 game-to-game transitions; **loser takes blue 62.2%** of the time,
  winner stays blue 37.8%.

The Play-In series suggested "winner stays blue" (7/8 transitions) — a
small-sample fluke. The DP uses `π_loser_blue = 0.622`.

### 7.5 The selection-bias trap (why §6 forbids raw score-state tables)

A naive table of "favorite win rate by score state" on the same data
gives (0,1) → 85.7% and (1,0) → 24.5%. Both are artifacts: the table
conditions on matches lasting ≥ 3 games, and a Bo3 that reaches game 3 is
by construction one where the game-2 leader lost. The clean,
unconditioned numbers are: in game 2, the game-1 winner wins again 63.1%
if it was the favorite and 54.8% if it was the underdog — a raw gap that
still confounds strength with momentum, since game-1 winners are
disproportionately the stronger team. Only the §6 regression (controlling
for pre-map `p̂`) can separate the two.

### 7.6 Fearless pool depletion (observed, not yet modeled)

Distinct champions used per team: KC–C9 25/25 (5 games), MVK–FUR 15/15
(3 games), KC–MVK 15/15, FUR–C9 20/20. In a 5-game Fearless series each
team burns through 25 distinct champions — the pool constraint is real,
and the `λ_pool` term has material to bite on. The prototype reports the
counts; it does not fit `λ_pool`.

## 8. Runtime data needs — what the updater needs, and what exists today

| Input | Exists today? | Notes |
|---|---|---|
| Series score `(a, b)` | **No.** Live tab auto-advances games (`fetchLiveGame`) and shows "Between games" / "Match over", but keeps no score state. | New: track score from observed game outcomes (winner from the final frame / game state). |
| Per-game sides (blue/red per game) | **Yes.** `fetchLiveGame` returns per-game `blueCode`/`redCode`. | Use directly for `side_term`. |
| Next game's draft (picks + bans) | **Partially.** Predict tab has manual draft entry (`DraftInput`); Live tab has no draft fetch. | Runtime options: (a) manual entry in the Live tab between games; (b) fetch draft from the lolesports match endpoint if/when it exposes picks/bans — verify before building on it. Until one exists, `draft_lift = 0` and the updater still works. |
| Live-model closing estimate `c` | **Partially.** `liveProba` + `snapshot` give the live prob at any poll; the t=25 bucket is the last bucket. | New: capture the **last in-game frame's** live prob when the game ends (the poll loop currently just moves on). Caveat: for games ending after 25 min, the t=25 bucket is a proxy — the final minutes are unmodeled. A t=30/35 bucket or continuous-time live model would tighten this; out of scope for v1. |
| Champions used per team (Fearless) | **No.** | New: record each game's picks per team (from the draft, when available) and accumulate. Needs the same draft source as above. |
| Pre-series `p_neutral`, `s` | **Yes.** `seriesConversion` already computes both orientations. | Reuse; store `μ₀`, `λ₀`, `s` in the series state at series start. |
| `team_champ_comfort` (for `depl`) | **Yes, in artifacts** (`TEAMS` / comfort table). | Reuse when `λ_pool` is fitted. |

## 9. Validation plan (before shipping)

1. **Backtest on historical series.** Reconstruct every Bo3/Bo5 in
   `features.csv` (match-prefix grouping, §7.4). For each game `k ≥ 1`,
   compute the stateful series prob from the pre-game state (using
   as-of-that-date model snapshots — *not* the current ones) and the
   static binomial prob. Score both on series outcome (log-loss, Brier)
   and on next-map outcome. The stateful model ships only if it beats
   static out-of-sample by a margin that survives a paired test.
2. **Calibrate σ₀ and w(c)** on the same backtest (grid search σ₀ ∈
   {0.3, 0.5, 0.75, 1.0}; compare `w(c)` mappings). Guard against
   overfitting: one global σ₀, not per-team.
3. **Series-prob calibration.** Bin stateful series probs (e.g. 0.2–0.4,
   0.4–0.6, …) and check observed series-win rates. A stateful model
   that is overconfident after one map (σ₀ too large) will show up here.
4. **Fit γ/δ/λ_pool** via the §6 regression on the same historical
   series, then re-run (1) with fitted values. If the fitted terms don't
   improve (1), ship with them at 0 — the strength update alone is the
   core improvement.
5. **Live shadow test.** Run the updater alongside the Live tab for a
   real series (e.g. the next Play-In/Swiss Bo5) without showing it to
   the user; check that score tracking, side detection, and closing-`c`
   capture all fire correctly.

## 10. Implementation sketch (app)

New module `src/model/seriesState.ts` (pure TS, no UI):

```ts
export interface SeriesState {
  teamA: string; teamB: string; bestOf: number; need: number;
  a: number; b: number;              // score
  mu: number; lambda: number;        // posterior over Δ
  s: number;                          // side logit bonus (fixed)
  gamma: Record<string, number>;      // score-state momentum (default {})
  delta: number;                      // streak term (default 0)
  lambdaPool: number;                 // Fearless coefficient (default 0)
  streakA: number;                    // net consecutive wins (A+ / B−)
  usedChamps: Record<string, string[]>; // per-team champions used
  nextBlue: string | null;            // side of next game, if known
  history: SeriesGameRecord[];        // per-game: p_map, closing c, outcome
}

export function initSeries(a, b, bestOf, pNeutral, s, params): SeriesState
export function mapWinProb(state, draftLift): number      // §3
export function seriesWinProb(state): number             // §4 DP
export function recordGame(state, outcome, closingC, side, draftLift, picks): SeriesState  // §5
```

Wiring:

- **Live tab:** on game end, call `recordGame` with the observed winner
  and the last frame's live prob; between games, show
  `seriesWinProb(state)` + `mapWinProb` for the next game (with
  `draftLift = 0` until a draft is entered). Persist `SeriesState` in the
  existing `storeGet`/`storeSet` mechanism so a refresh mid-series doesn't
  lose it. Game-end detection: the poll loop already notices when the
  current game leaves `inProgress` — capture the last `snapshot` before
  advancing.
- **Predict tab (series mode):** `seriesConversion` stays as the
  pre-series estimator (it *is* `initSeries` + `seriesWinProb` at 0–0);
  once a series has started, the Live tab's stateful number supersedes it.
- **Tests:** unit tests for the DP (0–0 Bo5 with `μ = 0` must equal the
  binomial at `p = 0.5` up to the side-transition structure; terminal
  states return 0/1); update tests (a win raises `μ`, a loss lowers it,
  larger `|c − 0.5|` moves `μ` more); a golden test replaying KC–C9 with
  the §7.1 numbers.

## 11. Limitations / open questions

- **σ₀ is a prior, not a fact.** 0.5 is a reasonable default (one map ≈
  0.1 logit of belief movement), but the backtest (§9) must pick it.
  Too large → the model overreacts to one map; too small → the update is
  cosmetic.
- **The DP holds (μ, λ) fixed in lookahead.** The true value of being up
  2–0 includes the *option* to learn more from future maps; ignoring it
  slightly understates the leader's edge. Acceptable for ≤ 4 remaining
  games.
- **Closing `c` is a t≤25 proxy.** Games ending after 25 min have
  unmodeled final minutes. The weight `w(c)` inherits that staleness.
- **Draft lift uses current snapshots.** Within a series the snapshot
  features (Elo, comfort) are stale by construction — that is fine, since
  `μ` absorbs series-specific information and the draft model only needs
  *relative* pick/ban values.
- **Momentum and adaptation are entangled.** A team down 0–2 may play
  worse (momentum) *and* draft worse (adaptation) *and* be revealed as
  weaker (strength). The §6 regression separates them only as well as the
  covariates allow; the design is robust to that ambiguity because the
  strength update works even if `γ = δ = λ_pool = 0`.
- **Bo1s are unaffected.** With no between-game updating, Bo1s keep the
  current per-map model. The stateful machinery only engages for Bo3/Bo5.

## Appendix — reproduction

Prototype script: `/tmp/series_proto.py` (scratch, not in the repo).
Inputs: `data/processed/features.csv`, `data/raw/2026.csv`,
`model/model.joblib` (v1.3.2 nodraft + full), `model/live_models.joblib`.
Key outputs quoted in §7 were produced by that script; the per-game
`p_map_nodraft`/`p_map_draft` are blue-perspective (P(blue wins)),
`p_state`/closing `c` are A-perspective.
