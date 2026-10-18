# Prop pricing model — method, validation, shipped / not shipped

Fitted params: `model/prop_models.json` (v1.0.0). Code: `src/prop_models.py`.
App copy: `src/data/prop_models.json` in the app repo (players converted
dict → compact tuples; see "App copy" below). App pricing + joint sim:
`src/model/props.ts`. App docs: `docs/derivatives.md`.

## Setup

- Data: Oracle's Elixir 2024-01-01 → 2026-10-16, 29,942 games.
- Time split: fit on games before 2026-07-01 (26,211), validate on
  2026-07-01 onward (3,731 games; 310 Bo5 series; 1,164 series of any
  best-of for the total-maps model).
- Ship gate: a prop ships only if it beats its naive baseline out-of-sample
  by **> 0.002 log-loss** (same bar as the v1.4.0 analyses). Log-loss
  deltas below are validation minus naive (positive = model wins).
- All trailing stats are walk-forward (shifted, strictly before the game).
  A game-2+ map may see earlier maps of the same series — correct for
  pre-game pricing.
- Elo convention: PLAIN `elo_diff` from `features.csv` (no region offset)
  in both Python and TS, so the two price identically. Region offset is
  second-order for pace/objective props. (Series props in the app use the
  app's `predictMatchup` per-map p, which is region-adjusted — consistent
  with the Python series validation, which uses the bundle's
  `elo_diff_adj`-trained p.)

## (a) Total kills O/U per map — SHIPPED

Poisson GLM on implied total
μ0 = (pace_for_blue + pace_against_blue + pace_for_red + pace_against_red) / 2
(both teams' trailing-20 scoring/conceding pace) + |Elo diff|.
Negative-binomial dispersion by moments. Naive: one global NB.

| | LL |
|---|---|
| Model | −0.3354 |
| Naive | −0.3726 |
| **Delta** | **+0.0372** |

## (b) Player kills O/U per map — SHIPPED

Poisson GLM on the player's shrunk trailing-20 kill rate
`(sum + 3 × pos_mean) / (n + 3)` + team pace + opponent conceding pace +
signed Elo diff + position intercepts, NB dispersion. Naive: per-position
global NB.

A naive fit over-dispersed out-of-sample (validation LL worse than naive),
so the served μ is shrunk toward the position mean:
`μ = pos_mean + s × (μ_raw − pos_mean)`, with `s` tuned on a late-train
slice only (fit < 2026-01-01, tune `s` on 2026-01-01..2026-06-30 by O/U LL
over a 0.00–1.00 grid), then refit on full train; α recomputed from
shrunk-μ residuals. The TS `pricePlayerKills` mirrors the shrinkage; the
sidecar carries `shrink` + `pos_means`.

| | LL |
|---|---|
| Model (shrunk) | −0.5309 |
| Naive | −0.5386 |
| **Delta** | **+0.0077** |

## (c) First blood / dragon / tower — ALL SHIPPED

Logistic on Elo diff + both teams' trailing first-X rates. Naive: global
base rate.

| Prop | Model LL | Base-rate LL | Delta |
|---|---|---|---|
| First blood | −0.6895 | −0.6920 | **+0.0025** |
| First dragon | −0.6586 | −0.6625 | **+0.0040** |
| First tower | −0.6366 | −0.6857 | **+0.0491** |

First blood clears the gate by only 0.0005 — it is the weakest shipped
prop. Treat its prices as the least reliable of the set.

## (d) Series exact score — SHIPPED; binomial total maps — NOT SHIPPED

Exact-score: binomial with fixed per-map p from the shipped nodraft model
(same conversion as `src/predict.py` `series_prob`). Naive: empirical
exact-score distribution of training series.

| | LL |
|---|---|
| Model (exact) | −1.6818 |
| Naive | −1.7703 |
| **Delta** | **+0.0884** (n = 310) |

Total maps via the same binomial: LL −1.0811 vs naive −1.0729
(delta **−0.0082**) — the binomial maps marginal LOSES. Favorites stomp
more than fixed-p binomial says (per-map p is not really fixed: after
winning game 1, the favorite's effective p rises). So the binomial maps
distribution does **not** ship.

## (d2) Total maps, conditional on p — SHIPPED

P(total maps | per-map p) fit directly: multinomial logistic for Bo5,
binary logistic for Bo3, on `[logit(p), |logit(p)|]`. This is what the
app's standalone total-maps prop uses.

| | LL |
|---|---|
| Model | −0.7654 |
| Naive (empirical) | −0.7695 |
| **Delta** | **+0.0042** (n = 1,164) |

## Joint simulation (related parlays) — SHIPPED

Same-game-parlay legs correlate, so naive independent multiplication
misprices combos. The joint sampler draws, per map: winner ~ Bernoulli(p),
then a (kill bucket, first-blood/dragon/tower combo) cell from an
empirical joint table stratified by (map winner, p-bin, map-index bucket,
tied?). Sampling kills and firsts JOINTLY from one cell preserves the
kills↔firsts correlation inside each stratum; the winner draw preserves
the winner↔everything correlation.

Three structure choices, each found by diagnosing a validation miss:

1. **Map-index bucket (1, 2, 3, 4+).** A game-number-blind table
   overestimated kills in long series because decider maps are slower:
   game-5 maps average ~29.5 kills vs ~33.3 for game 2.
2. **Tied flag.** Level series are slower (~31.3 kills when the series is
   tied going into the map vs ~33.4 when a side leads), consistently
   across p-bins.
3. **Raking.** Each sim is weighted by
   `P_cond(maps | p) / P_binom(maps | p)` so the sim's maps marginal
   matches the shipped conditional maps model instead of the fixed-p
   binomial (which overstates long series). This is the fix that moved
   the joint sim from tied-with-naive to clearly beating it, and it
   makes the related builder's maps marginal consistent with the
   standalone total-maps prop. The app applies the identical raking.

P-bins: [0, 0.40, 0.47, 0.53, 0.60, 1.0001]. Kill buckets: [0–15],
[16–18], [19–21], [22–24], [25–27], [28–30], [31–33], [34–36], [37–39],
[40–60] (10 buckets). Firsts combos: 8 (bits: blood, dragon, tower).
80 strata × 80 cells, Laplace smoothing 0.5/cell. Kills within a bucket
sampled as a uniform integer.

### Validation

Per validation Bo5 series: run sims, score the sim's joint P for 4
correlated combos as a probabilistic forecast of the actual joint outcome
(binary LL) vs the naive product of the sim's own (raked) marginals.
Kills line per series = floor(raked sim median) + 0.5.

| Combo | Joint LL | Naive LL | Delta | Mean joint P | Actual rate |
|---|---|---|---|---|---|
| A-win + A first dragon (map 1) | −0.5123 | −0.5125 | +0.0002 | 0.232 | 0.232 |
| A-win + over kills | −0.5541 | −0.5591 | +0.0050 | 0.258 | 0.252 |
| A-win + over 3.5 maps | −0.5987 | −0.6068 | +0.0081 | 0.311 | 0.290 |
| Over 3.5 maps + over kills | −0.6786 | −0.7181 | +0.0395 | 0.435 | 0.419 |
| **Pooled** | **−0.5859** | **−0.5991** | **+0.0132** | | |

All four combos beat or tie naive, and the joint means track actual
rates closely. Before raking, the pooled delta was +0.0005 and
over-maps+over-kills lost by −0.0108 — the raking is load-bearing.

The served joint table is refit on ALL games through the snapshot
(refit-after-validation); the validation numbers above use the
train-fit table.

## What was left out

- **Binomial total-maps distribution** — fails the gate (ΔLL −0.0082).
  The app never uses it; the standalone maps prop and the sim raking
  both use the conditional maps model.
- **Series-state / prev-win dynamics inside the joint sim** — the sim
  draws each map's winner from the same fixed p (no momentum, no
  side-swap effects). The series-state model (`src/model/seriesState.ts`)
  exists for in-progress series; the pre-series joint sim intentionally
  stays fixed-p for tractability. Its maps marginal is corrected by
  raking, not by dynamics.
- **Draft-adjusted props** — all props are pre-draft. Draft features
  (champion power, comfort) are not in the prop models; the draft is
  a veto, same as the main model.

## App copy

`model/prop_models.json` → app `src/data/prop_models.json` (135,707 bytes,
< 200 KB target). The only transformation: `players` goes from a dict
keyed by Oracle's Elixir playerid (a 42-char key, ~95 KB of dead weight)
to compact tuples `[name, team, position, rate, games]` (958 players).
All fitted params are copied verbatim; nothing is omitted (every prop
shipped). Reproduce with:

```bash
python -c "
import json
s = json.load(open('model/prop_models.json'))
s['players'] = [[v['name'], v['team'], v['position'], v['rate'], v['games']]
                 for v in s['players'].values()]
json.dump(s, open('../lol-predictor-app/src/data/prop_models.json', 'w'),
          separators=(',', ':'))
"
```

## TS/Python parity (checked 2026-10-17)

| Check | Python | TS | Verdict |
|---|---|---|---|
| Total kills μ / P(over 22.5) | 25.9952 / 0.6486 | 25.9914 / 0.6484 | Match within sidecar 3-decimal rounding |
| First dragon P(blue) | 0.4326 | 0.4326 | Exact |
| Series exact @ p=0.6 | 0.2160 / 0.2592 / 0.2074 / 0.1382 / 0.1152 / 0.0640 | same | Exact |
| Player kills (Sniper, top, rate 3.381) μ / P(over 3.5) | 2.9062 / 0.3293 | 2.9062 / 0.3293 | Exact |
| Joint sim @ p=0.6, 5,000 sims: P(A wins) | binomial 0.6826 | 0.6930 | Within MC noise (SE ≈ 0.0066) |
| Joint sim: mean kills / maps | — | 132.24 / 4.05 | Sane |

## Limitations

- First blood is a marginal ship (+0.0025, gate cleared by 0.0005).
- Validation series n = 310 (Bo5) — joint-sim deltas are meaningful
  but not huge-sample.
- Pace stats are trailing-20; a roster change makes a team's pace
  stale until 20 games pass. The app's diagnostic flag (|player − team
  gap| > 100) is the visible guard, not a correction.
- The joint table is fit on 2024–2026 games; a patch-era shift in
  kills pace would stale it until refit. Refit with each model refresh.
- Props are pre-draft only and assume the listed matchup's teams as
  given — no roster-change handling beyond the trailing stats.
