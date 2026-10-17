# Lane-by-lane: do winning lanes produce a gold differential?

**Follow-up to** `lane_gold_analysis.md` and `counterpick_analysis.md`, per the
user's refinement: **do not pool across the board — compare top vs top,
mid vs mid, ADC+sup vs ADC+sup, then add jungle to each unit.** Plus: is
+249 gold per draft-winning lane actually substantial, given how strongly
gold predicts wins?

**Short answers:**
1. **Lane-by-lane is much stronger than pooling suggested — for top.**
   Top matchup edge correlates **0.34** with top-lane gold (partial R²
   **0.113**), vs the pooled summed-edge correlation of 0.17 with team gold.
   Mid (0.20 / 0.040) and bot 2v2 (0.15 / 0.020) are weaker. Pooling hid
   top's signal in other lanes' noise.
2. **Adding jungle does NOT make the effect more notable.** Combined-unit
   partial R² *falls* (top 0.113 → top+jng 0.083; mid 0.040 → 0.037;
   bot 0.020 → 0.023, flat). The jungle edge itself is valid signal
   (slope 0.86 for its own gold, plus a small gank effect on top gold),
   but jungle gold is noisy, so merged units get noisier, not stronger.
3. **Bot-lane gold matters most for winning; top gold least** — the
   opposite of predictability. Bot 2v2 gold has the largest standardized
   win coefficient (0.89) and the biggest drop-one log-loss cost (+0.048);
   top is smallest (0.46 / +0.016).
4. **+249 gold is real and quantitatively coherent — worth ~+3.2pp win
   prob per lane — but it is a small mean shift on a huge noise
   distribution (SD 3,146).** That is why the mechanism is real and the
   predictive lift is still small. See §6 for the full reconciliation.
5. **The +249 survives the missing-data fix unchanged** (+244 naive →
   +249 clean). The ~8% NaN-gold games are all-or-nothing per game and
   only 1.8% of 2026 games.

**Verdict: NO-SHIP** for a per-unit matchup feature. The one exception to
watch: **top-only edge** is the sole variant whose CI excludes zero
(−0.00044 log-loss), but it is 4.5× below the 0.002 gate, accuracy is flat,
and it is 1 of 4 variants tested — a lead, not a ship.

---

## 1. Method

### Data
Oracle's Elixir per-player rows, `data/raw/2024.csv + 2025.csv + 2026.csv`,
joined to the game spine in `data/processed/features.csv` (Elo diff,
6 v1.3.2 features, outcome). Eval set: **9,524 complete 2026 games**
(all 5 positions present, gold not all-NaN — see §2).

### Matchup edges: expanding walk-forward table (stronger than the prior reports)
Both prior reports used a **fixed** table (2024–2025) to predict 2026.
Here the table is **expanding**: a game's edge uses all *earlier* games
only (2024–2025 plus earlier 2026). This is strictly walk-forward
(no leakage) and, as §3 shows, strictly better — same-era games carry
more relevant matchup information (consistent with the counterpick
report's finding that cross-era edges barely persist).

Two edge variants, mirroring the two prior reports:
- **Raw (primary, lane-gold style):** `edge = Σ gold_diff / (n + 3)`,
  shrinkage k=3 toward 0. Elo handled as a regression control.
- **Elo-residualized (robustness, counterpick style):** gold residuals
  from per-lane OLS `golddiffat15 ~ elo_signed` on the fit set
  (< 2026-07-01), shrunk with k=20. Win edges likewise residualized
  (residual = `y − P(win|elo)` from a fit-set Elo-only LR).

Edges are stored canonically by `(position, champA, champB)` with sign
flipped by side, so both orientations of a matchup pool into one estimate.
Unseen pairs → 0.

### Units
| Unit | Gold outcome (blue perspective) | Edge |
|---|---|---|
| top 1v1 | top `golddiffat15` | top edge |
| mid 1v1 | mid `golddiffat15` | mid edge |
| bot 2v2 | bot + sup `golddiffat15` | bot edge + sup edge (additive, **not** a 2v2-pair lookup — the counterpick report showed pair tables are unusable: 16,062 pairs, none significant) |
| top+jng 2v2 | top + jng gold | top edge + jng edge |
| mid+jng 2v2 | mid + jng gold | mid edge + jng edge |
| bot+jng 3v3 | bot + sup + jng gold | bot + sup + jng edges |

`golddiffat15` is the player's gold diff vs their lane opponent,
own-side perspective; blue-perspective lane gold = the blue player's
value (or −red's if blue's is missing).

### Regressions
Per unit: `unit_gold ~ unit_edge + elo_diff` (raw variant). Reported:
slope, 95% CI, correlation, full R², and **partial R²** of the edge
(variance explained beyond Elo alone — the number comparable to the
pooled 0.010). The residualized variant is `resid_gold ~ resid_edge`
(no Elo control needed — both sides already residualized).

---

## 2. Data-quality flag: the ~8% NaN-gold games

The parent's check verified: **7.96% of player rows have
`golddiffat15 = NaN`, and NaN is all-or-nothing per game** — a game
either has gold data for all 10 players or for none. A naive groupby
sum gives exactly 0 for those games (7.98% of games sum to exactly 0).

| Year | Games | All-NaN gold games | Share |
|---|---|---|---|
| 2024 | 10,204 | 1,393 | 13.7% |
| 2025 | 10,044 | 820 | 8.2% |
| 2026 | 9,694 | 170 | 1.8% |

**Handling:** all analyses here exclude all-NaN games (rather than
treating them as 0-gold games). The per-unit regressions additionally
require the unit's gold to be observed (automatic once all-NaN games
are excluded, since NaN is all-or-nothing).

**Does it change +249? No.** Exact replication of the lane-gold method
(fixed 2024–25 table, k=3, Elo control) on 2026 games:

| Version | n | Slope (gold per winning lane) | 95% CI |
|---|---|---|---|
| Naive (NaN → 0), as originally run | 9,694 | +244 | [191, 297] |
| Excluding all-NaN games | 9,524 | **+249** | **[196, 303]** |

The exclusion moves the estimate by **+5 gold (2%)** — the original
+249 was already the clean number, because 2026 gold coverage is 98.2%.
The fix matters far more for 2024 (13.7% NaN): any analysis pooling
2024–2026 gold without excluding all-NaN games attenuates its slopes
toward zero, since ~14% of 2024 games contribute a fake 0 regardless
of matchup. **All future gold analyses must exclude all-NaN games**
(or model missingness explicitly). Note the earlier claim in the
conversation that "team gold@15 SD ≈ 4,700" was computed on the naive
all-years sample; the clean 2026 SD is **3,146** (§6).

---

## 3. Per-unit results: lane-by-lane beats pooling (for top)

### Main regression (raw edges, k=3, Elo control; eval 2026, n=9,524)

| Unit | Slope (actual gold per 1 predicted) | 95% CI | Corr | Partial R² |
|---|---|---|---|---|
| **top 1v1** | **1.053** | [0.994, 1.112] | **0.340** | **0.1126** |
| mid 1v1 | 0.811 | [0.732, 0.891] | 0.199 | 0.0399 |
| bot 2v2 | 0.861 | [0.739, 0.982] | 0.147 | 0.0198 |
| top+jng 2v2 | 1.102 | [1.028, 1.176] | 0.294 | 0.0829 |
| mid+jng 2v2 | 0.901 | [0.809, 0.993] | 0.195 | 0.0373 |
| bot+jng 3v3 | 0.904 | [0.786, 1.023] | 0.159 | 0.0229 |

Pooled comparators: count-of-winning-lanes partial R² ≈ 0.010;
summed-edge R² = 0.029 (lane-gold report).

**Reading the table:**
- **Top is the standout.** Its matchup edge explains **11.3%** of
  top-lane gold variance beyond Elo — 11× the pooled count-R² and ~4×
  the pooled summed-edge R². Top is the most isolated 1v1 lane: no
  support, fewest roams, purest matchup expression. Pooling top's
  signal into team gold buried it under four other lanes' noise.
- **Slopes ≈ 1 mean the edges are well-calibrated in gold units**
  (1 gold predicted ≈ 1 gold realized). Top's CI covers 1.0 exactly.
  Mid (0.81) and bot (0.86) slightly overstate — their paper edges run
  a bit hot, consistent with mid/bot being more gank- and roam-shaped
  than the 1v1 table captures.
- **Jungle inclusion dilutes.** Top 0.113 → top+jng 0.083; mid 0.040 →
  0.037; bot 0.020 → 0.023 (flat). See §4 for why — the jungle edge
  is real, but jungle gold is noisy, so merged units get noisier.

### Robustness: Elo-residualized edges (k=20)
| Unit | Resid slope | 95% CI | R² |
|---|---|---|---|
| top 1v1 | 1.355 | [1.276, 1.433] | 0.1067 |
| mid 1v1 | 1.242 | [1.124, 1.359] | 0.0431 |
| bot 2v2 | 1.125 | [0.958, 1.292] | 0.0180 |
| top+jng | 1.475 | [1.374, 1.576] | 0.0795 |
| mid+jng | 1.347 | [1.210, 1.484] | 0.0376 |
| bot+jng | 1.234 | [1.064, 1.403] | 0.0209 |

The R²s agree with the raw variant (top 0.107 vs 0.113). The slopes
are > 1 **not** because the effect is stronger, but because k=20
shrinks the edges harder than k=3: the regressor is compressed toward
0, so the fitted slope compensates upward. Compare R² across
variants, not slopes.

### Expanding table beats fixed table (same-era games help)
| Unit | Fixed-table (2024–25) corr | Expanding-table corr |
|---|---|---|
| top | 0.298 | **0.340** |
| mid | 0.149 | **0.199** |
| bot 2v2 | 0.117 | **0.147** |

Every unit improves when the table includes earlier same-year games —
directly consistent with the counterpick report's non-persistence
finding (cross-era edges correlate only 0.17). A matchup table is a
perishable good: it must be rebuilt continuously, not frozen.

### Coverage (eval 2026, expanding table)
| Unit | All components n ≥ 1 | n ≥ 5 | n ≥ 20 |
|---|---|---|---|
| top 1v1 | 97.3% | 91.7% | 80.8% |
| mid 1v1 | 98.2% | 92.9% | 81.3% |
| bot 2v2 | 97.2% | 91.1% | 78.2% |
| top+jng | 94.7% | 83.6% | 64.1% |
| mid+jng | 95.5% | 84.6% | 64.4% |
| bot+jng | 94.6% | 82.6% | 61.1% |

Coverage is good for solo lanes; jungle-inclusive units are thinner
because *all* components must clear the bar.

---

## 4. Jungle: real signal, small gank channel, no R² gain

The jungle caveat matters — but it does not kill the jungle edge.
Junglers don't lane against each other, so "jungle edge" is a
farm/gank-pressure proxy, not a lane edge, and jungle gold mixes
champion classes (farmers vs gankers). Three tests separate what the
jungle edge can and cannot do:

**a) Jungle edge → jungle gold: real and calibrated.**
`gold_jng ~ e_jng + elo`: slope **0.856** [0.780, 0.932], corr **0.221**.
A jungler with a +500 paper edge actually banks ~+430 gold. The edge
is not garbage — jungle gold is just noisy (SD 935) and class-driven.

**b) Cross-lane (gank) channel: small, top-only.**
Two-regressor test — lane gold on lane edge + jungle edge:

| Outcome | Lane-edge slope | Jungle-edge slope | 95% CI (jungle) |
|---|---|---|---|
| top gold | 1.058 | **+0.141** | [0.054, 0.228] |
| mid gold | 0.813 | +0.056 | [−0.022, 0.135] |
| bot 2v2 gold | 0.863 | +0.096 | [−0.039, 0.230] |

The jungle edge significantly predicts **top** gold beyond the top
edge — the gank-pressure channel, on the isolated lane where a
jungler's visit actually moves the gold. No significant effect on mid
or bot gold (mid gets traffic from both sides; bot has a support
watching). The effect is small (+0.14 gold per 1 edge — a +500 jungle
edge is worth ~+70 top gold), but it is the one place the user's
"jungle included in all of those" hypothesis finds support.

**c) Merged units: no gain.** As §3 shows, forcing one combined edge
for top+jng / mid+jng / bot+jng does not raise R² — the combined
outcome inherits jungle gold's noise. The right modeling choice is
**not** merged units: keep the jungle edge as its own separate term
(and let the top equation absorb the small gank channel), exactly as
the counterpick report's solo-lane variant did by *dropping* jungle
rather than merging it.

**Bottom line on the jungle question:** the jungle edge carries real
signal for its own gold plus a small top-lane gank effect, but merging
it into lane units dilutes rather than strengthens. The caveat
matters for *interpretation* (never call it a "lane matchup") and for
*construction* (separate terms, not merged units) — not for validity.

---

## 5. Which unit's gold matters most for winning? Bot — by a lot

Logistic regression `y ~ elo_diff + unit golds` (per 1,000 gold),
fit on all 9,524 eval games for coefficients; honest H1→H2 for
drop-one-unit log-loss:

| Unit gold | Coef / 1,000 gold | Standardized (coef × SD) | Drop-one ΔLL (H1→H2) |
|---|---|---|---|
| bot 2v2 | 0.539 | **0.888** | **+0.0479** |
| mid | 0.575 | 0.556 | +0.0184 |
| jungle | 0.569 | 0.532 | +0.0182 |
| top | 0.416 | 0.462 | +0.0156 |

Bot-lane gold is worth roughly **3×** top-lane gold for predicting the
win (drop-one cost +0.048 vs +0.016), and its gold SD is the largest
of any unit (1,648 vs top 1,110, mid 967, jng 935, sup 532) — bot is
where games are decided.

**The asymmetry to remember: predictability ≠ value.**
Top gold is the *most* predictable from draft (corr 0.34) but the
*least* valuable for winning. Bot gold is the *least* predictable
from draft (corr 0.15) but the *most* valuable. A feature's worth is
predictability × value, and the draft-matchup feature is strongest
exactly where gold matters least. That is the structural reason the
feature keeps failing the ship gate despite top's strong per-unit R².

---

## 6. Reconciliation: is +249 gold per winning lane substantial?

**Yes — as a mechanism. Small — as a predictive feature. Both are true,
and the numbers reconcile exactly.**

### What +249 buys in win probability
Gold→win curve, fit on 2026 H1 (`y ~ elo_diff + team_gold`):
per-1,000-gold logit slope **0.5145**. At even Elo:

| Team gold@15 | P(blue wins) |
|---|---|
| 0 | 51.2% |
| **+249 (one winning lane)** | **54.4% (+3.2pp)** |
| +495 (three winning lanes, mean) | 57.5% (+6.3pp) |
| +1,000 | ~63.8% |

So one draft-winning lane is worth about **+3.2 percentage points** of
win probability through the gold channel. That is substantial in the
sense that a 3pp edge, if you could bet it at fair odds repeatedly,
would be a real edge. It is *not* substantial relative to the noise:
the clean 2026 team-gold SD is **3,146** (median |swing| 1,908), so
+249 is **8% of one standard deviation** of the thing it moves.

### Why P(positive gold) is only ~55% for 3 winning lanes
Three winning lanes shift the mean to +495 gold. On a normal
distribution with SD 3,146, the predicted P(gold > 0) is
Φ(495/3146) = **56.3%** — and the observed clean value is **55.1%**.
There is no mystery and no missing mechanism: a +423 mean shift on a
±3,146 noise distribution *must* give a coin-flip-ish probability.
The full bucket table (expanding-table edges, clean data):

| Gold-edge winning lanes | n | Mean gold | P(gold > 0) | Observed win | Gold-implied win |
|---|---|---|---|---|---|
| 0 | 287 | −793 | 39.4% | 47.4% | 44.0% |
| 1 | 1,510 | −346 | 43.7% | 51.8% | 47.8% |
| 2 | 3,098 | +120 | 51.2% | 52.7% | 52.0% |
| 3 | 3,024 | +495 | 55.1% | 53.8% | 55.6% |
| 4 | 1,370 | +907 | 61.8% | 57.4% | 58.9% |
| 5 | 235 | +1,570 | 68.9% | 58.7% | 64.6% |

Gold-implied win = the H1 gold model's prediction using each game's
actual Elo + realized gold. It tracks observed wins within noise
(the extreme buckets, n ≈ 250, have ±6pp sampling error; the middle
buckets track within ~2pp). The gold channel is quantitatively
coherent: draft edge → gold → win, with magnitudes that line up.

### The non-gold channel (why win edges beat gold edges)
Win edges (Elo-residualized, counterpick definition: edge > 0.02)
correlate only **0.20** with gold edges. Bucketing by *win-edge*
winning lanes, mean gold barely moves (+194 → +494) while win rate
climbs 49.0% → 60.2%:

| Win-edge winning lanes | n | Mean gold | Observed win | Gold-implied win | Gap (non-gold) |
|---|---|---|---|---|---|
| 0 | 1,430 | +194 | 49.0% | 52.6% | −3.6pp |
| 1 | 3,335 | +214 | 51.6% | 52.2% | −0.6pp |
| 2 | 2,976 | +288 | 54.9% | 54.0% | +0.9pp |
| 3 | 1,435 | +494 | 58.3% | 56.3% | +2.0pp |
| 4 | 309 | +459 | 60.2% | 54.9% | +5.3pp |
| 5 | 39 | +820 | 59.0% | 57.3% | +1.7pp |

The gap (observed − gold-implied) grows with more win-edge-winning
lanes: counterpick value increasingly comes from **non-gold channels**
— scaling curves, teamfight utility, map pressure — exactly as the
counterpick report concluded (Mundo wins games while losing lane
gold). So there are *two* draft-edge mechanisms: a small gold channel
(+249/lane, fully coherent) and a larger non-gold channel that the
gold analysis cannot see. The win-edge feature (counterpick report)
captures both and still failed the gate — the non-gold channel is
real but just as noisy.

### Plain-language summary
Gold is a *strong predictor when observed* (in-game) because by then
it summarizes everything that happened. Pre-game, the draft only
gets to *nudge the expectation* of gold by ~250 per lane, while the
actual gold outcome swings by ±3,000 for reasons the draft cannot
foresee (jungle pathing, fights, execution, roams). A real mechanism
with a 250:3,000 signal-to-noise ratio gives a small prediction lift
— that is the entire story. The pooled analysis understated top's
piece of it (11% of top-gold variance), but even top's piece is worth
only ~0.0004 log-loss, because top gold is the least valuable gold.

---

## 7. Ship / no-ship: per-unit matchup feature

Honest test: fit LR on the 6 v1.3.2 features + per-unit edges on
2026 H1 (n=5,888), evaluate on 2026 H2 (n=3,636). Same 0.002 gate.

| Model | H2 log-loss | Accuracy | Paired ΔLL | 95% CI |
|---|---|---|---|---|
| v1.3.2 features (base) | 0.6333 | 0.6460 | — | — |
| \+ top/mid/bot2v2 edges | 0.6338 | 0.6460 | +0.00051 | [−0.00058, +0.00160] |
| **\+ top edge only** | **0.6328** | 0.6460 | **−0.00044** | **[−0.00083, −0.00006]** |
| \+ all 5 lane edges | 0.6337 | 0.6455 | +0.00041 | [−0.00068, +0.00151] |
| \+ top+jng/mid+jng/bot+jng edges | 0.6338 | 0.6460 | +0.00049 | [−0.00049, +0.00146] |

**Verdict: NO-SHIP.** Every multi-unit variant is neutral-to-worse.
Top-only is the sole variant whose CI excludes zero — consistent with
top being the strongest per-unit signal — but −0.00044 is **4.5× below
the 0.002 gate**, accuracy is unchanged, and it is 1 significant result
out of 4 variants (multiple-comparison caveat: ~19% chance at least
one of four 95% CIs excludes zero under the null). Treat top-only as
the lead refinement to re-test on a fresh holdout (e.g. post-Worlds
games), not as a ship. If it replicates there with ΔLL ≤ −0.002, ship
it; otherwise the matchup-edge line is closed.

---

## 8. Limitations

- **Matchup edges pool across leagues.** A top edge estimated mostly
  from minor-league games may not transfer to LCK/LPL; no league split
  was run. The region-strength analysis (separate report) suggests
  league context matters.
- **Patch pooling within the expanding window.** The expanding table
  mixes patches; per-patch tables are too thin (n = 2–5 per matchup).
  The expanding-vs-fixed improvement (§3) suggests recency helps, but
  the patch-aware variant (prior-patch fallback, mirroring `d_power`)
  remains untested.
- **Residualization baselines** (Elo-only win LR, per-lane gold OLS)
  are fit on < 2026-07-01 and applied everywhere, including inside the
  fit set — the same negligible 2-parameter leakage the counterpick
  report accepted. The primary (raw) analysis has no such leakage.
- **Lane assignments** are Oracle's Elixir `position` labels; lane
  swaps and flex picks can mislabel a matchup.
- **Gold-implied wins** use an H1-fit model applied to all of 2026
  (H1 in-sample, H2 honest); the bucket comparison is descriptive,
  not a formal mediation analysis.
- **Top-only significance** is nominal and post-hoc (1 of 4 variants);
  it needs a pre-registered re-test, not a ship.

---

## 9. Reproduction

Scripts (scratch, not part of the model): `/tmp/lbl/analysis.py`
(expanding table, fixed-table replication, missingness audit),
`/tmp/lbl/analysis2.py` (per-unit regressions, coverage, gold→win,
model test, reconciliation), `/tmp/lbl/analysis3.py` (fixed vs
expanding, two-regressor jungle tests). Frames: `/tmp/lbl/games.csv`
(all games with walk-forward edges), `/tmp/lbl/eval2026_full.csv`
(2026 eval set). Run with
`~/workspace/lol-worlds-model/.venv/bin/python`. Read-only w.r.t.
model files; no model files were modified.
