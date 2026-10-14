# Optimal betting timing: pre-draft vs post-draft vs live

Question from Thomas (2026-10-13): should he bet at draft start, or wait and use live odds?

## 1. Draft-info lift (measured)

Holdout 2026-07-01 → 2026-10-01, n=3,368, same games both models (`model/metrics.json`):

| Model | Accuracy | Log-loss | AUC |
|---|---|---|---|
| Pre-draft (`nodraft`: elo, form, blue) | 0.6431 | 0.6330 | 0.6849 |
| Draft-adjusted (`full`: + d_power, d_comfort, ban_edge) | 0.6452 | 0.6310 | 0.6865 |
| **Lift** | **+0.0021 (~7 games)** | **−0.0020** | **+0.0016** |

Knowing the draft is worth **0.002 in log-loss** — real but small. Why so small: pro drafts are usually competent, so draft differentials are narrow, and Elo already absorbs most team-strength signal. The draft features' value concentrates in outlier drafts (big comfort mismatches, off-meta power picks on the patch).

## 2. Live-model strength (measured)

Per-bucket LR on [pre-game logit, gold diff/1000, kill diff], isotonic-calibrated (v1.3.0, `src/train_live.py`):

| t (min) | Live ll | Pre-game ll | Δll | Live acc |
|---|---|---|---|---|
| 10 | 0.535 | 0.631 | −0.092 | 72.0% |
| 15 | 0.475 | 0.631 | −0.147 | 76.4% |
| 20 | 0.415 | 0.632 | −0.208 | 79.1% |
| 25 | 0.360 | 0.652 | −0.282 | 82.6% |

Gold diff does almost all the work; kills add ~nothing early. Note the pre-game baseline itself degrades on long games (0.652 at t=25): games that go long were coin-flips pre-game — live state helps most exactly where pre-game is weakest.

**Hard constraint: no model read before t=10.** "Wait 5 minutes into the game" is not supported — the live choice starts at the 10-minute bucket.

## 3. Where the edge vs the market should peak (structural)

Key assumption: Pinnacle's pre-match line is approximately efficient; Thomas's edge comes from model calibration plus patch-aware champion power and comfort specifics — not private information.

- **Pre-draft:** model and market both price Elo/form/patch priors. Edge exists only where the model is better calibrated or the market is slow (early patch cycle — Worlds is on 16.18, so champion-power edges are largest in Play-In/early Swiss before the market learns).
- **Post-draft (champ select complete, pre-match):** draft is public ~30–40 min before start; sharps adjust in minutes. The model's own draft lift is only Δll 0.002, so the draft rarely flips a bet — but the info is free (short wait), and there is a plausible brief stale-line window right after draft completes. No reason to bet blind pre-draft when the draft is 30 min away, *unless* the pre-draft line already shows ≥threshold edge (locking a good price beats waiting for tiny draft info).
- **Live (t≥10):** absolute probability quality is far better, but everyone sees the same gold lead — the edge is calibration skill vs the live line, not information. Live markets also carry wider margins (higher vig eats edge). Unproven vs market; treat as experimental.

Net: **primary window = post-draft pre-match** (full model, free info, possible stale-line window). **Live = secondary, higher bar** (edge ≥8%+, t≥10 only). Pre-draft bets only to lock an already-good price.

## 4. Empirical measurement (post-Worlds)

The edge-watch worker logs every odds check to `reports/odds_snapshots.csv`:

```
timestamp_utc, match_id, blue, red, book, odds_blue, odds_red,
model_p_blue_predraft, model_p_blue_draft, snapshot_type, edge_blue, implied_blue
```

- `snapshot_type`: `pre_draft` | `post_draft` | `live_t10` | `live_t15` | `live_t20` | `live_t25`
- `model_p_blue_draft`: filled once draft is known (null before)
- Log at first sighting, on material line moves, and at close (last pre-match snapshot). Live odds are NOT available from The Odds API (pre-match only) — live-timing analysis is limited to model-side paper trading unless a live-odds source is added.

Post-tournament analysis (`src/analyze_timing.py`, to be written after Worlds):
1. **CLV per timing:** for each placed bet, taken price vs closing line. Positive CLV = timing beat the close.
2. **Edge decay:** mean model edge at first sighting vs at close, split by `pre_draft`/`post_draft`. Does post-draft edge survive line adjustment?
3. **Draft value-add:** among bets placed post-draft, fraction where `model_p_blue_draft` changed the side or stake vs pre-draft p.
4. **Bet calibration by timing:** actual win rate vs model p for bets placed in each window.
5. **Live paper-trade:** live model p vs live market (if sourced) — otherwise report live model calibration on Worlds games only.

This will replace the structural reasoning above with measured answers.
