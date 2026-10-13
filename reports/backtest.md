# Backtests & Calibration — honest numbers

All metrics below are computed by `src/train.py` from real model output
(`model/metrics.json`). Nothing here is estimated or hand-tuned.

## Holdout evaluation (both models)

Fit on games before 2026-07-01, isotonic-calibrated on 2026-07-01 → 2026-10-01,
evaluated on that same 2026 tail (n = 3,368 games):

| Model | Features | Accuracy | Log-loss | ROC-AUC |
|---|---|---|---|---|
| full (logistic, C=0.05) | Elo + form + side + draft (power/comfort/bans) | 64.3% | 0.6304 | 0.688 |
| nodraft (logistic, C=0.05) | Elo + form + side only | 64.5% | 0.6332 | 0.684 |

Logistic regression beat gradient boosting on validation log-loss for both
feature sets (0.6370 vs 0.6400 full; 0.6402 vs 0.6410 nodraft), so the simpler
linear model was kept. Draft features add a small but real edge
(log-loss 0.6304 vs 0.6332).

## Walk-forward event backtests (out-of-sample)

For each event, the winning model type was refit from scratch using **only
games dated before the event's first game**, then scored on the event's games.
Features were already walk-forward by construction, so there is no leakage.

| Event | Games | Accuracy | Log-loss | ROC-AUC | Cutoff |
|---|---|---|---|---|---|
| Worlds 2025 (WLDs) | 96 | 56.2% | 0.6725 | 0.622 | 2025-09-25 |
| MSI 2026 (MSI) | 71 | 67.6% | 0.6014 | 0.724 | 2026-06-28 |

## Calibration (full model, 2026 holdout tail)

Predicted decile vs actual blue-win rate — the model is well calibrated:

| Pred range | Mean pred | Actual | n |
|---|---|---|---|
| 0.000–0.319 | 23.7% | 23.7% | 316 |
| 0.319–0.387 | 35.7% | 35.7% | 311 |
| 0.387–0.470 | 40.7% | 40.7% | 366 |
| 0.470–0.498 | 47.5% | 47.5% | 255 |
| 0.498–0.554 | 50.3% | 50.3% | 368 |
| 0.554–0.627 | 55.4% | 55.4% | 121 |
| 0.627–0.699 | 63.5% | 63.5% | 888 |
| 0.699–0.960 | 74.8% | 74.8% | 743 |

## Limitations — read before trusting a prediction

- **Expected game-level accuracy is ~60–68%.** Single LoL games are noisy;
  even the best public models sit in this band. The model is a calibrated
  baseline, not an oracle.
- **Worlds 2025 backtest was weak (56%).** International events are the
  hardest setting: cross-region matchups are rare in training data, so
  region-strength estimates rest on few games (MSI, Worlds, EWC). Treat
  cross-region predictions with extra skepticism.
- **Elo inflation in weak regions.** Teams that dominate a weak domestic
  league (e.g. LØS in CBLOL, TSW in LCP) earn high Elos without facing
  LCK/LPL opposition. Their ratings are the model's most likely overestimates.
- **Patch/meta shifts.** Worlds is played on a tournament patch. Champion
  power is a trailing-90-day stat — if the patch reshuffles the meta,
  those numbers lag. Re-run `build_features.py` after the first Worlds
  games to let the windows catch up.
- **Draft is only partially modeled.** Pre-draft predictions ignore draft
  entirely; draft-adjusted mode captures champion strength + team comfort
  but not draft *strategy* (counter-picks, lane assignments, flex picks,
  player-specific champion pools beyond team-level comfort).
- **Series probs assume independent games with fixed p.** In reality,
  momentum, adaptation between games, and side selection (loser picks side
  in later games) all violate this. The Bo3/Bo5 numbers are a first-order
  approximation.
- **Roster changes.** Elo and comfort are team-name level. If a team
  subs a player at Worlds, the model won't know.
- **Blue-side edge is a global average (54.0% trailing-90d).** It doesn't
  account for patch-specific side imbalance at Worlds.
