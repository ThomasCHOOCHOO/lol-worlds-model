# Model history

Versioned phone artifacts (the exact JSON the app consumes) for past models.
Binaries (`*.joblib`) are not versioned here; retrain from the tagged source.

- `v1.2.0/` — core pre-game/draft model: Elo + form_diff + blue_edge (+ draft
  features), patch-aware champion power, 180-day recency. Holdout n=3,368:
  acc 0.6452 / log-loss 0.6310 / AUC 0.6865. Git tag `v1.2.0`.
- `v1.2.0-live/` — v1.2.0 core + live in-game models (per-bucket logistic
  regression at t=10/15/20/25 on pre-game logit + gold/kill diffs, isotonic
  calibrated; log-loss 0.535/0.475/0.415/0.360). Commit `df14937`.
- `v1.3.0/` (current, `model/phone_artifacts.json`) — `form_diff` replaced by
  `eg_gold_diff` (trailing-20 mean own-side gold@15, shrunk). Holdout:
  acc 0.6508 / log-loss 0.6271 / AUC 0.695. Git tag `v1.3.0`.
