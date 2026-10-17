# Phase 1 Wave 2 — model export work

**Date:** 2026-10-17. **Status:** done, verified. Nothing pushed, no bets, no app files touched.

## What was added

**`src/export_artifacts.py`** — the exported phone artifact now includes:
- `models.{full,nodraft}.calibration_honest` — nested-split honest maps in the
  exact shape of the existing `calibration` field
  (`{xs, ys, x_min, x_max, y_min, y_max}`). Full-model map read from the
  canonical `model/honest_calibration.json` and **verified against a
  recomputation from the exported bundle** (export raises on drift — a stale
  map paired with fresh coefficients would silently miscalibrate). Nodraft
  map computed the same way via `train.honest_map_from_bundle` (it already
  supported any key; no extension needed). `x_min/x_max` are the coarsened
  map's first/last thresholds (= the fitted map's input range);
  `y_min/y_max` are the clip bounds `[0.001, 0.999]`.
- Top-level `player_gaps`: `{team: gap}` (883 teams) from
  `model/player_ratings.json`. Keys are full team names, same namespace as
  the artifact's `teams` dict. `+ calibration_honest_note`.
- Top-level `live_models_role_gold`: per-bucket variant-A live models in
  the **same shape** as baseline `live_models` (features
  `[pre_logit, top_gold, jng_gold, mid_gold, bot_gold, sup_gold, killdiff]`,
  coef, intercept, isotonic calibration). Baseline `live_models` untouched.
  `+ live_models_role_gold_note` (feed-only; OCR keeps baseline; shadow test
  pending before Kelly sizing).

**`src/build_player_ratings.py`** — the player snapshot builder, moved from
`/tmp/player_ratings_snapshot.py` into `src/` (only change: `BASE` is now
repo-relative instead of a hardcoded home path). Rebuilds the player Elo
replay per `reports/player_ratings_analysis.md` §2: playerid keying,
`(player, role)` key, init 1500, K=24, 40% year-boundary regression,
debutant prior = mean of rated teammates, current-5 player_elo vs team_elo.

**`src/train_live_role_gold.py`** (new) — re-runs the variant-A fit per
`reports/live_role_gold.md` §1 (same protocol as `src/train_live.py`:
LR C=1.0, train < 2026-07-01, isotonic cal 2026-07-01..2026-10-01).
Role golds from player rows only (no team-row double-count); `killdiff`
kept at the shipped 2× definition; games missing any role gold excluded
per bucket. Writes the **deployable** `model/live_models_role_gold.joblib`
`{t: IsotonicCalibrated}` and adds an `export_fit` section to
`model/live_models_role_gold.json` (the analysis sections are preserved).
`pre_logit` comes from the current `model.joblib` (region-offset model), so
the live models stay consistent with the exported pre-game model.

**`src/refresh.py`** — new steps, in order: `player_ratings`,
`train_live_role_gold`, both before `export_artifacts` (export embeds their
outputs). Docstring updated.

**`src/validate_export.py`** — extended with checks for all three additions
(honest maps recomputed from the bundle; `player_gaps` vs
`player_ratings.json`; role-gold JSON math vs the joblib wrappers) and the
two new top-level keys in the schema list.

## Verification

- **Player snapshot reproduction:** the `src/` builder reproduces the
  Wave 1 `model/player_ratings.json` **byte-identically** (`diff` → IDENTICAL):
  5,333 players, 883 teams, 5,863 (player, role) ratings. Play-In
  |gap| range 2.0–24.2 (report: 1–24; max off by 0.2).
- **Role-gold re-fit vs report Table §3:** coefficients match to ≤0.017
  (role golds identical; `pre_logit` shifted slightly by the region-offset
  model). ADC > support ordering holds at all four buckets. Honest-split
  log-loss for A: 0.5553 / 0.4888 / 0.4181 / 0.3813 (report:
  0.5602 / 0.4885 / 0.4242 / 0.3802) — same picture.
- **Honest maps:** full 13 steps, nodraft 9 steps; both match recomputation
  exactly; monotone; ys within [0.001, 0.999] (no 0/1 outputs).
- **`validate_export.py` on the draft artifact: ALL CHECKS PASSED.**
  TS/Python parity: per-case diffs 0.00e+00; isotonic max diff 5.55e-17;
  live baseline 6.94e-18; live role-gold 2.78e-17.

## Version incident (caught and fixed)

`export_artifacts.py` had a stale fallback default `MODEL_VERSION = "1.3.0"`.
It silently mislabeled **two** exports on 2026-10-17: the region agent's
draft (15:22) and my first export, which overwrote the shipped
`model/phone_artifacts.json` as "1.3.0". Caught on the post-export
spot-check. Fix applied:
- Shipped artifact **restored byte-identically** from the pristine app copy
  (`~/workspace/lol-predictor-app/src/data/phone_artifacts.json`, same
  mtime/content): version 1.3.2, snapshot 2026-10-16, no new keys.
- The new export lives at `model/phone_artifacts_1.4.0_draft.json` with
  explicit `MODEL_VERSION_OVERRIDE=1.4.0` (the 1.4.0 line per the Phase 1
  plan; the shipped 1.3.2 artifact is untouched until Phase 1 ships).
- The stale default is now `"0.0.0-dev"` — a value that can never be
  mistaken for a release; real exports pass `MODEL_VERSION_OVERRIDE`
  explicitly (refresh.py always does). Comment in code explains why.

## Regeneration commands

```bash
.venv/bin/python src/build_player_ratings.py        # -> model/player_ratings.json
.venv/bin/python src/train_live_role_gold.py        # -> model/live_models_role_gold.joblib + .json (export_fit)
MODEL_VERSION_OVERRIDE=1.4.0 \
  PHONE_ARTIFACTS_OUT=model/phone_artifacts_1.4.0_draft.json \
  .venv/bin/python src/export_artifacts.py          # draft artifact (never the shipped file)
ARTIFACTS_PATH=model/phone_artifacts_1.4.0_draft.json \
  .venv/bin/python src/validate_export.py           # must print ALL CHECKS PASSED
```

## For the app-side agent / promotion

- Consume the **draft** artifact: `model/phone_artifacts_1.4.0_draft.json`
  (version 1.4.0). Do not read the new keys from the shipped 1.3.2 artifact
  — they are not there.
- Contract: `models.{full,nodraft}.calibration_honest` (toggle),
  `player_gaps` (diagnostic flag), `live_models_role_gold` (feed-only
  variant A; OCR keeps baseline `live_models`).
- Promotion (draft → shipped → app → push) is a parent decision: region
  gate, prev-win validation, and the role-gold shadow test are still
  pending/outstanding elsewhere.
