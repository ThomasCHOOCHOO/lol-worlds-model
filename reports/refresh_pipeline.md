# Post-game refresh pipeline

Keeps the model current as Worlds 2026 games are played. New games flow into
Elo, form, and patch-aware champion-power snapshots (and get top recency
weight in training); the exported phone bundle is regenerated and copied to
the iPhone app.

## How it works — `src/refresh.py`

1. Reads the current `snapshot_date` from `model/phone_artifacts.json`.
2. Downloads the latest Oracle's Elixir 2026 CSV from public Google Drive
   (file `1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm`, linked from
   oracleselixir.com/tools/downloads — the old S3 bucket is dead). Tries
   multiple Drive URL patterns; validates the `gameid` header.
3. Compares the newest game date in the download vs the snapshot.
   - **No newer games** (or download unavailable, e.g. Drive quota) → logs,
     exits 0, touches nothing. (`REFRESH_STATUS=noop`)
4. **New games found** → backs up `data/raw/2026.csv`, installs the new one,
   and runs the proven pipeline in order:
   `build_features.py` → `train.py` → `export_artifacts.py`
   (with `MODEL_VERSION_OVERRIDE` set to the next patch version, e.g.
   1.2.0 → 1.2.1; patch bumps are data refreshes, minor/major are model
   changes) → `validate_export.py`.
   - Any step failing → restores the backup CSV, logs, exits 1.
     (`REFRESH_STATUS=failed`)
5. Copies the fresh `model/phone_artifacts.json` to
   `~/workspace/lol-predictor-app/src/data/phone_artifacts.json` so the app
   picks it up on next launch (the app reads version/snapshot from the JSON,
   nothing hardcoded).
6. Appends a run entry to `reports/refresh_log.md`.
   (`REFRESH_STATUS=refreshed`)

Note: `train.py` keeps its fixed fit/calibration windows
(`FIT_END`/`CAL_END`); new games update the trailing snapshots (Elo, form,
champion power) and receive the highest recency weight. Model coefficients
are refit on the same windows — deliberately conservative.

Data-source caveat (verified 2026-10-13): Google Drive intermittently
returns "quota exceeded" for the 2026 file (resets within ~24h). Refresh
treats this as a clean no-op and retries at the next scheduled run.

## Schedule — one-off runs after each Worlds 2026 stage

Per Thomas 2026-10-13: NOT daily. One run the morning after each major
stage completes, ~06:00 America/Toronto (cron `runonce` jobs, owned by
goal `league-of-legends-worlds-2026-prediction-model`):

| run | stage ends | refreshes at |
|---|---|---|
| post-Play-In | Oct 18 (Play-In Oct 15–18) | Oct 19, 06:00 |
| post-Swiss | Oct 31 (Swiss Oct 23–31) | Nov 1, 06:00 |
| post-quarterfinals | Nov 6 (QF Nov 3–6) | Nov 7, 06:00 |
| post-semifinals | Nov 8 (SF Nov 7–8) | Nov 9, 06:00 |
| post-final | Nov 14 (Final) | Nov 15, 06:00 |

Stage dates verified against multiple live sources (gosugamers, talkesport,
namu.wiki, gamedaily) on 2026-10-13; all agree.

Each run: executes `src/refresh.py` via the project venv. No chat message
on no-op; one concise message (new version, snapshot date, games added)
only when a refresh actually completed.

## Manual run

```bash
cd ~/workspace/lol-worlds-model
.venv/bin/python src/refresh.py            # full refresh if new games exist
.venv/bin/python src/refresh.py --check    # report only, change nothing
```

Run log: `reports/refresh_log.md`.
