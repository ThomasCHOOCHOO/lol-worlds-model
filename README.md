# LoL Worlds 2026 Match Predictor

Statistical model that predicts the winner of individual League of Legends
pro matches — built for **Worlds 2026** (Oct 16 – Nov 14), with **draft
(champion picks/bans) as model inputs**.

## How it works

1. **Data**: ~29,800 pro games (2024–2026) from Oracle's Elixir public CSVs.
2. **Features** — all walk-forward (a game at time *t* only ever sees games
   before *t*):
   - `elo_diff` — team Elo diff (init 1500, K=24, 40% offseason regression)
   - `eg_gold_diff` — trailing-20 mean own-side gold diff at 15 min, shrunk
     toward 0 (replaced win-rate `form_diff` in v1.3.0; strictly more
     informative than binary W/L)
   - `blue_edge` — trailing-90-day global blue-side win rate (~54%)
   - `d_power` — draft power edge: trailing-90d global champion win% (min 10
     games, empirical-Bayes shrunk), summed over each side's 5 picks
   - `d_comfort` — team-champion comfort edge: team's trailing-365d win% on
     each picked champion (min 3 games, shrunk toward team form)
   - `ban_edge` — champion-power sum of your bans minus theirs
3. **Model**: logistic regression (beat gradient boosting on validation
   log-loss), isotonic-calibrated. Two variants: `full` (with draft) and
   `nodraft` (pre-draft predictions).

## Honest accuracy expectations

| Evaluation | Accuracy | Log-loss | ROC-AUC |
|---|---|---|---|
| 2026 holdout tail (n=3,368) | 65.1% | 0.627 | 0.70 |
| Worlds 2025 backtest, out-of-sample (n=96) | 55.2% | 0.680 | 0.60 |
| MSI 2026 backtest, out-of-sample (n=71) | 62.0% | 0.621 | 0.68 |

Game-level accuracy of ~60–68% is the realistic band for pro LoL. International
events are the hardest setting (sparse cross-region data). Full calibration
table, backtests, and limitations: [`reports/backtest.md`](reports/backtest.md).

## Setup (Windows PowerShell)

```powershell
cd C:\path\to\lol-worlds-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
cd ~/workspace/lol-worlds-model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## App (interactive)

```powershell
streamlit run app.py
```

Opens a dashboard in your browser with four tabs: **Match predictor**
(team dropdowns, best-of selector, optional draft entry with champion
dropdowns + bans), **Team ratings**, **Play-In simulator** (pick the CBLOL
seed, price every pairing), and **Backtest** (metrics, calibration,
limitations).

## Usage (CLI)

Scripts live in `src/`. Run them from the project root.

**1. Download data** (2024/2025 CSVs; 2026 ships with the project):

```powershell
python src/download_data.py
```

**2. Build features** (~30–60 s):

```powershell
python src/build_features.py
```

**3. Train** (~10 s):

```powershell
python src/train.py
```

**4. Predict.**

Pre-draft single game:

```powershell
python src/predict.py --blue T1 --red BLG
```

Pre-draft Bo5 series:

```powershell
python src/predict.py --blue T1 --red BLG --best-of 5
```

Draft-adjusted (comma-separated champion names; bans optional):

```powershell
python src/predict.py --blue T1 --red BLG `
  --blue-picks "Azir,Vi,Ahri,Kai'Sa,Rakan" `
  --red-picks "Renekton,Sejuani,Orianna,Ziggs,Nautilus" `
  --blue-bans "Kalista,Yone" --red-bans "Vi,Neeko"
```

Batch from CSV (columns: `blue,red,best_of,label`):

```powershell
python src/predict.py --schedule data/worlds2026_schedule_example.csv
```

Team tags are case-insensitive: `T1 BLG GEN HLE DK AL TES IG G2 MKOI KC TL TLAW
LYON C9 TSW CFO MVK LOS FUR` etc. — see `python src/predict.py --list-teams`.

## Refreshing during Worlds

As Worlds games are played, update the snapshots so Elo/form/champion-power
reflect tournament form:

```powershell
# re-download the 2026 CSV (Oracle's Elixir updates daily), then:
python src/build_features.py
python src/train.py
```

(Or append Worlds games to `data/raw/2026.csv` manually — same schema.)

## Project layout

```
lol-worlds-model/
├── app.py                 # Streamlit app: predictor, ratings, play-in sim, backtest
├── src/
│   ├── download_data.py   # fetch Oracle's Elixir CSVs → data/raw/
│   ├── build_features.py  # walk-forward features → data/processed/features.csv
│   ├── train.py           # tune, compare, calibrate, backtest → model/
│   ├── predict.py         # CLI: pre-draft, draft-adjusted, batch
│   └── model_utils.py     # shared calibrated-model wrapper
├── data/
│   ├── raw/               # 2024.csv, 2025.csv, 2026.csv (Oracle's Elixir)
│   ├── processed/         # features.csv
│   └── worlds2026_schedule_example.csv
├── model/
│   ├── model.joblib      # calibrated models + scalers
│   ├── snapshots.joblib  # Elo / form / champ-power / comfort as of latest game
│   └── metrics.json      # all evaluation numbers
├── reports/
│   ├── ratings.md            # ranked power ratings, all 19 Worlds teams
│   ├── playin_predictions.md # all 6 pairwise Bo5s (both CBLOL#2 variants)
│   └── backtest.md           # honest backtests + limitations
└── requirements.txt
```

## Data credit

Match data © Oracle's Elixir (Tim Sevenhuysen), free for analysts/fans,
updated daily: <https://oracleselixir.com/tools/downloads>
