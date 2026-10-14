# LoL Worlds 2026 Prediction Model — User Guide

Statistical match predictor for pro League of Legends, built for Worlds 2026
(Oct 15 – Nov 14). Predicts per-map win probability, converts to series
probability for Bo3/Bo5, and sizes bets with adaptive Kelly. Two interfaces:
the **iPhone app** (primary) and the **Python CLI** (power use, retraining).

Current model: **v1.3.0** (snapshot 2026-10-08; 29,793 games).

---

## 1. The iPhone app (primary interface)

### Install (Windows PowerShell)

```powershell
cd C:\path\to\lol-predictor-app
npm install        # once
npx expo start
```

1. Install **Expo Go** from the App Store.
2. iPhone and PC on the **same Wi-Fi**.
3. Scan the terminal QR code with the iPhone camera.
4. The app loads inside Expo Go. Reinstall after each model update.

The app is **fully offline** except the Live tab (match feed) and Odds tab
(odds API), which need internet.

### Tabs

**Predict** — pick blue/red teams (search understands tags: `t1`, `g2`, `c9`).
Two modes: pre-draft, or draft mode (5 picks + 5 bans per side, entered after
champ select). Patch selector defaults to the Worlds patch (16.18). Each team
card shows roster-adjusted Elo, raw Elo, roster Δ, turnover note, and the 2026
roster.

**Results** — the big calibrated win % *is* the confidence figure. Below it: a
calibration honesty note ("when the model said 63–67%, that side won 63% of
773 holdout games"), feature contributions (blue − red), and a metrics
footnote (holdout accuracy, ECE, snapshot date).

**Kelly** — enter the book's **decimal odds** manually (or tap a price in the
Odds tab to send it here). Shows implied probability, your edge, and
full/half/quarter-Kelly stakes. **NO BET** when edge ≤ 0. The recommended
fraction is adaptive:
- edge < 3% or thin calibration bin (n < 30) → quarter Kelly
- edge ≥ 8% and deep bin (n ≥ 300) → full Kelly
- otherwise → half Kelly
- if Pinnacle's de-vigged price disagrees with the model's edge at a soft
  book → fraction halved (floor: quarter Kelly). This is the smart-money
  agreement check: when the sharp book says you're wrong, bet smaller.

**Odds** — pulls moneyline odds via The Odds API (`esports_lol`). Setup: get a
free key at the-odds-api.com (500 requests/month), paste it in the tab (stored
securely on-device), tap refresh — **manual refresh only**, each tap costs one
request. Best price per team is highlighted; "My books" filter persists. Tap
any price to send it to Kelly. Cross-book arbs are flagged automatically with
a two-book stake split when `1/bestA + 1/bestB < 1`.

**Live** — in-game win probability. Pick the live match; the tab polls the
LoL Esports feed every ~12 seconds and shows live win %, gold/kill diffs,
dragons, towers. Uses per-minute models (t = 10/15/20/25). If you ran a draft
prediction for the same matchup, its probability seeds the live model as the
prior. No model read before t = 10.

**Parlay** — ranks two-leg and three-leg combos by expected value
(`EV = P × D − 1`, P = product of model probabilities, D = product of decimal
odds). Same-book legs only (a parlay is one ticket). Positive EV = VALUE;
negative EV shown as cover cost. Same-match combos get structural correlation
notes, not fake precise probabilities (e.g. favorite ML + over maps is
negatively correlated — the book may overpay).

**Model** — version, backtests, honest metrics, roster formula. Read this
before trusting any number the app shows.

---

## 2. Python CLI

Setup (Windows PowerShell):

```powershell
cd C:\path\to\lol-worlds-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Predictions:

```powershell
# pre-draft, Bo5 series
.venv\Scripts\python.exe src\predict.py --blue KC --red C9 --best-of 5

# with draft (comma-separated champion names, after champ select)
.venv\Scripts\python.exe src\predict.py --blue KC --red C9 --best-of 5 `
  --blue-picks "Aatrox,Vi,Ahri,Kai'Sa,Renata Glasc" `
  --red-picks "Gwen,Xin Zhao,Taliyah,Senna,Nautilus" `
  --blue-bans "Azir,Kalista,Neeko,Rumble,Sejuani" `
  --red-bans "Yone,Wukong,Orianna,Varus,Rell"

# batch from CSV (columns: blue,red,best_of,label)
.venv\Scripts\python.exe src\predict.py --schedule my_matches.csv

# interactive dashboard
streamlit run app.py
```

Side assignment: `--blue`/`--red` are the actual sides when known; pre-draw,
either order works — the model's blue edge (~54%) is the only side effect.

---

## 3. Betting workflow

The intended loop, in order:

1. **~60–75 min before a match**, check the Odds tab (or wait for the
   automated watch alert). The watch checks each match once and pings you
   only on **edge ≥ 5%** or **arb profit ≥ 1%**.
2. **Lock the price then.** Drafts finish ~1 minute before play — there is no
   usable post-draft betting window.
3. **Draft = veto power.** Enter the draft in the app. If the draft-adjusted
   probability kills the edge, **skip**. Never force a bet the draft
   disagrees with.
4. **Size with adaptive Kelly** on a $100/book bankroll. The app does the
   math; the smart-money check trims the fraction when Pinnacle disagrees.
5. **Live betting is secondary**: only from t = 10 onward, bar 8%+ edge.
6. **APPROVE or SKIP every alert.** Nothing is ever placed automatically.

What the model is and isn't:
- It predicts **per-map** win probability; series probability comes from the
  binomial (independent maps). Fearless draft is handled *between* maps by
  entering the actual draft — burned champions simply can't be picked.
- The draft adds ~0.002 log-loss over pre-draft. Small but real; its main
  value is vetoing bad spots and seeding the live model.
- Skip deposit bonuses with wagering requirements: at ~14 bets per Worlds
  and ~$5 average stakes, clearing 10× on $100 is mathematically out of
  reach, and bonus-abuse flags get winnings voided.

---

## 4. The model (v1.3.0)

**Features** (all walk-forward — a game at time *t* only sees games before *t*):
- `elo_diff` — team Elo diff (init 1500, K=24, 40% offseason regression)
- `eg_gold_diff` — trailing-20 mean own-side gold diff at 15 min, shrunk
  toward 0 (replaced win-rate form in v1.3.0; strictly more informative)
- `blue_edge` — trailing-90-day global blue-side win rate (~54%)
- `d_power` — patch-aware champion power summed over each side's 5 picks
- `d_comfort` — team-champion comfort (trailing-365d, min 3 games)
- `ban_edge` — champion power of your bans minus theirs

Logistic regression (beat gradient boosting on validation log-loss),
isotonic-calibrated. Two variants: `full` (with draft) and `nodraft`.

**Honest metrics** (v1.3.0):

| Evaluation | Accuracy | Log-loss | AUC |
|---|---|---|---|
| 2026 holdout (n=3,368) | 65.1% | 0.6271 | 0.695 |
| MSI 2026 backtest (n=71) | 62.0% | 0.6214 | 0.676 |
| Worlds 2025 backtest (n=96) | 55.2% | 0.6803 | 0.597 |

Calibration note: isotonic calibration was fit on the recent tail, so the
near-zero ECE is not independent out-of-sample proof. International events
are the hardest setting (sparse cross-region data); expect the low end of
the 60–68% band there.

**Version history** (`model/history/` holds past artifacts):
- v1.2.0 — Elo + win-rate form + blue edge; holdout 64.5% / 0.6310 / 0.6865
- v1.2.0-live — v1.2.0 core + live in-game models (t=10/15/20/25)
- v1.3.0 — `form_diff` → `eg_gold_diff`; holdout 65.1% / 0.6271 / 0.695

Dropped experiments (tested, failed the bar, not shipped): champion-pairing
synergy, comp-archetype features, range v2, range×engage interaction,
fearless-draft pool depth, and 7 of 8 screened features (objective control,
first-blood rate, patch-adaptation speed, meta-comfort alignment, flex picks,
rest/fatigue, knockout experience — reversed! — comeback factor). Details in
`reports/`.

---

## 5. Retraining & upkeep

Full rebuild (after new games are played):

```powershell
.venv\Scripts\python.exe src\build_features.py   # walk-forward features
.venv\Scripts\python.exe src\train.py            # fit + calibrate + metrics
.venv\Scripts\python.exe src\train_live.py       # in-game models
.venv\Scripts\python.exe src\export_artifacts.py  # phone_artifacts.json
.venv\Scripts\python.exe src\validate_export.py   # must print ALL CHECKS PASSED
```

`src/refresh.py` automates this end-to-end (download → backup → retrain →
version bump → validate → copy artifacts); it no-ops cleanly when
Oracle's Elixir is unreachable. Data refresh runs automatically after each
tournament stage during Worlds.

To ship a new model to the phone: copy the regenerated
`model/phone_artifacts.json` to the app's `src/data/phone_artifacts.json`,
re-run the parity check (`node scripts/parity_check.mjs` in the app repo —
must pass), reinstall via Expo Go.

---

## 6. Limitations

- Snapshot as of 2026-10-08; retrain as Worlds games are played.
- Elo inflates for weak-region dominators (a team crushing a minor region
  looks stronger than it is against LCK/LPL opposition).
- Series conversion assumes independent maps at fixed p; fearless draft and
  within-series adaptation make this approximate (handled per-map via the
  draft model, not in the pre-series number).
- The Odds API is pre-match only; there is no live-odds feed.
- Ontario books may limit winning accounts; keep stakes recreational and
  never chase. Nothing here is financial advice.
