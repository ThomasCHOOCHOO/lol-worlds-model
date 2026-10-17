"""Train per-bucket live in-game models with ROLE-WEIGHTED gold (variant A).

Same walk-forward protocol as the shipped model (src/train_live.py):
per-bucket (t = 10/15/20/25) logistic regression, C = 1.0, train on games
with date < 2026-07-01, isotonic calibration on 2026-07-01..2026-10-01.
Games with gamelength <= t*60 excluded.

Variant A (reports/live_role_gold.md): total gold is REPLACED by five
per-role diffs. Features:
  [pre_logit, top_gold, jng_gold, mid_gold, bot_gold, sup_gold, killdiff]
Role golds are blue-minus-red goldat{t} per position, in 1,000s, summed
over PLAYER rows only (the team row's gold is the sum of its players --
including it would double-count, the shipped 2x-total bug). killdiff is
kept EXACTLY as the shipped definition (sum over ALL rows incl. the team
row, i.e. 2x the true kill diff), so variant A is comparable to the
baseline on the same killdiff scale. Games missing any role gold at a
bucket are EXCLUDED for that bucket (the shipped groupby-sum turns those
into fake 0-gold games; exclusion is apples-to-apples).

Outputs:
  model/live_models_role_gold.joblib -> {t: IsotonicCalibrated wrapper}
    (the DEPLOYABLE variant-A models; the pre-existing
    model/live_models_role_gold.json was metrics/coefficients only)
  model/live_models_role_gold.json gains an "export_fit" section with the
    fresh fit's features/coefs/intercept/calibration metrics per bucket
    (the analysis sections of that file are preserved untouched).

pre_logit comes from the CURRENT model.joblib (region-offset model), so
these live models stay consistent with the exported pre-game model.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import IsotonicCalibrated

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "model")
FIT_END = "2026-07-01"
CAL_END = "2026-10-01"
BUCKETS = (10, 15, 20, 25)
ROLES = ("top", "jng", "mid", "bot", "sup")
ROLE_FEATS = [f"{r}_gold" for r in ROLES]
FEATS = ["pre_logit"] + ROLE_FEATS + ["killdiff"]


def main():
    bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    feats = pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"),
                        parse_dates=["date"])
    X = feats[bundle["features_full"]].to_numpy(float)
    Xs = bundle["scaler_full"].transform(X)
    pre_p = np.clip(bundle["model_full"].predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    feats["pre_p"] = pre_p
    feats["pre_logit"] = np.log(pre_p / (1 - pre_p))

    # ---- per-game role golds + shipped-definition killdiff ----
    frames = []
    for yr in (2024, 2025, 2026):
        raw = pd.read_csv(
            os.path.join(BASE, "data", "raw", f"{yr}.csv"),
            usecols=["gameid", "side", "position", "gamelength", "result"] +
                    [f"{s}at{t}" for t in BUCKETS for s in ("gold", "kills")])
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    assert set(raw["position"].unique()) <= {"top", "jng", "mid", "bot", "sup", "team"}, \
        f"unexpected positions: {sorted(raw['position'].unique())}"

    # Team-level (ALL rows incl. team row): gamelength + killdiff, shipped def.
    team = raw.groupby(["gameid", "side"]).agg(
        gamelength=("gamelength", "first"),
        **{f"kills{t}": (f"killsat{t}", "sum") for t in BUCKETS}).reset_index()
    wide = team.pivot(index=["gameid", "gamelength"], columns="side")
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()

    # Player-level role golds (player rows only).
    players = raw[raw["position"] != "team"]
    pg = players.pivot_table(index=["gameid", "side", "position"],
                             values=[f"goldat{t}" for t in BUCKETS],
                             aggfunc="first").reset_index()
    pg_wide = pg.pivot(index=["gameid", "side"], columns="position")
    pg_wide.columns = [f"{c[0]}_{c[1]}" for c in pg_wide.columns]
    pg_wide = pg_wide.reset_index()

    df = feats[["gameid", "date", "y", "pre_p", "pre_logit"]].merge(
        wide, on="gameid", how="inner")
    # df is one row per game (wide is per-game after the side pivot).
    # Per-game role golds are merged inside the per-bucket loop (gold_p).
    print(f"games with at-t data: {len(df)}")

    live = {}
    export_fit = {}
    for t in BUCKETS:
        d = df[df["gamelength"] > t * 60].copy()
        # Role gold diffs: blue-minus-red per position, in 1000s.
        # pg_wide has one row per (gameid, side); use it directly.
        gold = pg_wide[["gameid", "side"] +
                       [f"goldat{t}_{r}" for r in ROLES]].copy()
        gold_p = gold.pivot(index="gameid", columns="side")
        gold_p.columns = [f"{c[0]}_{c[1]}" for c in gold_p.columns]
        gold_p = gold_p.reset_index()
        dd = d.merge(gold_p, on="gameid", how="left")
        for r in ROLES:
            dd[f"{r}_gold"] = (dd[f"goldat{t}_{r}_Blue"] -
                               dd[f"goldat{t}_{r}_Red"]) / 1000.0
        dd["killdiff"] = dd[f"kills{t}_Blue"] - dd[f"kills{t}_Red"]
        # Exclude games missing ANY role gold at this bucket (all-or-nothing
        # per game in practice; a partial NaN is still excluded, not zero-filled).
        before = len(dd)
        dd = dd.dropna(subset=[f"{r}_gold" for r in ROLES])
        tr = dd["date"] < FIT_END
        ca = (dd["date"] >= FIT_END) & (dd["date"] < CAL_END)
        Xtr, ytr = dd.loc[tr, FEATS].to_numpy(float), dd.loc[tr, "y"].to_numpy(int)
        Xca, yca = dd.loc[ca, FEATS].to_numpy(float), dd.loc[ca, "y"].to_numpy(int)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
        cal = IsotonicCalibrated(lr).fit(Xca, yca)
        p = np.clip(cal.predict_proba(Xca)[:, 1], 1e-6, 1 - 1e-6)
        ll = log_loss(yca, p)
        acc = accuracy_score(yca, p > 0.5)
        base_ll = log_loss(yca, dd.loc[ca, "pre_p"])
        # Honest-split check (reported only): isotonic fit Jul-Aug -> eval Sep-Oct.
        ca_fit = (dd["date"] >= FIT_END) & (dd["date"] < "2026-09-01")
        ca_ev = (dd["date"] >= "2026-09-01") & (dd["date"] < CAL_END)
        cal_h = IsotonicCalibrated(lr).fit(
            dd.loc[ca_fit, FEATS].to_numpy(float), dd.loc[ca_fit, "y"].to_numpy(int))
        p_h = np.clip(cal_h.predict_proba(
            dd.loc[ca_ev, FEATS].to_numpy(float))[:, 1], 1e-6, 1 - 1e-6)
        ll_h = log_loss(dd.loc[ca_ev, "y"].to_numpy(int), p_h)
        live[t] = cal
        export_fit[str(t)] = {
            "features": list(FEATS),
            "coef": [round(float(c), 4) for c in lr.coef_[0]],
            "intercept": round(float(lr.intercept_[0]), 4),
            "n_train": int(tr.sum()),
            "n_cal": int(ca.sum()),
            "n_excluded_missing_role_gold": int(before - len(dd)),
            "cal_log_loss": round(float(ll), 4),
            "cal_accuracy": round(float(acc), 4),
            "pregame_log_loss": round(float(base_ll), 4),
            "honest_split_log_loss": round(float(ll_h), 4),
            "honest_split_n": int(ca_ev.sum()),
            "note": ("Variant A, shipped protocol (isotonic fit on the full cal "
                     "window, graded on the same window -- relative comparison "
                     "only). honest_split is isotonic Jul-Aug -> eval Sep-Oct, "
                     "reported only, not exported."),
        }
        print(f"t={t}: n_tr={tr.sum()} n_cal={ca.sum()} "
              f"excluded_missing={before - len(dd)} "
              f"coef={export_fit[str(t)]['coef']} b={export_fit[str(t)]['intercept']} "
              f"cal_ll={ll:.4f} (pre-game {base_ll:.4f}) acc={acc:.4f} "
              f"honest_ll={ll_h:.4f} (n={int(ca_ev.sum())})")

    joblib.dump(live, os.path.join(MODEL_DIR, "live_models_role_gold.joblib"))
    json_path = os.path.join(MODEL_DIR, "live_models_role_gold.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            summary = json.load(f)
    else:
        summary = {}
    summary["export_fit"] = export_fit
    summary["export_fit_note"] = (
        "Fresh fit from src/train_live_role_gold.py against the CURRENT "
        "model.joblib (pre_logit from the region-offset model). The "
        "per-bucket analysis sections of this file (variants, drop-one, "
        "correlations) are the original 2026-10-17 analysis against the "
        "pre-region-offset bundle and are preserved untouched; export_fit "
        "is the deployable fit. Coefficients should be close to the "
        "analysis's Table §3 (small shifts from the region-offset pre_logit).")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote model/live_models_role_gold.joblib and updated "
          "model/live_models_role_gold.json (export_fit)")


if __name__ == "__main__":
    main()
