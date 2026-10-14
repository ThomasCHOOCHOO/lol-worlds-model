"""Train per-bucket live in-game win-probability models.

For each t in {10, 15, 20, 25} minutes: logistic regression on
[pre_logit, golddiff_k, killdiff], then isotonic calibration fit on the
2026 tail (same convention as v1.2.0: IsotonicRegression fit on the
calibration window, unweighted, out_of_bounds="clip").

Walk-forward: train date < 2026-07-01, calibration 2026-07-01..2026-10-01.
Games with gamelength <= t*60 excluded. No leakage (all features known at t).

Output: model/live_models.joblib -> {t: IsotonicCalibrated wrapper}
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import IsotonicCalibrated

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "model")
FIT_END = "2026-07-01"
CAL_END = "2026-10-01"
BUCKETS = (10, 15, 20, 25)
FEATS = ["pre_logit", "golddiff_k", "killdiff"]


def main():
    bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    feats = pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"),
                        parse_dates=["date"])
    X = feats[bundle["features_full"]].to_numpy(float)
    Xs = bundle["scaler_full"].transform(X)
    pre_p = np.clip(bundle["model_full"].predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    feats["pre_p"] = pre_p
    feats["pre_logit"] = np.log(pre_p / (1 - pre_p))

    frames = []
    for yr in (2024, 2025, 2026):
        raw = pd.read_csv(
            os.path.join(BASE, "data", "raw", f"{yr}.csv"),
            usecols=["gameid", "side", "gamelength", "result"] + [
                f"{s}at{t}" for t in BUCKETS for s in ("gold", "kills")])
        g = raw.groupby(["gameid", "side"]).agg(
            gamelength=("gamelength", "first"),
            **{f"{s}{t}": (f"{s}at{t}", "sum")
               for t in BUCKETS for s in ("gold", "kills")}).reset_index()
        frames.append(g)
    team = pd.concat(frames, ignore_index=True)
    wide = team.pivot(index=["gameid", "gamelength"], columns="side")
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()

    df = feats[["gameid", "date", "y", "pre_p", "pre_logit"]].merge(
        wide, on="gameid", how="inner")
    print(f"games with at-t data: {len(df)}")

    live = {}
    summary = {}
    for t in BUCKETS:
        d = df[df["gamelength"] > t * 60].copy()
        d["golddiff_k"] = (d[f"gold{t}_Blue"] - d[f"gold{t}_Red"]) / 1000.0
        d["killdiff"] = d[f"kills{t}_Blue"] - d[f"kills{t}_Red"]
        tr = d["date"] < FIT_END
        ca = (d["date"] >= FIT_END) & (d["date"] < CAL_END)
        Xtr, ytr = d.loc[tr, FEATS].to_numpy(float), d.loc[tr, "y"].to_numpy(int)
        Xca, yca = d.loc[ca, FEATS].to_numpy(float), d.loc[ca, "y"].to_numpy(int)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
        cal = IsotonicCalibrated(lr).fit(Xca, yca)
        p = np.clip(cal.predict_proba(Xca)[:, 1], 1e-6, 1 - 1e-6)
        ll = log_loss(yca, p)
        acc = accuracy_score(yca, p > 0.5)
        base_ll = log_loss(yca, d.loc[ca, "pre_p"])
        live[t] = cal
        summary[t] = {
            "n_train": int(tr.sum()), "n_cal": int(ca.sum()),
            "coef": [round(float(c), 4) for c in lr.coef_[0]],
            "intercept": round(float(lr.intercept_[0]), 4),
            "cal_log_loss": round(float(ll), 4),
            "cal_accuracy": round(float(acc), 4),
            "pregame_log_loss": round(float(base_ll), 4),
        }
        print(f"t={t}: n_tr={tr.sum()} n_cal={ca.sum()} "
              f"coef={summary[t]['coef']} b={summary[t]['intercept']} "
              f"cal_ll={ll:.4f} (pre-game {base_ll:.4f}) acc={acc:.4f}")

    joblib.dump(live, os.path.join(MODEL_DIR, "live_models.joblib"))
    with open(os.path.join(MODEL_DIR, "live_models.json"), "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print("wrote model/live_models.joblib and model/live_models.json")


if __name__ == "__main__":
    main()
