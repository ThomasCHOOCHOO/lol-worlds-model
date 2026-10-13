"""One-off: tune the recency half-life for sample weights.

Grid: unweighted baseline + 90/180/270/365 days. For each:
  * tune C (weighted TimeSeriesSplit), fit LR vs HGB with sample weights,
    select by UNWEIGHTED validation log-loss on the 2026 tail
  * fit isotonic UNWEIGHTED on the tail (sanity-check vs weighted isotonic)
  * holdout metrics (acc, log-loss, AUC, ECE -- all unweighted, comparable)
  * honest walk-forward backtests (Worlds 2025, MSI 2026), weights anchored
    at each event's cutoff

Pick by holdout log-loss with ECE as tiebreak; sanity-check backtests so we
don't overfit one event. Run: python src/tune_recency.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model_utils import IsotonicCalibrated
from train import (CAL_END, FIT_END, FULL_FEATURES, calibration_table,
                   fit_winner, metrics, recency_weights)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")


def ece(y_true, p, n_bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    out = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (p >= lo) & (p <= hi)
        if m.sum():
            out += m.sum() / len(p) * abs(y_true[m].mean() - p[m].mean())
    return float(out)


def run_half_life(df, X, y, half_life, t_now):
    fit_mask = (df["date"] < FIT_END).to_numpy()
    cal_mask = ((df["date"] >= FIT_END) & (df["date"] < CAL_END)).to_numpy()
    sw = recency_weights(df["date"], t_now, half_life)
    sw_fit = None if sw is None else sw[fit_mask]

    scaler = StandardScaler().fit(X[fit_mask], sample_weight=sw_fit)
    Xs = scaler.transform(X)
    name, model, info = fit_winner(Xs[fit_mask], y[fit_mask],
                                  Xs[cal_mask], y[cal_mask], sw_fit)

    # isotonic: unweighted (pipeline choice) vs weighted (sanity check)
    p_raw = np.clip(model.predict_proba(Xs[cal_mask])[:, 1], 1e-6, 1 - 1e-6)
    ir_u = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir_u.fit(p_raw, y[cal_mask])
    ir_w = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir_w.fit(p_raw, y[cal_mask], sample_weight=(None if sw is None
                                                else sw[cal_mask]))
    from sklearn.metrics import log_loss
    p_u = np.clip(ir_u.predict(p_raw), 1e-6, 1 - 1e-6)
    p_w = np.clip(ir_w.predict(p_raw), 1e-6, 1 - 1e-6)
    iso_check = {"unweighted_ll": float(log_loss(y[cal_mask], p_u)),
                 "weighted_ll": float(log_loss(y[cal_mask], p_w))}

    calib = IsotonicCalibrated(model).fit(Xs[cal_mask], y[cal_mask])
    p_cal = calib.predict_proba(Xs[cal_mask])[:, 1]
    m = metrics(y[cal_mask], p_cal)
    m["ece"] = ece(y[cal_mask], p_cal)
    m["model_type"] = name

    # backtests
    bt = {}
    for label, league, year in (("worlds_2025", "WLDs", 2025),
                               ("msi_2026", "MSI", 2026)):
        ev = df[(df["league"] == league) & (df["date"].dt.year == year)]
        if len(ev) == 0:
            continue
        cutoff = ev["date"].min()
        pre = (df["date"] < cutoff).to_numpy()
        sw_pre = recency_weights(df["date"], cutoff, half_life)
        sw_pre = None if sw_pre is None else sw_pre[pre]
        sc = StandardScaler().fit(X[pre], sample_weight=sw_pre)
        Xs2 = sc.transform(X)
        if name == "logistic_regression":
            from train import tune_c
            c, _ = tune_c(Xs2[pre], y[pre], sw_pre)
            mod = LogisticRegression(C=c, max_iter=2000).fit(
                Xs2[pre], y[pre], sample_weight=sw_pre)
        else:
            mod = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, random_state=7).fit(
                Xs2[pre], y[pre], sample_weight=sw_pre)
        p = np.clip(mod.predict_proba(Xs2[ev.index])[:, 1], 1e-6, 1 - 1e-6)
        bm = metrics(y[ev.index], p)
        bt[label] = {"n": bm["n"], "acc": round(bm["accuracy"], 4),
                     "ll": round(bm["log_loss"], 4),
                     "auc": round(bm["roc_auc"], 4)}
    return m, bt, iso_check, info


def main():
    df = pd.read_csv(os.path.join(PROC, "features.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    X = df[FULL_FEATURES].to_numpy(float)
    y = df["y"].to_numpy(int)
    t_now = df["date"].max()
    print(f"{len(df)} games, t_now={t_now.date()}")

    print(f"{'half_life':>10} | {'holdout acc':>11} {'ll':>7} {'auc':>6} {'ece':>6} | "
          f"{'w25 acc/ll':>14} {'msi acc/ll':>14} | model")
    print("-" * 105)
    results = {}
    for hl in [None, 90, 180, 270, 365]:
        m, bt, iso_check, info = run_half_life(df, X, y, hl, t_now)
        results[hl] = (m, bt)
        w25 = bt.get("worlds_2025", {})
        msi = bt.get("msi_2026", {})
        print(f"{str(hl):>10} | {m['accuracy']:>11.4f} {m['log_loss']:>7.4f} "
              f"{m['roc_auc']:>6.4f} {m['ece']:>6.4f} | "
              f"{w25.get('acc', float('nan')):>6.3f}/{w25.get('ll', float('nan')):<6.4f} "
              f"{msi.get('acc', float('nan')):>6.3f}/{msi.get('ll', float('nan')):<6.4f} | "
              f"{m['model_type']} C={info.get('C', '-')}")
        print(f"           isotonic check (tail ll): unweighted="
              f"{iso_check['unweighted_ll']:.4f} weighted={iso_check['weighted_ll']:.4f}")


if __name__ == "__main__":
    main()
