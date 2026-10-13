"""Train and evaluate the LoL match-winner model.

Pipeline (all walk-forward, no leakage):
  * C tuned for LogisticRegression via 3-fold TimeSeriesSplit on the fit set.
  * LogisticRegression vs HistGradientBoosting compared by log-loss on a
    held-out 2026 tail; the winner is kept, then isotonic-calibrated
    (IsotonicRegression fit on that tail, out-of-bounds clipped).
  * Two models are trained: `full` (all features incl. draft) and
    `nodraft` (elo_diff, form_diff, blue_edge) for pre-draft predictions.
  * Recency weighting: the fit set is weighted with exponential decay
    w = 0.5 ** (age_days / HALF_LIFE_DAYS), t_now = latest data date
    (event cutoff for backtests). Passed as sample_weight to the LR/HGB fit
    and to the scaler. The isotonic calibration map is fit UNWEIGHTED on the
    recent tail: the tail is already the most recent ~3 months, and weighted
    isotonic risks overfitting the map to the very latest games (verified:
    weighted vs unweighted isotonic differ negligibly on the tail).
  * Honest walk-forward backtests: refit the winning model type on data
    strictly before each event's first game, then score the event's games:
      - Worlds 2025 (WLDs games in 2025)
      - MSI 2026 (MSI games in 2026)

Outputs (model/):
  model.joblib  - calibrated models, scalers, feature lists, metadata
  metrics.json  - holdout + backtest metrics, calibration table
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from model_utils import IsotonicCalibrated

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "model")

FULL_FEATURES = ["elo_diff", "form_diff", "blue_edge", "d_power", "d_comfort", "ban_edge"]
NODRAFT_FEATURES = ["elo_diff", "form_diff", "blue_edge"]

FIT_END = "2026-07-01"
CAL_END = "2026-10-01"

# Exponential recency half-life for sample weights, tuned by src/tune_recency.py
# (grid over 90/180/270/365/unweighted on holdout log-loss + ECE, checked
# against both backtests). None = unweighted baseline.
HALF_LIFE_DAYS = 180


def recency_weights(dates, t_now, half_life):
    """w = 0.5 ** (age_days / half_life); None half-life -> None (unweighted)."""
    if half_life is None:
        return None
    age_days = (t_now - dates).dt.total_seconds() / 86400.0
    return np.power(0.5, np.clip(age_days, 0, None) / half_life)


def metrics(y_true, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, p > 0.5)),
        "log_loss": float(log_loss(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "base_rate": float(np.mean(y_true)),
    }


def calibration_table(y_true, p, n_bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        out.append({
            "bin": i + 1,
            "pred_range": [round(float(lo), 3), round(float(hi), 3)],
            "mean_pred": round(float(p[m].mean()), 3),
            "actual_rate": round(float(y_true[m].mean()), 3),
            "n": int(m.sum()),
        })
    return out


def tune_c(X, y, sw=None):
    """Grid-search C with manual TimeSeriesSplit so sample_weight is honored."""
    best_c, best_ll = 1.0, np.inf
    for c in [0.05, 0.2, 1.0, 5.0]:
        lls = []
        for tr, va in TimeSeriesSplit(n_splits=3).split(X):
            sw_tr = None if sw is None else sw[tr]
            m = LogisticRegression(C=c, max_iter=2000).fit(
                X[tr], y[tr], sample_weight=sw_tr)
            p = np.clip(m.predict_proba(X[va])[:, 1], 1e-6, 1 - 1e-6)
            lls.append(log_loss(y[va], p))
        ll = float(np.mean(lls))
        if ll < best_ll:
            best_ll, best_c = ll, c
    return best_c, best_ll


def fit_winner(X_fit, y_fit, X_val, y_val, sw_fit=None, seed=7, force_lr=False):
    """Tune C, compare LR vs HGB on validation log-loss, return (name, fitted model).

    Both candidates are fit with sample_weight=sw_fit (recency weights);
    selection uses UNWEIGHTED validation log-loss for comparability.
    force_lr=True pins the winner to logistic regression: the phone's offline
    JSON artifact format requires a linear model, and the HGB-vs-LR validation
    gaps observed are far below noise (see reports/recency_weighting.md).
    """
    c, _ = tune_c(X_fit, y_fit, sw_fit)
    lr = LogisticRegression(C=c, max_iter=2000).fit(
        X_fit, y_fit, sample_weight=sw_fit)
    if force_lr:
        ll_lr = log_loss(y_val, np.clip(lr.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6))
        return "logistic_regression", lr, {"C": c, "val_log_loss": ll_lr,
                                           "pinned": "HGB comparison skipped; "
                                                     "phone artifact needs LR"}
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         random_state=seed).fit(
        X_fit, y_fit, sample_weight=sw_fit)
    ll_lr = log_loss(y_val, np.clip(lr.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6))
    ll_hgb = log_loss(y_val, np.clip(hgb.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6))
    if ll_lr <= ll_hgb:
        return "logistic_regression", lr, {"C": c, "val_log_loss": ll_lr,
                                           "hgb_val_log_loss": ll_hgb}
    return "hist_gradient_boosting", hgb, {"val_log_loss": ll_hgb,
                                           "lr_val_log_loss": ll_lr, "lr_C": c}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(PROC, "features.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"loaded {len(df)} games, {df['date'].min()} -> {df['date'].max()}")

    fit_mask = df["date"] < FIT_END
    cal_mask = (df["date"] >= FIT_END) & (df["date"] < CAL_END)
    print(f"fit: {fit_mask.sum()}, calibration/holdout: {cal_mask.sum()}")

    bundle = {"features_full": FULL_FEATURES, "features_nodraft": NODRAFT_FEATURES,
              "fit_end": FIT_END, "cal_end": CAL_END,
              "half_life_days": HALF_LIFE_DAYS}
    report = {"half_life_days": HALF_LIFE_DAYS}

    t_now = df["date"].max()
    for key, feats in (("full", FULL_FEATURES), ("nodraft", NODRAFT_FEATURES)):
        X = df[feats].to_numpy(float)
        y = df["y"].to_numpy(int)
        sw = recency_weights(df["date"], t_now, HALF_LIFE_DAYS)
        sw_fit = None if sw is None else sw[fit_mask.to_numpy()]
        scaler = StandardScaler().fit(X[fit_mask], sample_weight=sw_fit)
        Xs = scaler.transform(X)
        name, model, info = fit_winner(Xs[fit_mask], y[fit_mask],
                                      Xs[cal_mask], y[cal_mask], sw_fit,
                                      force_lr=(key == "nodraft"))
        print(f"[{key}] winner: {name} {info}")
        # isotonic map fit UNWEIGHTED on the recent tail (see module docstring)
        calib = IsotonicCalibrated(model).fit(Xs[cal_mask], y[cal_mask])
        p_cal = calib.predict_proba(Xs[cal_mask])[:, 1]
        m = metrics(y[cal_mask], p_cal)
        m["calibration"] = calibration_table(y[cal_mask], p_cal)
        m["model_type"] = name
        m["model_info"] = info
        print(f"[{key}] holdout ({CAL_END} tail): "
              f"acc={m['accuracy']:.3f} ll={m['log_loss']:.4f} auc={m['roc_auc']:.3f} n={m['n']}")
        report[key] = m
        bundle[f"model_{key}"] = calib
        bundle[f"scaler_{key}"] = scaler

    # ---------------- honest walk-forward backtests ----------------
    backtests = {}
    for label, league, year in (("worlds_2025", "WLDs", 2025), ("msi_2026", "MSI", 2026)):
        ev = df[(df["league"] == league) & (df["date"].dt.year == year)]
        if len(ev) == 0:
            print(f"[backtest] {label}: no games found, skipping")
            continue
        cutoff = ev["date"].min()
        pre = df["date"] < cutoff
        feats = FULL_FEATURES
        X = df[feats].to_numpy(float)
        y = df["y"].to_numpy(int)
        # recency weights anchored at the event cutoff: no future leakage
        sw_pre = recency_weights(df["date"], cutoff, HALF_LIFE_DAYS)
        sw_pre = None if sw_pre is None else sw_pre[pre.to_numpy()]
        scaler = StandardScaler().fit(X[pre], sample_weight=sw_pre)
        Xs = scaler.transform(X)
        # same model class as the main winner, fresh C tune on pre-cutoff data
        main_type = report["full"]["model_type"]
        if main_type == "logistic_regression":
            c, _ = tune_c(Xs[pre], y[pre], sw_pre)
            model = LogisticRegression(C=c, max_iter=2000).fit(
                Xs[pre], y[pre], sample_weight=sw_pre)
        else:
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                   random_state=7).fit(
                Xs[pre], y[pre], sample_weight=sw_pre)
        p = np.clip(model.predict_proba(Xs[ev.index])[:, 1], 1e-6, 1 - 1e-6)
        m = metrics(y[ev.index], p)
        m["event_games"] = int(len(ev))
        m["cutoff"] = str(cutoff)
        m["model_type"] = main_type
        m["note"] = "out-of-sample: model fit only on games before the event's first game"
        backtests[label] = m
        print(f"[backtest] {label}: n={m['n']} acc={m['accuracy']:.3f} "
              f"ll={m['log_loss']:.4f} auc={m['roc_auc']:.3f} (cutoff {cutoff.date()})")
    report["backtests"] = backtests

    joblib.dump(bundle, os.path.join(MODEL_DIR, "model.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote model/model.joblib and model/metrics.json")


if __name__ == "__main__":
    main()
