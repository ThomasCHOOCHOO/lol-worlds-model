"""Train and evaluate the LoL match-winner model.

Pipeline (all walk-forward, no leakage):
  * C tuned for LogisticRegression via 3-fold TimeSeriesSplit on the fit set.
  * LogisticRegression vs HistGradientBoosting compared by log-loss on a
    held-out 2026 tail; the winner is kept, then isotonic-calibrated
    (IsotonicRegression fit on that tail, out-of-bounds clipped).
  * Two models are trained: `full` (all features incl. draft) and
    `nodraft` (elo_diff_adj, eg_gold_diff, blue_edge) for pre-draft predictions.
  * Region-strength offsets (src/region_offsets.py): both models train on
    elo_diff_adj = elo_diff + off[region_blue] - off[region_red], which
    corrects weak-region Elo inflation. train.py requires the elo_diff_adj
    column (run src/region_offsets.py after build_features.py). The
    backtests run BOTH the baseline (raw elo_diff) and the offset variant
    (cutoff-scoped offsets + cutoff-scoped team-year regions); the offset
    variant is the shipped number, and delta_log_loss vs baseline is the
    region-offset gate (must confirm the standalone validation
    ΔLL ≈ -0.049 to -0.051).
  * Recency weighting: the fit set is weighted with exponential decay
    w = 0.5 ** (age_days / HALF_LIFE_DAYS), t_now = latest data date
    (event cutoff for backtests). Passed as sample_weight to the LR/HGB fit
    and to the scaler. The isotonic calibration map is fit UNWEIGHTED on the
    recent tail: the tail is already the most recent ~3 months, and weighted
    isotonic risks overfitting the map to the very latest games (verified:
    weighted vs unweighted isotonic differ negligibly on the tail).
  * HONEST calibration (v1.4.0): the shipped map is still fit on the full
    tail, but every retrain must also report NESTED-SPLIT metrics -- the
    isotonic map fit on [FIT_END, CAL_FIT_END) and graded on the untouched
    slice [CAL_FIT_END, CAL_END). In-sample numbers (map graded on the games
    that built it) are reported only as a comparison and must never be
    quoted as calibration quality. The honest map (fit slice only,
    coarsened) is written to model/honest_calibration.json for the app's
    calibration toggle.
  * Predictions are clipped to [PRED_MIN, PRED_MAX] = [0.001, 0.999]
    (see model_utils.py) and calibration maps are coarsened so no step
    rests on fewer than CAL_MIN_STEP_COUNT fit games.
  * Honest walk-forward backtests: refit the winning model type on data
    strictly before each event's first game, then score the event's games:
      - Worlds 2025 (WLDs games in 2025)
      - MSI 2026 (MSI games in 2026)

Outputs (model/):
  model.joblib  - calibrated models, scalers, feature lists, metadata
  metrics.json  - holdout + backtest metrics, calibration table,
                  honest (nested-split) calibration metrics per model
  honest_calibration.json - the honest map (fit-slice only, coarsened)
    for the app toggle: {xs, ys, fitted_on, n, honest_eval{logloss, ece, n}}
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

from model_utils import (PRED_MAX, PRED_MIN, IsotonicCalibrated,
                           apply_isotonic_map, coarsen_isotonic)
from region_offsets import (appearances_from_features, apply_offsets,
                            fit_offsets, team_year_regions)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "model")

FULL_FEATURES = ["elo_diff_adj", "eg_gold_diff", "blue_edge", "d_power", "d_comfort", "ban_edge"]
NODRAFT_FEATURES = ["elo_diff_adj", "eg_gold_diff", "blue_edge"]
# Backtest baseline only: unadjusted Elo, for the region-offset gate.
FULL_FEATURES_BASE = ["elo_diff", "eg_gold_diff", "blue_edge", "d_power", "d_comfort", "ban_edge"]

FIT_END = "2026-07-01"
CAL_END = "2026-10-01"
# Nested split: the honest calibration map is fit on [FIT_END, CAL_FIT_END)
# and graded on the untouched slice [CAL_FIT_END, CAL_END).
CAL_FIT_END = "2026-09-01"
# Coarsening: no calibration step may rest on fewer than this many fit games
# (caps the binomial SE of any step at sqrt(0.25/100) = 0.05).
CAL_MIN_STEP_COUNT = 100

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


def ece(y_true, p, n_bins=10):
    """Expected calibration error: 10 equal-width bins, weighted
    |mean_pred - observed_rate|. Computed on UNTOUCHED games only to be
    meaningful; an ECE of ~0 on the map's own fit games is by construction
    and must never be reported as calibration quality."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (p >= lo) & (p <= hi) if b == n_bins - 1 else (p >= lo) & (p < hi)
        if m.sum():
            total += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(total)


def honest_map_from_bundle(bundle, df, key="full"):
    """Build the HONEST calibration map from an existing trained bundle.

    The base model (fitted on pre-FIT_END games) is untouched. The isotonic
    map is fit ONLY on [FIT_END, CAL_FIT_END), then coarsened, then graded
    on the untouched slice [CAL_FIT_END, CAL_END). Returns
    (honest_json, report_dict) where honest_json matches the exact schema
    the app toggle consumes:
      {xs, ys, fitted_on, n, honest_eval{logloss, ece, n}}
    and report_dict carries the full honest-vs-shipped comparison.
    """
    feats = bundle[f"features_{key}"]
    scaler = bundle[f"scaler_{key}"]
    base_model = bundle[f"model_{key}"].base_model
    X = df[feats].to_numpy(float)
    y = df["y"].to_numpy(int)
    Xs = scaler.transform(X)
    p_raw = np.clip(base_model.predict_proba(Xs)[:, 1], PRED_MIN, PRED_MAX)

    fit_m = (df["date"] >= FIT_END) & (df["date"] < CAL_FIT_END)
    eval_m = (df["date"] >= CAL_FIT_END) & (df["date"] < CAL_END)
    tail_m = (df["date"] >= FIT_END) & (df["date"] < CAL_END)
    y_fit, y_eval, y_tail = y[fit_m], y[eval_m], y[tail_m]
    p_fit, p_eval, p_tail = p_raw[fit_m], p_raw[eval_m], p_raw[tail_m]

    # Honest map: fit on the fit slice only, then coarsened.
    honest_cal = IsotonicCalibrated(base_model).fit(Xs[fit_m], y_fit)
    xs_c, ys_c = coarsen_isotonic(honest_cal.ir, p_fit, CAL_MIN_STEP_COUNT)
    p_eval_honest = apply_isotonic_map(p_eval, xs_c, ys_c)
    # Same map WITHOUT coarsening, for transparency on what coarsening costs.
    p_eval_honest_raw = np.clip(honest_cal.predict_proba(Xs[eval_m])[:, 1],
                                PRED_MIN, PRED_MAX)
    # Shipped-style map: fit on the FULL tail, graded on the SAME tail
    # (in-sample -- reported only as a comparison, never as quality).
    shipped_cal = IsotonicCalibrated(base_model).fit(Xs[tail_m], y_tail)
    p_tail_shipped = np.clip(shipped_cal.predict_proba(Xs[tail_m])[:, 1],
                             PRED_MIN, PRED_MAX)

    honest_json = {
        "xs": [float(v) for v in xs_c],
        "ys": [float(v) for v in ys_c],
        "fitted_on": "2026-07-01..2026-08-31",
        "n": int(fit_m.sum()),
        "honest_eval": {
            "logloss": float(log_loss(y_eval, p_eval_honest)),
            "ece": float(ece(y_eval, p_eval_honest)),
            "n": int(eval_m.sum()),
        },
    }
    report = {
        "key": key,
        "fitted_on": "2026-07-01..2026-08-31",
        "n_fit": int(fit_m.sum()),
        "n_eval": int(eval_m.sum()),
        "n_steps_coarsened": len(xs_c),
        "n_steps_uncoarsened": len(honest_cal.ir.X_thresholds_),
        "honest_coarsened": {
            "logloss": float(log_loss(y_eval, p_eval_honest)),
            "ece": float(ece(y_eval, p_eval_honest)),
        },
        "honest_uncoarsened": {
            "logloss": float(log_loss(y_eval, p_eval_honest_raw)),
            "ece": float(ece(y_eval, p_eval_honest_raw)),
        },
        "raw_uncalibrated_on_eval": {
            "logloss": float(log_loss(y_eval, p_eval)),
            "ece": float(ece(y_eval, p_eval)),
        },
        "shipped_style_in_sample_on_full_tail": {
            "logloss": float(log_loss(y_tail, p_tail_shipped)),
            "ece": float(ece(y_tail, p_tail_shipped)),
            "n": int(tail_m.sum()),
            "note": ("IN-SAMPLE: map graded on the games that built it. "
                     "Do not quote as calibration quality."),
        },
    }
    return honest_json, report


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


def backtest_fit_predict(X, y, pre, sw_pre, main_type, ev_index):
    """Fit the main model class on pre-cutoff data; predict event games.

    Same model class as the main winner, fresh C tune on pre-cutoff data.
    """
    scaler = StandardScaler().fit(X[pre], sample_weight=sw_pre)
    Xs = scaler.transform(X)
    if main_type == "logistic_regression":
        c, _ = tune_c(Xs[pre], y[pre], sw_pre)
        model = LogisticRegression(C=c, max_iter=2000).fit(
            Xs[pre], y[pre], sample_weight=sw_pre)
    else:
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               random_state=7).fit(
            Xs[pre], y[pre], sample_weight=sw_pre)
    return np.clip(model.predict_proba(Xs[ev_index])[:, 1], 1e-6, 1 - 1e-6)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(PROC, "features.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"loaded {len(df)} games, {df['date'].min()} -> {df['date'].max()}")
    if "elo_diff_adj" not in df.columns:
        raise SystemExit(
            "features.csv lacks elo_diff_adj: run src/region_offsets.py "
            "after src/build_features.py to fit offsets and add the column")

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

    # ---------------- honest (nested-split) calibration ----------------
    # The shipped maps above are fit on the full tail and graded on the same
    # tail (in-sample). Every retrain must also produce the honest map
    # (fit slice only) + honest metrics on the untouched eval slice.
    for key in ("full", "nodraft"):
        honest_json, honest_report = honest_map_from_bundle(bundle, df, key)
        report[key]["honest_calibration"] = honest_report
        print(f"[{key}] honest (map fit Jul-Aug, graded on untouched Sep): "
              f"ll={honest_report['honest_coarsened']['logloss']:.4f} "
              f"ece={honest_report['honest_coarsened']['ece']:.4f} "
              f"n={honest_report['n_eval']} steps={honest_report['n_steps_coarsened']} "
              f"(raw ll={honest_report['raw_uncalibrated_on_eval']['logloss']:.4f}, "
              f"shipped-style in-sample ll="
              f"{honest_report['shipped_style_in_sample_on_full_tail']['logloss']:.4f})")
    honest_json, _ = honest_map_from_bundle(bundle, df, "full")
    with open(os.path.join(MODEL_DIR, "honest_calibration.json"), "w") as f:
        json.dump(honest_json, f, indent=2)
    print("wrote model/honest_calibration.json (full-model honest map, "
          "for the app calibration toggle)")

    # ---------------- honest walk-forward backtests ----------------
    # Each event is scored TWICE: a baseline on raw elo_diff, and the offset
    # variant on elo_diff_adj recomputed from CUTOFF-SCOPED offsets (fit only
    # on cross-region games before the event) + cutoff-scoped team-year
    # regions. The offset variant is the shipped number; delta_log_loss vs
    # baseline is the region-offset gate.
    backtests = {}
    for label, league, year in (("worlds_2025", "WLDs", 2025), ("msi_2026", "MSI", 2026)):
        ev = df[(df["league"] == league) & (df["date"].dt.year == year)]
        if len(ev) == 0:
            print(f"[backtest] {label}: no games found, skipping")
            continue
        cutoff = ev["date"].min()
        pre = df["date"] < cutoff
        y = df["y"].to_numpy(int)
        # recency weights anchored at the event cutoff: no future leakage
        sw_pre = recency_weights(df["date"], cutoff, HALF_LIFE_DAYS)
        sw_pre = None if sw_pre is None else sw_pre[pre.to_numpy()]
        main_type = report["full"]["model_type"]
        # baseline: unadjusted Elo
        X_base = df[FULL_FEATURES_BASE].to_numpy(float)
        p_base = backtest_fit_predict(X_base, y, pre, sw_pre, main_type, ev.index)
        m_base = metrics(y[ev.index], p_base)
        # offset variant: cutoff-scoped offsets + cutoff-scoped regions
        team_year_cut = team_year_regions(appearances_from_features(df), cutoff=cutoff)
        off_cut = fit_offsets(df, team_year_cut, cutoff=cutoff)["offsets_elo"]
        df_off = df.copy()
        df_off["elo_diff_adj"] = apply_offsets(df_off, off_cut, team_year_cut)
        X_off = df_off[FULL_FEATURES].to_numpy(float)
        p_off = backtest_fit_predict(X_off, y, pre, sw_pre, main_type, ev.index)
        m = metrics(y[ev.index], p_off)
        m["baseline_log_loss"] = m_base["log_loss"]
        m["baseline_accuracy"] = m_base["accuracy"]
        m["delta_log_loss"] = m["log_loss"] - m_base["log_loss"]
        m["offsets_used"] = {r: round(float(v), 1) for r, v in off_cut.items()}
        m["event_games"] = int(len(ev))
        m["cutoff"] = str(cutoff)
        m["model_type"] = main_type
        m["note"] = ("out-of-sample: model fit only on games before the event's first game; "
                     "offsets fit only on cross-region games before the event")
        backtests[label] = m
        print(f"[backtest] {label}: n={m['n']} acc={m['accuracy']:.3f} "
              f"ll={m['log_loss']:.4f} auc={m['roc_auc']:.3f} "
              f"(baseline ll={m_base['log_loss']:.4f}, "
              f"delta_ll={m['delta_log_loss']:+.4f}) (cutoff {cutoff.date()})")
    report["backtests"] = backtests

    joblib.dump(bundle, os.path.join(MODEL_DIR, "model.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote model/model.joblib and model/metrics.json")


if __name__ == "__main__":
    main()
