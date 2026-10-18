"""Validate the tier-conditional player/team Elo blend (offline; default OFF).

    python src/train_blend.py

Protocol (mirrors src/train.py and reports/player_ratings_analysis.md):
- Fit set: games with date < 2026-07-01. Tail (honest OOS): 2026-07-01..2026-10-01.
- The blend weight w(inst) = sigmoid(a + b*inst) is FIT by minimizing
  3-fold TimeSeriesSplit CV log-loss on the fit set only (never the tail).
  The final LR (C tuned, 180-day recency weights, scaler fit on fit set)
  is then evaluated RAW (no isotonic) on the tail.
- Primary comparison: blend vs M0 (team Elo only), paired bootstrap
  (20k resamples) on per-game log-loss differences.
- Hard gate: tier-1+intl must be NO WORSE than M0 (non-inferiority: the
  95% CI upper bound on blend-minus-M0 log-loss must be < +0.002, the
  project's shipping gate) AND the pooled tail gain must survive
  (CI excludes 0).
- Also reported: minors, academy/dev (NACL + LCK Challengers), KeSPA Cup,
  the disagreement subset (|player_diff - team_diff| > 100), honest
  walk-forward backtests (Worlds 2025, MSI 2026; (a, b) refit on pre-event
  data only), and a tier-augmented w as a sensitivity check.
- Roster-strength check: roster_strength.py's delta_2026 is validated the
  same way (LR on fit set, RAW tail + early-2026 window, base vs adjusted).

Outputs (only when the script is run explicitly -- nothing here touches
the default training path):
- model/model_blend.joblib: blend LR bundles (full + nodraft) + (a, b)
- model/player_blend_snapshot.json: blend_params filled in from the fit
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from model_utils import IsotonicCalibrated  # noqa: E402
from roster_strength import compute_adjustments  # noqa: E402
from train import (CAL_END, FIT_END, HALF_LIFE_DAYS, metrics,  # noqa: E402
                   recency_weights, tune_c)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "model")
RNG = np.random.default_rng(7)
N_BOOT = 20000

TIER1 = {"LCK", "LPL", "LEC", "LCS", "LTA", "LTA N", "LTA S", "LCP"}
INTL = {"MSI", "WLDs", "EWC"}
OTHER_FEATURES_FULL = ["eg_gold_diff", "blue_edge", "d_power", "d_comfort", "ban_edge"]
OTHER_FEATURES_NODRAFT = ["eg_gold_diff", "blue_edge"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blend_feature(df, a, b, tier_c=0.0):
    """elo_diff_blend: per-side blend, then the region-offset diff.

    Each side's rating = w(inst_side) * player_elo + (1 - w) * team_elo,
    w = sigmoid(a + b*inst + tier_c*is_tier1). The region correction is
    orthogonal to the player/team question, so it is added after blending
    (elo_diff_adj - elo_diff from features.csv)."""
    is_t1 = df["league"].isin(TIER1).to_numpy(float)
    w_b = sigmoid(a + b * df["instability_blue"].to_numpy() + tier_c * is_t1)
    w_r = sigmoid(a + b * df["instability_red"].to_numpy() + tier_c * is_t1)
    team_b = df["team_elo_blue"].to_numpy()
    team_r = df["team_elo_red"].to_numpy()
    play_b = df["player_elo_blue"].to_numpy()
    play_r = df["player_elo_red"].to_numpy()
    off = (df["elo_diff_adj"] - df["elo_diff"]).to_numpy()
    return (w_b * play_b + (1 - w_b) * team_b
            - (w_r * play_r + (1 - w_r) * team_r) + off)


def oof_predictions(df_fit, feature_vec, c_fixed=1.0):
    """Out-of-fold RAW probs for one candidate feature vector (3-fold TSS).

    LR with fixed C, recency weights, scaler fit per fold. TimeSeriesSplit
    never tests the earliest quarter, so those rows stay NaN; cv_metrics
    drops them. The covered ~3/4 of the fit set still holds ~19.6k games
    for the (a, b) search, and the final LR is fit on the full fit set."""
    X_other = np.column_stack([df_fit[c].to_numpy() for c in OTHER_FEATURES_FULL])
    X = np.column_stack([feature_vec, X_other])
    y = df_fit["y"].to_numpy(int)
    dates = df_fit["date"]
    sw = recency_weights(dates, dates.max(), HALF_LIFE_DAYS)
    oof = np.full(len(df_fit), np.nan)
    for tr, va in TimeSeriesSplit(n_splits=3).split(X):
        sc = StandardScaler().fit(X[tr], sample_weight=None if sw is None else sw[tr])
        m = LogisticRegression(C=c_fixed, max_iter=500).fit(
            sc.transform(X[tr]), y[tr],
            sample_weight=None if sw is None else sw[tr])
        oof[va] = np.clip(m.predict_proba(sc.transform(X[va]))[:, 1], 1e-6, 1 - 1e-6)
    return oof


def cv_metrics(df_fit, oof):
    """(pooled CV log-loss, tier-1+intl CV log-loss) for OOF predictions.

    Only rows covered by a test fold are scored (the earliest quarter has
    no OOF prediction under TimeSeriesSplit)."""
    m = ~np.isnan(oof)
    y = df_fit["y"].to_numpy(int)[m]
    o = oof[m]
    pooled = log_loss(y, o)
    mt = df_fit["league"].isin(TIER1 | INTL).to_numpy()[m]
    return pooled, log_loss(y[mt], o[mt])


def fit_ab(df_fit, tier_c_values=(0.0,), cap_extra=0.002):
    """Fit (a, b) -- and optionally tier_c -- by CONSTRAINED CV log-loss.

    The unconstrained pooled-CV optimum degenerates to w = 1 everywhere
    (pure player Elo), because the fit set is ~76% minor-league games where
    player Elo wins. That reproduces the M2 failure the blend exists to fix.
    So (a, b) is fit subject to the non-inferiority constraint on the
    fit set's tier-1+intl games: tier-1+intl CV log-loss must be no more
    than M0's + 0.002 (the shipping gate). Among feasible candidates, pick
    the one minimizing pooled CV log-loss. M0 (w = 0) is always feasible,
    so the feasible set is non-empty.

    Returns dict with a, b, tier_c, cv_logloss (pooled), cv_tier1, and the
    unconstrained best for the report.
    """
    y = df_fit["y"].to_numpy(int)
    team_vec = df_fit["elo_diff_adj"].to_numpy()
    oof_m0 = oof_predictions(df_fit, team_vec)
    pooled_m0, tier1_m0 = cv_metrics(df_fit, oof_m0)
    cap = tier1_m0 + cap_extra
    print(f"  [fit_ab] M0 CV: pooled={pooled_m0:.4f} tier-1+intl={tier1_m0:.4f} "
          f"(constraint cap={cap:.4f})")

    a_grid = [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
    b_grid = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    best = None          # (pooled_cv, a, b, tier_c, tier1_cv) over all tier_c
    best_tier0 = None    # same, restricted to tier_c == 0
    best_by_tierc = {}   # tier_c -> best feasible (pooled_cv, a, b, tier1_cv)
    best_uncon = None    # (pooled_cv, a, b, tier_c, tier1_cv)
    for tier_c in tier_c_values:
        for a in a_grid:
            for b in b_grid:
                vec = blend_feature(df_fit, a, b, tier_c)
                oof = oof_predictions(df_fit, vec)
                pooled, tier1 = cv_metrics(df_fit, oof)
                if best_uncon is None or pooled < best_uncon[0]:
                    best_uncon = (pooled, a, b, tier_c, tier1)
                if tier1 <= cap:
                    if best is None or pooled < best[0]:
                        best = (pooled, a, b, tier_c, tier1)
                    if tier_c == 0.0 and (best_tier0 is None or pooled < best_tier0[0]):
                        best_tier0 = (pooled, a, b, tier_c, tier1)
                    cur = best_by_tierc.get(tier_c)
                    if cur is None or pooled < cur[0]:
                        best_by_tierc[tier_c] = (pooled, a, b, tier1)
    if best is None:
        # Cannot happen (w -> 0 approaches M0, which satisfies the cap), but
        # fail loudly rather than silently shipping a constraint violation.
        raise RuntimeError("no feasible (a, b): even w -> 0 violates the tier-1 cap")
    out = {"a": float(best[1]), "b": float(best[2]), "tier_c": float(best[3]),
           "cv_logloss": float(best[0]), "cv_tier1": float(best[4]),
           "m0_cv_logloss": float(pooled_m0), "m0_cv_tier1": float(tier1_m0),
           "best_tier0": {"a": float(best_tier0[1]), "b": float(best_tier0[2]),
                          "cv_logloss": float(best_tier0[0]),
                          "cv_tier1": float(best_tier0[4])},
           "best_by_tierc": {str(tc): {"a": float(v[1]), "b": float(v[2]),
                                       "cv_logloss": float(v[0]),
                                       "cv_tier1": float(v[3])}
                             for tc, v in best_by_tierc.items()},
           "unconstrained": {"a": float(best_uncon[1]), "b": float(best_uncon[2]),
                             "tier_c": float(best_uncon[3]),
                             "cv_logloss": float(best_uncon[0]),
                             "cv_tier1": float(best_uncon[4])}}
    return out


def fit_lr_eval(df_fit, df_eval, feature_fn, other_features, t_now=None):
    """Fit LR (C tuned, recency weights) on df_fit; return RAW probs on df_eval."""
    X_fit = np.column_stack([feature_fn(df_fit)]
                            + [df_fit[c].to_numpy() for c in other_features])
    X_ev = np.column_stack([feature_fn(df_eval)]
                           + [df_eval[c].to_numpy() for c in other_features])
    y_fit = df_fit["y"].to_numpy(int)
    t_now = df_fit["date"].max() if t_now is None else t_now
    sw = recency_weights(df_fit["date"], t_now, HALF_LIFE_DAYS)
    sc = StandardScaler().fit(X_fit, sample_weight=sw)
    c, _ = tune_c(sc.transform(X_fit), y_fit, sw)
    m = LogisticRegression(C=c, max_iter=2000).fit(
        sc.transform(X_fit), y_fit, sample_weight=sw)
    p = np.clip(m.predict_proba(sc.transform(X_ev))[:, 1], 1e-6, 1 - 1e-6)
    return p, {"C": c}


def paired_boot(diff, n=N_BOOT):
    """95% percentile CI + P(diff < 0) for per-game log-loss differences."""
    diff = np.asarray(diff)
    idx = RNG.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi), float((means < 0).mean())


def adopt_tier_c(fit_result):
    """Adopt the tier_c term only if it clearly beats instability-only.

    Bar: constrained pooled CV log-loss at least 0.002 better than the
    best tier_c = 0 candidate. Otherwise the simpler instability-only
    blend ships (one less context variable, no league plumbing needed
    at predict time). Returns (a, b, tier_c)."""
    best = fit_result
    t0 = fit_result["best_tier0"]
    if best["tier_c"] != 0.0 and best["cv_logloss"] < t0["cv_logloss"] - 0.002:
        return float(best["a"]), float(best["b"]), float(best["tier_c"])
    return float(t0["a"]), float(t0["b"]), 0.0


def subset_report(name, sub, p_blend, p_m0, p_m1=None):
    """RAW log-loss + paired-bootstrap diffs (blend - M0, M1 - M0) on a subset."""
    y = sub["y"].to_numpy(int)
    ll_b = log_loss(y, p_blend)
    ll_0 = log_loss(y, p_m0)
    per_b = -(y * np.log(p_blend) + (1 - y) * np.log(1 - p_blend))
    per_0 = -(y * np.log(p_m0) + (1 - y) * np.log(1 - p_m0))
    lo, hi, pbetter = paired_boot(per_b - per_0)
    line = (f"{name:22s} n={len(sub):5d} M0={ll_0:.4f} blend={ll_b:.4f} "
            f"diff={ll_b - ll_0:+.4f} CI[{lo:+.4f},{hi:+.4f}] P(blend better)={pbetter:.3f}")
    if p_m1 is not None:
        ll_1 = log_loss(y, p_m1)
        per_1 = -(y * np.log(p_m1) + (1 - y) * np.log(1 - p_m1))
        lo1, hi1, pb1 = paired_boot(per_1 - per_0)
        line += (f" | M1(pure player)={ll_1:.4f} diff={ll_1 - ll_0:+.4f} "
                 f"CI[{lo1:+.4f},{hi1:+.4f}]")
    print(line)
    return {"n": int(len(sub)), "m0": float(ll_0), "blend": float(ll_b),
            "diff": float(ll_b - ll_0), "ci": [lo, hi], "p_better": float(pbetter)}


def main():
    feat = pd.read_csv(os.path.join(PROC, "features.csv"), parse_dates=["date"])
    pf = pd.read_csv(os.path.join(PROC, "player_features.csv"), parse_dates=["date"])
    df = feat.merge(pf[["gameid", "team_elo_blue", "team_elo_red",
                        "player_elo_blue", "player_elo_red", "player_elo_diff",
                        "team_elo_diff",
                        "instability_blue", "instability_red", "instability_max"]],
                    on="gameid", how="inner", validate="one_to_one")
    print(f"merged {len(df)} games ({len(feat)} in features.csv, "
          f"{len(pf)} in player_features.csv)")
    assert len(df) == len(feat), "player_features.csv is missing games"
    df = df.sort_values("date").reset_index(drop=True)

    # roster_strength deltas (computed once, used by the RS validation below;
    # defined here so fit/tail slices inherit the delta_diff column).
    adj = compute_adjustments()
    delta = {t: (r.get("delta_2026") or 0.0) for t, r in adj.items()}
    df["delta_diff"] = df["blue"].map(delta).fillna(0.0) - df["red"].map(delta).fillna(0.0)

    fit = df[df["date"] < FIT_END].copy()
    tail = df[(df["date"] >= FIT_END) & (df["date"] < CAL_END)].copy()
    print(f"fit: {len(fit)}, tail: {len(tail)}")

    # ---- fit (a, b) [and tier_c] on the fit set only, constrained ----
    ab = fit_ab(fit, tier_c_values=(0.0, -1.0, -2.0))
    a, b, tier_c = adopt_tier_c(ab)
    print(f"fitted blend: a={a:.3f} b={b:.3f} tier_c={tier_c:.3f} "
          f"(constrained CV log-loss {ab['cv_logloss']:.4f}; "
          f"tier-1+intl CV {ab['cv_tier1']:.4f} vs M0 {ab['m0_cv_tier1']:.4f})")
    print(f"  unconstrained optimum (for the report): a={ab['unconstrained']['a']:.2f} "
          f"b={ab['unconstrained']['b']:.2f} tier_c={ab['unconstrained']['tier_c']:.2f} "
          f"pooled CV {ab['unconstrained']['cv_logloss']:.4f} "
          f"tier-1+intl CV {ab['unconstrained']['cv_tier1']:.4f} -- degenerates to w=1")
    print(f"  best tier_c=0: a={ab['best_tier0']['a']:.3f} b={ab['best_tier0']['b']:.3f} "
          f"pooled CV {ab['best_tier0']['cv_logloss']:.4f}")
    for inst in (0.0, 0.2, 0.5, 1.0):
        print(f"  w(inst={inst}) = {sigmoid(a + b * inst + tier_c * 0):.3f} "
              f"(minor); w(inst={inst}, tier-1) = {sigmoid(a + b * inst + tier_c):.3f}")
    blend_fn = lambda d: blend_feature(d, a, b, tier_c)
    player_fn = lambda d: d["player_elo_diff"].to_numpy() + (d["elo_diff_adj"] - d["elo_diff"]).to_numpy()
    team_fn = lambda d: d["elo_diff_adj"].to_numpy()

    # ---- full-model evaluation on the tail ----
    results = {}
    p_blend, _ = fit_lr_eval(fit, tail, blend_fn, OTHER_FEATURES_FULL)
    p_m0, _ = fit_lr_eval(fit, tail, team_fn, OTHER_FEATURES_FULL)
    p_m1, _ = fit_lr_eval(fit, tail, player_fn, OTHER_FEATURES_FULL)
    tail = tail.copy()
    tail["p_blend"], tail["p_m0"], tail["p_m1"] = p_blend, p_m0, p_m1

    print("\n=== Full model, RAW tail log-loss (blend vs M0 team-Elo, M1 pure player) ===")
    t1intl = tail[tail["league"].isin(TIER1 | INTL)]
    minors = tail[~tail["league"].isin(TIER1 | INTL)]
    acad = tail[tail["league"].isin({"NACL", "LCKC"})]
    kespa = tail[tail["league"] == "KeSPA Cup"]
    disag = tail[(tail["player_elo_diff"] - tail["team_elo_diff"]).abs() > 100]
    results["tail"] = {
        "tier1_intl": subset_report("Tier-1 + internationals", t1intl,
                                    t1intl["p_blend"].to_numpy(), t1intl["p_m0"].to_numpy(),
                                    t1intl["p_m1"].to_numpy()),
        "pooled": subset_report("Pooled tail", tail,
                                tail["p_blend"].to_numpy(), tail["p_m0"].to_numpy(),
                                tail["p_m1"].to_numpy()),
        "minors": subset_report("Minors", minors,
                                minors["p_blend"].to_numpy(), minors["p_m0"].to_numpy(),
                                minors["p_m1"].to_numpy()),
        "academy_dev": subset_report("Academy/dev (NACL+LCKC)", acad,
                                     acad["p_blend"].to_numpy(), acad["p_m0"].to_numpy(),
                                     acad["p_m1"].to_numpy()),
        "kespa": subset_report("KeSPA Cup", kespa,
                               kespa["p_blend"].to_numpy(), kespa["p_m0"].to_numpy(),
                               kespa["p_m1"].to_numpy()),
        "disagreement_gt100": subset_report("|player-team| > 100", disag,
                                            disag["p_blend"].to_numpy(), disag["p_m0"].to_numpy(),
                                            disag["p_m1"].to_numpy()),
    }
    # nodraft variant (same (a, b); the blend is about the Elo feature)
    p_blend_nd, _ = fit_lr_eval(fit, tail, blend_fn, OTHER_FEATURES_NODRAFT)
    p_m0_nd, _ = fit_lr_eval(fit, tail, team_fn, OTHER_FEATURES_NODRAFT)
    t1 = tail[tail["league"].isin(TIER1 | INTL)]
    print("\n=== Nodraft model, RAW tail ===")
    results["nodraft"] = {
        "tier1_intl": subset_report("Tier-1 + intl (nodraft)", t1,
                                    p_blend_nd[tail["league"].isin(TIER1 | INTL).to_numpy()],
                                    p_m0_nd[tail["league"].isin(TIER1 | INTL).to_numpy()]),
        "pooled": subset_report("Pooled (nodraft)", tail, p_blend_nd, p_m0_nd),
    }

    # ---- gate verdict ----
    gate = results["tail"]["tier1_intl"]
    pooled = results["tail"]["pooled"]
    noninf = gate["ci"][1] < 0.002
    pooled_gain = pooled["ci"][1] < 0.0
    print(f"\nGATE: tier-1+intl non-inferiority (CI upper < +0.002): "
          f"{'PASS' if noninf else 'FAIL'} (upper={gate['ci'][1]:+.4f}); "
          f"pooled gain (CI excludes 0): {'PASS' if pooled_gain else 'FAIL'} "
          f"(CI [{pooled['ci'][0]:+.4f},{pooled['ci'][1]:+.4f}])")
    results["gate"] = {"noninferiority_pass": bool(noninf),
                       "pooled_gain_pass": bool(pooled_gain),
                       "both_pass": bool(noninf and pooled_gain)}

    # ---- sensitivity 1: tier-term variants (best feasible per tier_c) ----
    # The primary (tier_c = 0) failed the gate. Theory says the tier term is
    # NECESSARY, not optional: 73% of games have instability 0 in BOTH
    # regimes, and at inst = 0 the regimes want opposite weights (minors ->
    # player Elo, tier-1 -> team Elo). Only tier separates them there.
    # These variants are exploratory (not the pre-registered primary);
    # their tail numbers are a sensitivity check, not a gate-passing claim.
    # If one passes the tail gate here, it still needs re-validation on
    # fresh data (Worlds 2026 Swiss+knockout) before shipping.
    results["sensitivity"] = {}
    t1m = tail["league"].isin(TIER1 | INTL).to_numpy()
    for tc in (-1.0, -2.0):
        rec = ab["best_by_tierc"].get(str(tc))
        if rec is None:
            print(f"\n=== Sensitivity: tier_c={tc}: no feasible candidate ===")
            results["sensitivity"][f"tier_c_{tc}"] = {"note": "no feasible candidate"}
            continue
        fn_tc = lambda d, a=rec["a"], b=rec["b"], tc=tc: blend_feature(d, a, b, tc)
        p_tc, _ = fit_lr_eval(fit, tail, fn_tc, OTHER_FEATURES_FULL)
        print(f"\n=== Sensitivity: tier_c={tc} blend "
              f"(a={rec['a']:.2f} b={rec['b']:.2f}) on the tail ===")
        results["sensitivity"][f"tier_c_{tc}"] = {
            "a": rec["a"], "b": rec["b"], "tier_c": tc,
            "tier1_intl": subset_report(f"Tier-1+intl (tier_c={tc})", t1,
                                        p_tc[t1m], t1["p_m0"].to_numpy()),
            "pooled": subset_report(f"Pooled (tier_c={tc})", tail, p_tc,
                                    tail["p_m0"].to_numpy()),
        }

    # ---- sensitivity 2: strict-cap variant (fit-set tier-1 CV <= M0's) ----
    ab_strict = fit_ab(fit, tier_c_values=(0.0,), cap_extra=0.0)
    a_s, b_s, _ = adopt_tier_c(ab_strict)
    fn_s = lambda d: blend_feature(d, a_s, b_s, 0.0)
    p_s, _ = fit_lr_eval(fit, tail, fn_s, OTHER_FEATURES_FULL)
    t1m = tail["league"].isin(TIER1 | INTL).to_numpy()
    print(f"\n=== Sensitivity: strict-cap blend (a={a_s:.2f} b={b_s:.2f}) ===")
    results["sensitivity"]["strict_cap"] = {
        "a": a_s, "b": b_s, "tier_c": 0.0,
        "tier1_intl": subset_report("Tier-1+intl (strict-cap)", t1,
                                    p_s[t1m], t1["p_m0"].to_numpy()),
        "pooled": subset_report("Pooled (strict-cap)", tail, p_s,
                                tail["p_m0"].to_numpy()),
    }

    # ---- honest walk-forward backtests (refit (a, b) + LR pre-event only) ----
    print("\n=== Backtests (honest: (a, b) + LR refit on pre-event data only) ===")
    results["backtests"] = {}
    for label, league, year in (("worlds_2025", "WLDs", 2025), ("msi_2026", "MSI", 2026)):
        ev = df[(df["league"] == league) & (df["date"].dt.year == year)].copy()
        if len(ev) == 0:
            continue
        cutoff = ev["date"].min()
        pre = df[df["date"] < cutoff].copy()
        ab_ev = fit_ab(pre, tier_c_values=(0.0, -1.0, -2.0))
        a_ev, b_ev, c_ev = adopt_tier_c(ab_ev)
        fn_ev = lambda d, a_ev=a_ev, b_ev=b_ev, c_ev=c_ev: blend_feature(d, a_ev, b_ev, c_ev)
        p_b, _ = fit_lr_eval(pre, ev, fn_ev, OTHER_FEATURES_FULL, t_now=cutoff)
        p_0, _ = fit_lr_eval(pre, ev, team_fn, OTHER_FEATURES_FULL, t_now=cutoff)
        y = ev["y"].to_numpy(int)
        ll_b, ll_0 = log_loss(y, p_b), log_loss(y, p_0)
        per = -(y * np.log(p_b) + (1 - y) * np.log(1 - p_b)) + \
              (y * np.log(p_0) + (1 - y) * np.log(1 - p_0))
        lo, hi, pb = paired_boot(per)
        print(f"{label}: n={len(ev)} M0={ll_0:.4f} blend={ll_b:.4f} "
              f"diff={ll_b - ll_0:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"(fitted a={a_ev:.2f} b={b_ev:.2f} tier_c={c_ev:.2f} on pre-event data)")
        results["backtests"][label] = {"n": int(len(ev)), "m0": float(ll_0),
                                        "blend": float(ll_b),
                                        "diff": float(ll_b - ll_0),
                                        "ci": [lo, hi], "a": a_ev, "b": b_ev,
                                        "tier_c": c_ev}

    # ---- roster_strength.py validation ----
    print("\n=== roster_strength.py delta_2026 validation (base vs adjusted) ===")
    t1_teams = {t for t in delta if any(
        (df["league"].isin(TIER1 | INTL)) & ((df["blue"] == t) | (df["red"] == t)))}
    dvals = [abs(delta[t]) for t in t1_teams]
    print(f"tier-1+intl teams with a delta: {len(dvals)}; "
          f"|delta| mean={np.mean(dvals):.1f} median={np.median(dvals):.1f} "
          f"max={max(dvals):.1f}")
    adj_fn = lambda d: (d["elo_diff_adj"] + d["delta_diff"]).to_numpy()
    p_adj, _ = fit_lr_eval(fit, tail, adj_fn, OTHER_FEATURES_FULL)
    t1 = tail[tail["league"].isin(TIER1 | INTL)]
    results["roster_strength"] = {
        "tail_tier1_intl": subset_report("RS: tier-1+intl (adj vs base)",
                                         t1, p_adj[tail["league"].isin(TIER1 | INTL).to_numpy()],
                                         t1["p_m0"].to_numpy()),
        "tail_pooled": subset_report("RS: pooled (adj vs base)", tail, p_adj,
                                     tail["p_m0"].to_numpy()),
    }
    # early-2026 window: LR fit on 2024-2025 only, evaluate Jan-Mar 2026
    pre26 = df[df["date"] < "2026-01-01"].copy()
    early = df[(df["date"] >= "2026-01-01") & (df["date"] < "2026-04-01")].copy()
    p_adj_e, _ = fit_lr_eval(pre26, early, adj_fn, OTHER_FEATURES_FULL,
                             t_now=pd.Timestamp("2025-12-31"))
    p_0_e, _ = fit_lr_eval(pre26, early, team_fn, OTHER_FEATURES_FULL,
                           t_now=pd.Timestamp("2025-12-31"))
    early_t1 = early[early["league"].isin(TIER1 | INTL)]
    m_early = early["league"].isin(TIER1 | INTL).to_numpy()
    results["roster_strength"]["early2026_tier1_intl"] = subset_report(
        "RS: early-2026 tier-1+intl", early_t1, p_adj_e[m_early], p_0_e[m_early])
    results["roster_strength"]["early2026_pooled"] = subset_report(
        "RS: early-2026 pooled", early, p_adj_e, p_0_e)
    print("RS caveat: delta_2026 uses full-year 2026 rosters, so the 2026 "
          "windows above have mild roster leakage (see report).")

    # ---- write the blend bundle + snapshot params ----
    bundle = {}
    for key, other in (("full", OTHER_FEATURES_FULL), ("nodraft", OTHER_FEATURES_NODRAFT)):
        X_fit = np.column_stack([blend_fn(fit)] + [fit[c].to_numpy() for c in other])
        y_fit = fit["y"].to_numpy(int)
        sw = recency_weights(fit["date"], fit["date"].max(), HALF_LIFE_DAYS)
        sc = StandardScaler().fit(X_fit, sample_weight=sw)
        c, _ = tune_c(sc.transform(X_fit), y_fit, sw)
        m = LogisticRegression(C=c, max_iter=2000).fit(
            sc.transform(X_fit), y_fit, sample_weight=sw)
        X_tail = np.column_stack([blend_fn(tail)] + [tail[c].to_numpy() for c in other])
        y_tail = tail["y"].to_numpy(int)
        calib = IsotonicCalibrated(m).fit(sc.transform(X_tail), y_tail)
        bundle[f"features_{key}"] = ["elo_diff_blend"] + list(other)
        bundle[f"model_{key}"] = calib
        bundle[f"scaler_{key}"] = sc
        p_cal = calib.predict_proba(sc.transform(X_tail))[:, 1]
        print(f"[blend bundle {key}] C={c} tail RAW ll="
              f"{log_loss(y_tail, np.clip(m.predict_proba(sc.transform(X_tail))[:, 1], 1e-6, 1 - 1e-6)):.4f} "
              f"CAL ll={log_loss(y_tail, p_cal):.4f} (CAL in-sample for the map)")
    bundle.update({"blend_a": float(a), "blend_b": float(b),
                   "blend_tier_c": float(tier_c),
                   "fit_end": FIT_END, "cal_end": CAL_END,
                   "half_life_days": HALF_LIFE_DAYS,
                   "gate": results["gate"]})
    joblib.dump(bundle, os.path.join(MODEL_DIR, "model_blend.joblib"))
    print("wrote model/model_blend.joblib (blend bundle; BINARY -- local only, not pushed)")
    snap_path = os.path.join(MODEL_DIR, "player_blend_snapshot.json")
    with open(snap_path) as f:
        snap = json.load(f)
    snap["blend_params"] = {"a": float(a), "b": float(b), "tier_c": float(tier_c),
                            "cv_logloss": float(ab["cv_logloss"]),
                            "gate": results["gate"]}
    with open(snap_path, "w") as f:
        json.dump(snap, f, indent=1, sort_keys=True)
    print("filled blend_params in model/player_blend_snapshot.json")

    with open(os.path.join(BASE, "reports", "tier_conditional_blend_validation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote reports/tier_conditional_blend_validation.json")


if __name__ == "__main__":
    main()
