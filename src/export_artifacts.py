"""Export ONE self-contained JSON for fully offline phone-app inference.

Reads model/model.joblib + model/snapshots.joblib + src/roster_strength.py and
writes model/phone_artifacts.json with everything the app needs, no server:

  * feature order, logistic-regression coefficients + intercept, scaler
    mean/scale (both the `full` draft model and the `nodraft` pre-draft model)
  * isotonic calibration maps as plain arrays (per model): clip p_raw to
    [x_min, x_max], linearly interpolate over (xs, ys), clip to [1e-6, 1-1e-6]
    -- exactly what sklearn 1.9 IsotonicRegression(out_of_bounds="clip") does
  * per model, a `calibration_honest` map (same shape): the nested-split
    honest map (fit Jul-Aug only, graded on untouched Sep games). The app
    toggle switches between `calibration` (shipped) and
    `calibration_honest`. Full-model honest map comes from the canonical
    model/honest_calibration.json (verified against a recomputation from
    the exported bundle); nodraft honest map is computed the same way.
  * `player_gaps`: {team: player_elo - team_elo} from
    model/player_ratings.json, for the app's diagnostic flag
    (|gap| > 100 -> warning, app display only).
  * `live_models_role_gold`: per-bucket variant-A live models (role golds
    replace total gold; same shape as `live_models`). Feed-only: OCR games
    keep the baseline `live_models` (OCR has team gold only). Variant A is
    conditional-ship pending the Swiss shadow test -- see the note field.
  * roster-adjusted team ratings (see src/roster_strength.py) + raw snapshot
    Elo + delta, per-team form, yearly starting rosters, turnover
  * region-strength offsets (src/region_offsets.py): per-region Elo
    adjustments + each team's region, so the app can compute
    elo_diff_adj = elo_diff + off[region_blue] - off[region_red] itself
    (the exported feature lists name elo_diff_adj; the app must apply the
    adjustment before scoring -- see region_offsets note in the artifact)
  * team tag aliases, champion power map, champion list, team-champion comfort
  * blue-side edge constant
  * calibration table (2026 holdout, 10 quantile bins) + ECE + accuracy, so the
    app can show an honest confidence tier per prediction
  * backtest summary (indicative; those backtests refit the model, they don't
    use this artifact)

Versioning: model_version + snapshot_date bump whenever artifacts are
regenerated (new year, new rosters, retrained model).

Recomputes holdout predictions with pure-Python JSON math (the same code path
the phone uses) to derive the calibration table / ECE / accuracy -- this also
end-to-end validates the exported numbers.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd

from model_utils import PRED_MAX, PRED_MIN  # noqa: E402
from predict import ALIASES, load_bundle  # noqa: E402  (comfort keys normalized here)
from region_offsets import load_offsets  # noqa: E402
from roster_strength import adjusted_snapshot_ratings, compute_adjustments  # noqa: E402
from train import honest_map_from_bundle  # noqa: E402  (nodraft honest map + verification)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "model")
PROC = os.path.join(BASE, "data", "processed")

MODEL_VERSION = os.environ.get("MODEL_VERSION_OVERRIDE", "0.0.0-dev")
# The fallback MUST NOT be a plausible release number: a stale "1.3.0"
# default silently mislabeled two exports on 2026-10-17 (the region-offset
# draft and the Phase 1 Wave 2 export). Real exports pass
# MODEL_VERSION_OVERRIDE explicitly (refresh.py bumps it); a manual run
# without an override gets the unmistakable 0.0.0-dev marker.
TIER_RULE = {
    "rule": "margin = |p_calibrated - 0.5|; bin = calibration bin containing p_calibrated",
    "high": {"min_margin": 0.20, "min_bin_n": 100},
    "medium": {"min_margin": 0.10, "min_bin_n": 30},
    "low": "otherwise (margin < 0.10 or bin n < 30)",
    "note": ("Tier reflects discrimination (margin) plus calibration evidence "
             "(bin sample size). Show the bin's observed accuracy as the honest "
             "confidence note, e.g. 'model was right 78% of the time in this "
             "range (n=312)'."),
}


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def calibrate(p_raw, cal):
    """Pure-Python replica of the sklearn isotonic calibration map."""
    p = min(max(p_raw, cal["x_min"]), cal["x_max"])
    xs, ys = cal["xs"], cal["ys"]
    # np.interp-equivalent linear interpolation
    if p <= xs[0]:
        y = ys[0]
    elif p >= xs[-1]:
        y = ys[-1]
    else:
        import bisect
        i = bisect.bisect_right(xs, p) - 1
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        y = y0 + (p - x0) * (y1 - y0) / (x1 - x0) if x1 > x0 else y0
    return min(max(y, 1e-6), 1 - 1e-6)


def predict_pure(features, model_json):
    z = sum((f - m) / s * c
            for f, m, s, c in zip(features, model_json["scaler_mean"],
                                  model_json["scaler_scale"], model_json["coef"]))
    z += model_json["intercept"]
    return calibrate(sigmoid(z), model_json["calibration"])


def iso_to_json(ir):
    return {
        "xs": [float(x) for x in np.asarray(ir.X_thresholds_).ravel()],
        "ys": [float(y) for y in np.asarray(ir.y_thresholds_).ravel()],
        "x_min": float(ir.X_min_),
        "x_max": float(ir.X_max_),
        "y_min": float(ir.y_min),
        "y_max": float(ir.y_max),
    }


def model_to_json(bundle, key):
    m = bundle[f"model_{key}"]
    sc = bundle[f"scaler_{key}"]
    lr = m.base_model
    if type(lr).__name__ != "LogisticRegression":
        raise TypeError(
            f"model_{key} is {type(lr).__name__}: the phone artifact format "
            "requires a linear model (train.py pins nodraft to LR for this reason)")
    return {
        "model_type": type(lr).__name__,
        "features": list(bundle[f"features_{key}"]),
        "coef": [float(c) for c in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "scaler_mean": [float(x) for x in sc.mean_],
        "scaler_scale": [float(x) for x in sc.scale_],
        "calibration": iso_to_json(m.ir),
    }


def honest_to_cal(honest):
    """Coarsened honest map ({xs, ys, ...}) -> artifact calibration shape.

    The coarsened xs keep the fitted map's first/last thresholds, so
    x_min/x_max are the map's input range; ys are already bounded by the
    IsotonicCalibrated clip [PRED_MIN, PRED_MAX].
    """
    xs = [float(x) for x in honest["xs"]]
    ys = [float(y) for y in honest["ys"]]
    return {
        "xs": xs,
        "ys": ys,
        "x_min": xs[0],
        "x_max": xs[-1],
        "y_min": PRED_MIN,
        "y_max": PRED_MAX,
    }


LIVE_FEATURES = ["pre_logit", "golddiff_k", "killdiff"]
ROLE_LIVE_FEATURES = ["pre_logit", "top_gold", "jng_gold", "mid_gold",
                      "bot_gold", "sup_gold", "killdiff"]


def live_to_json(cal_wrapper, features=LIVE_FEATURES):
    """Export one per-bucket live model (no scaler: raw features)."""
    lr = cal_wrapper.base_model
    if type(lr).__name__ != "LogisticRegression":
        raise TypeError("live bucket model must be a LogisticRegression")
    return {
        "model_type": type(lr).__name__,
        "features": list(features),
        "coef": [float(c) for c in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "calibration": iso_to_json(cal_wrapper.ir),
    }


def main():
    bundle, snaps, champ_power = load_bundle()
    snapshot_date = snaps["as_of"][:10]

    full_json = model_to_json(bundle, "full")
    nodraft_json = model_to_json(bundle, "nodraft")

    # ---- live in-game models (per minute bucket; no scaler, raw features) ----
    live_models = joblib.load(os.path.join(MODEL_DIR, "live_models.joblib"))
    live_json = {str(t): live_to_json(live_models[t]) for t in (10, 15, 20, 25)}

    # ---- live variant A (role golds replace total gold; feed-only) ----
    live_role_gold = joblib.load(
        os.path.join(MODEL_DIR, "live_models_role_gold.joblib"))
    live_role_gold_json = {
        str(t): live_to_json(live_role_gold[t], ROLE_LIVE_FEATURES)
        for t in (10, 15, 20, 25)}

    # ---- roster-adjusted team ratings ----
    adjustments = compute_adjustments()
    adj_ratings = adjusted_snapshot_ratings(adjustments)
    # team regions: snapshots carry them (build_features.py); fall back to
    # model/region_offsets.json (canonical file) for older snapshots.
    offsets, team_region = load_offsets()
    if "region_offsets" in snaps and "team_region" in snaps:
        offsets, team_region = snaps["region_offsets"], snaps["team_region"]
    teams = {}
    for team, elo in snaps["elo"].items():
        rec = adjustments.get(team, {})
        teams[team] = {
            "elo": adj_ratings[team],          # roster-adjusted (what the app uses)
            "elo_raw": round(float(elo), 2),   # pre-adjustment snapshot Elo
            "roster_delta": round(adj_ratings[team] - float(elo), 2),
            "form": round(float(snaps["form"].get(team, 0.5)), 4),
            "eg_gold": round(float(snaps["eg_gold"].get(team, 0.0)), 1),
            "region": team_region.get(team),
            "turnover_2025_2026": rec.get("turnover_2025_2026"),
            "roster_2025": rec.get("roster_2025", []),
            "roster_2026": rec.get("roster_2026", []),
        }

    # ---- champion data (lowercase keys, matching CLI/app lookup) ----
    comfort = {}
    for k, v in snaps["team_champ_comfort"].items():
        t, c = k.rsplit("||", 1)  # already lowercased by load_bundle
        comfort.setdefault(t, {})[c] = round(float(v), 4)

    # per-patch champion power: {patch: {champ_lower: value}} for recent patches,
    # plus the current (Worlds 2026) patch designation. The app should use
    # champion_power_by_patch[current_patch] (== top-level champion_power).
    patch_tables = {
        p: {k: round(float(v), 4) for k, v in tbl.items()}
        for p, tbl in snaps.get("patch_tables", {}).items()
    }
    current_patch = snaps.get("current_patch")

    # ---- player-team diagnostic gaps (app flag; |gap| > 100 -> warning) ----
    with open(os.path.join(MODEL_DIR, "player_ratings.json")) as f:
        player_ratings = json.load(f)
    player_gaps = {t: float(v["gap"]) for t, v in player_ratings["teams"].items()}

    # ---- confidence: recompute holdout predictions with pure JSON math ----
    df = pd.read_csv(os.path.join(PROC, "features.csv"), parse_dates=["date"])

    # ---- honest calibration maps (nested-split; for the app toggle) ----
    # Full-model map comes from the canonical file; verify it matches a
    # recomputation from THIS bundle (fail loudly on drift -- a stale map
    # paired with fresh coefficients would silently miscalibrate).
    with open(os.path.join(MODEL_DIR, "honest_calibration.json")) as f:
        honest_full_file = json.load(f)
    honest_full_re, _ = honest_map_from_bundle(bundle, df, "full")
    if (len(honest_full_file["xs"]) != len(honest_full_re["xs"]) or
            any(abs(a - b) >= 1e-9 for a, b in
                zip(honest_full_file["xs"], honest_full_re["xs"])) or
            any(abs(a - b) >= 1e-9 for a, b in
                zip(honest_full_file["ys"], honest_full_re["ys"]))):
        raise RuntimeError(
            "model/honest_calibration.json does not match the current bundle; "
            "rerun train.py (it regenerates the file) before exporting")
    honest_nodraft, _ = honest_map_from_bundle(bundle, df, "nodraft")
    full_json["calibration_honest"] = honest_to_cal(honest_full_file)
    nodraft_json["calibration_honest"] = honest_to_cal(honest_nodraft)

    cal = df[(df["date"] >= bundle["fit_end"]) & (df["date"] < bundle["cal_end"])].copy()
    cal = cal.sort_values("date").reset_index(drop=True)
    feats = bundle["features_full"]
    X = cal[feats].to_numpy(float)
    y = cal["y"].to_numpy(int)
    p = np.array([predict_pure(row, full_json) for row in X])

    acc = float(np.mean((p > 0.5) == y))
    # ECE: 10 equal-width bins
    ece, ece_n = 0.0, 0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (p > lo) & (p <= hi) if b < 9 else (p >= lo) & (p <= hi)
        if b == 0:
            m = (p >= lo) & (p <= hi)
        if m.sum():
            ece += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
            ece_n += 1
    # calibration table: 10 quantile bins (same binning as train.py)
    edges = np.quantile(p, np.linspace(0, 1, 11))
    cal_table = []
    for i in range(10):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (p >= lo) & (p <= hi) if i == 9 else (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        cal_table.append({
            "bin": i + 1,
            "pred_range": [round(lo, 3), round(hi, 3)],
            "mean_pred": round(float(p[m].mean()), 3),
            "observed_acc": round(float(y[m].mean()), 3),
            "n": int(m.sum()),
        })

    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    backtests = {
        k: {"n": v["n"], "accuracy": round(v["accuracy"], 4),
            "log_loss": round(v["log_loss"], 4), "note": v.get("note", "")}
        for k, v in metrics.get("backtests", {}).items()
    }

    artifact = {
        "model_version": MODEL_VERSION,
        "snapshot_date": snapshot_date,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {"full": full_json, "nodraft": nodraft_json},
        "calibration_honest_note": (
            "Each model carries calibration_honest: the nested-split honest "
            "map (isotonic fit on 2026-07-01..2026-08-31 only, graded on "
            "untouched Sep games). The shipped `calibration` map was fit on "
            "the full tail and graded on the same tail (in-sample). The app "
            "toggle switches between them; default is the shipped map."),
        "live_models": live_json,
        "live_models_role_gold": live_role_gold_json,
        "live_models_role_gold_note": (
            "Variant A (reports/live_role_gold.md): role golds replace total "
            "gold. Feed-only -- OCR games keep the baseline live_models "
            "(OCR has team gold only). CONDITIONAL SHIP: shadow-test on live "
            "Swiss games before letting variant A drive Kelly fractions; "
            "if the shadow test comes back <= 0, revert to baseline."),
        "player_gaps": player_gaps,
        "blue_edge": round(float(snaps["blue_edge"]), 4),
        "teams": teams,
        "team_aliases": ALIASES,
        "champion_power": {k: round(float(v), 4) for k, v in champ_power.items()},
        "current_patch": current_patch,
        "recent_patches": sorted(patch_tables.keys()),
        "champion_power_by_patch": patch_tables,
        "champions": sorted(snaps["champ_power"].keys()),
        "team_champion_comfort": comfort,
        "roster_formula": ("rating = snapshot_elo + delta_2026; delta_2026 = A - U; "
                           "A = 1500 + 0.6*(1-0.5*t)*((1-t)*E_prev + t*incoming - 1500); "
                           "U = 1500 + 0.6*(E_prev - 1500); t = 2025->2026 starter turnover; "
                           "incoming = mean prior-year Elo of incoming starters' old teams "
                           "(1500 for rookies). See reports/roster_strength.md."),
        "calibration_table": cal_table,
        "calibration_note": ("2026 holdout (fit_end <= date < cal_end), full draft model, "
                             "pure-JSON recomputation. Backtest figures below are indicative: "
                             "those backtests refit the model on pre-event data and do not "
                             "use this artifact."),
        "backtests": backtests,
        "confidence_tiers": TIER_RULE,
        "region_offsets": {
            "offsets_elo": {r: round(float(v), 1) for r, v in offsets.items()},
            "note": ("The exported models were TRAINED on elo_diff_adj, not "
                     "elo_diff. Before scoring, the app must compute "
                     "elo_diff_adj = elo_diff + off[region_blue] - "
                     "off[region_red] using each team's region below "
                     "(teams[].region) and this table; unknown region -> "
                     "offset 0. Offsets are constant (no decay) and refit "
                     "from cross-region games on every refresh -- see "
                     "reports/region_strength_analysis.md."),
            "fitted_regions": sorted(offsets),
            "minor_regions_offset_0": ["Japan", "Oceania", "Turkiye"],
        },
        "metrics": {
            "holdout_n": int(len(cal)),
            "holdout_accuracy": round(acc, 4),
            "holdout_ece": round(float(ece), 4),
            "holdout_ece_note": ("In-sample: the isotonic map was fit on this same holdout, "
                                 "so ECE~0 reflects fit quality, not out-of-sample calibration. "
                                 "Treat per-bin observed accuracies as indicative."),
            "holdout_period": [bundle["fit_end"], bundle["cal_end"]],
        },
    }

    # PHONE_ARTIFACTS_OUT overrides the output path (draft exports must
    # never overwrite the shipped phone_artifacts.json).
    out_path = os.environ.get("PHONE_ARTIFACTS_OUT",
                              os.path.join(MODEL_DIR, "phone_artifacts.json"))
    with open(out_path, "w") as f:
        json.dump(artifact, f)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KB)")
    print(f"  holdout: n={len(cal)} acc={acc:.4f} ece={ece:.4f}")
    print(f"  teams={len(teams)} champs={len(champ_power)} "
          f"comfort_entries={sum(len(v) for v in comfort.values())}")


if __name__ == "__main__":
    main()
