"""Validate model/phone_artifacts.json against src/predict.py.

Pure-Python (no sklearn) JSON inference must reproduce predict.py's
probabilities within 1e-6, using the RAW snapshot Elo (predict.py does not
apply the roster adjustment; the phone app uses the adjusted "elo" field,
which is raw + delta -- checked separately below).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from model_utils import PRED_MAX, PRED_MIN
from predict import draft_features, load_bundle, parse_champs, predict_game, resolve_team

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-6


def load_artifacts():
    # ARTIFACTS_PATH overrides the artifact path (draft exports must be
    # validated against the draft file, never the shipped artifact).
    path = os.environ.get("ARTIFACTS_PATH",
                          os.path.join(BASE, "model", "phone_artifacts.json"))
    with open(path) as f:
        return json.load(f)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def calibrate(p_raw, cal):
    """Pure-Python replica of the sklearn isotonic calibration map.

    Clips to [PRED_MIN, PRED_MAX] to match model_utils.IsotonicCalibrated
    (the wrapper's clip bounds, not 1e-6).
    """
    p = min(max(p_raw, cal["x_min"]), cal["x_max"])
    xs, ys = cal["xs"], cal["ys"]
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
    return min(max(y, PRED_MIN), PRED_MAX)


def json_predict(art, blue_tag, red_tag, draft_args, model_key, use_adjusted=False):
    blue, red = resolve_team(blue_tag), resolve_team(red_tag)
    tb, tr = art["teams"][blue], art["teams"][red]
    elo_b = tb["elo" if use_adjusted else "elo_raw"]
    elo_r = tr["elo" if use_adjusted else "elo_raw"]
    feats = {
        "elo_diff": elo_b - elo_r,
        "eg_gold_diff": tb["eg_gold"] - tr["eg_gold"],
        "blue_edge": art["blue_edge"],
    }
    # region-strength adjustment: the exported models train on elo_diff_adj,
    # so the JSON path must apply the same adjustment predict.py does.
    # Unknown region -> offset 0.
    off = art.get("region_offsets", {}).get("offsets_elo", {})
    off_b = off.get(tb.get("region"), 0.0) if tb.get("region") else 0.0
    off_r = off.get(tr.get("region"), 0.0) if tr.get("region") else 0.0
    feats["elo_diff_adj"] = feats["elo_diff"] + off_b - off_r
    if draft_args is not None:
        cp = art["champion_power"]
        comf = art["team_champion_comfort"]
        bp = [c.strip().lower() for c in draft_args["blue_picks"].split(",") if c.strip()]
        rp = [c.strip().lower() for c in draft_args["red_picks"].split(",") if c.strip()]
        bb = [c.strip().lower() for c in draft_args.get("blue_bans", "").split(",") if c.strip()]
        rb = [c.strip().lower() for c in draft_args.get("red_bans", "").split(",") if c.strip()]
        feats["d_power"] = sum(cp.get(c, 0.5) for c in bp) - sum(cp.get(c, 0.5) for c in rp)
        feats["d_comfort"] = (sum(comf.get(blue, {}).get(c, tb["form"]) for c in bp)
                              - sum(comf.get(red, {}).get(c, tr["form"]) for c in rp))
        feats["ban_edge"] = sum(cp.get(c, 0.5) for c in bb) - sum(cp.get(c, 0.5) for c in rb)
    mj = art["models"][model_key]
    z = sum((feats[f] - m) / s * c for f, m, s, c in
            zip(mj["features"], mj["scaler_mean"], mj["scaler_scale"], mj["coef"]))
    z += mj["intercept"]
    return calibrate(sigmoid(z), mj["calibration"]), feats


def ref_predict(blue_tag, red_tag, draft_args, bundle, snaps, champ_power):
    blue, red = resolve_team(blue_tag), resolve_team(red_tag)
    draft = None
    if draft_args is not None:
        draft = draft_features(
            blue, red,
            parse_champs(draft_args["blue_picks"], champ_power, "p"),
            parse_champs(draft_args["red_picks"], champ_power, "p"),
            parse_champs(draft_args.get("blue_bans"), champ_power, "b"),
            parse_champs(draft_args.get("red_bans"), champ_power, "b"),
            snaps, champ_power)
    model_key = "full" if draft is not None else "nodraft"
    p, _ = predict_game(blue, red, draft, bundle, snaps, champ_power)
    return p, model_key


CASES = [
    ("t1", "blg", None),
    ("t1", "blg", {"blue_picks": "Azir,Vi,Ahri,Kai'Sa,Rakan",
                   "red_picks": "Renekton,Sejuani,Orianna,Ziggs,Nautilus",
                   "blue_bans": "Kalista,Yone,Smolder",
                   "red_bans": "Vi,Neeko,Rakan"}),
    ("gen", "g2", None),
    ("c9", "kc", {"blue_picks": "Aurora,Skarner,Tristana,Ashe,Leona",
                  "red_picks": "Ambessa,Nocturne,Akali,Caitlyn,Thresh",
                  "blue_bans": "Kalista,Varus",
                  "red_bans": "Aurora,Azir"}),
    ("tsw", "fur", None),
]


def main():
    art = load_artifacts()
    bundle, snaps, champ_power = load_bundle()
    ok = True
    print(f"{'matchup':28s} {'model':8s} {'predict.py':>10s} {'json':>10s} {'|diff|':>10s}")
    for blue_tag, red_tag, draft_args in CASES:
        p_ref, model_key = ref_predict(blue_tag, red_tag, draft_args,
                                       bundle, snaps, champ_power)
        p_json, feats = json_predict(art, blue_tag, red_tag, draft_args, model_key)
        d = abs(p_ref - p_json)
        status = "OK " if d < TOL else "FAIL"
        if d >= TOL:
            ok = False
        label = f"{blue_tag}-vs-{red_tag}" + ("+draft" if draft_args else "")
        print(f"{label:28s} {model_key:8s} {p_ref:10.6f} {p_json:10.6f} {d:10.2e} {status}")

    # roster adjustment identity: elo == elo_raw + delta
    for team in ("T1", "Karmine Corp", "Bilibili Gaming"):
        t = art["teams"][team]
        assert abs(t["elo"] - (t["elo_raw"] + t["roster_delta"])) < 1e-9, team
    print("roster identity (elo == elo_raw + delta): OK")

    # isotonic map vs sklearn wrapper on random raw probs
    rng = np.random.default_rng(1)
    worst = 0.0
    for key in ("full", "nodraft"):
        ir = bundle[f"model_{key}"].ir
        pts = rng.uniform(0, 1, 5000)
        sk = np.clip(ir.predict(np.clip(pts, 1e-6, 1 - 1e-6)), 1e-6, 1 - 1e-6)
        mine = np.array([calibrate(p, art["models"][key]["calibration"]) for p in pts])
        worst = max(worst, float(np.max(np.abs(sk - mine))))
    print(f"isotonic map vs sklearn wrapper max diff: {worst:.2e} "
          f"({'OK' if worst < TOL else 'FAIL'})")
    ok = ok and worst < TOL

    # schema sanity
    required = ["model_version", "snapshot_date", "models", "blue_edge", "teams",
                "team_aliases", "champion_power", "champions",
                "team_champion_comfort", "calibration_table", "confidence_tiers",
                "metrics", "backtests", "live_models", "region_offsets",
                "player_gaps", "live_models_role_gold"]
    missing = [k for k in required if k not in art]
    print(f"schema keys: {'OK' if not missing else 'MISSING ' + str(missing)}")
    ok = ok and not missing

    # region offsets in the artifact must match the canonical JSON
    from region_offsets import load_offsets
    canon_off, _ = load_offsets()
    art_off = art.get("region_offsets", {}).get("offsets_elo", {})
    off_match = (set(art_off) == set(canon_off)
                 and all(abs(art_off[r] - canon_off[r]) < 1e-6 for r in canon_off))
    print(f"region_offsets match model/region_offsets.json: "
          f"{'OK' if off_match else 'FAIL ' + str(art_off)}")
    ok = ok and off_match

    # live bucket models: pure-Python JSON math vs sklearn wrapper
    import joblib as _joblib
    live = _joblib.load(os.path.join(BASE, "model", "live_models.joblib"))
    live_feats = ["pre_logit", "golddiff_k", "killdiff"]
    worst_live = 0.0
    rng2 = np.random.default_rng(7)
    for t in (10, 15, 20, 25):
        lj = art["live_models"][str(t)]
        assert lj["features"] == live_feats, f"live {t} feature order"
        wrapper = live[t]
        # random states + realistic game states
        X = np.vstack([
            rng2.uniform([-3, -8, -10], [3, 8, 10], size=(2000, 3)),
            np.array([[0.8, 2.5, 3],    # blue slight favorite, ahead
                      [-0.5, -4.0, -5],  # blue underdog, behind
                      [0.0, 0.0, 0]]),   # dead even
        ])
        sk = np.clip(wrapper.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        z = X @ np.array(lj["coef"]) + lj["intercept"]
        mine = np.array([calibrate(sigmoid(v), lj["calibration"]) for v in z])
        worst_live = max(worst_live, float(np.max(np.abs(sk - mine))))
    print(f"live bucket JSON vs sklearn wrapper max diff: {worst_live:.2e} "
          f"({'OK' if worst_live < TOL else 'FAIL'})")
    ok = ok and worst_live < TOL

    # ---- Phase 1 Wave 2 additions ----
    # (a) calibration_honest: recompute from the bundle; must match artifact.
    import pandas as _pd
    from train import honest_map_from_bundle as _honest_map
    df_full = _pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"),
                           parse_dates=["date"])
    honest_ok = True
    for key in ("full", "nodraft"):
        hj, _ = _honest_map(bundle, df_full, key)
        cj = art["models"][key].get("calibration_honest")
        if cj is None:
            print(f"calibration_honest[{key}]: MISSING")
            honest_ok = False
            continue
        xs_match = (len(cj["xs"]) == len(hj["xs"]) and
                    all(abs(a - b) < 1e-9 for a, b in zip(cj["xs"], hj["xs"])))
        ys_match = (len(cj["ys"]) == len(hj["ys"]) and
                    all(abs(a - b) < 1e-9 for a, b in zip(cj["ys"], hj["ys"])))
        mono = all(b >= a for a, b in zip(cj["ys"], cj["ys"][1:]))
        bounds = (abs(cj["x_min"] - cj["xs"][0]) < 1e-12 and
                  abs(cj["x_max"] - cj["xs"][-1]) < 1e-12 and
                  all(PRED_MIN <= y <= PRED_MAX for y in cj["ys"]))
        good = xs_match and ys_match and mono and bounds
        print(f"calibration_honest[{key}]: steps={len(cj['xs'])} "
              f"xs_match={xs_match} ys_match={ys_match} monotone={mono} "
              f"bounds={bounds} ({'OK' if good else 'FAIL'})")
        honest_ok = honest_ok and good
    ok = ok and honest_ok

    # (b) player_gaps: must match model/player_ratings.json team gaps.
    with open(os.path.join(BASE, "model", "player_ratings.json")) as f:
        pr = json.load(f)
    gaps_ok = (set(art.get("player_gaps", {})) == set(pr["teams"]) and
               all(abs(art["player_gaps"][t] - pr["teams"][t]["gap"]) < 1e-9
                   for t in pr["teams"]))
    print(f"player_gaps match model/player_ratings.json "
          f"(n={len(pr['teams'])}): {'OK' if gaps_ok else 'FAIL'}")
    ok = ok and gaps_ok

    # (c) live_models_role_gold: JSON math vs variant-A joblib wrappers.
    live_rg = _joblib.load(os.path.join(BASE, "model",
                                        "live_models_role_gold.joblib"))
    role_feats = ["pre_logit", "top_gold", "jng_gold", "mid_gold",
                  "bot_gold", "sup_gold", "killdiff"]
    worst_rg = 0.0
    rng3 = np.random.default_rng(11)
    for t in (10, 15, 20, 25):
        lj = art["live_models_role_gold"][str(t)]
        assert lj["features"] == role_feats, f"role-gold live {t} feature order"
        wrapper = live_rg[t]
        X = np.vstack([
            rng3.uniform([-3, -3, -3, -3, -3, -3, -10],
                         [3, 3, 3, 3, 3, 3, 10], size=(2000, 7)),
            np.array([[0.8, 1.0, 0.5, 0.8, 1.5, 0.2, 3],
                      [-0.5, -1.0, -0.5, -0.8, -1.5, -0.2, -5],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]]),
        ])
        sk = np.clip(wrapper.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        z = X @ np.array(lj["coef"]) + lj["intercept"]
        mine = np.array([calibrate(sigmoid(v), lj["calibration"]) for v in z])
        worst_rg = max(worst_rg, float(np.max(np.abs(sk - mine))))
    print(f"live role-gold JSON vs sklearn wrapper max diff: {worst_rg:.2e} "
          f"({'OK' if worst_rg < TOL else 'FAIL'})")
    ok = ok and worst_rg < TOL

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
