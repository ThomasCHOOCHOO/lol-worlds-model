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

from predict import draft_features, load_bundle, parse_champs, predict_game, resolve_team

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-6


def load_artifacts():
    with open(os.path.join(BASE, "model", "phone_artifacts.json")) as f:
        return json.load(f)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def calibrate(p_raw, cal):
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
    return min(max(y, 1e-6), 1 - 1e-6)


def json_predict(art, blue_tag, red_tag, draft_args, model_key, use_adjusted=False):
    blue, red = resolve_team(blue_tag), resolve_team(red_tag)
    tb, tr = art["teams"][blue], art["teams"][red]
    elo_b = tb["elo" if use_adjusted else "elo_raw"]
    elo_r = tr["elo" if use_adjusted else "elo_raw"]
    feats = {
        "elo_diff": elo_b - elo_r,
        "form_diff": tb["form"] - tr["form"],
        "blue_edge": art["blue_edge"],
    }
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
                "metrics", "backtests"]
    missing = [k for k in required if k not in art]
    print(f"schema keys: {'OK' if not missing else 'MISSING ' + str(missing)}")
    ok = ok and not missing

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
