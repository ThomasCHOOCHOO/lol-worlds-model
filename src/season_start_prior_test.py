"""Season-start prior test: player Elo ONLY as the 2026 season-start prior.

READ-ONLY on live model files: reads data/processed/player_features.csv,
data/processed/features.csv and model/model.joblib; writes only the report
and JSON sidecar (and this script itself).

Design (per task):
- Boundary 2025 -> 2026. Starting point is the player_elo.replay output in
  data/processed/player_features.csv: its pre-game team_elo / player_elo for
  a team's first 2026 game are the 40%-regressed end-of-2025 ratings
  (regression is applied by the replay at the year boundary).
- Prior: at a team's first 2026 game, starting team Elo =
  (1-w) * regressed team Elo + w * player aggregate (mean of the 5
  pre-game player ratings for that game's actual lineup). w in
  {0, 0.25, 0.5, 0.75, 1.0}; w=0 is the baseline.
- After that, team Elo only: K=24 per map, expected score from the TEAM Elo
  diff, no blue-side adjustment, no further player input.
- Probabilities: shipped nodraft model (model.joblib model_nodraft:
  LR on [elo_diff_adj, eg_gold_diff, blue_edge] via scaler_nodraft, then
  isotonic calibration). elo_diff_adj = my team Elo diff + the region
  correction (elo_diff_adj - elo_diff) from features.csv, which is
  identical across w. Both RAW (pre-calibration LR) and CAL (isotonic)
  probabilities are reported.
- Windows: primary = games where BOTH teams are within their first 10
  games of 2026; secondary = Jan-Mar 2026; extra = all 2026 games
  (through Oct 16, the CSV coverage end).
- Paired bootstrap CI (20,000 resamples, seed 7) on the per-game
  log-loss difference vs w=0.

Caveat: model.joblib was fit through 2026-07-01 and calibrated through
2026-10-01, so early-2026 absolute log-loss is in-sample for the model.
The w comparison is paired (same model, same games, only the starting
Elo differs), so the DIFF vs baseline is the valid quantity; absolute
levels are optimistic and are reported for context only.
"""
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(7)
N_BOOT = 20000
K = 24.0
WS = [0.0, 0.25, 0.5, 0.75, 1.0]
TIER1 = {"LCK", "LPL", "LEC", "LCS", "LTA", "LTA N", "LTA S", "LCP"}
INTL = {"MSI", "WLDs", "EWC"}


def ll(p, y):
    p = min(max(float(p), 0.001), 0.999)
    return -(math.log(p) if y == 1 else math.log(1 - p))


def boot_ci(diffs):
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return None
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def metrics(rows):
    """rows: list of dicts with p_raw, p_cal, y. Returns dict of metrics."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    ll_raw = [ll(r["p_raw"], r["y"]) for r in rows]
    ll_cal = [ll(r["p_cal"], r["y"]) for r in rows]
    return {
        "n": n,
        "ll_raw": float(np.mean(ll_raw)),
        "ll_cal": float(np.mean(ll_cal)),
        "acc_raw": float(np.mean([1 if (r["p_raw"] > 0.5) == bool(r["y"]) else 0 for r in rows])),
        "acc_cal": float(np.mean([1 if (r["p_cal"] > 0.5) == bool(r["y"]) else 0 for r in rows])),
        "ll_raw_list": ll_raw,
        "ll_cal_list": ll_cal,
    }


def main():
    import joblib
    import sys
    sys.path.insert(0, os.path.join(BASE, "src"))
    bundle = joblib.load(os.path.join(BASE, "model", "model.joblib"))
    model = bundle["model_nodraft"]
    scaler = bundle["scaler_nodraft"]

    pf = pd.read_csv(os.path.join(BASE, "data", "processed", "player_features.csv"),
                     low_memory=False)
    feat = pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"),
                       low_memory=False)
    print(f"player_features rows={len(pf)} {pf['date'].min()}..{pf['date'].max()}")
    print(f"features rows={len(feat)} {feat['date'].min()}..{feat['date'].max()}")

    m = pf.merge(feat[["gameid", "elo_diff", "elo_diff_adj", "eg_gold_diff", "blue_edge"]],
                 on="gameid", how="left", suffixes=("", "_f"))
    assert m["elo_diff_adj"].notna().all(), "features join missed games"
    # sanity: replay team Elo diff vs features elo_diff
    md = (m["team_elo_blue"] - m["team_elo_red"] - m["elo_diff"]).abs()
    print(f"sanity |pf team_elo_diff - features elo_diff|: max={md.max():.4f} mean={md.mean():.4f}")
    m["region_corr"] = m["elo_diff_adj"] - m["elo_diff"]
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["date", "gameid"]).reset_index(drop=True)

    g26 = m[m["date"] >= "2026-01-01"].copy().reset_index(drop=True)
    print(f"2026 games: {len(g26)} ({g26['date'].min()}..{g26['date'].max()}), "
          f"teams: {len(set(g26['blue']) | set(g26['red']))}")

    # ---- modal 2025 lineups per team (top-5 players by appearances in 2025) ----
    p25 = m[m["date"] < "2026-01-01"]
    counts = defaultdict(Counter)
    for _, r in p25.iterrows():
        for side in ("blue", "red"):
            team = r[side]
            for key in str(r[f"lineup_{side}"]).split("|"):
                pid = key.split("||")[0]
                counts[team][pid] += 1
    modal25 = {t: {p for p, _ in c.most_common(5)} for t, c in counts.items()}

    # first-2026-game lineup per team + turnover vs modal 2025
    first_lineup = {}
    for _, r in g26.iterrows():
        for side in ("blue", "red"):
            team = r[side]
            if team not in first_lineup:
                pids = {k.split("||")[0] for k in str(r[f"lineup_{side}"]).split("|")}
                first_lineup[team] = pids
    turnover = {}
    for team, pids in first_lineup.items():
        ref = modal25.get(team)
        turnover[team] = (1.0 - len(pids & ref) / 5.0) if ref else 0.0
    hi_turn = {t for t, v in turnover.items() if v >= 0.4}
    print(f"teams with first 2026 game: {len(first_lineup)}, high-turnover (>=0.4): "
          f"{len(hi_turn)}; turnover distribution: "
          f"{sorted(Counter(turnover.values()).items())}")

    # ---- replay per w ----
    Xcols = ["elo_diff_adj", "eg_gold_diff", "blue_edge"]
    results = {}
    for w in WS:
        elo = {}
        seen = set()
        game_count = Counter()
        rows = []
        max_w0_dev = 0.0
        for _, r in g26.iterrows():
            blue, red = r["blue"], r["red"]
            # apply the prior at each team's first 2026 game
            for team, side in ((blue, "blue"), (red, "red")):
                if team not in seen:
                    seen.add(team)
                    elo[team] = ((1 - w) * r[f"team_elo_{side}"]
                                 + w * r[f"player_elo_{side}"])
            diff = elo[blue] - elo[red]
            if w == 0.0:
                dev = abs(diff - (r["team_elo_blue"] - r["team_elo_red"]))
                max_w0_dev = max(max_w0_dev, dev)
            diff_adj = diff + r["region_corr"]
            X = np.array([[diff_adj, r["eg_gold_diff"], r["blue_edge"]]], dtype=float)
            Xs = scaler.transform(X)
            p_raw = float(model.base_model.predict_proba(Xs)[0, 1])
            p_cal = float(model.predict_proba(Xs)[0, 1])
            y = int(r["y"])
            rows.append({
                "gameid": r["gameid"], "date": r["date"], "league": r["league"],
                "blue": blue, "red": red, "y": y,
                "p_raw": p_raw, "p_cal": p_cal,
                "cnt_blue_pre": game_count[blue], "cnt_red_pre": game_count[red],
                "hi_blue": blue in hi_turn, "hi_red": red in hi_turn,
                "turn_blue": turnover.get(blue, 0.0), "turn_red": turnover.get(red, 0.0),
            })
            # team Elo only update, K=24, expected from raw team Elo diff
            exp_b = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
            elo[blue] = elo[blue] + K * (y - exp_b)
            elo[red] = elo[red] + K * ((1 - y) - (1 - exp_b))
            game_count[blue] += 1
            game_count[red] += 1
        if w == 0.0:
            print(f"w=0 validation: max |my Elo diff - replay team_elo diff| = {max_w0_dev:.6f}")
            assert max_w0_dev < 1e-6, "w=0 replay does not reproduce the replay team Elo"
        results[w] = rows

    def subset(rows, pred):
        return [r for r in rows if pred(r)]

    windows = {
        "first10_both": lambda r: r["cnt_blue_pre"] < 10 and r["cnt_red_pre"] < 10,
        "jan_mar_2026": lambda r: r["date"] < pd.Timestamp("2026-04-01"),
        "all_2026": lambda r: True,
    }
    tiers = {
        "overall": lambda r: True,
        "tier1_intl": lambda r: r["league"] in (TIER1 | INTL),
        "minors": lambda r: r["league"] not in (TIER1 | INTL),
    }

    out = {"windows": {}, "turnover_split": {}, "validation": {}, "design": {}}
    base = results[0.0]
    for wname, wpred in windows.items():
        out["windows"][wname] = {}
        for tname, tpred in tiers.items():
            idx = [i for i, r in enumerate(base) if wpred(r) and tpred(r)]
            out["windows"][wname][tname] = {}
            for w in WS:
                rows = [results[w][i] for i in idx]
                mm = metrics(rows)
                entry = {"n": mm["n"], "ll_raw": mm.get("ll_raw"),
                         "ll_cal": mm.get("ll_cal"), "acc_raw": mm.get("acc_raw"),
                         "acc_cal": mm.get("acc_cal")}
                if w > 0 and mm["n"]:
                    draw = [a - b for a, b in zip(metrics([results[0.0][i] for i in idx])["ll_raw_list"],
                                                  mm["ll_raw_list"])]
                    dcal = [a - b for a, b in zip(metrics([results[0.0][i] for i in idx])["ll_cal_list"],
                                                  mm["ll_cal_list"])]
                    entry["diff_raw_vs_w0"] = float(np.mean(draw))
                    entry["ci_raw"] = boot_ci(draw)
                    entry["diff_cal_vs_w0"] = float(np.mean(dcal))
                    entry["ci_cal"] = boot_ci(dcal)
                out["windows"][wname][tname][str(w)] = entry

    # turnover split at w=0.5 (and best w if different), primary window only
    for w in [0.5]:
        idx = [i for i, r in enumerate(base)
               if windows["first10_both"](r)]
        for label, pred in (("high_turnover_either", lambda r: r["hi_blue"] or r["hi_red"]),
                            ("low_turnover_both", lambda r: (not r["hi_blue"]) and (not r["hi_red"]))):
            sub = [i for i in idx if pred(base[i])]
            mb = metrics([base[i] for i in sub])
            mw = metrics([results[w][i] for i in sub])
            draw = [a - b for a, b in zip(mb["ll_raw_list"], mw["ll_raw_list"])]
            dcal = [a - b for a, b in zip(mb["ll_cal_list"], mw["ll_cal_list"])]
            out["turnover_split"][label] = {
                "w": w, "n": mb["n"],
                "ll_raw_w0": mb.get("ll_raw"), "ll_raw_w": mw.get("ll_raw"),
                "diff_raw": float(np.mean(draw)) if draw else None,
                "ci_raw": boot_ci(draw) if draw else None,
                "ll_cal_w0": mb.get("ll_cal"), "ll_cal_w": mw.get("ll_cal"),
                "diff_cal": float(np.mean(dcal)) if dcal else None,
                "ci_cal": boot_ci(dcal) if dcal else None,
            }

    out["validation"] = {
        "w0_reproduces_replay_team_elo": True,
        "max_pf_vs_features_elo_diff": float(md.max()),
        "n_2026_games": int(len(g26)),
        "n_teams_2026": int(len(set(g26["blue"]) | set(g26["red"]))),
        "n_high_turnover_teams": int(len(hi_turn)),
        "turnover": {t: round(v, 2) for t, v in sorted(turnover.items())},
        "model": "shipped model.joblib model_nodraft (LR coef [0.5694, 0.2598, 0.0128] on "
                 "[elo_diff_adj, eg_gold_diff, blue_edge], scaler_nodraft, isotonic cal; "
                 "fit_end 2026-07-01, cal_end 2026-10-01)",
        "caveat": "model fit/calibration windows cover early 2026, so absolute log-loss "
                  "is in-sample for the model; paired diffs vs w=0 are the valid quantity.",
        "roster_strength_comparator": "skipped: delta_2026 uses full-year 2026 rosters "
                  "(leakage); a clean version would need re-deriving adjustments from "
                  "first-game lineups, a different formula/test.",
    }
    out["design"] = {
        "boundary": "2025->2026", "K": K,
        "prior": "starting Elo = (1-w)*regressed team Elo + w*player aggregate at "
                 "first 2026 game (actual lineup, no full-year roster use)",
        "w_grid": WS, "gate": "diff > 0.002 AND 95% CI excludes 0",
        "primary_window": "both teams within first 10 games of 2026",
    }

    os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)
    with open(os.path.join(BASE, "reports", "season_start_prior_test.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out["windows"], indent=1))
    print(json.dumps(out["turnover_split"], indent=1))
    return out


if __name__ == "__main__":
    main()
