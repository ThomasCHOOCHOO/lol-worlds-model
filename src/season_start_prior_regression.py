"""Regression check for the SHIPPED season-start prior (w=1.0).

Replays 2025->2026 through the shipped code paths and confirms the result
from reports/season_start_prior_test.md reproduces: w=1.0 raw log-loss
gain about +0.0143 on the primary first-10-games window (n=1,434),
95% CI excluding 0. Target numbers (test): raw diff +0.01433
CI [+0.00723, +0.02148]; cal diff +0.01575 CI [+0.00810, +0.02337].

Part A: src/player_elo.py replay() with w=1.0 vs w=0.0, scored with the
shipped model.joblib. Tests the replay the sidecar / snapshot use.

Part B: src/build_features.py run in a scratch copy of the repo
(/tmp/ssp_regression, data/raw symlinked, own data/processed + model),
once with the default w=1.0 and once with --season-start-prior-w 0.
Tests the replay that builds features.csv / snapshots.joblib. Also checks
  (i) w=0 scratch features.csv reproduces the live features.csv elo_diff
      (i.e. the edits did not change baseline behaviour), and
  (ii) w=1.0 scratch elo_diff matches the Part A replay team_elo_diff.

Writes only reports/season_start_prior_regression.json (and the scratch
dir under /tmp). Never touches live artifacts: the scratch build writes
into /tmp, and Part A runs replay() in-process without write_outputs.

Usage:
  .venv/bin/python src/season_start_prior_regression.py [--part a|b|all]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from season_start_prior_test import boot_ci, ll  # noqa: E402

SCRATCH = "/tmp/ssp_regression"
TARGET_RAW_DIFF = 0.01433
TARGET_RAW_CI = [0.00723, 0.02148]


def load_model():
    import joblib
    bundle = joblib.load(os.path.join(BASE, "model", "model.joblib"))
    return bundle["model_nodraft"], bundle["scaler_nodraft"]


def feat_lookup():
    """gameid -> (eg_gold_diff, blue_edge, region_corr) from live features.csv."""
    feat = pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"),
                       low_memory=False)
    feat["region_corr"] = feat["elo_diff_adj"] - feat["elo_diff"]
    return {r["gameid"]: (r["eg_gold_diff"], r["blue_edge"], r["region_corr"])
            for _, r in feat.iterrows()}


def score_replay(per_game, lookup, model, scaler):
    """Score a replay's 2026 games; returns rows with p_raw/p_cal/y/counts."""
    from collections import Counter
    pg = per_game.copy()
    pg["date"] = pd.to_datetime(pg["date"])
    g26 = pg[pg["date"] >= "2026-01-01"].sort_values(
        ["date", "gameid"]).reset_index(drop=True)
    counts = Counter()
    rows = []
    for _, r in g26.iterrows():
        blue, red = r["blue"], r["red"]
        diff = r["team_elo_blue"] - r["team_elo_red"]
        eg, be, rc = lookup[r["gameid"]]
        X = np.array([[diff + rc, eg, be]], dtype=float)
        Xs = scaler.transform(X)
        rows.append({
            "gameid": r["gameid"], "date": r["date"], "league": r["league"],
            "y": int(r["y"]),
            "p_raw": float(model.base_model.predict_proba(Xs)[0, 1]),
            "p_cal": float(model.predict_proba(Xs)[0, 1]),
            "cnt_blue_pre": counts[blue], "cnt_red_pre": counts[red],
        })
        counts[blue] += 1
        counts[red] += 1
    return rows


def primary_metrics(rows_w, rows_0):
    """Primary window (both teams within first 10 games of 2026)."""
    idx = [i for i, r in enumerate(rows_0)
           if r["cnt_blue_pre"] < 10 and r["cnt_red_pre"] < 10]
    assert idx, "primary window is empty"
    out = {"n": len(idx)}
    for key, col in (("raw", "p_raw"), ("cal", "p_cal")):
        lls_w = [ll(rows_w[i][col], rows_w[i]["y"]) for i in idx]
        lls_0 = [ll(rows_0[i][col], rows_0[i]["y"]) for i in idx]
        diffs = [a - b for a, b in zip(lls_0, lls_w)]
        out[f"ll_{key}_w"] = float(np.mean(lls_w))
        out[f"ll_{key}_w0"] = float(np.mean(lls_0))
        out[f"diff_{key}"] = float(np.mean(diffs))
        out[f"ci_{key}"] = boot_ci(diffs)
    return out


def part_a():
    print("=== Part A: player_elo.replay, w=1.0 vs w=0.0 ===", flush=True)
    from player_elo import load_player_rows, replay
    model, scaler = load_model()
    lookup = feat_lookup()
    df = load_player_rows()
    print(f"  player rows: {len(df)}, games: {df['gameid'].nunique()}", flush=True)
    results = {}
    per_games = {}
    for w in (0.0, 1.0):
        print(f"  replay w={w} ...", flush=True)
        per_game, _, skipped = replay(df, season_start_prior_w=w)
        print(f"    {len(per_game)} games, skipped {skipped}", flush=True)
        per_games[w] = per_game
        results[w] = score_replay(per_game, lookup, model, scaler)
        os.makedirs(SCRATCH, exist_ok=True)
        per_game.to_csv(os.path.join(SCRATCH, f"per_game_w{w}.csv"), index=False)
    m = primary_metrics(results[1.0], results[0.0])
    print(f"  PRIMARY n={m['n']}: raw diff {m['diff_raw']:.5f} "
          f"CI [{m['ci_raw'][0]:.5f}, {m['ci_raw'][1]:.5f}]; "
          f"cal diff {m['diff_cal']:.5f} "
          f"CI [{m['ci_cal'][0]:.5f}, {m['ci_cal'][1]:.5f}]", flush=True)
    return {"part_a": m, "per_games": per_games}


def setup_scratch():
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(os.path.join(SCRATCH, "data", "processed"))
    os.makedirs(os.path.join(SCRATCH, "model"))
    os.symlink(os.path.join(BASE, "data", "raw"),
               os.path.join(SCRATCH, "data", "raw"))
    shutil.copytree(os.path.join(BASE, "src"), os.path.join(SCRATCH, "src"))
    for f in ("region_offsets.json", "model.joblib"):
        shutil.copy2(os.path.join(BASE, "model", f),
                     os.path.join(SCRATCH, "model", f))
    print(f"  scratch ready at {SCRATCH}", flush=True)


def run_build_features(w):
    """Run build_features.py in the scratch repo; return its features.csv."""
    args = [sys.executable, os.path.join(SCRATCH, "src", "build_features.py")]
    if w == 0.0:
        args += ["--season-start-prior-w", "0"]
    log_path = os.path.join(SCRATCH, f"build_w{w}.log")
    with open(log_path, "w") as lf:
        p = subprocess.run(args, cwd=SCRATCH, stdout=lf, stderr=subprocess.STDOUT,
                           text=True, timeout=7200)
    if p.returncode != 0:
        raise SystemExit(f"scratch build_features w={w} FAILED (see {log_path})")
    out = os.path.join(SCRATCH, f"features_w{w}.csv")
    shutil.copy2(os.path.join(SCRATCH, "data", "processed", "features.csv"), out)
    print(f"  build_features w={w} ok -> {out}", flush=True)
    return out


def score_features_csv(path, model, scaler):
    """Score a features.csv on the primary window (its own elo_diff_adj)."""
    from collections import Counter
    feat = pd.read_csv(path, low_memory=False)
    feat["date"] = pd.to_datetime(feat["date"])
    g26 = feat[feat["date"] >= "2026-01-01"].sort_values(
        ["date", "gameid"]).reset_index(drop=True)
    counts = Counter()
    rows = []
    for _, r in g26.iterrows():
        X = np.array([[r["elo_diff_adj"], r["eg_gold_diff"], r["blue_edge"]]],
                     dtype=float)
        Xs = scaler.transform(X)
        rows.append({
            "gameid": r["gameid"], "y": int(r["y"]),
            "p_raw": float(model.base_model.predict_proba(Xs)[0, 1]),
            "p_cal": float(model.predict_proba(Xs)[0, 1]),
            "cnt_blue_pre": counts[r["blue"]], "cnt_red_pre": counts[r["red"]],
        })
        counts[r["blue"]] += 1
        counts[r["red"]] += 1
    return rows, feat


def part_b(per_games_a):
    print("=== Part B: build_features.py in scratch, w=1.0 vs w=0.0 ===",
          flush=True)
    model, scaler = load_model()
    setup_scratch()
    paths = {w: run_build_features(w) for w in (1.0, 0.0)}
    rows = {}
    feats = {}
    for w, path in paths.items():
        rows[w], feats[w] = score_features_csv(path, model, scaler)
    m = primary_metrics(rows[1.0], rows[0.0])
    print(f"  PRIMARY n={m['n']}: raw diff {m['diff_raw']:.5f} "
          f"CI [{m['ci_raw'][0]:.5f}, {m['ci_raw'][1]:.5f}]; "
          f"cal diff {m['diff_cal']:.5f} "
          f"CI [{m['ci_cal'][0]:.5f}, {m['ci_cal'][1]:.5f}]", flush=True)

    # (i) w=0 scratch reproduces live features.csv elo_diff (baseline unchanged)
    live = pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"),
                       usecols=["gameid", "elo_diff"], low_memory=False)
    merged0 = feats[0.0].merge(live, on="gameid", suffixes=("_scratch", "_live"))
    d0 = (merged0["elo_diff_scratch"] - merged0["elo_diff_live"]).abs()
    print(f"  w=0 scratch vs live features.csv elo_diff: max |diff| = {d0.max():.6f} "
          f"over {len(merged0)} games", flush=True)

    # (ii) w=1.0 scratch elo_diff matches Part A replay team_elo_diff
    pg1 = per_games_a[1.0][["gameid", "team_elo_blue", "team_elo_red"]].copy()
    pg1["team_elo_diff"] = pg1["team_elo_blue"] - pg1["team_elo_red"]
    merged1 = feats[1.0].merge(pg1[["gameid", "team_elo_diff"]], on="gameid")
    d1 = (merged1["elo_diff"] - merged1["team_elo_diff"]).abs()
    print(f"  w=1.0 scratch elo_diff vs Part A replay team_elo_diff: "
          f"max |diff| = {d1.max():.6f} over {len(merged1)} games", flush=True)
    return {"part_b": m,
            "w0_scratch_vs_live_max_diff": float(d0.max()),
            "w1_scratch_vs_replay_max_diff": float(d1.max())}


def main():
    ap = argparse.ArgumentParser(description="Regression check for the "
                                            "shipped season-start prior")
    ap.add_argument("--part", choices=["a", "b", "all"], default="all")
    args = ap.parse_args()
    out = {"target": {"raw_diff": TARGET_RAW_DIFF, "raw_ci": TARGET_RAW_CI,
                      "n_primary": 1434}}
    per_games = None
    if args.part in ("a", "all"):
        res_a = part_a()
        out.update({k: v for k, v in res_a.items() if k != "per_games"})
        per_games = res_a["per_games"]
    if args.part in ("b", "all"):
        if per_games is None:
            per_games = {}
            for w in (0.0, 1.0):
                per_games[w] = pd.read_csv(
                    os.path.join(SCRATCH, f"per_game_w{w}.csv"), low_memory=False)
        out.update(part_b(per_games))

    def verdict(m):
        ok = (m["diff_raw"] > 0.002 and m["ci_raw"][0] > 0
              and abs(m["diff_raw"] - TARGET_RAW_DIFF) < 0.002)
        return "PASS" if ok else "FAIL"

    for part in ("part_a", "part_b"):
        if part in out:
            out[part]["verdict"] = verdict(out[part])
            print(f"{part}: {out[part]['verdict']} "
                  f"(raw diff {out[part]['diff_raw']:.5f} vs target "
                  f"{TARGET_RAW_DIFF})", flush=True)
    with open(os.path.join(BASE, "reports",
                           "season_start_prior_regression.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote reports/season_start_prior_regression.json")


if __name__ == "__main__":
    main()
