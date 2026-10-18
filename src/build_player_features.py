"""Build per-game player-Elo + instability features and the blend snapshot.

    python src/build_player_features.py

Writes:
- data/processed/player_features.csv (one row per game; join to
  data/processed/features.csv on gameid for the blend validation)
- model/player_blend_snapshot.json (final ratings + per-team last/modal
  lineups; the roster resolver predict.py --elo-blend consumes)

Validation (must all pass; exits nonzero otherwise):
- final team Elos match model/snapshots.joblib Elo (same replay constants)
- per-team player_elo matches model/player_ratings.json (same replay)
- per-game team_elo_diff matches data/processed/features.csv elo_diff
- Play-In Oct 2026 |player_diff - team_diff| range is 1-24 (report §9 check)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd

from player_elo import load_player_rows, replay, write_outputs  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("loading player rows...")
    df = load_player_rows()
    print(f"  {len(df)} player rows, {df['gameid'].nunique()} games")
    per_game, snapshot, skipped = replay(df)
    print(f"  replayed {len(per_game)} games, skipped {skipped}")
    feat_path, snap_path = write_outputs(per_game, snapshot)
    print(f"wrote {feat_path} ({len(per_game)} games)")
    print(f"wrote {snap_path} "
          f"({len(snapshot['players'])} (player, role) ratings, "
          f"{len(snapshot['teams'])} teams)")

    # ---- validation 1: final team Elo vs snapshots.joblib ----
    # snapshots.joblib carries the v1.3.3 Elo PATCH (MVK +32.29, FURIA -32.29
    # from the Oct 17 MVK 3-1 FURIA series; reports/ratings_patch_1.3.3.md),
    # while this replay is the pure game-by-game Elo. So the only teams
    # allowed to differ are exactly those two, by exactly the patch amounts.
    snaps = joblib.load(os.path.join(BASE, "model", "snapshots.joblib"))
    diffs = {}
    for team, rec in snapshot["teams"].items():
        if team in snaps["elo"]:
            diffs[team] = rec["team_elo"] - snaps["elo"][team]
    big = {t: d for t, d in diffs.items() if abs(d) >= 0.51}
    print(f"final team Elo vs snapshots.joblib: max |diff| = "
          f"{max(abs(d) for d in diffs.values()):.2f} over {len(diffs)} teams; "
          f"teams differing by >= 0.51: {big}")
    assert set(big) == {"MVK Esports", "FURIA"}, \
        f"unexpected team Elo divergence (expected only the 1.3.3 patch): {big}"
    assert abs(big["MVK Esports"] + 32.29) < 0.51, big
    assert abs(big["FURIA"] - 32.29) < 0.51, big

    # ---- validation 2: per-team player_elo vs player_ratings.json ----
    with open(os.path.join(BASE, "model", "player_ratings.json")) as f:
        pr = json.load(f)
    diffs = []
    for team, rec in snapshot["teams"].items():
        if team in pr["teams"]:
            diffs.append(abs(rec["player_elo"] - pr["teams"][team]["player_elo"]))
    print(f"per-team player_elo vs player_ratings.json: max |diff| = "
          f"{max(diffs):.2f} over {len(diffs)} teams")
    assert max(diffs) < 0.51, "player Elo replay diverged from build_player_ratings.py"

    # ---- validation 3: per-game team_elo_diff vs features.csv elo_diff ----
    feat = pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"),
                       usecols=["gameid", "elo_diff"])
    m = per_game.merge(feat, on="gameid", how="inner", suffixes=("", "_feat"))
    print(f"  merged {len(m)} games with features.csv")
    d = (m["team_elo_diff"] - m["elo_diff"]).abs()
    print(f"per-game team_elo_diff vs features.csv elo_diff: "
          f"max |diff| = {d.max():.2f}, mean = {d.mean():.4f}")
    assert d.max() < 0.51, "per-game team Elo replay diverged from build_features.py"

    # ---- validation 4: Play-In gap check (report §9: 1-24) ----
    pg = per_game.copy()
    pg["date"] = pd.to_datetime(pg["date"])
    pi = pg[(pg["league"] == "WLDs") & (pg["date"] >= "2026-10-15")].copy()
    if len(pi):
        pi["gap"] = (pi["player_elo_diff"] - pi["team_elo_diff"]).abs()
        print(f"Play-In |player_diff - team_diff| range: {pi['gap'].min():.1f} - "
              f"{pi['gap'].max():.1f} (report: 1-24)")
        assert pi["gap"].min() >= 0.5 and pi["gap"].max() <= 24.5

    # ---- summary stats ----
    print(f"instability: mean={per_game['instability_max'].mean():.3f}, "
          f"share>0={(per_game['instability_max'] > 0).mean():.3f}, "
          f"share==1={(per_game['instability_max'] == 1).mean():.4f}")
    print("ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
