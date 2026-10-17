"""Build model/player_ratings.json: player Elo snapshot for the app's

diagnostic flag (|player_elo - team_elo| gap per team). Wired into
src/refresh.py so the snapshot regenerates on every refresh.

Rebuilds the player Elo replay per reports/player_ratings_analysis.md section 2:
- players keyed by playerid (null playerid falls back to name), key = (player, role)
- init 1500, K=24, 40% regression toward 1500 at each calendar-year boundary
- team aggregate = mean of the 5 pre-game player ratings
- every player on a side updates by K * (team result - expected team score),
  where the expected score uses the player-aggregate diff
- debutant prior = mean of rated teammates (1500 if none rated)
Team Elo is replayed in the same pass with the same constants (mirroring
src/build_features.py), so player_elo and team_elo are internally consistent.

Emits model/player_ratings.json with EXACTLY:
  {"players": {"<playerid>": {"name": str, "role": str, "rating": float}},
   "teams": {"<team>": {"player_elo": float, "team_elo": float, "gap": float}}}
- "current 5" per team = last-seen lineup from the most recent game
- player_elo = mean of those 5 players' CURRENT (end-of-replay) ratings,
  using the (player, role) rating matching each lineup slot
- gap = player_elo - team_elo

Two data-forced conventions (schema shape unchanged):
- null-playerid players are keyed "name:<playername>" (1.3% of player rows).
- a playerid with multiple roles gets ONE players-dict entry: the role they
  played most recently, with that (player, role) rating.
"""
import json
import math
import os
from collections import defaultdict

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELO_INIT = 1500.0
K = 24.0
KEEP = 0.6  # 40% regression toward 1500 each new calendar year


def elo_exp(diff):
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def main():
    cols = ["gameid", "date", "league", "side", "position", "playername",
            "playerid", "teamname", "result"]
    raw = pd.concat(
        [pd.read_csv(os.path.join(BASE, "data", "raw", f"{y}.csv"),
                     usecols=cols) for y in (2024, 2025, 2026)],
        ignore_index=True,
    )
    players = raw[raw["position"] != "team"].copy()  # drop team aggregate rows
    players["date"] = pd.to_datetime(players["date"])
    assert (players.groupby(["gameid", "side"]).size() == 5).all(), \
        "every game-side must have exactly 5 player rows"

    # games in chronological order (datetime, then gameid for determinism)
    gdate = players.groupby("gameid")["date"].first()
    game_ids = sorted(gdate.index, key=lambda g: (gdate[g], g))
    by_game = {gid: grp for gid, grp in players.groupby("gameid")}

    team_elo = defaultdict(lambda: ELO_INIT)
    player_rating = {}   # (key, role) -> rating
    player_meta = {}     # (key, role) -> (name, last_seen)
    games_played = defaultdict(int)  # (key, role) -> games
    last_lineup = {}     # team -> list of 5 (key, role)
    pregame = []         # validation records (Play-In gap check)

    cur_year = None
    for gid in game_ids:
        grp = by_game[gid]
        t = grp["date"].iloc[0]
        if cur_year is None:
            cur_year = t.year
        elif t.year != cur_year:
            for team in list(team_elo.keys()):
                team_elo[team] = ELO_INIT + KEEP * (team_elo[team] - ELO_INIT)
            for key in list(player_rating.keys()):
                player_rating[key] = ELO_INIT + KEEP * (player_rating[key] - ELO_INIT)
            cur_year = t.year

        blue_rows = grp[grp["side"] == "Blue"]
        red_rows = grp[grp["side"] == "Red"]
        assert len(blue_rows) == 5 and len(red_rows) == 5, gid
        blue_team = blue_rows["teamname"].iloc[0]
        red_team = red_rows["teamname"].iloc[0]
        blue_win = 1 if blue_rows["result"].iloc[0] == 1 else 0

        def side_pre(rows):
            keys = []
            for _, r in rows.iterrows():
                pid = r["playerid"]
                key = pid if pd.notna(pid) else "name:" + str(r["playername"])
                keys.append((str(key), str(r["position"])))
            rated = [player_rating[k] for k in keys if k in player_rating]
            prior = float(sum(rated) / len(rated)) if rated else ELO_INIT
            pre = [player_rating.get(k, prior) for k in keys]
            return keys, pre

        blue_keys, blue_pre = side_pre(blue_rows)
        red_keys, red_pre = side_pre(red_rows)
        agg_b = sum(blue_pre) / 5.0
        agg_r = sum(red_pre) / 5.0

        eb, er = team_elo[blue_team], team_elo[red_team]
        exp_team = elo_exp(eb - er)
        exp_player = elo_exp(agg_b - agg_r)
        pregame.append({
            "gameid": gid, "date": t, "league": grp["league"].iloc[0] if "league" in grp else None,
            "blue": blue_team, "red": red_team,
            "team_diff": eb - er, "player_diff": agg_b - agg_r,
        })

        team_elo[blue_team] = eb + K * (blue_win - exp_team)
        team_elo[red_team] = er + K * ((1 - blue_win) - (1 - exp_team))
        for k, pre in zip(blue_keys, blue_pre):
            player_rating[k] = pre + K * (blue_win - exp_player)
            games_played[k] += 1
        for k, pre in zip(red_keys, red_pre):
            player_rating[k] = pre + K * ((1 - blue_win) - (1 - exp_player))
            games_played[k] += 1
        for rows, keys in ((blue_rows, blue_keys), (red_rows, red_keys)):
            for (_, r), k in zip(rows.iterrows(), keys):
                name = r["playername"]
                prev = player_meta.get(k)
                if prev is None or t > prev[1]:
                    player_meta[k] = ("" if pd.isna(name) else str(name), t)

        last_lineup[blue_team] = blue_keys
        last_lineup[red_team] = red_keys

    # ---- players dict: one entry per playerid (most recent role) ----
    by_pid = defaultdict(list)
    for (pid, role) in player_rating:
        by_pid[pid].append(role)
    players_out = {}
    for pid, roles in by_pid.items():
        best_role, best_date = None, None
        for role in roles:
            _, last = player_meta[(pid, role)]
            if best_date is None or last > best_date:
                best_role, best_date = role, last
        name, _ = player_meta[(pid, best_role)]
        players_out[pid] = {
            "name": name,
            "role": best_role,
            "rating": round(float(player_rating[(pid, best_role)]), 2),
        }

    # ---- teams dict: current-5 player_elo vs team_elo ----
    teams_out = {}
    for team, lineup in last_lineup.items():
        assert len(lineup) == 5, team
        p_elo = sum(player_rating[k] for k in lineup) / 5.0
        t_elo = team_elo[team]
        teams_out[team] = {
            "player_elo": round(float(p_elo), 2),
            "team_elo": round(float(t_elo), 2),
            "gap": round(float(p_elo - t_elo), 2),
        }

    out = {"players": players_out, "teams": teams_out}
    path = os.path.join(BASE, "model", "player_ratings.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {path}: {len(players_out)} players, {len(teams_out)} teams")
    print(f"(player, role) ratings tracked: {len(player_rating)}")

    # ---- validation ----
    import numpy as np
    games_per_pid = defaultdict(int)
    for (pid, _), n in games_played.items():
        games_per_pid[pid] += n
    med = float(np.median(list(games_per_pid.values())))
    med_key = float(np.median(list(games_played.values())))
    med_ge20 = float(np.median([v for v in games_per_pid.values() if v >= 20]))
    print(f"distinct playerids: {len(games_per_pid)} (report ~5,333); "
          f"median games/player: {med:.0f} (report ~28); "
          f"median per (player,role) key: {med_key:.0f}; "
          f"median among players with >=20 games: {med_ge20:.0f}")

    pre = pd.DataFrame(pregame)
    pre["date"] = pd.to_datetime(pre["date"])
    pi = pre[(pre["league"] == "WLDs") & (pre["date"] >= "2026-10-15")].copy()
    pi["gap_diff"] = (pi["player_diff"] - pi["team_diff"]).abs()
    pi["signed"] = pi["player_diff"] - pi["team_diff"]
    for (b, r), g in pi.groupby(["blue", "red"]):
        print(f"Play-In {b} vs {r}: |player_diff - team_diff| per game = "
              f"{[round(v, 1) for v in g['gap_diff']]}; "
              f"signed = {[round(v, 1) for v in g['signed']]}")
    print("Play-In |gap| range:", round(pi["gap_diff"].min(), 1),
          "-", round(pi["gap_diff"].max(), 1), "(report: 1-24)")


if __name__ == "__main__":
    main()
