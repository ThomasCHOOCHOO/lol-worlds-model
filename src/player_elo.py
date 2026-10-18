"""Player-Elo replay + roster-instability features (tier-conditional blend).

Rebuilds the player Elo replay per reports/player_ratings_analysis.md section 2
(same constants as build_features.py team Elo, so scales are comparable):
- players keyed by Oracle's Elixir playerid (null playerid falls back to
  "name:<playername>"); rating key is (player, role)
- init 1500, K=24, 40% regression toward 1500 at each calendar-year boundary
- team aggregate = mean of the 5 pre-game player ratings
- every player on a side updates by K * (team result - expected team score),
  where the expected score uses the PLAYER-aggregate diff
- debutant prior = mean of rated teammates (1500 if none rated)

Plus the roster-instability context variable (see reports/
tier_conditional_blend.md):
- modal lineup of a team at game time = the 5 players with the most games
  for that team in the trailing 30 days before the game (ties: most recent
  appearance wins, then player key). Overlap compares PLAYER keys, not
  (player, role) keys: a role swap by the same five humans is not a lineup
  change for the purpose of "does team Elo still describe these five?".
- instability = 1 - |today's 5 players ∩ modal 5| / 5.
- Fallbacks: if fewer than 5 distinct players appear in the trailing-30-day
  window, compare against the team's most recent lineup instead; if the
  team has no prior game at all, instability = 0 (nothing to compare --
  and for a debut team player Elo and team Elo coincide anyway).

Outputs (via src/build_player_features.py):
- data/processed/player_features.csv: one row per game with
  player_elo_diff, per-side player/team Elos, per-side + max instability.
- model/player_blend_snapshot.json: final per-(player, role) ratings,
  per-team last + modal lineups, and per-team instability as of the latest
  game -- the roster resolver predict.py --elo-blend consumes.
"""
import json
import os
from collections import defaultdict, deque
from datetime import timedelta

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
PROC_DIR = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "model")

ELO_INIT = 1500.0
K = 24.0
KEEP = 0.6  # 40% regression toward 1500 each new calendar year
WINDOW_DAYS = 30
POSITION_ORDER = ["top", "jng", "mid", "bot", "sup"]

COLS = ["gameid", "date", "league", "side", "position", "playername",
        "playerid", "teamname", "result"]


def elo_exp(diff):
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def player_key(playerid, playername):
    """playerid when present, else 'name:<playername>' (1.3% of rows)."""
    return str(playerid) if pd.notna(playerid) else "name:" + str(playername)


def key_str(pid, role):
    return f"{pid}||{role}"


def split_key(s):
    pid, role = s.rsplit("||", 1)
    return pid, role


def load_player_rows():
    frames = []
    for year in ("2024", "2025", "2026"):
        path = os.path.join(RAW_DIR, f"{year}.csv")
        df = pd.read_csv(path, usecols=COLS, low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"] != "team"].copy()  # drop team aggregate rows
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "gameid", "side", "teamname"])
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    return df


def _modal_lineup(window):
    """Top-5 players by appearances in a window deque of (date, player-set).

    Returns (modal_players, n_distinct). Ties: most recent appearance wins,
    then player key (deterministic)."""
    counts = defaultdict(int)
    last_seen = {}
    for d, players in window:
        for p in players:
            counts[p] += 1
            if p not in last_seen or d > last_seen[p]:
                last_seen[p] = d
    ranked = sorted(counts, key=lambda p: (-counts[p], -last_seen[p].timestamp(), p))
    return ranked[:5], len(ranked)


def replay(df):
    """Walk-forward player-Elo + team-Elo replay.

    Returns (per_game DataFrame, snapshot dict). Per-game rows carry
    pre-game ratings only (no leakage: state updates after the row is built).
    """
    players = df.copy()
    gdate = players.groupby("gameid")["date"].first()
    game_ids = sorted(gdate.index, key=lambda g: (gdate[g], g))
    by_game = {gid: grp for gid, grp in players.groupby("gameid")}

    team_elo = defaultdict(lambda: ELO_INIT)
    player_rating = {}    # (pid, role) -> rating
    player_meta = {}      # (pid, role) -> (name, last_seen date)
    last_lineup = {}      # team -> list of (pid, role) keys, position order
    last_league = {}      # team -> league of its most recent game
    windows = defaultdict(deque)  # team -> deque of (date, {player keys})
    rows = []

    cur_year = None
    skipped = 0
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
        if len(blue_rows) != 5 or len(red_rows) != 5:
            skipped += 1
            continue
        blue_team = blue_rows["teamname"].iloc[0]
        red_team = red_rows["teamname"].iloc[0]
        bw = blue_rows["result"].iloc[0]
        if pd.isna(bw):
            skipped += 1
            continue
        blue_win = 1 if bw == 1 else 0
        league = grp["league"].iloc[0] if "league" in grp else None

        def side_pre(team, rows_):
            keys = []
            for _, r in rows_.iterrows():
                pid = player_key(r["playerid"], r["playername"])
                keys.append((pid, str(r["position"])))
            # canonical position order for stable lineups
            order = {p: i for i, p in enumerate(POSITION_ORDER)}
            keys = sorted(keys, key=lambda kr: order.get(kr[1], 99))
            rated = [player_rating[k] for k in keys if k in player_rating]
            prior = float(sum(rated) / len(rated)) if rated else ELO_INIT
            pre = [player_rating.get(k, prior) for k in keys]
            # instability: today's player set vs trailing-30-day modal lineup
            win = windows[team]
            cutoff = t - timedelta(days=WINDOW_DAYS)
            while win and win[0][0] < cutoff:
                win.popleft()
            modal, n_distinct = _modal_lineup(win)
            if n_distinct >= 5:
                ref = modal
            elif team in last_lineup:
                ref = [pid for pid, _ in last_lineup[team]]
            else:
                ref = None
            today = {pid for pid, _ in keys}
            if ref is None:
                inst = 0.0
            else:
                inst = 1.0 - len(today & set(ref)) / 5.0
            return keys, pre, inst, modal

        blue_keys, blue_pre, inst_b, modal_b = side_pre(blue_team, blue_rows)
        red_keys, red_pre, inst_r, modal_r = side_pre(red_team, red_rows)
        agg_b = sum(blue_pre) / 5.0
        agg_r = sum(red_pre) / 5.0
        eb, er = team_elo[blue_team], team_elo[red_team]
        exp_team = elo_exp(eb - er)
        exp_player = elo_exp(agg_b - agg_r)

        rows.append({
            "gameid": gid, "date": t, "league": league,
            "blue": blue_team, "red": red_team,
            "team_elo_blue": eb, "team_elo_red": er,
            "player_elo_blue": agg_b, "player_elo_red": agg_r,
            "team_elo_diff": eb - er, "player_elo_diff": agg_b - agg_r,
            "instability_blue": inst_b, "instability_red": inst_r,
            "instability_max": max(inst_b, inst_r),
            "lineup_blue": "|".join(key_str(*k) for k in blue_keys),
            "lineup_red": "|".join(key_str(*k) for k in red_keys),
            "modal_blue": "|".join(modal_b),
            "modal_red": "|".join(modal_r),
            "y": blue_win,
        })

        # ---- update state (strictly after feature computation) ----
        team_elo[blue_team] = eb + K * (blue_win - exp_team)
        team_elo[red_team] = er + K * ((1 - blue_win) - (1 - exp_team))
        for k, pre in zip(blue_keys, blue_pre):
            player_rating[k] = pre + K * (blue_win - exp_player)
        for k, pre in zip(red_keys, red_pre):
            player_rating[k] = pre + K * ((1 - blue_win) - (1 - exp_player))
        for rows_, keys in ((blue_rows, blue_keys), (red_rows, red_keys)):
            for (_, r), k in zip(rows_.iterrows(), keys):
                name = r["playername"]
                prev = player_meta.get(k)
                if prev is None or t > prev[1]:
                    player_meta[k] = ("" if pd.isna(name) else str(name), t)
        for team, keys in ((blue_team, blue_keys), (red_team, red_keys)):
            last_lineup[team] = keys
            last_league[team] = league
            windows[team].append((t, {pid for pid, _ in keys}))

    per_game = pd.DataFrame(rows)
    t_max = per_game["date"].max() if len(per_game) else None

    # ---- snapshot: final ratings + per-team last/modal lineups ----
    players_out = {}
    for (pid, role), rating in player_rating.items():
        name, last = player_meta.get((pid, role), ("", t_max))
        players_out[key_str(pid, role)] = {
            "name": name, "role": role,
            "rating": round(float(rating), 2),
            "last_seen": str(last.date()) if hasattr(last, "date") else str(last),
        }
    teams_out = {}
    for team, lineup in last_lineup.items():
        p_elo = sum(player_rating[k] for k in lineup) / 5.0
        win = windows[team]
        if t_max is not None:
            cutoff = t_max - timedelta(days=WINDOW_DAYS)
            while win and win[0][0] < cutoff:
                win.popleft()
        modal, n_distinct = _modal_lineup(win)
        if n_distinct >= 5:
            ref = modal
        else:
            ref = [pid for pid, _ in lineup]
        today = {pid for pid, _ in lineup}
        inst = 1.0 - len(today & set(ref)) / 5.0 if ref else 0.0
        teams_out[team] = {
            "team_elo": round(float(team_elo[team]), 2),
            "player_elo": round(float(p_elo), 2),
            "gap": round(float(p_elo - team_elo[team]), 2),
            "league": last_league.get(team),
            "last_lineup": [key_str(*k) for k in lineup],
            "modal_lineup": modal,
            "instability": round(float(inst), 3),
        }
    snapshot = {
        "as_of": str(t_max),
        "players": players_out,
        "teams": teams_out,
        "blend_params": None,  # filled by src/train_blend.py after fitting (a, b)
        "notes": ("Player Elo replay per reports/player_ratings_analysis.md §2. "
                  "instability = 1 - |last lineup ∩ trailing-30-day modal lineup| / 5. "
                  "blend_params is None until src/train_blend.py fits (a, b)."),
    }
    return per_game, snapshot, skipped


def write_outputs(per_game, snapshot):
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    feat_path = os.path.join(PROC_DIR, "player_features.csv")
    per_game.to_csv(feat_path, index=False)
    snap_path = os.path.join(MODEL_DIR, "player_blend_snapshot.json")
    with open(snap_path, "w") as f:
        json.dump(snapshot, f, indent=1, sort_keys=True)
    return feat_path, snap_path
