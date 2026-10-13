"""Roster-aware team strength for the LoL Worlds phone app.

Problem: team rosters change every year (offseason shuffle), but the base
model (build_features.py) applies the SAME 40% Elo regression toward 1500 to
every team at each year boundary. A team that kept all 5 starters is treated
identically to a team that replaced 3 of them.

This module derives each team's starting roster per calendar year from
Oracle's Elixir player rows, measures roster turnover at year boundaries, and
defines a roster-adjusted team strength that blends:
  * the team's previous-season year-end Elo, and
  * the prior-season Elo of the teams the incoming players came from,
with extra shrinkage toward the league mean (1500) proportional to turnover.

Formula (for the boundary between year y-1 and year y):

    roster(team, y)  = 5 players with the most games for `team` in year y
                       (substitutes beyond the top 5 are ignored)
    t(team, y)       = |roster(team,y) \\ roster(team,y-1)| / 5
                       = fraction of starting slots filled by new players
                       (rookies with no year-(y-1) games count toward t)

    E_prev(team)     = team's Elo after the last game of year y-1
                       (Elo replay uses build_features constants: init 1500, K=24)
    U(team, y)       = 1500 + 0.6 * (E_prev(team) - 1500)
                       # the existing model's uniform prior at season start

    q(p)             = Elo_{y-1} of player p's main team in year y-1
                       (team p played most games for; 1500 if p has no
                       year-(y-1) games, i.e. a rookie / import)
    incoming(team,y) = mean(q(p)) over incoming starters, or 1500 if none
    target           = (1 - t) * E_prev(team) + t * incoming(team, y)
    A(team, y)       = 1500 + 0.6 * (1 - 0.5 * t) * (target - 1500)
                       # roster-adjusted prior at season start

    delta(team)      = A(team, 2026) - U(team, 2026)
    rating_adjusted  = snapshot_elo(team) + delta(team)

Properties:
  * t = 0  ->  A = U exactly (no roster change => no change vs the base model).
  * t = 1  ->  A = 1500 + 0.3 * (incoming - 1500): a fully rebuilt roster is
    judged mostly by the quality of the incoming players' old teams, heavily
    regressed toward the mean (high uncertainty).
  * The delta is computed once at the latest year boundary and carried forward
    to the current snapshot; games played since then already move the Elo.

Outputs:
  * roster_strength(team, year) / adjusted ratings dicts (importable)
  * reports/roster_strength.md - full write-up with a worked numeric example
  * consumed by src/export_artifacts.py for the phone-app JSON bundle
"""

import os
from collections import Counter, defaultdict

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
MODEL_DIR = os.path.join(BASE, "model")

ELO_INIT = 1500.0
ELO_K = 24
YEAR_KEEP = 0.6
ROSTER_SIZE = 5
YEAR_PAIRS = [(2024, 2025), (2025, 2026)]

COLS = ["gameid", "date", "teamname", "playername", "position", "result"]


def load_players():
    frames = []
    for year in ("2024", "2025", "2026"):
        path = os.path.join(RAW_DIR, f"{year}.csv")
        df = pd.read_csv(path, usecols=COLS, low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"] != "team"]  # player rows only
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "gameid", "teamname", "playername"])
    df["year"] = df["date"].dt.year
    return df


def starting_rosters(df):
    """{(year, team): [player, ...]} - top-5 players by games played that year."""
    games = df.groupby(["year", "teamname", "playername"])["gameid"].nunique()
    rosters = {}
    for (year, team), grp in games.groupby(level=[0, 1]):
        top = grp.sort_values(ascending=False).head(ROSTER_SIZE)
        rosters[(year, team)] = list(top.index.get_level_values("playername"))
    return rosters


def year_end_elos_v2():
    """Replay team Elo over 2024-2026 and snapshot each team's Elo at year end.

    Uses the same Elo constants as build_features.py (init 1500, K=24,
    40% regression toward 1500 at each calendar-year boundary).
    Returns {(year, team): elo_after_last_game_of_year}.
    """
    # Per-game sides/teams/results come from team rows (player rows alone
    # don't carry 'side'), so reload the CSVs with the needed columns.
    frames = []
    for year in ("2024", "2025", "2026"):
        f = pd.read_csv(os.path.join(RAW_DIR, f"{year}.csv"),
                        usecols=["gameid", "date", "side", "teamname", "position",
                                 "result"],
                        low_memory=False)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    team_rows = df[df["position"] == "team"].dropna(subset=["date", "side"])
    side_team = team_rows.groupby(["gameid", "side"])["teamname"].first()
    side_won = team_rows.groupby(["gameid", "side"])["result"].first()
    game_date = team_rows.groupby("gameid")["date"].first()

    game_ids = sorted(game_date.index, key=lambda g: game_date[g])
    elo = defaultdict(lambda: ELO_INIT)
    year_end = {}
    cur_year = None
    for gid in game_ids:
        t = game_date[gid]
        if cur_year is None:
            cur_year = t.year
        elif t.year != cur_year:
            for team in list(elo.keys()):
                year_end[(cur_year, team)] = elo[team]
            for team in list(elo.keys()):
                elo[team] = ELO_INIT + YEAR_KEEP * (elo[team] - ELO_INIT)
            cur_year = t.year
        try:
            blue = side_team[(gid, "Blue")]
            red = side_team[(gid, "Red")]
        except KeyError:
            continue
        bw = side_won[(gid, "Blue")]
        if pd.isna(bw):
            continue
        blue_win = 1 if bw == 1 else 0
        eb, er = elo[blue], elo[red]
        exp_blue = 1.0 / (1.0 + 10.0 ** (-(eb - er) / 400.0))
        elo[blue] = eb + ELO_K * (blue_win - exp_blue)
        elo[red] = er + ELO_K * ((1 - blue_win) - (1 - exp_blue))
    for team in list(elo.keys()):
        year_end[(cur_year, team)] = elo[team]
    return year_end


def compute_adjustments():
    """Compute the roster adjustment for every team at the 2025->2026 boundary
    (and 2024->2025 for the report).

    Returns dict with per-team: roster_{2024,2025,2026}, turnover per boundary,
    E_prev, incoming_mean, U, A, delta (2026 boundary), plus the raw inputs.
    """
    players = load_players()
    rosters = starting_rosters(players)
    year_end = year_end_elos_v2()

    # player -> main team per year (most games)
    pgames = players.groupby(["year", "playername", "teamname"])["gameid"].nunique()
    main_team = {}
    for (year, player), grp in pgames.groupby(level=[0, 1]):
        main_team[(year, player)] = grp.idxmax()[2]

    out = {}
    teams = sorted({team for (y, team) in rosters})
    for team in teams:
        rec = {"team": team}
        for y in (2024, 2025, 2026):
            rec[f"roster_{y}"] = rosters.get((y, team), [])
        e24 = year_end.get((2024, team))
        e25 = year_end.get((2025, team))
        rec["elo_2024_end"] = None if e24 is None else round(e24, 1)
        rec["elo_2025_end"] = None if e25 is None else round(e25, 1)
        for y_prev, y in YEAR_PAIRS:
            r_prev = set(rosters.get((y_prev, team), []))
            r_new = rosters.get((y, team), [])
            if not r_new or not r_prev:
                rec[f"turnover_{y_prev}_{y}"] = None
                rec[f"A_{y}"] = rec[f"U_{y}"] = None
                continue
            incoming_players = [p for p in r_new if p not in r_prev]
            t = len(incoming_players) / ROSTER_SIZE
            e_prev = year_end.get((y_prev, team), ELO_INIT)
            qs = []
            for p in incoming_players:
                pt = main_team.get((y_prev, p))
                qs.append(year_end.get((y_prev, pt), ELO_INIT) if pt else ELO_INIT)
            incoming_mean = sum(qs) / len(qs) if qs else ELO_INIT
            U = ELO_INIT + YEAR_KEEP * (e_prev - ELO_INIT)
            target = (1 - t) * e_prev + t * incoming_mean
            A = ELO_INIT + YEAR_KEEP * (1 - 0.5 * t) * (target - ELO_INIT)
            rec[f"turnover_{y_prev}_{y}"] = round(t, 3)
            rec[f"incoming_{y}"] = [p for p in incoming_players]
            rec[f"incoming_mean_{y}"] = round(incoming_mean, 1)
            rec[f"U_{y}"] = round(U, 1)
            rec[f"A_{y}"] = round(A, 1)
        rec["delta_2026"] = (round(rec["A_2026"] - rec["U_2026"], 1)
                             if rec.get("A_2026") is not None else 0.0)
        out[team] = rec
    return out


def adjusted_snapshot_ratings(adjustments):
    """{team: roster-adjusted rating} = snapshot Elo + 2026-boundary delta."""
    snaps = joblib.load(os.path.join(MODEL_DIR, "snapshots.joblib"))
    ratings = {}
    for team, elo in snaps["elo"].items():
        delta = adjustments.get(team, {}).get("delta_2026", 0.0) or 0.0
        ratings[team] = round(elo + delta, 2)
    return ratings


def main():
    adj = compute_adjustments()
    n = len(adj)
    with_t = sum(1 for r in adj.values() if r.get("turnover_2025_2026") is not None)
    print(f"{n} teams, {with_t} with a 2025->2026 turnover measurement")
    deltas = [r["delta_2026"] for r in adj.values()]
    print(f"delta_2026: min {min(deltas):.1f}, max {max(deltas):.1f}, "
          f"mean {sum(deltas)/len(deltas):+.1f}")
    # worked example for the report: team with the biggest |delta|
    ex = max(adj.values(), key=lambda r: abs(r["delta_2026"]))
    print("\nWorked example (largest |delta_2026|):")
    for k in ("team", "elo_2025_end", "roster_2025", "roster_2026",
              "turnover_2025_2026", "incoming_2026", "incoming_mean_2026",
              "U_2026", "A_2026", "delta_2026"):
        print(f"  {k}: {ex.get(k)}")
    # also print T1 for familiarity
    t1 = adj.get("T1")
    if t1:
        print("\nT1:")
        for k in ("elo_2025_end", "roster_2025", "roster_2026",
                  "turnover_2025_2026", "incoming_2026", "incoming_mean_2026",
                  "U_2026", "A_2026", "delta_2026"):
            print(f"  {k}: {t1.get(k)}")
    return adj


if __name__ == "__main__":
    main()
