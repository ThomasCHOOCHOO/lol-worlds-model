"""Build walk-forward (no-leakage) game-level features from Oracle's Elixir CSVs.

Every trailing stat for a game at datetime t uses only games with datetime < t.

Features (blue-side perspective):
  elo_diff   - Elo(blue) - Elo(red); Elo init 1500, K=24, 40% regression to 1500 each new year
  form_diff  - REPLACED by eg_gold_diff in v1.3.0 (kept in snapshots only as the
               comfort fallback); trailing-20-game win-rate diff, shrunk toward
               0.5 with 8 pseudo-games
  eg_gold_diff - trailing-20-game mean gold differential at 15 minutes
               (own-team perspective), shrunk toward 0 with 8 pseudo-games;
               blue minus red. Games with null golddiffat15 are skipped.
  blue_edge  - trailing-90-day global blue-side win rate (mild shrinkage, 10 pseudo-games)
  d_power    - sum(champion power over blue's 5 picks) - sum(red's 5);
               champion power is PATCH-AWARE: this champion's win% on the SAME
               patch, games strictly before t, shrunk as
               (w_P + 15*prior)/(n_P + 15) where prior is the previous patch
               version's shrunk value (w_Q + 15*0.5)/(n_Q + 15) on games
               strictly before t (0.5 if no previous patch). Cold start
               (n_P = 0) -> previous patch's value.
  d_comfort  - same with team-champion comfort: team's trailing-365d win% fielding
               that champion (min 3 games), shrunk toward team's trailing-20 win
               rate with 10 pseudo-games. Kept patch-global on purpose: a team's
               (champion, patch) sample is ~1-5 games, pure noise; the 365d
               window is dominated by recent-patch games anyway.
  ban_edge   - sum(power of champions banned BY blue) - sum(power of champions banned BY red)
               (same-patch power, same formula as d_power)
  y          - 1 if blue won, else 0

Picks are derived from player rows (position != 'team'), NOT from pick1..pick5
columns (those have format issues in 2026 data). Bans come from team rows.

Outputs:
  data/processed/features.csv - one row per game with the features above
  model/snapshots.joblib          - final Elo/form/eg_gold/champion-power/team-comfort
                                   snapshots (as of the latest game date) for predict.py
"""
import os
from collections import defaultdict, deque
from datetime import timedelta

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
PROC_DIR = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "model")

ELO_K = 24
ELO_INIT = 1500.0
YEAR_KEEP = 0.6  # 40% regression toward 1500 each new calendar year
FORM_N = 20
FORM_PSEUDO = 8
BLUE_PSEUDO = 10
CHAMP_WINDOW = timedelta(days=90)
PATCH_PSEUDO = 15  # pseudo-games; prior = previous patch's shrunk value (see champ_power)
COMFORT_WINDOW = timedelta(days=365)
COMFORT_MIN_GAMES = 3
COMFORT_PSEUDO = 10

COLS = ["gameid", "date", "league", "side", "position", "teamname", "champion",
        "result", "ban1", "ban2", "ban3", "ban4", "ban5", "patch",
        "golddiffat15"]


def parse_patch(v):
    """'16.18' -> (16, 18). Returns None for unparseable values."""
    try:
        major, minor = str(v).strip().split(".")[:2]
        return (int(major), int(minor))
    except (ValueError, AttributeError):
        return None


class Trail:
    """Trailing-window win/loss tracker: deque of (datetime, won) + running counts."""

    __slots__ = ("dq", "wins", "total")

    def __init__(self):
        self.dq = deque()
        self.wins = 0
        self.total = 0

    def evict(self, t, window):
        cutoff = t - window
        dq = self.dq
        while dq and dq[0][0] < cutoff:
            _, won = dq.popleft()
            self.total -= 1
            self.wins -= won

    def add(self, t, won):
        self.dq.append((t, won))
        self.total += 1
        self.wins += won


def load_games():
    frames = []
    for year in ("2024", "2025", "2026"):
        path = os.path.join(RAW_DIR, f"{year}.csv")
        # patch must stay a string: "16.10" would otherwise coerce to float 16.1
        df = pd.read_csv(path, usecols=COLS, dtype={"patch": str}, low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "gameid", "side", "teamname"])
    # result must be usable: keep games where both sides have a decided result
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    ok_games = df.dropna(subset=["result"]).groupby("gameid")["side"].nunique()
    ok_games = ok_games[ok_games == 2].index
    df = df[df["gameid"].isin(ok_games)]
    return df


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("loading CSVs...")
    df = load_games()
    print(f"  {len(df)} rows, {df['gameid'].nunique()} games")

    players = df[df["position"] != "team"]
    teams = df[df["position"] == "team"]

    # per-game picks from player rows
    picks = (players.groupby(["gameid", "side"])["champion"]
             .apply(lambda s: [c for c in s if pd.notna(c)]))
    # per-game bans + team names from team rows
    ban_cols = ["ban1", "ban2", "ban3", "ban4", "ban5"]
    bans = (teams.groupby(["gameid", "side"])[ban_cols]
            .first()
            .apply(lambda r: [c for c in r if pd.notna(c)], axis=1))
    side_team = teams.groupby(["gameid", "side"])["teamname"].first()
    side_won = teams.groupby(["gameid", "side"])["result"].first()
    # golddiffat15 is own-side perspective (blue row = blue-red, red row = red-blue;
    # verified exact negatives), so no sign flip is needed per team.
    game_gold = teams.groupby(["gameid", "side"])["golddiffat15"].first()
    game_date = df.groupby("gameid")["date"].first()
    game_league = df.groupby("gameid")["league"].first()
    # one patch per game (mode of row patches; Oracle's data is one patch/game)
    game_patch = (df.dropna(subset=["patch"])
                    .groupby("gameid")["patch"]
                    .agg(lambda s: s.mode().iloc[0]))
    game_patch = game_patch.reindex(game_date.index).fillna("unknown")

    # patch version ordering, e.g. 15.24 < 16.01; used for the prior-patch lookup
    patch_order = sorted(game_patch.unique(),
                         key=lambda p: parse_patch(p) or (0, 0))
    patch_idx = {p: i for i, p in enumerate(patch_order)}
    print(f"  patches: {len(patch_order)} "
          f"({patch_order[0]} .. {patch_order[-1]})")

    game_ids = sorted(game_date.index, key=lambda g: game_date[g])
    print(f"  processing {len(game_ids)} games chronologically...")

    elo = defaultdict(lambda: ELO_INIT)
    form = defaultdict(deque)          # team -> deque(maxlen=FORM_N) of 1/0
    eg = defaultdict(deque)            # team -> deque(maxlen=FORM_N) of own-side gold@15 diffs
    blue_hist = Trail()                # global blue-side results, 90d window
    # (champion, patch) -> [wins, total], games strictly before the current
    # game (updated only after feature computation). No time-window eviction:
    # the patch itself is the window.
    champ_patch = defaultdict(lambda: [0, 0])
    tc_hist = defaultdict(Trail)       # (team, champion) -> 365d results

    def form_rate(team):
        dq = form[team]
        n = len(dq)
        return (sum(dq) + FORM_PSEUDO * 0.5) / (n + FORM_PSEUDO)

    def eg_gold(team):
        """Trailing mean own-side gold diff at 15, shrunk toward 0.

        (sum + 8*0) / (n + 8): same pseudo-count convention as form_rate,
        but the prior is 0 (a margin, not a rate). Null-gold games are never
        appended, so n <= FORM_N counts only games with gold data."""
        dq = eg[team]
        n = len(dq)
        return sum(dq) / (n + FORM_PSEUDO)

    def champ_power(champ, patch):
        """Patch-aware champion strength, strictly walk-forward.

        power = (w_P + a*prior) / (n_P + a),  a = PATCH_PSEUDO (15)
          w_P, n_P : champ wins/games on THIS patch, games strictly before t
          prior    : shrunk value from the previous patch version
                     (w_Q + a*0.5)/(n_Q + a) on games strictly before t,
                     or 0.5 when there is no previous patch.
        Cold start (n_P = 0) -> power = previous patch's value. Large n_P ->
        power -> this patch's win%. No min-games gate; shrinkage handles it.
        """
        w, n = champ_patch[(champ, patch)]
        qi = patch_idx.get(patch, 0) - 1
        if qi >= 0:
            qw, qn = champ_patch[(champ, patch_order[qi])]
            prior = (qw + PATCH_PSEUDO * 0.5) / (qn + PATCH_PSEUDO)
        else:
            prior = 0.5
        return (w + PATCH_PSEUDO * prior) / (n + PATCH_PSEUDO)

    def comfort(team, champ, t, fr):
        tr = tc_hist[(team, champ)]
        tr.evict(t, COMFORT_WINDOW)
        if tr.total >= COMFORT_MIN_GAMES:
            return (tr.wins + COMFORT_PSEUDO * fr) / (tr.total + COMFORT_PSEUDO)
        return fr

    rows = []
    cur_year = None
    for i, gid in enumerate(game_ids):
        t = game_date[gid]
        if cur_year is None:
            cur_year = t.year
        elif t.year != cur_year:
            # offseason roster shuffle: regress all Elos 40% toward 1500
            for team in list(elo.keys()):
                elo[team] = ELO_INIT + YEAR_KEEP * (elo[team] - ELO_INIT)
            cur_year = t.year

        try:
            blue = side_team[(gid, "Blue")]
            red = side_team[(gid, "Red")]
        except KeyError:
            continue
        blue_win = 1 if side_won[(gid, "Blue")] == 1 else 0

        eb, er = elo[blue], elo[red]
        fb, fr = form_rate(blue), form_rate(red)
        egb, egr = eg_gold(blue), eg_gold(red)
        gp = game_patch[gid]

        blue_hist.evict(t, CHAMP_WINDOW)
        blue_edge = ((blue_hist.wins + BLUE_PSEUDO * 0.5)
                     / (blue_hist.total + BLUE_PSEUDO))

        bp = picks.get((gid, "Blue"), [])
        rp = picks.get((gid, "Red"), [])
        if len(bp) == 5 and len(rp) == 5:
            d_power = (sum(champ_power(c, gp) for c in bp)
                       - sum(champ_power(c, gp) for c in rp))
            d_comfort = (sum(comfort(blue, c, t, fb) for c in bp)
                         - sum(comfort(red, c, t, fr) for c in rp))
            bb = bans.get((gid, "Blue"), [])
            rb = bans.get((gid, "Red"), [])
            ban_edge = (sum(champ_power(c, gp) for c in bb)
                        - sum(champ_power(c, gp) for c in rb))
            picks_ok = True
        else:
            d_power = d_comfort = ban_edge = 0.0
            picks_ok = False

        rows.append({
            "gameid": gid, "date": t, "league": game_league[gid],
            "blue": blue, "red": red,
            "elo_diff": eb - er, "eg_gold_diff": egb - egr, "blue_edge": blue_edge,
            "d_power": d_power, "d_comfort": d_comfort, "ban_edge": ban_edge,
            "picks_ok": picks_ok, "y": blue_win,
        })

        # ---- update state with this game (strictly after feature computation) ----
        exp_blue = 1.0 / (1.0 + 10.0 ** (-(eb - er) / 400.0))
        elo[blue] = eb + ELO_K * (blue_win - exp_blue)
        elo[red] = er + ELO_K * ((1 - blue_win) - (1 - exp_blue))
        fb_dq = form[blue]
        fb_dq.append(blue_win)
        if len(fb_dq) > FORM_N:
            fb_dq.popleft()
        fr_dq = form[red]
        fr_dq.append(1 - blue_win)
        if len(fr_dq) > FORM_N:
            fr_dq.popleft()
        # own-side gold@15 diffs (skip games with null gold data)
        gb = game_gold.get((gid, "Blue"), float("nan"))
        gr = game_gold.get((gid, "Red"), float("nan"))
        if gb == gb:  # not NaN
            eg_bq = eg[blue]
            eg_bq.append(float(gb))
            if len(eg_bq) > FORM_N:
                eg_bq.popleft()
        if gr == gr:  # not NaN
            eg_rq = eg[red]
            eg_rq.append(float(gr))
            if len(eg_rq) > FORM_N:
                eg_rq.popleft()
        blue_hist.add(t, blue_win)
        if picks_ok:
            for c in bp:
                rec = champ_patch[(c, gp)]
                rec[0] += blue_win
                rec[1] += 1
                tc_hist[(blue, c)].add(t, blue_win)
            for c in rp:
                rec = champ_patch[(c, gp)]
                rec[0] += 1 - blue_win
                rec[1] += 1
                tc_hist[(red, c)].add(t, 1 - blue_win)

        if (i + 1) % 10000 == 0:
            print(f"  ... {i + 1}/{len(game_ids)}")

    feat = pd.DataFrame(rows)
    feat_path = os.path.join(PROC_DIR, "features.csv")
    feat.to_csv(feat_path, index=False)
    print(f"wrote {feat_path} ({len(feat)} games)")
    print(f"  picks_ok rate: {feat['picks_ok'].mean():.3f}")
    print(f"  blue win rate overall: {feat['y'].mean():.3f}")

    # ---- snapshots for live prediction (as of latest game date) ----
    t_max = feat["date"].max()
    print(f"building snapshots as of {t_max}...")

    # final per-(champion, patch) counts -> shrunk power with prior-patch blend
    # (same formula as champ_power, evaluated on the full data).
    # Includes champions seen on the previous patch even with zero games on
    # this patch: cold-start champs get their prior-patch value instead of
    # vanishing from the table (live prediction must not drop them).
    def final_patch_power(patch):
        qi = patch_idx.get(patch, 0) - 1
        qpatch = patch_order[qi] if qi >= 0 else None
        champs = {c for (c, p) in champ_patch if p == patch}
        if qpatch:
            champs |= {c for (c, p) in champ_patch if p == qpatch}
        out = {}
        for champ in champs:
            w, n = champ_patch.get((champ, patch), (0, 0))
            if qpatch:
                qw, qn = champ_patch.get((champ, qpatch), (0, 0))
                prior = (qw + PATCH_PSEUDO * 0.5) / (qn + PATCH_PSEUDO)
            else:
                prior = 0.5
            out[champ] = (w + PATCH_PSEUDO * prior) / (n + PATCH_PSEUDO)
        return out

    # recent patches: trailing ~6 months of patch versions (enough for the app's
    # patch selector; older tables are stale for live prediction)
    recent_cutoff = t_max - timedelta(days=180)
    patch_last_seen = {}
    for gid2 in game_ids:
        patch_last_seen[game_patch[gid2]] = game_date[gid2]
    recent_patches = [p for p in patch_order
                      if p != "unknown" and patch_last_seen.get(p, t_max) >= recent_cutoff]
    # current (Worlds 2026) patch: highest version with substantial pro games
    patch_game_counts = {p: sum(1 for gid2 in game_ids if game_patch[gid2] == p)
                         for p in patch_order}
    current_patch = max((p for p in patch_order
                         if p != "unknown" and patch_game_counts.get(p, 0) >= 100),
                        key=lambda p: parse_patch(p))
    print(f"  current_patch={current_patch} "
          f"(n={patch_game_counts[current_patch]} games); "
          f"exporting {len(recent_patches)} recent patches")

    champ_power_by_patch = {p: final_patch_power(p) for p in recent_patches}
    champ_power_snap = champ_power_by_patch[current_patch]
    tc_snap = {}
    for (team, champ), tr in tc_hist.items():
        tr.evict(t_max, COMFORT_WINDOW)
        fr = form_rate(team)
        tc_snap[(team, champ)] = (
            (tr.wins + COMFORT_PSEUDO * fr) / (tr.total + COMFORT_PSEUDO)
            if tr.total >= COMFORT_MIN_GAMES else fr)
    blue_hist.evict(t_max, CHAMP_WINDOW)
    snapshots = {
        "as_of": str(t_max),
        "elo": dict(elo),
        "form": {team: form_rate(team) for team in set(list(elo.keys()) + list(form.keys()))},
        "eg_gold": {team: eg_gold(team) for team in set(list(elo.keys()) + list(eg.keys()))},
        "blue_edge": (blue_hist.wins + BLUE_PSEUDO * 0.5) / (blue_hist.total + BLUE_PSEUDO),
        # current-patch table (kept for backward compat: app.py / CLI default)
        "champ_power": champ_power_snap,
        "current_patch": current_patch,
        "recent_patches": recent_patches,
        "champ_power_by_patch": champ_power_by_patch,
        "team_champ_comfort": {f"{team}||{champ}": v for (team, champ), v in tc_snap.items()},
    }
    joblib.dump(snapshots, os.path.join(MODEL_DIR, "snapshots.joblib"))
    print("wrote model/snapshots.joblib")


if __name__ == "__main__":
    main()
