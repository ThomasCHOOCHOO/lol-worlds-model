"""Prop pricing models for the LoL Worlds predictor (v1.0).

Prices single-match props and feeds the Derivatives tab's related-parlay
joint simulator. Every model is fit on games strictly before 2026-07-01 and
validated on games from 2026-07-01 onward. A prop SHIPS only if it beats its
naive baseline out-of-sample by more than the repo's 0.002 log-loss gate
(same bar as the v1.4.0 feature analyses).

Props
-----
(a) Total kills O/U per map: Poisson GLM on an implied-total feature
    (both teams' trailing scoring/conceding pace) + |Elo diff|, with a
    negative-binomial dispersion fitted by moments. Naive baseline: one
    global NB for all games.
(b) Player kills O/U per map: Poisson GLM on the player's shrunk trailing-20
    kill rate + team pace + opponent conceding pace + signed Elo diff +
    position intercepts, NB dispersion. Naive: per-position global NB.
(c) First blood / first dragon / first tower (P(blue gets it)): logistic on
    Elo diff + both teams' trailing first-X rates. Naive: global base rate.
(d) Series exact score + total maps: binomial with fixed per-map p from the
    shipped nodraft model (same series conversion as src/predict.py).
    Naive: empirical exact-score distribution of training series. Total maps
    gets its own p-conditional model (multinomial logistic on logit(p)): the
    binomial maps marginal loses to the empirical distribution, so the
    standalone total-maps prop uses the conditional model, not the binomial.

Joint simulation (related parlays)
----------------------------------
Same-game-parlay legs correlate (a series win and over-kills reinforce each
other). The joint sampler draws, per map: winner ~ Bernoulli(p), then a
(kill bucket, first-blood/dragon/tower combo) cell from an empirical joint
table stratified by (map winner, p-bin, map-index bucket, tied?). Sampling kills
and firsts JOINTLY from one cell preserves the kills<->firsts correlation
inside each stratum; the winner draw preserves the winner<->everything
correlation; the map-index bucket (1, 2, 3, 4+) captures decider maps being
slower (game-5 maps average ~29.5 kills vs ~33.3 for game 2); the tied flag
captures level series being slower (~31.3 kills when tied vs ~33.4 when a
side leads, consistently across p-bins). Sims are RAKED: each sim is
weighted by P_cond(maps | p) / P_binom(maps | p) so the sim's maps marginal
matches the shipped p-conditional maps model instead of the fixed-p binomial
(which overstates long series). The app runs
the identical sampler in TypeScript (src/model/props.ts) — same strata,
same buckets, same cell probabilities, same seeded PRNG family.

Pre-game only: no game-length, no in-game state. All trailing stats are
walk-forward (shifted, strictly-before-t), so a game 2+ map may see earlier
maps of the same series — exactly what a bettor pricing map 2 knows.

Elo convention: prop models use the PLAIN elo_diff from features.csv (no
region offset), so the Python fits and the app's TypeScript price the same
numbers. The region offset moves winner probabilities by up to ~300 Elo for
extreme pairs, but props are driven by pace/objective rates; the offset's
second-order effect is not worth a second plumbing path.

Run:  .venv/bin/python src/prop_models.py
Writes: model/prop_models.json (sidecar for the app) + prints validation.
"""

import json
import math
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, PoissonRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_utils  # noqa: F401  (needed to unpickle model.joblib)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
MODEL_DIR = os.path.join(BASE, "model")
CUT = pd.Timestamp("2026-07-01")
GATE = 0.002  # repo log-loss improvement bar

# Joint-table strata
P_BINS = [0.0, 0.40, 0.47, 0.53, 0.60, 1.0001]
KILL_BUCKETS = [  # [lo, hi] inclusive; last bucket is [40, 60]
    [0, 15], [16, 18], [19, 21], [22, 24], [25, 27],
    [28, 30], [31, 33], [34, 36], [37, 39], [40, 60],
]
# Firsts combo index: bit0 = first blood, bit1 = first dragon, bit2 = first tower
FIRST_COMBOS = 8
SMOOTH = 0.5  # Laplace smoothing per joint-table cell


# ---------------------------------------------------------------- data

def load_oe():
    frames = []
    for year in (2024, 2025, 2026):
        df = pd.read_csv(os.path.join(RAW_DIR, f"{year}.csv"), low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["series_id"] = df["gameid"].astype(str).str.replace(r"_game_\d+$", "", regex=True)
    return df


def build_team_games(df):
    """One row per (game, team) with own/opp kills and first-X flags."""
    team = df[df["position"] == "team"].copy()
    team = team[["gameid", "series_id", "date", "league", "game", "side",
                 "teamname", "result", "teamkills",
                 "firstblood", "firstdragon", "firsttower"]].copy()
    team = team.rename(columns={"game": "game_no", "teamname": "team",
                                "teamkills": "kills"})
    # Opponent kills per game: the other team's kills in the same game.
    opp = team.groupby("gameid")["kills"].sum().rename("total_kills")
    team = team.join(opp, on="gameid")
    team["opp_kills"] = team["total_kills"] - team["kills"]
    # Per-game blue/red mapping for the game-level frame.
    piv = team.pivot_table(index="gameid", columns="side", values="team",
                           aggfunc="first")
    res = team.pivot_table(index="gameid", columns="side", values="result",
                           aggfunc="first")
    team = team.sort_values(["team", "date", "game_no", "gameid"]).reset_index(drop=True)
    return team, piv, res


def add_trailing(team):
    """Walk-forward trailing-20 stats per team (shifted: strictly before t).

    A game-2+ map sees earlier maps of the same series (date, game_no order),
    which is exactly what a bettor pricing that map knows.
    """
    cols = ["kills", "opp_kills", "firstblood", "firstdragon", "firsttower"]
    team = team.sort_values(["team", "date", "game_no", "gameid"]).reset_index(drop=True)
    out = team.copy()
    for c in cols:
        shifted = team.groupby("team")[c].shift(1)
        roll = shifted.groupby(team["team"]).rolling(20, min_periods=1).mean()
        out[f"tr_{c}"] = roll.reset_index(level=0, drop=True)
        cnt = shifted.groupby(team["team"]).rolling(20, min_periods=1).count()
        out[f"tr_{c}_n"] = cnt.reset_index(level=0, drop=True)
    return out


def build_game_frame(team, piv, res, feats):
    """One row per game, blue perspective, with trailing stats for both sides."""
    g = team.drop_duplicates("gameid").copy()
    g["blue"] = g["gameid"].map(piv.get("Blue"))
    g["red"] = g["gameid"].map(piv.get("Red"))
    g["y_blue"] = g["gameid"].map(res.get("Blue"))
    g = g[["gameid", "series_id", "date", "league", "game_no", "blue", "red",
           "y_blue", "total_kills"]].copy()
    # Trailing stats live on the team-game rows; merge each side's row
    # (merges, not per-row .loc: the frame is 60k rows).
    stat_cols = ["kills", "opp_kills", "firstblood", "firstdragon", "firsttower"]
    keep = ["gameid", "team", "side"] + [f"tr_{c}" for c in stat_cols] \
        + [f"tr_{c}_n" for c in stat_cols]
    tt = team[keep]
    for side_name, team_col in (("Blue", "blue"), ("Red", "red")):
        sub = tt[tt["side"] == side_name].drop(columns="side")
        rename = {"team": team_col}
        for c in stat_cols:
            rename[f"tr_{c}"] = f"{team_col}_{c}"
            rename[f"tr_{c}_n"] = f"{team_col}_{c}_n"
        sub = sub.rename(columns=rename)
        g = g.merge(sub, on=["gameid", team_col], how="left")
    # First-X flags from the blue team's row.
    fb = team[team["side"] == "Blue"].set_index("gameid")["firstblood"]
    fd = team[team["side"] == "Blue"].set_index("gameid")["firstdragon"]
    ft = team[team["side"] == "Blue"].set_index("gameid")["firsttower"]
    g["blue_fb"] = g["gameid"].map(fb)
    g["blue_fd"] = g["gameid"].map(fd)
    g["blue_ft"] = g["gameid"].map(ft)
    # Model features: plain elo_diff (no region offset) + eg_gold + blue_edge.
    f = feats[["gameid", "elo_diff", "eg_gold_diff", "blue_edge"]].drop_duplicates("gameid")
    g = g.merge(f, on="gameid", how="left")
    return g


def build_player_games(df, team):
    """One row per (game, player) with shrunk trailing kill rate."""
    p = df[df["position"].isin(["top", "jng", "mid", "bot", "sup"])].copy()
    p = p[["gameid", "date", "game", "position", "playername", "playerid",
           "teamname", "kills"]].copy()
    p = p.rename(columns={"game": "game_no", "teamname": "team"})
    p = p.sort_values(["playerid", "date", "game_no", "gameid"]).reset_index(drop=True)
    # Trailing-20 kills sum + count per player (shifted), shrunk to position mean.
    shifted = p.groupby("playerid")["kills"].shift(1)
    ssum = shifted.groupby(p["playerid"]).rolling(20, min_periods=1).sum()
    scnt = shifted.groupby(p["playerid"]).rolling(20, min_periods=1).count()
    p["tr_kills_sum"] = ssum.reset_index(level=0, drop=True)
    p["tr_kills_n"] = scnt.reset_index(level=0, drop=True)
    pos_mean = p.groupby("position")["kills"].mean().to_dict()
    p["pos_mean"] = p["position"].map(pos_mean)
    p["rate"] = (p["tr_kills_sum"].fillna(0) + 3 * p["pos_mean"]) / (p["tr_kills_n"].fillna(0) + 3)
    # Team trailing pace + opponent conceding pace at this game, from team frame.
    pace = team[["gameid", "team", "tr_kills"]].rename(columns={"tr_kills": "team_pace"})
    p = p.merge(pace, on=["gameid", "team"], how="left")
    return p, pos_mean


# ---------------------------------------------------------------- NB helpers

def nb_params(mu, alpha):
    """Return (n, p) for scipy.stats.nbinom with mean mu, Var = mu + alpha*mu^2."""
    alpha = max(float(alpha), 1e-6)
    n = 1.0 / alpha
    p = n / (n + np.asarray(mu, dtype=float))
    return n, p


def nb_over_prob(mu, alpha, line):
    """P(X > line) for NB(mu, alpha); line like 22.5 -> P(X >= 23)."""
    n, p = nb_params(mu, alpha)
    return float(1.0 - stats.nbinom.cdf(math.floor(line), n, p))


def fit_dispersion(y, mu):
    """NB2 moment estimator for alpha on the training fit."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    num = float(np.mean((y - mu) ** 2) - np.mean(mu))
    den = float(np.mean(mu ** 2))
    return max(num / den, 1e-4) if den > 0 else 0.05


def bin_ll(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------- (a) total kills

def fit_total_kills(g):
    tr = g.dropna(subset=["blue_kills", "red_kills", "blue_opp_kills", "red_opp_kills",
                          "elo_diff", "total_kills"]).copy()
    mu0 = (tr["blue_kills"] + tr["blue_opp_kills"] + tr["red_kills"] + tr["red_opp_kills"]) / 2.0
    X = np.column_stack([np.ones(len(tr)), np.log(np.maximum(mu0, 1.0)),
                         np.abs(tr["elo_diff"].values) / 100.0])
    model = PoissonRegressor(alpha=1.0, fit_intercept=False, max_iter=1000)
    model.fit(X, tr["total_kills"].values)
    mu = model.predict(X)
    alpha = fit_dispersion(tr["total_kills"].values, mu)
    return {"coef": [float(c) for c in model.coef_], "alpha": float(alpha),
            "n_train": int(len(tr))}


def total_kills_mu(params, pace):
    """pace: dict with blue/red kills + opp_kills trailing means + elo_diff."""
    mu0 = (pace["blue_kills"] + pace["blue_opp_kills"] + pace["red_kills"]
           + pace["red_opp_kills"]) / 2.0
    c = params["coef"]
    return float(np.exp(c[0] + c[1] * math.log(max(mu0, 1.0))
                        + c[2] * abs(pace["elo_diff"]) / 100.0))


def total_kills_over(params, pace, line):
    return nb_over_prob(total_kills_mu(params, pace), params["alpha"], line)


# ---------------------------------------------------------------- (b) player kills

POSITIONS = ["top", "jng", "mid", "bot", "sup"]


def fit_player_kills(pg, g):
    # Opponent's conceding pace + signed Elo diff from the player's side.
    gmap = g.set_index("gameid")
    rows = []
    for r in pg.itertuples(index=False):
        try:
            grow = gmap.loc[r.gameid]
        except KeyError:
            continue
        if r.team == grow.blue:
            opp_pace = grow.red_opp_kills
            elo_signed = grow.elo_diff
        elif r.team == grow.red:
            opp_pace = grow.blue_opp_kills
            elo_signed = -grow.elo_diff
        else:
            continue
        rows.append((r.date, r.rate, r.team_pace, opp_pace, elo_signed, r.position, r.kills))
    d = pd.DataFrame(rows, columns=["date", "rate", "team_pace", "opp_pace", "elo", "pos", "kills"])
    d = d.dropna()
    pos_means = {pos: float(d[d["pos"] == pos]["kills"].mean()) for pos in POSITIONS}

    def design(dd):
        return np.column_stack(
            [np.ones(len(dd)),
             np.log(np.maximum(dd["rate"].values, 0.05)),
             dd["team_pace"].values, dd["opp_pace"].values, dd["elo"].values / 100.0]
            + [(dd["pos"].values == pos).astype(float) for pos in POSITIONS])

    def raw_mu(model, dd):
        return model.predict(design(dd))

    # Shrinkage: the raw GLM's mu is over-dispersed OOS (player trailing
    # rates are noisy, so raw mu spreads too far from the position mean).
    # Estimate the shrink factor s (mu = pos_mean + s*(mu_raw - pos_mean))
    # on a late-train tuning slice ONLY — never on the validation period.
    tune_cut = pd.Timestamp("2026-01-01")
    fit_d = d[d["date"] < tune_cut]
    tune_d = d[d["date"] >= tune_cut]
    tune_model = PoissonRegressor(alpha=1.0, fit_intercept=False, max_iter=1000)
    tune_model.fit(design(fit_d), fit_d["kills"].values)
    tune_mu = raw_mu(tune_model, tune_d)
    tune_alpha = fit_dispersion(fit_d["kills"].values, raw_mu(tune_model, fit_d))
    tune_pos = tune_d["pos"].map(pos_means).values
    best_s, best_ll = 1.0, -np.inf
    for s in np.arange(0.0, 1.01, 0.05):
        mu_s = tune_pos + s * (tune_mu - tune_pos)
        ll = _ou_ll(mu_s, tune_alpha, tune_d["kills"].values)
        if ll > best_ll:
            best_ll, best_s = ll, float(s)
    model = PoissonRegressor(alpha=1.0, fit_intercept=False, max_iter=1000)
    model.fit(design(d), d["kills"].values)
    mu_raw = raw_mu(model, d)
    mu = d["pos"].map(pos_means).values + best_s * (mu_raw - d["pos"].map(pos_means).values)
    alpha = fit_dispersion(d["kills"].values, mu)
    return {"coef": [float(c) for c in model.coef_], "alpha": float(alpha),
            "n_train": int(len(d)), "positions": POSITIONS,
            "shrink": best_s, "pos_means": pos_means,
            "tune": {"cut": "2026-01-01", "n_tune": int(len(tune_d)),
                     "ou_ll_at_best_s": best_ll}}


def _ou_ll(mus, alpha, ys, lines=(1.5, 2.5, 3.5, 4.5, 5.5)):
    ps = []
    for mu, y in zip(np.asarray(mus, dtype=float), np.asarray(ys, dtype=float)):
        for line in lines:
            ps.append((y > line, nb_over_prob(float(mu), float(alpha), line)))
    dd = pd.DataFrame(ps, columns=["y", "p"])
    return bin_ll(dd["y"].values, dd["p"].values)


def player_kills_mu(params, rate, team_pace, opp_pace, elo_signed, position):
    c = params["coef"]
    pos_idx = params["positions"].index(position)
    z = (c[0] + c[1] * math.log(max(rate, 0.05)) + c[2] * team_pace
         + c[3] * opp_pace + c[4] * elo_signed / 100.0 + c[5 + pos_idx])
    mu_raw = float(np.exp(z))
    # Shrinkage toward the position mean (estimated on late-train only).
    s = params.get("shrink", 1.0)
    pos_mean = params.get("pos_means", {}).get(position, mu_raw)
    return float(pos_mean + s * (mu_raw - pos_mean))


def player_kills_over(params, rate, team_pace, opp_pace, elo_signed, position, line):
    mu = player_kills_mu(params, rate, team_pace, opp_pace, elo_signed, position)
    return nb_over_prob(mu, params["alpha"], line)


# ---------------------------------------------------------------- (c) first-X

def fit_first_x(g, col):
    """col: blue_fb / blue_fd / blue_ft. Features: elo_diff, blue rate, red rate."""
    rate_col = {"blue_fb": "firstblood", "blue_fd": "firstdragon", "blue_ft": "firsttower"}[col]
    d = g.dropna(subset=[col, "elo_diff", f"blue_{rate_col}", f"red_{rate_col}"]).copy()
    X = np.column_stack([d["elo_diff"].values / 100.0,
                         d[f"blue_{rate_col}"].values, d[f"red_{rate_col}"].values])
    y = d[col].values.astype(int)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    return {"coef": [float(c) for c in model.coef_[0]],
            "intercept": float(model.intercept_[0]),
            "base_rate": float(y.mean()), "n_train": int(len(d))}


def first_x_prob(params, elo_diff, own_rate, opp_rate):
    c = params["coef"]
    z = params["intercept"] + c[0] * elo_diff / 100.0 + c[1] * own_rate + c[2] * opp_rate
    return float(1.0 / (1.0 + math.exp(-z)))


# ---------------------------------------------------------------- (d) series

def series_exact_probs(p):
    """Exact-score distribution for a Bo5, fixed per-map p (P(A wins a map))."""
    q = 1 - p
    return {
        "3-0": p ** 3, "3-1": 3 * p ** 3 * q, "3-2": 6 * p ** 3 * q ** 2,
        "2-3": 6 * p ** 2 * q ** 3, "1-3": 3 * p * q ** 3, "0-3": q ** 3,
    }


def series_total_maps_probs(p):
    e = series_exact_probs(p)
    return {"3": e["3-0"] + e["0-3"], "4": e["3-1"] + e["1-3"], "5": e["3-2"] + e["2-3"]}


def series_probs_bo(p, best_of):
    """Exact-score + total-maps distributions for Bo1/Bo3/Bo5, fixed per-map p."""
    q = 1 - p
    if best_of == 1:
        return {"exact": {"1-0": p, "0-1": q}, "maps": {"1": 1.0}}
    if best_of == 3:
        e = {"2-0": p ** 2, "2-1": 2 * p ** 2 * q,
             "1-2": 2 * p * q ** 2, "0-2": q ** 2}
        return {"exact": e, "maps": {"2": e["2-0"] + e["0-2"], "3": e["2-1"] + e["1-2"]}}
    e = series_exact_probs(p)
    return {"exact": e, "maps": series_total_maps_probs(p)}


def _logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_maps_model(s_train):
    """P(total maps | per-map p): the binomial maps marginal loses to the
    empirical distribution (favorites stomp more than fixed-p binomial says),
    so fit it directly: multinomial logistic for Bo5, binary for Bo3."""
    out = {}
    t5 = s_train[(s_train["best_of"] == 5)].dropna(subset=["p_game1"])
    X5 = np.column_stack([[ _logit(p) for p in t5["p_game1"]],
                          [abs(_logit(p)) for p in t5["p_game1"]]])
    m5 = LogisticRegression(C=1.0, max_iter=1000)
    m5.fit(X5, t5["n_maps"].values)
    out["bo5"] = {"classes": [int(c) for c in m5.classes_],
                  "coef": [[float(c) for c in row] for row in m5.coef_],
                  "intercept": [float(c) for c in m5.intercept_],
                  "n_train": int(len(t5))}
    t3 = s_train[(s_train["best_of"] == 3)].dropna(subset=["p_game1"])
    X3 = np.column_stack([[ _logit(p) for p in t3["p_game1"]],
                          [abs(_logit(p)) for p in t3["p_game1"]]])
    m3 = LogisticRegression(C=1.0, max_iter=1000)
    m3.fit(X3, (t3["n_maps"] == 3).astype(int).values)
    out["bo3"] = {"coef": [float(c) for c in m3.coef_[0]],
                  "intercept": float(m3.intercept_[0]),
                  "n_train": int(len(t3))}
    return out


def maps_model_probs(params, p, best_of):
    """P(total maps) from the fitted conditional model (NOT the binomial)."""
    if best_of <= 1:
        return {"1": 1.0}
    z = [_logit(p), abs(_logit(p))]
    if best_of == 3:
        c = params["bo3"]["coef"]
        zlin = params["bo3"]["intercept"] + c[0] * z[0] + c[1] * z[1]
        p3 = 1.0 / (1.0 + math.exp(-zlin))
        return {"2": 1.0 - p3, "3": p3}
    logits = []
    for row, b in zip(params["bo5"]["coef"], params["bo5"]["intercept"]):
        logits.append(b + row[0] * z[0] + row[1] * z[1])
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    tot = sum(exps)
    return {str(c): e / tot for c, e in zip(params["bo5"]["classes"], exps)}


def validate_maps_model(params, s_valid, s_train):
    """Multiclass LL of the conditional maps model vs empirical naive."""
    ll_m, ll_n = [], []
    for best_of, key in ((5, "n_maps"), (3, "n_maps")):
        v = s_valid[s_valid["best_of"] == best_of].dropna(subset=["p_game1"])
        t = s_train[s_train["best_of"] == best_of]
        naive = {str(mm): float((t["n_maps"] == mm).mean())
                 for mm in sorted(t["n_maps"].unique())}
        for r in v.itertuples(index=False):
            pr = maps_model_probs(params, float(r.p_game1), best_of)
            ll_m.append(math.log(max(pr[str(r.n_maps)], 1e-9)))
            ll_n.append(math.log(max(naive[str(r.n_maps)], 1e-9)))
    return {"n": len(ll_m), "ll": float(np.mean(ll_m)) if ll_m else float("nan"),
            "naive_ll": float(np.mean(ll_n)) if ll_n else float("nan")}

# ---------------------------------------------------------------- joint sim

def pbin_index(p):
    for i in range(len(P_BINS) - 1):
        if P_BINS[i] <= p < P_BINS[i + 1]:
            return i
    return len(P_BINS) - 2


def kill_bucket_idx(kills):
    for i, (lo, hi) in enumerate(KILL_BUCKETS):
        if lo <= kills <= hi:
            return i
    return len(KILL_BUCKETS) - 1


def add_score_before(g):
    """Score state before each map: A := game-1 blue team. tied_before = 1 if
    the series is level going into this map. Tied maps are slower (~31.3
    kills vs ~33.4 when a side leads, consistently across p-bins), so the
    joint table conditions on it."""
    g = g.sort_values(["league", "blue", "red", "date", "game_no"]).copy()
    g["pair"] = [tuple(sorted((b, r))) for b, r in zip(g["blue"], g["red"])]
    g["day"] = g["date"].dt.date
    g["skey"] = [f"{l}|{' vs '.join(p)}|{d}"
                 for l, p, d in zip(g["league"], g["pair"], g["day"])]
    g = g.sort_values(["skey", "game_no"])
    g["a_team"] = g.groupby("skey")["blue"].transform("first")
    g["winner_team"] = np.where(g["y_blue"] == 1, g["blue"], g["red"])
    g["a_won"] = (g["winner_team"] == g["a_team"]).astype(int)
    g["b_won"] = 1 - g["a_won"]
    g["a_before"] = g.groupby("skey")["a_won"].cumsum() - g["a_won"]
    g["b_before"] = g.groupby("skey")["b_won"].cumsum() - g["b_won"]
    g["tied_before"] = (g["a_before"] == g["b_before"]).astype(int)
    return g


def build_joint_table(maps):
    """maps: game frame with p_map (bundle nodraft P(blue wins)), total_kills,
    blue firsts, game_no, tied_before. A := blue.
    Returns table[winner][pbin][game_bucket][tied] = 80-cell probs, where
    game_bucket = min(game_no, 4) - 1 (game 4+ pooled: decider maps are
    slower — game-5 maps average ~29.5 kills vs ~33.3 for game 2 — so a
    game-number-blind table overestimates kills in long series), and tied =
    series level going into the map (tied maps average ~31.3 kills vs ~33.4
    when a side leads)."""
    table = {}
    for winner in ("A", "B"):
        table[winner] = []
        for _b in range(len(P_BINS) - 1):
            table[winner].append(
                [[np.full(len(KILL_BUCKETS) * FIRST_COMBOS, SMOOTH) for _ in range(2)]
                 for _ in range(4)])
    for r in maps.itertuples(index=False):
        if pd.isna(r.total_kills) or pd.isna(r.blue_fb) or pd.isna(r.blue_fd) or pd.isna(r.blue_ft):
            continue
        winner = "A" if r.y_blue == 1 else "B"
        b = pbin_index(r.p_map)
        gb = min(int(r.game_no), 4) - 1
        tb = int(r.tied_before)
        kb = kill_bucket_idx(int(r.total_kills))
        combo = (int(r.blue_fb) | (int(r.blue_fd) << 1) | (int(r.blue_ft) << 2))
        table[winner][b][gb][tb][kb * FIRST_COMBOS + combo] += 1.0
    for winner in ("A", "B"):
        for b in range(len(P_BINS) - 1):
            for gb in range(4):
                for tb in range(2):
                    arr = table[winner][b][gb][tb]
                    table[winner][b][gb][tb] = (arr / arr.sum()).tolist()
    return table


def _sample_cell(rng, probs):
    return int(rng.choice(len(probs), p=np.asarray(probs, dtype=float)))


def simulate_series(rng, p, best_of, table):
    """One simulated series. Returns dict with winner, maps, kills, map1 firsts."""
    need = best_of // 2 + 1
    wins = {"A": 0, "B": 0}
    total_kills = 0
    maps = 0
    map1_firsts = None
    b = pbin_index(p)
    while wins["A"] < need and wins["B"] < need:
        w = "A" if rng.random() < p else "B"
        gb = min(maps + 1, 4) - 1  # map index: decider maps are slower
        tb = 1 if wins["A"] == wins["B"] else 0  # series level going in?
        cell = _sample_cell(rng, table[w][b][gb][tb])
        kb, combo = divmod(cell, FIRST_COMBOS)
        lo, hi = KILL_BUCKETS[kb]
        kills = int(rng.integers(lo, hi + 1))
        if maps == 0:
            map1_firsts = {"fb": combo & 1, "fd": (combo >> 1) & 1, "ft": (combo >> 2) & 1}
        wins[w] += 1
        total_kills += kills
        maps += 1
    return {"winner": "A" if wins["A"] == need else "B",
            "a_wins": wins["A"], "b_wins": wins["B"],
            "maps": maps, "kills": total_kills, "map1": map1_firsts}


# ---------------------------------------------------------------- series set

def reconstruct_series(g):
    """Completed series, grouped by (league, team pair, calendar day).

    NOTE: Oracle's Elixir gameids do NOT identify a series. Old-format ids
    (``10660-10660_game_1``) share a prefix per series, but new-format ids
    (``LOLTMNT02_444480``) are per-GAME ids — two maps of one series have
    unrelated prefixes. Same pair + same league + same day is the reliable
    series key (a pair never plays two separate series in one day in pro
    leagues). A := the game-1 blue team; a_wins counts A's wins (sides swap
    within a series, so counting blue wins would be wrong).
    """
    out = []
    g = g.copy()
    g["pair"] = [tuple(sorted((b, r))) for b, r in zip(g["blue"], g["red"])]
    g["day"] = g["date"].dt.date
    for (_, _pair, _day), grp in g.groupby(["league", "pair", "day"]):
        grp = grp.sort_values("date")
        nums = grp["game_no"].tolist()
        if nums != list(range(1, len(nums) + 1)):
            continue
        n = len(nums)
        first = grp.iloc[0]
        a_team = first["blue"]
        winner = [row["blue"] if row["y_blue"] == 1 else row["red"]
                  for _, row in grp.iterrows()]
        a_wins = int(sum(1 for w in winner if w == a_team))
        b_wins = n - a_wins
        if n == 1:
            best_of, need = 1, 1
        elif n == 2:
            best_of, need = 3, 2
        elif n == 3:
            best_of, need = (5, 3) if a_wins == 3 or a_wins == 0 else (3, 2)
        elif n in (4, 5):
            best_of, need = 5, 3
        else:
            continue
        if max(a_wins, b_wins) < need:
            continue
        # The winner must have reached `need` exactly at the last game.
        last_a = winner[-1] == a_team
        prior_a = a_wins - (1 if last_a else 0)
        prior_b = b_wins - (0 if last_a else 1)
        if max(prior_a, prior_b) >= need:
            continue
        out.append({"series_id": f"{first['league']}|{' vs '.join(_pair)}|{_day}",
                    "start": grp["date"].min(), "best_of": best_of,
                    "a": a_team, "b": first["red"], "a_wins": a_wins,
                    "b_wins": b_wins, "n_maps": n,
                    "p_game1": first["p_map"] if "p_map" in first else np.nan,
                    "kills": int(grp["total_kills"].sum()),
                    "map1_fb": grp.iloc[0]["blue_fb"], "map1_fd": grp.iloc[0]["blue_fd"],
                    "map1_ft": grp.iloc[0]["blue_ft"]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- validation

def validate_total_kills(params, g_valid):
    lines = [18.5, 20.5, 22.5, 24.5, 26.5]
    d = g_valid.dropna(subset=["blue_kills", "red_kills", "blue_opp_kills",
                               "red_opp_kills", "elo_diff", "total_kills"])
    rows = []
    for r in d.itertuples(index=False):
        pace = {"blue_kills": r.blue_kills, "blue_opp_kills": r.blue_opp_kills,
                "red_kills": r.red_kills, "red_opp_kills": r.red_opp_kills,
                "elo_diff": r.elo_diff}
        for line in lines:
            rows.append((r.total_kills > line, total_kills_over(params, pace, line)))
    d = pd.DataFrame(rows, columns=["y", "p"])
    return bin_ll(d["y"].values, d["p"].values)


def naive_total_kills_ll(g_train, g_valid):
    mu = float(g_train["total_kills"].mean())
    alpha = fit_dispersion(g_train["total_kills"].values, np.full(len(g_train), mu))
    lines = [18.5, 20.5, 22.5, 24.5, 26.5]
    rows = []
    for tot in g_valid["total_kills"].values:
        for line in lines:
            rows.append((tot > line, nb_over_prob(mu, alpha, line)))
    d = pd.DataFrame(rows, columns=["y", "p"])
    return bin_ll(d["y"].values, d["p"].values)


def validate_player_kills(params, pg_valid, g_valid):
    gmap = g_valid.set_index("gameid")
    lines = [1.5, 2.5, 3.5, 4.5, 5.5]
    rows = []
    for r in pg_valid.itertuples(index=False):
        try:
            grow = gmap.loc[r.gameid]
        except KeyError:
            continue
        if r.team == grow.blue:
            opp_pace, elo = grow.red_opp_kills, grow.elo_diff
        elif r.team == grow.red:
            opp_pace, elo = grow.blue_opp_kills, -grow.elo_diff
        else:
            continue
        if pd.isna(opp_pace) or pd.isna(r.team_pace):
            continue
        for line in lines:
            rows.append((r.kills > line,
                         nb_over_prob(player_kills_mu(params, r.rate, r.team_pace,
                                                      opp_pace, elo, r.position),
                                      params["alpha"], line)))
    d = pd.DataFrame(rows, columns=["y", "p"])
    return bin_ll(d["y"].values, d["p"].values)


def naive_player_kills_ll(pg_train, pg_valid):
    # Per-position global NB.
    stats_by_pos = {}
    for pos, grp in pg_train.groupby("position"):
        mu = float(grp["kills"].mean())
        stats_by_pos[pos] = (mu, fit_dispersion(grp["kills"].values, np.full(len(grp), mu)))
    lines = [1.5, 2.5, 3.5, 4.5, 5.5]
    rows = []
    for r in pg_valid.itertuples(index=False):
        mu, alpha = stats_by_pos[r.position]
        for line in lines:
            rows.append((r.kills > line, nb_over_prob(mu, alpha, line)))
    d = pd.DataFrame(rows, columns=["y", "p"])
    return bin_ll(d["y"].values, d["p"].values)


def validate_first_x(params, g_valid, col):
    rate_col = {"blue_fb": "firstblood", "blue_fd": "firstdragon", "blue_ft": "firsttower"}[col]
    d = g_valid.dropna(subset=[col, "elo_diff", f"blue_{rate_col}", f"red_{rate_col}"])
    p = [first_x_prob(params, r.elo_diff, getattr(r, f"blue_{rate_col}"),
                      getattr(r, f"red_{rate_col}")) for r in d.itertuples(index=False)]
    return bin_ll(d[col].values.astype(int), np.array(p))


def validate_series(series_valid, series_train):
    v = series_valid[series_valid["best_of"] == 5].dropna(subset=["p_game1"])
    # Naive: empirical exact-score distribution of training Bo5 series.
    t = series_train[series_train["best_of"] == 5]
    keys = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
    naive = {}
    for k in keys:
        a, b = map(int, k.split("-"))
        naive[k] = ((t["a_wins"] == a) & (t["b_wins"] == b)).mean()
    ll_model, ll_naive, ll_maps_model, ll_maps_naive = [], [], [], []
    for r in v.itertuples(index=False):
        e = series_exact_probs(r.p_game1)
        m = series_total_maps_probs(r.p_game1)
        k = f"{r.a_wins}-{r.b_wins}"
        ll_model.append(math.log(max(e[k], 1e-9)))
        ll_naive.append(math.log(max(naive[k], 1e-9)))
        mk = str(r.n_maps)
        naive_maps = {str(mm): ((t["n_maps"] == mm).mean()) for mm in (3, 4, 5)}
        ll_maps_model.append(math.log(max(m[mk], 1e-9)))
        ll_maps_naive.append(math.log(max(naive_maps[mk], 1e-9)))
    return {"n": int(len(v)), "exact_ll": float(np.mean(ll_model)),
            "exact_naive_ll": float(np.mean(ll_naive)),
            "maps_ll": float(np.mean(ll_maps_model)),
            "maps_naive_ll": float(np.mean(ll_maps_naive))}


def validate_joint(table, series_valid, maps_params, n_sims=1000, seed=7):
    """Joint-sim forecast quality vs the naive independent product.

    For each validation Bo5 series: run the sim, score the sim's joint
    probability for 4 correlated combos as a probabilistic forecast of the
    actual joint outcome (binary LL), and do the same for the naive product
    of the sim's own marginals.

    Sims are raked: each sim is weighted by
    P_cond(maps | p) / P_binom(maps | p) so the sim's maps marginal matches
    the shipped p-conditional maps model (same raking the app applies).
    """
    v = series_valid[series_valid["best_of"] == 5].dropna(subset=["p_game1"]).copy()
    rng = np.random.default_rng(seed)
    rows = []
    for r in v.itertuples(index=False):
        p = float(r.p_game1)
        sims = [simulate_series(rng, p, 5, table) for _ in range(n_sims)]
        cond = maps_model_probs(maps_params, p, 5)
        binom = series_probs_bo(p, 5)["maps"]
        w = np.array([cond[str(s["maps"])] / max(binom[str(s["maps"])], 1e-9)
                      for s in sims])
        w = w / w.sum()
        kills = np.array([s["kills"] for s in sims])
        # Raked median kills line.
        order = np.argsort(kills)
        cumw = np.cumsum(w[order])
        line = math.floor(float(kills[order[np.searchsorted(cumw, 0.5)]])) + 0.5
        a_win = np.array([s["winner"] == "A" for s in sims])
        over_k = kills > line
        over_m = np.array([s["maps"] > 3.5 for s in sims])
        a_fd = np.array([s["map1"]["fd"] == 1 for s in sims])
        actual = {
            "A_win+overK": (r.a_wins == 3) and (r.kills > line),
            "A_win+overM": (r.a_wins == 3) and (r.n_maps > 3.5),
            "A_win+A_fd": (r.a_wins == 3) and (r.map1_fd == 1),
            "overM+overK": (r.n_maps > 3.5) and (r.kills > line),
        }
        wmean = lambda mask: float(np.sum(w[mask]))
        joint = {
            "A_win+overK": wmean(a_win & over_k),
            "A_win+overM": wmean(a_win & over_m),
            "A_win+A_fd": wmean(a_win & a_fd),
            "overM+overK": wmean(over_m & over_k),
        }
        marg = {"A_win": wmean(a_win), "overK": wmean(over_k),
                "overM": wmean(over_m), "A_fd": wmean(a_fd)}
        naive = {
            "A_win+overK": marg["A_win"] * marg["overK"],
            "A_win+overM": marg["A_win"] * marg["overM"],
            "A_win+A_fd": marg["A_win"] * marg["A_fd"],
            "overM+overK": marg["overM"] * marg["overK"],
        }
        for k in actual:
            rows.append({"combo": k, "actual": int(actual[k]),
                         "joint": joint[k], "naive": naive[k]})
    d = pd.DataFrame(rows)
    out = {"n_series": int(len(v)), "n_sims": n_sims, "combos": {}}
    for k, grp in d.groupby("combo"):
        out["combos"][k] = {
            "joint_ll": bin_ll(grp["actual"].values, grp["joint"].values),
            "naive_ll": bin_ll(grp["actual"].values, grp["naive"].values),
            "mean_joint": float(grp["joint"].mean()),
            "actual_rate": float(grp["actual"].mean()),
        }
    out["joint_ll"] = float(np.mean([c["joint_ll"] for c in out["combos"].values()]))
    out["naive_ll"] = float(np.mean([c["naive_ll"] for c in out["combos"].values()]))
    return out


# ---------------------------------------------------------------- export

def current_team_stats(team):
    """Trailing-20 pace/objective rates as of the latest game per team.

    Only teams with a game in 2026 (recent enough for pace to mean anything).
    """
    out = {}
    for tm, grp in team.groupby("team"):
        grp = grp.sort_values(["date", "game_no", "gameid"]).tail(20)
        if len(grp) < 3 or grp["date"].max() < pd.Timestamp("2026-01-01"):
            continue
        out[tm] = {
            "pace_for": round(float(grp["kills"].mean()), 3),
            "pace_against": round(float(grp["opp_kills"].mean()), 3),
            "fb": round(float(grp["firstblood"].mean()), 3),
            "fd": round(float(grp["firstdragon"].mean()), 3),
            "ft": round(float(grp["firsttower"].mean()), 3),
            "games": int(len(grp)),
            "last_date": str(grp["date"].max().date()),
        }
    return out


def current_player_stats(p, pos_mean):
    """Players with a game since 2026-09-01: shrunk trailing-20 kill rate."""
    recent = p[p["date"] >= pd.Timestamp("2026-09-01")].copy()
    out = {}
    for (pid, name), grp in recent.groupby(["playerid", "playername"]):
        grp = grp.sort_values(["date", "game_no", "gameid"])
        last = grp.iloc[-1]
        tail = grp.tail(20)
        if len(tail) < 3:
            continue
        s, n = float(tail["kills"].sum()), len(tail)
        rate = (s + 3 * pos_mean[last["position"]]) / (n + 3)
        out[str(pid)] = {"name": str(name), "team": str(last["team"]),
                         "position": str(last["position"]),
                         "rate": round(float(rate), 3), "games": int(n)}
    return out


def r4(x):
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r4(v) for v in x]
    if isinstance(x, float):
        return round(x, 4)
    return x


def main():
    print("loading Oracle's Elixir data ...")
    df = load_oe()
    team, piv, res = build_team_games(df)
    team = add_trailing(team)
    feats = pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"),
                        usecols=["gameid", "elo_diff", "eg_gold_diff", "blue_edge"])
    g = build_game_frame(team, piv, res, feats)
    g = g.dropna(subset=["elo_diff", "blue", "red"])
    print(f"games: {len(g)} ({g['date'].min().date()} .. {g['date'].max().date()})")

    # Score state before each map (for the joint table's tied dimension).
    g = add_score_before(g)

    # Per-map p from the shipped nodraft model (calibrated), for series + joint.
    bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    X = bundle["scaler_nodraft"].transform(
        g[["elo_diff", "eg_gold_diff", "blue_edge"]].values)
    # NOTE: bundle was trained on elo_diff_adj; features.csv carries the plain
    # elo_diff here, so this p is the app-equivalent (unadjusted) one.
    g["p_map"] = bundle["model_nodraft"].predict_proba(X)[:, 1]

    pg, pos_mean = build_player_games(df, team)

    g_train = g[g["date"] < CUT].copy()
    g_valid = g[g["date"] >= CUT].copy()
    pg_train = pg[pg["date"] < CUT].copy()
    pg_valid = pg[pg["date"] >= CUT].copy()
    print(f"train games: {len(g_train)}, valid games: {len(g_valid)}")

    # (a) total kills
    tk = fit_total_kills(g_train)
    tk_ll = validate_total_kills(tk, g_valid)
    tk_naive = naive_total_kills_ll(g_train, g_valid)
    tk["validation"] = {"n": int(len(g_valid)), "ll": tk_ll, "naive_ll": tk_naive,
                        "delta": tk_ll - tk_naive}
    tk["shipped"] = (tk_ll - tk_naive) > GATE
    print(f"(a) total kills: LL {tk_ll:.4f} vs naive {tk_naive:.4f} "
          f"(delta {tk_ll - tk_naive:+.4f}) shipped={tk['shipped']}")

    # (b) player kills
    pk = fit_player_kills(pg_train, g_train)
    pk_ll = validate_player_kills(pk, pg_valid, g_valid)
    pk_naive = naive_player_kills_ll(pg_train, pg_valid)
    pk["validation"] = {"n": int(len(pg_valid)), "ll": pk_ll, "naive_ll": pk_naive,
                        "delta": pk_ll - pk_naive}
    pk["shipped"] = (pk_ll - pk_naive) > GATE
    print(f"(b) player kills: LL {pk_ll:.4f} vs naive {pk_naive:.4f} "
          f"(delta {pk_ll - pk_naive:+.4f}) shipped={pk['shipped']}")

    # (c) first-X
    fx = {}
    for col, key in (("blue_fb", "blood"), ("blue_fd", "dragon"), ("blue_ft", "tower")):
        m = fit_first_x(g_train, col)
        ll = validate_first_x(m, g_valid, col)
        d = g_valid.dropna(subset=[col])
        naive = bin_ll(d[col].values.astype(int),
                       np.full(len(d), m["base_rate"]))
        m["validation"] = {"n": int(len(d)), "ll": ll, "naive_ll": naive,
                           "delta": ll - naive}
        m["shipped"] = (ll - naive) > GATE
        fx[key] = m
        print(f"(c) first {key}: LL {ll:.4f} vs base-rate {naive:.4f} "
              f"(delta {ll - naive:+.4f}) shipped={m['shipped']}")

    # (d) series
    series = reconstruct_series(g)
    s_train = series[series["start"] < CUT]
    s_valid = series[series["start"] >= CUT]
    s_val = validate_series(s_valid, s_train)
    s_val["shipped"] = (s_val["exact_ll"] - s_val["exact_naive_ll"]) > GATE
    print(f"(d) series exact: LL {s_val['exact_ll']:.4f} vs naive "
          f"{s_val['exact_naive_ll']:.4f} (delta "
          f"{s_val['exact_ll'] - s_val['exact_naive_ll']:+.4f}) n={s_val['n']} "
          f"shipped={s_val['shipped']}; total-maps LL {s_val['maps_ll']:.4f} vs "
          f"{s_val['maps_naive_ll']:.4f}")

    # (d2) total maps, conditional on p (the binomial maps marginal fails the gate)
    maps_model = fit_maps_model(s_train)
    maps_val = validate_maps_model(maps_model, s_valid, s_train)
    maps_model["validation"] = r4({**maps_val, "delta": maps_val["ll"] - maps_val["naive_ll"]})
    maps_model["shipped"] = (maps_val["ll"] - maps_val["naive_ll"]) > GATE
    print(f"(d2) total maps | p: LL {maps_val['ll']:.4f} vs naive "
          f"{maps_val['naive_ll']:.4f} (delta {maps_val['ll'] - maps_val['naive_ll']:+.4f}) "
          f"n={maps_val['n']} shipped={maps_model['shipped']}")

    # Joint simulation (validated on the train-fit table; the served table
    # is refit on all games through the snapshot — refit-after-validation).
    joint = build_joint_table(g_train)
    j_val = validate_joint(joint, s_valid, maps_model)
    print(f"joint sim: LL {j_val['joint_ll']:.4f} vs naive-product "
          f"{j_val['naive_ll']:.4f} (delta {j_val['joint_ll'] - j_val['naive_ll']:+.4f}) "
          f"n_series={j_val['n_series']}")
    for k, c in j_val["combos"].items():
        print(f"  {k}: joint LL {c['joint_ll']:.4f} vs naive {c['naive_ll']:.4f}; "
              f"mean joint P {c['mean_joint']:.3f} vs actual {c['actual_rate']:.3f}")

    joint_served = build_joint_table(g)
    served_table = {}
    for w, pbs in joint_served.items():
        served_table[w] = [[[[round(x, 4) for x in cells] for cells in tbs]
                            for tbs in gbs] for gbs in pbs]
    sidecar = {
        "version": "1.0.0",
        "fitted_on": "2024-01-01..2026-06-30",
        "validated_on": "2026-07-01..2026-10-16",
        "gate": GATE,
        "total_kills": r4(tk),
        "player_kills": r4(pk),
        "first_x": r4(fx),
        "series": r4({"method": "binomial_fixed_p", "validation": s_val,
                      "shipped": s_val["shipped"], "maps_model": maps_model}),
        "joint": {"p_bins": P_BINS, "kill_buckets": KILL_BUCKETS,
                  "table": served_table,
                  "validation": r4(j_val),
                  "note": ("Validation used the train-fit table; served table refit on all "
                           "games through the snapshot.")},
        "teams": current_team_stats(team),
        "players": current_player_stats(pg, pos_mean),
    }
    out_path = os.path.join(MODEL_DIR, "prop_models.json")
    with open(out_path, "w") as f:
        json.dump(sidecar, f)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB; "
          f"{len(sidecar['teams'])} teams, {len(sidecar['players'])} players)")

    # Worked parity example (fixed matchup) for the TS port to match.
    ex_pace = {"blue_kills": 13.0, "blue_opp_kills": 12.0, "red_kills": 12.5,
               "red_opp_kills": 13.2, "elo_diff": 100.0}
    print("parity example: total_kills mu = "
          f"{total_kills_mu(tk, ex_pace):.4f}, P(over 22.5) = "
          f"{total_kills_over(tk, ex_pace, 22.5):.4f}")
    print("parity example: first dragon P(blue) = "
          f"{first_x_prob(fx['dragon'], 100.0, 0.55, 0.45):.4f}")
    print("parity example: series exact @p=0.6: "
          + ", ".join(f"{k} {v:.4f}" for k, v in series_exact_probs(0.6).items()))


if __name__ == "__main__":
    main()
