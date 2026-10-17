"""Region-strength offsets: correct weak-region Elo inflation.

Elo systematically overrates weak regions (LATAM, Vietnam, Taiwan) and
underrates LCK in cross-region games (see reports/region_strength_analysis.md).
This module fits hierarchical region offsets and applies them to elo_diff:

    logit(p) = a + b * logit(p_elo) + off[region_blue] - off[region_red]

fit on cross-region games only (MSI / Worlds / First Stand), ridge lambda=1
on the offsets, via IRLS. Games where either side's region is minor,
unknown, or otherwise unfitted are excluded from the fit (the original
analysis excluded Japan's 4 games for the same reason -- its Protocol B
uses n=560, not 564). Only the offsets are shipped -- the intercept
(blue-edge stand-in; production already has blue_edge) and the slope
(adds nothing once offsets are in) are fit-but-discarded.

Region assignment: a team's region is its modal top-flight domestic league
that year. LCP (merged APAC league, 2025+) has no single region, so LCP
teams resolve via their most recent prior-year domestic league. Manual
overrides (flagged assumptions): MVK Esports -> Vietnam, Team Secret
Whales -> Vietnam. Teams/regions with no evidence get offset 0.

Offsets are CONSTANT (no decay -- decay was tested and rejected: any decay
rule hurts held-out log-loss monotonically). They are REFIT from data on
every run (they drift -- LPL's edge shrank ~40% from 2024 to 2025-26).

Pipeline order: build_features.py runs first (produces features.csv with
raw elo_diff), then this module fits on features.csv's raw elo_diff and
appends the elo_diff_adj column. build_features.py also computes
elo_diff_adj from an existing region_offsets.json when present, but this
module's post-fit refresh is authoritative -- it guarantees train.py sees
offsets fit on the current features.csv. The fit ALWAYS uses raw elo_diff,
never elo_diff_adj (fitting on adjusted Elo would be circular).

For backtests, pass cutoff=... to fit on cross-region games strictly
before the cutoff (no future leakage).
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
PROC_DIR = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "model")
OFFSETS_PATH = os.path.join(MODEL_DIR, "region_offsets.json")
FEATURES_PATH = os.path.join(PROC_DIR, "features.csv")

# Top-flight domestic league -> region. Developmental leagues of top
# regions (same org ecosystem) map to the parent region.
LEAGUE_TO_REGION = {
    "LCK": "LCK",
    "LPL": "LPL",
    "LEC": "EMEA",
    "LCS": "NA",
    "LTA N": "NA",
    "LTA S": "LATAM",
    "CBLOL": "LATAM",
    "CBLOLA": "LATAM",
    "LLA": "LATAM",
    "VCS": "Vietnam",
    "PCS": "Taiwan",
    "LJL": "Japan",
    "LCO": "Oceania",
    "TCL": "Turkiye",
    "KeSPA Cup": "LCK",
    "LCKC": "LCK",
    "LDL": "LPL",
    "LCP": None,  # resolved via most recent prior-year domestic league
}
# Leagues excluded from the modal-domestic computation: internationals and
# cross-conference play carry no region signal.
INTERNATIONAL_LEAGUES = {"MSI", "WLDs", "FST", "EWC", "ASI", "LTA"}
# Primary fit set (per the analysis): MSI + Worlds + First Stand.
CROSS_REGION_LEAGUES = {"MSI", "WLDs", "FST"}
# Manual overrides, flagged assumptions (see analysis SS1):
# MVK = MGN Vikings' 2026 slot/rebrand, no pre-2026 record under its own
# name; Team Secret Whales = ex-Team Whales (VCS 2024), no 2024 record
# under its current name.
MANUAL_OVERRIDES = {
    "MVK Esports": "Vietnam",
    "Team Secret Whales": "Vietnam",
}
# Minor regions are never fitted (too few cross-region games) -> offset 0.
MINOR_REGIONS = {"Japan", "Oceania", "Turkiye"}
UNKNOWN = "Unknown"
RIDGE_LAMBDA = 1.0
MIN_GAMES_TO_FIT = 15  # region appearances in the fit set; revisit below this
LOGIT_TO_ELO = 400.0 / math.log(10.0)  # d(Elo)/d(logit)
ELO_TO_LOGIT = math.log(10.0) / 400.0


def load_team_appearances(raw_dir=RAW_DIR):
    """Team rows from the raw Oracle's Elixir CSVs -> (team, league, date)."""
    frames = []
    for year in ("2024", "2025", "2026"):
        path = os.path.join(raw_dir, f"{year}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["date", "league", "position", "teamname"],
                         low_memory=False)
        frames.append(df[df["position"] == "team"])
    apps = pd.concat(frames, ignore_index=True)
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps = apps.dropna(subset=["date", "teamname", "league"])
    return apps.rename(columns={"teamname": "team"})[["team", "league", "date"]]


def appearances_from_features(df):
    """Same (team, league, date) frame, built from features.csv rows."""
    blue = df[["blue", "league", "date"]].rename(columns={"blue": "team"})
    red = df[["red", "league", "date"]].rename(columns={"red": "team"})
    apps = pd.concat([blue, red], ignore_index=True)
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    return apps.dropna(subset=["date", "team", "league"])


def team_year_regions(appearances, cutoff=None):
    """Resolve {team: {year: region}} from game appearances.

    Modal domestic league per (team, year); LCP resolves via the most
    recent prior year with a mapped non-LCP domestic league; a year with
    no domestic games inherits the most recent prior year's region.
    With cutoff set, only appearances with date < cutoff are used (for
    honest backtests).
    """
    apps = appearances
    if cutoff is not None:
        apps = apps[apps["date"] < pd.Timestamp(cutoff)]
    apps = apps.copy()
    apps["year"] = apps["date"].dt.year
    domestic = apps[~apps["league"].isin(INTERNATIONAL_LEAGUES)]

    # modal domestic league per (team, year)
    modal = {}
    for (team, year), grp in domestic.groupby(["team", "year"]):
        modal[(team, year)] = grp["league"].value_counts().idxmax()

    teams = sorted(apps["team"].unique())
    out = {}
    for team in teams:
        years = sorted({y for (t, y) in modal if t == team}
                       | {y for (t, y) in
                          zip(apps.loc[apps["team"] == team, "team"],
                              apps.loc[apps["team"] == team, "year"])})
        resolved = {}
        memo = {}

        def resolve(year, _seen=frozenset()):
            if year in memo:
                return memo[year]
            lg = modal.get((team, year))
            if lg is None:
                # no domestic games that year: inherit most recent prior year
                prev = [y for y in resolved if y < year]
                region = resolved[max(prev)] if prev else UNKNOWN
            elif lg == "LCP":
                # walk back to the most recent prior year with a mapped
                # non-LCP domestic league
                region = UNKNOWN
                for y2 in sorted([y for (t, y) in modal
                                  if t == team and y < year], reverse=True):
                    lg2 = modal[(team, y2)]
                    if lg2 != "LCP" and lg2 in LEAGUE_TO_REGION:
                        region = LEAGUE_TO_REGION[lg2]
                        break
            elif lg in LEAGUE_TO_REGION:
                region = LEAGUE_TO_REGION[lg]
            else:
                region = UNKNOWN
            memo[year] = region
            return region

        for year in years:
            resolved[year] = resolve(year)
        # manual overrides win for every year the team appears
        if team in MANUAL_OVERRIDES:
            for year in resolved:
                resolved[year] = MANUAL_OVERRIDES[team]
        out[team] = resolved
    return out


def team_region_latest(team_year, team):
    """A team's region in its most recent year with data (for prediction)."""
    years = team_year.get(team)
    if not years:
        return UNKNOWN
    return years[max(years)]


def _regions_for_games(df, team_year):
    """Per-game (blue_region, red_region) from the team-year table."""
    years = df["date"].dt.year
    blue_reg = [team_year.get(b, {}).get(y, UNKNOWN)
                for b, y in zip(df["blue"], years)]
    red_reg = [team_year.get(r, {}).get(y, UNKNOWN)
               for r, y in zip(df["red"], years)]
    return np.array(blue_reg), np.array(red_reg)


def _irls_logit(X, y, ridge_diag, max_iter=100, tol=1e-10):
    """IRLS for penalized logistic regression.

    ridge_diag[i] is the ridge penalty on coefficient i (0 = unpenalized).
    Returns the coefficient vector.
    """
    n, p = X.shape
    beta = np.zeros(p)
    lam = np.asarray(ridge_diag, dtype=float)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(prob * (1.0 - prob), 1e-6, None)
        z = eta + (y - prob) / w
        XtW = X.T * w
        A = XtW @ X + np.diag(lam)
        b = XtW @ z
        try:
            new_beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            new_beta = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta
    return beta


def fit_offsets(features, team_year, cutoff=None, ridge=RIDGE_LAMBDA,
                min_games=MIN_GAMES_TO_FIT):
    """Fit hierarchical region offsets on cross-region games.

    Uses features.csv's RAW elo_diff (never elo_diff_adj -- fitting on
    adjusted Elo would be circular). With cutoff set, only games with
    date < cutoff are used (backtests). Returns a result dict with
    offsets in Elo points plus fit diagnostics.
    """
    df = features[features["league"].isin(CROSS_REGION_LEAGUES)].copy()
    if cutoff is not None:
        df = df[df["date"] < pd.Timestamp(cutoff)]
    df = df.reset_index(drop=True)
    blue_reg, red_reg = _regions_for_games(df, team_year)

    # fitted regions: enough appearances, not minor, not unknown
    appearances = {}
    for r in list(blue_reg) + list(red_reg):
        appearances[r] = appearances.get(r, 0) + 1
    fitted = sorted([r for r, n in appearances.items()
                     if n >= min_games and r not in MINOR_REGIONS
                     and r != UNKNOWN])
    idx = {r: i for i, r in enumerate(fitted)}

    # Fit ONLY on games where both sides have fitted regions. Games
    # against minor / unknown / unfitted regions are excluded: they
    # cannot identify the fitted offsets (their dummy is an arbitrary 0),
    # and including them breaks the ridge centering -- the original
    # analysis excluded Japan's 4 games for exactly this reason
    # (its Protocol B uses n=560, not 564).
    unknown_games = int(((blue_reg == UNKNOWN) | (red_reg == UNKNOWN)).sum())
    keep = np.array([(rb in idx) and (rr in idx)
                     for rb, rr in zip(blue_reg, red_reg)])
    df = df[keep].reset_index(drop=True)
    blue_reg, red_reg = blue_reg[keep], red_reg[keep]

    y = df["y"].to_numpy(float)
    logit_p_elo = df["elo_diff"].to_numpy(float) * ELO_TO_LOGIT
    X = np.zeros((len(df), 2 + len(fitted)))
    X[:, 0] = 1.0  # intercept (fit but NOT shipped)
    X[:, 1] = logit_p_elo  # slope (fit but NOT shipped)
    for i, (rb, rr) in enumerate(zip(blue_reg, red_reg)):
        if rb in idx:
            X[i, 2 + idx[rb]] += 1.0
        if rr in idx:
            X[i, 2 + idx[rr]] -= 1.0
    ridge_diag = np.zeros(X.shape[1])
    ridge_diag[2:] = ridge
    beta = _irls_logit(X, y, ridge_diag)

    offsets_logit = {r: float(beta[2 + idx[r]]) for r in fitted}
    offsets_elo = {r: v * LOGIT_TO_ELO for r, v in offsets_logit.items()}
    return {
        "offsets_elo": offsets_elo,
        "offsets_logit": offsets_logit,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "n_games": int(len(df)),
        "fitted_regions": fitted,
        "region_appearances": {r: int(n) for r, n in appearances.items()},
        "unknown_region_games": unknown_games,
        "cutoff": str(cutoff) if cutoff is not None else None,
    }


def apply_offsets(features, offsets_elo, team_year):
    """elo_diff_adj = elo_diff + off[region_blue] - off[region_red].

    Unknown region -> offset 0. Returns a Series aligned to features.
    """
    blue_reg, red_reg = _regions_for_games(features, team_year)
    off_b = pd.Series(blue_reg).map(offsets_elo).fillna(0.0).to_numpy(float)
    off_r = pd.Series(red_reg).map(offsets_elo).fillna(0.0).to_numpy(float)
    return features["elo_diff"].to_numpy(float) + off_b - off_r


def bootstrap_cis(features, team_year, cutoff=None, n_boot=500, seed=7):
    """Bootstrap 95% CIs for the offsets (resample cross-region games)."""
    df = features[features["league"].isin(CROSS_REGION_LEAGUES)].copy()
    if cutoff is not None:
        df = df[df["date"] < pd.Timestamp(cutoff)]
    rng = np.random.default_rng(seed)
    base = fit_offsets(features, team_year, cutoff=cutoff)
    samples = {r: [] for r in base["fitted_regions"]}
    for _ in range(n_boot):
        resampled = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        try:
            fit = fit_offsets(resampled, team_year, cutoff=cutoff)
        except Exception:  # noqa: BLE001 - a resample may separate; skip it
            continue
        for r in samples:
            samples[r].append(fit["offsets_elo"].get(r, 0.0))
    cis = {}
    for r, vals in samples.items():
        if len(vals) >= 50:
            cis[r] = [round(float(np.percentile(vals, 2.5)), 1),
                      round(float(np.percentile(vals, 97.5)), 1)]
    return cis


def load_team_year(path=OFFSETS_PATH):
    """Team-year region table from region_offsets.json (int year keys).

    Returns {} when the file does not exist yet (first run).
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {t: {int(y): r for y, r in yrs.items()}
            for t, yrs in data.get("team_year_regions", {}).items()}


def load_offsets(path=OFFSETS_PATH):
    """Load (offsets_elo, team_region) from region_offsets.json.

    Returns ({}, {}) when the file does not exist yet (first run).
    """
    if not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    offsets = data.get("offsets_elo", {})
    team_year = {t: {int(y): r for y, r in yrs.items()}
                 for t, yrs in data.get("team_year_regions", {}).items()}
    team_region = {t: team_region_latest(team_year, t) for t in team_year}
    return offsets, team_region


def main():
    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    print(f"loaded {len(features)} games from {FEATURES_PATH}")

    appearances = load_team_appearances()
    team_year = team_year_regions(appearances)
    fit = fit_offsets(features, team_year)
    print(f"fit on {fit['n_games']} cross-region games "
          f"({', '.join(fit['fitted_regions'])}); "
          f"unknown-region games: {fit['unknown_region_games']}")

    team_region = {t: team_region_latest(team_year, t) for t in team_year}
    out = {
        "offsets_elo": {r: round(v, 1) for r, v in fit["offsets_elo"].items()},
        "offsets_logit": {r: round(v, 4) for r, v in fit["offsets_logit"].items()},
        "intercept": round(fit["intercept"], 4),
        "slope": round(fit["slope"], 4),
        "fit_date": str(features["date"].max().date()),
        "n_games": fit["n_games"],
        "method": ("logit(p) = a + b*logit(p_elo) + off[blue] - off[red], "
                   "ridge lambda=1 on offsets, IRLS; intercept/slope fit but "
                   "not shipped; offsets constant (no decay)"),
        "lambda": RIDGE_LAMBDA,
        "min_games_to_fit": MIN_GAMES_TO_FIT,
        "fitted_regions": fit["fitted_regions"],
        "region_appearances": fit["region_appearances"],
        "unknown_region_games": fit["unknown_region_games"],
        "league_to_region": LEAGUE_TO_REGION,
        "manual_overrides": MANUAL_OVERRIDES,
        "team_year_regions": {t: {str(y): r for y, r in yrs.items()}
                              for t, yrs in team_year.items()},
        "team_region": team_region,
    }
    with open(OFFSETS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OFFSETS_PATH}")
    for r in fit["fitted_regions"]:
        print(f"  {r}: {fit['offsets_elo'][r]:+.1f} Elo pts")

    # authoritative refresh of the elo_diff_adj column in features.csv
    features["elo_diff_adj"] = apply_offsets(features, fit["offsets_elo"], team_year)
    features.to_csv(FEATURES_PATH, index=False)
    print(f"refreshed elo_diff_adj in {FEATURES_PATH}")


if __name__ == "__main__":
    main()
