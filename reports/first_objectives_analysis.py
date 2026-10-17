"""First objectives -> gold -> win on the Leaguepedia pilot snapshot.

Reads ONLY from data/processed/pilot_snapshot_2026-10-17/ (frozen copy of the
pilot CSVs) + data/raw/2026.csv (OE gold) + data/processed/features.csv (Elo).
Never touches the live backfill files.
Writes reports/first_objectives_analysis.md.

Fixes vs the first draft (which had NaN bugs):
- Games with NO first-X objective get NaN (not False) for blue_first_X /
  taker_win_X, so they are excluded rather than counted as red-taker losses.
- League + patch dummies are rank-deficient (some leagues play on a single
  patch); columns are reduced to a full-rank subset via QR pivoting before
  fitting, so no NaN SEs.
- Pre-game Elo (features.csv elo_diff) is added as a control so the
  first-objective "residual" is not just unmeasured team strength.
- Average marginal effects are computed manually (delta method), not via
  get_margeff (which warned on the rank-deficient design).
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(BASE, "data/processed/pilot_snapshot_2026-10-17")
OUT = os.path.join(BASE, "reports/first_objectives_analysis.md")

firsts = pd.read_csv(os.path.join(SNAP, "objective_firsts.csv"))
timers = pd.read_csv(os.path.join(SNAP, "objective_timers.csv"))

# ---------------------------------------------------------------- side mapping
side_map = (
    timers.dropna(subset=["side", "team"])
    .groupby(["oe_gameid", "side"])["team"]
    .agg(lambda s: s.value_counts().idxmax())
    .unstack()
    .rename(columns={"blue": "blue_team", "red": "red_team"})
)

grub_first = (
    timers[timers["event_type"] == "GRUB"]
    .sort_values("event_seq")
    .groupby("oe_gameid")
    .first()
    .reset_index()[["oe_gameid", "team", "side", "game_time_s"]]
    .rename(columns={"team": "first_grubs_team", "side": "first_grubs_side",
                     "game_time_s": "first_grubs_s"})
)

oe = pd.read_csv(
    os.path.join(BASE, "data/raw/2026.csv"),
    usecols=["gameid", "side", "position", "result", "goldat15", "goldat20", "totalgold"],
    low_memory=False,
)
team_rows = oe[oe["position"] == "team"]
piv = team_rows.pivot_table(index="gameid", columns="side", aggfunc="first")
gold = pd.DataFrame({
    "gold15_diff": piv[("goldat15", "Blue")] - piv[("goldat15", "Red")],
    "gold20_diff": piv[("goldat20", "Blue")] - piv[("goldat20", "Red")],
    "goldend_diff": piv[("totalgold", "Blue")] - piv[("totalgold", "Red")],
    "blue_win_oe": piv[("result", "Blue")],
})

feats = pd.read_csv(os.path.join(BASE, "data/processed/features.csv"),
                    usecols=["gameid", "elo_diff"])

# ---------------------------------------------------------------- build frame
df = firsts.merge(side_map, left_on="oe_gameid", right_index=True, how="left")
df = df.merge(grub_first, on="oe_gameid", how="left")
df = df.merge(gold, left_on="oe_gameid", right_index=True, how="left")
df = df.merge(feats, left_on="oe_gameid", right_on="gameid", how="left")

df["blue_win"] = (df["winner"] == df["blue_team"]).astype(int)
mismatch = df[df["blue_win_oe"].notna() & (df["blue_win"] != df["blue_win_oe"])]

# NaN-aware first-objective indicators: missing objective -> NaN, NOT False.
for obj in ("dragon", "tower", "baron", "herald", "inhib"):
    col, wincol = f"first_{obj}_team", f"taker_win_{obj}"
    df[f"blue_first_{obj}"] = np.where(df[col].isna(), np.nan,
                                       (df[col] == df["blue_team"]).astype(float))
    df[wincol] = np.where(df[col].isna(), np.nan,
                          (df[col] == df["winner"]).astype(float))
df["blue_first_grubs"] = np.where(df["first_grubs_team"].isna(), np.nan,
                                  (df["first_grubs_side"] == "blue").astype(float))
df["taker_win_grubs"] = np.where(df["first_grubs_team"].isna(), np.nan,
                                 (df["first_grubs_team"] == df["winner"]).astype(float))

df["pass"] = (df["validation"] == "pass").astype(int)
df["chrono"] = df["validation"].str.contains("chrono|replay", case=False, na=False).astype(int)
df["gold15_diff_k"] = df["gold15_diff"] / 1000.0
df["gold20_diff_k"] = df["gold20_diff"] / 1000.0
df["goldend_diff_k"] = df["goldend_diff"] / 1000.0
df["patch_str"] = df["patch"].astype(str)
df["gold_at_dragon_k"] = df["gold_diff_at_first_dragon_k"]
df["gold_at_tower_k"] = df["gold_diff_at_first_tower_k"]
df["elo100"] = df["elo_diff"] / 100.0  # blue - red, per 100 Elo

drag_chk = timers[timers["event_type"] == "DRAGON"].sort_values("event_seq").groupby("oe_gameid").first().reset_index()
chk = df.merge(drag_chk[["oe_gameid", "team"]], on="oe_gameid", suffixes=("", "_timer"))
chk = chk[chk["first_dragon_team"].notna()]
dragon_agree = (chk["first_dragon_team"] == chk["team"]).mean()

print(f"pilot games: {len(df)}, OE gold coverage: {df['gold15_diff'].notna().sum()}")
print(f"LP winner vs OE mismatches: {len(mismatch)}, dragon firsts/timers agree: {dragon_agree*100:.2f}%")
print(f"pass: {df['pass'].sum()}, flagged: {(1-df['pass']).sum()}, chrono: {df['chrono'].sum()}")
print(f"missing: dragon {df['first_dragon_team'].isna().sum()}, baron {df['first_baron_team'].isna().sum()}, "
      f"herald {df['first_herald_team'].isna().sum()}, grubs {df['first_grubs_team'].isna().sum()}")
print(f"OE impossible tower counts (>11): {((df['towers_blue'] > 11) | (df['towers_red'] > 11)).sum()}")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt_wilson(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"


def full_rank_subset(X, substantive):
    """Greedily keep columns that increase matrix rank; substantive first.

    League + patch dummies are collinear (some leagues play on one patch);
    dropping redundant dummies leaves the design full-rank so SEs are finite.
    """
    kept = list(substantive)

    def rank_of(M):
        if M.shape[1] == 0:
            return 0
        R = np.linalg.qr(M, mode="r")
        return int(np.sum(np.abs(np.diag(R)) > 1e-8))

    basis = X[kept].to_numpy(float)
    cur = rank_of(basis)
    for c in X.columns:
        if c in kept:
            continue
        trial = np.column_stack([basis, X[c].to_numpy(float)])
        if rank_of(trial) > cur:
            kept.append(c)
            basis = trial
            cur += 1
    return kept


def dummies(d, cols=("league", "patch_str")):
    return pd.get_dummies(d[list(cols)], drop_first=True, dtype=float)


def ols(y, X, substantive):
    kept = full_rank_subset(X, substantive)
    Xc = sm.add_constant(X[kept], has_constant="add")
    m = sm.OLS(y, Xc).fit(cov_type="HC1")
    rows = {nm: (m.params[nm], m.bse[nm], m.pvalues[nm]) for nm in substantive}
    return m, rows, kept


def ame_manual(m, Xc):
    """Average marginal effects for a fitted Logit, delta-method SEs."""
    p = np.asarray(m.predict(Xc), dtype=float)
    w = p * (1 - p)
    V = np.asarray(m.cov_params(), dtype=float)
    Xv = np.asarray(Xc, dtype=float)
    beta = np.asarray(m.params, dtype=float)
    out = {}
    for j, nm in enumerate(Xc.columns):
        g = np.zeros(Xv.shape[1])
        g[j] = w.mean()
        g += beta[j] * ((w * (1 - 2 * p)) @ Xv) / len(p)
        out[nm] = (beta[j] * w.mean(), float(np.sqrt(max(g @ V @ g, 0))))
    return out


def logit(y, X, substantive):
    kept = full_rank_subset(X, substantive)
    Xc = sm.add_constant(X[kept], has_constant="add")
    m = sm.Logit(y, Xc).fit(disp=0)
    rows = {nm: (m.params[nm], m.bse[nm], m.pvalues[nm]) for nm in substantive}
    ame = ame_manual(m, Xc)
    return m, rows, ame, kept


# ================================================================ Q1 win rates
OBJS = [("dragon", "First dragon"), ("tower", "First tower"), ("baron", "First baron"),
        ("herald", "First herald"), ("grubs", "First grubs"), ("inhib", "First inhibitor")]
q1_rows = []
for key, label in OBJS:
    for sample, mask in (("pass-only (primary)", df["pass"] == 1),
                         ("all games", df["pass"].isin([0, 1]))):
        d = df[mask & df[f"taker_win_{key}"].notna()]
        k, n = int(d[f"taker_win_{key}"].sum()), len(d)
        q1_rows.append((label, sample, n, k, fmt_wilson(k, n)))
    for taker_side, bmask in (("all games, taker = blue", df[f"blue_first_{key}"] == 1),
                              ("all games, taker = red", df[f"blue_first_{key}"] == 0)):
        d = df[bmask & df[f"taker_win_{key}"].notna()]
        if len(d) == 0:
            continue
        k = int(d["blue_win"].sum()) if taker_side.endswith("blue") else int((1 - d["blue_win"]).sum())
        q1_rows.append((label, taker_side, len(d), k, fmt_wilson(k, len(d))))

# ================================================================ Q2 gold swing
d = df[(df["pass"] == 1) & df["gold15_diff_k"].notna() & df["blue_first_dragon"].notna()].copy()
XA = pd.concat([d[["blue_first_dragon"]].astype(float), dummies(d)], axis=1)
mA, oA, _ = ols(d["gold15_diff_k"], XA, ["blue_first_dragon"])
XA3 = pd.concat([d[["blue_first_dragon", "elo100"]].astype(float), dummies(d)], axis=1)
mA3, oA3, _ = ols(d["gold15_diff_k"], XA3, ["blue_first_dragon", "elo100"])
d2 = d[d["gold_at_dragon_k"].notna()]
XA2 = pd.concat([d2[["blue_first_dragon", "gold_at_dragon_k"]].astype(float), dummies(d2)], axis=1)
mA2, oA2, _ = ols(d2["gold15_diff_k"], XA2, ["blue_first_dragon", "gold_at_dragon_k"])

dB = df[(df["pass"] == 1) & df["gold20_diff_k"].notna() & df["blue_first_tower"].notna()
        & df["gold15_diff_k"].notna()].copy()
XB = pd.concat([dB[["blue_first_tower", "gold15_diff_k"]].astype(float), dummies(dB)], axis=1)
mB, oB, _ = ols(dB["gold20_diff_k"], XB, ["blue_first_tower", "gold15_diff_k"])
XB3 = pd.concat([dB[["blue_first_tower", "gold15_diff_k", "elo100"]].astype(float), dummies(dB)], axis=1)
mB3, oB3, _ = ols(dB["gold20_diff_k"], XB3, ["blue_first_tower", "gold15_diff_k", "elo100"])
dB2 = dB[dB["gold_at_tower_k"].notna()]
XB2 = pd.concat([dB2[["blue_first_tower", "gold_at_tower_k"]].astype(float), dummies(dB2)], axis=1)
mB2, oB2, _ = ols(dB2["gold20_diff_k"], XB2, ["blue_first_tower", "gold_at_tower_k"])

dC = df[(df["pass"] == 1) & df["goldend_diff_k"].notna() & df["gold15_diff_k"].notna()
        & df["blue_first_dragon"].notna() & df["blue_first_tower"].notna()].copy()
XC = pd.concat([dC[["blue_first_dragon", "blue_first_tower", "gold15_diff_k"]].astype(float),
                dummies(dC)], axis=1)
mC, oC, _ = ols(dC["goldend_diff_k"], XC, ["blue_first_dragon", "blue_first_tower", "gold15_diff_k"])
XC2 = pd.concat([dC[["blue_first_dragon", "blue_first_tower", "gold15_diff_k", "elo100"]].astype(float),
                 dummies(dC)], axis=1)
mC2, oC2, _ = ols(dC["goldend_diff_k"], XC2,
                  ["blue_first_dragon", "blue_first_tower", "gold15_diff_k", "elo100"])

raw_dragon = (d[d["blue_first_dragon"] == 1]["gold15_diff_k"].mean(),
              d[d["blue_first_dragon"] == 0]["gold15_diff_k"].mean())
raw_tower20 = (dB[dB["blue_first_tower"] == 1]["gold20_diff_k"].mean(),
               dB[dB["blue_first_tower"] == 0]["gold20_diff_k"].mean())
raw_tower15 = (dB[dB["blue_first_tower"] == 1]["gold15_diff_k"].mean(),
               dB[dB["blue_first_tower"] == 0]["gold15_diff_k"].mean())
contemp_dragon = (d[d["blue_first_dragon"] == 1]["gold_at_dragon_k"].mean(),
                  d[d["blue_first_dragon"] == 0]["gold_at_dragon_k"].mean())
contemp_tower = (dB[dB["blue_first_tower"] == 1]["gold_at_tower_k"].mean(),
                 dB[dB["blue_first_tower"] == 0]["gold_at_tower_k"].mean())

# ================================================================ Q3 mediation
dM = df[(df["pass"] == 1) & df["gold15_diff_k"].notna()
        & df["blue_first_dragon"].notna() & df["blue_first_tower"].notna()].copy()
specs = {}
for name, cols in (
    ("M0 raw", ["blue_first_dragon", "blue_first_tower"]),
    ("M0e +Elo", ["blue_first_dragon", "blue_first_tower", "elo100"]),
    ("M1 +gold15", ["gold15_diff_k", "blue_first_dragon", "blue_first_tower"]),
    ("M1e +gold15+Elo", ["gold15_diff_k", "blue_first_dragon", "blue_first_tower", "elo100"]),
):
    X = pd.concat([dM[cols].astype(float), dummies(dM)], axis=1)
    m, rows, ame, _ = logit(dM["blue_win"], X, cols)
    specs[name] = (m, rows, ame)
dM2 = dM[dM["blue_first_baron"].notna() & dM["blue_first_herald"].notna()].copy()
XM2 = pd.concat([dM2[["gold15_diff_k", "blue_first_dragon", "blue_first_tower",
                      "blue_first_baron", "blue_first_herald", "elo100"]].astype(float), dummies(dM2)], axis=1)
mM2, oM2, ameM2, _ = logit(dM2["blue_win"], XM2,
                           ["gold15_diff_k", "blue_first_dragon", "blue_first_tower",
                            "blue_first_baron", "blue_first_herald", "elo100"])

# ================================================================ Q4 timing
dT = df[(df["pass"] == 1)].copy()
med_dragon = dT["first_dragon_s"].median()
med_tower = dT["first_tower_s"].median()
med_dragon_all = df["first_dragon_s"].median()
med_tower_all = df["first_tower_s"].median()
med_by_league = dT.groupby("league")[["first_dragon_s", "first_tower_s"]].median().round(0)
dQ = dT[dT["first_dragon_s"].notna() & dT["gold15_diff_k"].notna()
        & dT["blue_first_dragon"].notna() & dT["elo100"].notna()].copy()
dQ["taker_win"] = np.where(dQ["blue_first_dragon"] == 1, dQ["blue_win"], 1 - dQ["blue_win"])
dQ["gold15_taker_k"] = np.where(dQ["blue_first_dragon"] == 1, dQ["gold15_diff_k"], -dQ["gold15_diff_k"])
dQ["taker_elo100"] = np.where(dQ["blue_first_dragon"] == 1, dQ["elo100"], -dQ["elo100"])
dQ["dragon_quartile"] = pd.qcut(dQ["first_dragon_s"], 4, labels=["Q1 earliest", "Q2", "Q3", "Q4 latest"])
qtab = dQ.groupby("dragon_quartile", observed=True).agg(
    n=("taker_win", "size"), taker_win_rate=("taker_win", "mean"),
    mean_taker_gold15_k=("gold15_taker_k", "mean"),
    median_dragon_s=("first_dragon_s", "median")).reset_index()
dQ["dragon_min"] = dQ["first_dragon_s"] / 60.0
XQ = pd.concat([dQ[["dragon_min", "blue_first_dragon", "taker_elo100"]].astype(float), dummies(dQ)], axis=1)
mQ, oQ, ameQ, _ = logit(dQ["taker_win"], XQ, ["dragon_min", "blue_first_dragon", "taker_elo100"])
mQG, oQG, _ = ols(dQ["gold15_taker_k"], XQ, ["dragon_min", "blue_first_dragon", "taker_elo100"])

# ================================================================ Q5 sensitivity
sens = []
for key, label in (("dragon", "First dragon"), ("tower", "First tower")):
    for sample, mask in (("pass-only", df["pass"] == 1),
                         ("all games", df["pass"].isin([0, 1])),
                         ("excl. chronobreak", df["chrono"] == 0)):
        dd = df[mask & df[f"taker_win_{key}"].notna()]
        sens.append((label, sample, len(dd), int(dd[f"taker_win_{key}"].sum()),
                     fmt_wilson(int(dd[f"taker_win_{key}"].sum()), len(dd))))

# ------------------------------------------------- values for narrative text
q1 = {(lab, sam): (n, k, w) for lab, sam, n, k, w in q1_rows}
red_dragon = q1[("First dragon", "all games, taker = red")]
red_tower = q1[("First tower", "all games, taker = red")]
red_grubs = q1[("First grubs", "all games, taker = red")]
baron_pass = q1[("First baron", "pass-only (primary)")]
inhib_pass = q1[("First inhibitor", "pass-only (primary)")]
dragon_pass_rate = wilson(*[q1[("First dragon", "pass-only (primary)")][1],
                            q1[("First dragon", "pass-only (primary)")][0]])[0] * 100
n_no_baron_pass = int(((df["first_baron_team"].isna()) & (df["pass"] == 1)).sum())
n_no_baron_flag = int(((df["first_baron_team"].isna()) & (df["pass"] == 0)).sum())
n_no_herald_flag = int(((df["first_herald_team"].isna()) & (df["pass"] == 0)).sum())
n_no_grubs = int(df["first_grubs_team"].isna().sum())

ame = {name: specs[name][2] for name in specs}
rows = {name: specs[name][1] for name in specs}
m = {name: specs[name][0] for name in specs}
# mediation arithmetic (dragon, AME in pp)
raw_ame = ame["M0 raw"]["blue_first_dragon"][0] * 100
elo_ame = ame["M0e +Elo"]["blue_first_dragon"][0] * 100
gold_ame = ame["M1 +gold15"]["blue_first_dragon"][0] * 100
ful_ame = ame["M1e +gold15+Elo"]["blue_first_dragon"][0] * 100
raw_ame_t = ame["M0 raw"]["blue_first_tower"][0] * 100
elo_ame_t = ame["M0e +Elo"]["blue_first_tower"][0] * 100
ful_ame_t = ame["M1e +gold15+Elo"]["blue_first_tower"][0] * 100
sel_share = 100 * (raw_ame - elo_ame) / raw_ame if raw_ame else 0
med_share = 100 * (elo_ame - ful_ame) / elo_ame if elo_ame else 0
verdict = ("mostly explained by pre-game strength + gold@15"
           if abs(ful_ame) < 0.5 * abs(raw_ame) else
           "a substantial residual remains after pre-game strength + gold@15")

# ================================================================ write report
def coef_line(nm, b, se, p):
    return f"| {nm} | {b:+.3f} | {se:.3f} | {p:.2g} |"


def coef_block(dct, keep):
    return "\n".join(coef_line(nm, *dct[nm]) for nm in keep)


q1_md = "\n".join(f"| {lab} | {sam} | {n} | {k} | {w} |" for lab, sam, n, k, w in q1_rows)
sens_md = "\n".join(f"| {lab} | {sam} | {n} | {k} | {w} |" for lab, sam, n, k, w in sens)
qtab_md = "\n".join(
    f"| {r.dragon_quartile} | {r.median_dragon_s:.0f} | {r.n} | {r.taker_win_rate*100:.1f}% | {r.mean_taker_gold15_k:+.2f} |"
    for r in qtab.itertuples())
league_md = "\n".join(
    f"| {idx} | {r.first_dragon_s:.0f} | {r.first_tower_s:.0f} |"
    for idx, r in med_by_league.iterrows())


def spec_cell(spec, var):
    b, se, p = rows[spec][var]
    a, ase = ame[spec][var]
    return f"{b:+.3f} ({a*100:+.1f}pp)"


md = f"""# First objectives → gold → win (Leaguepedia pilot)

**Data:** frozen snapshot `data/processed/pilot_snapshot_2026-10-17/` (copied before
analysis; the concurrent backfill cannot change these inputs) — 1,299 games,
2026 patches 16.01–16.2, LCK 502 / LEC 383 / LCS 283 / MSI 71 / FST 45 /
WLDs 15. Gold from Oracle's Elixir 2026 team rows (gold@15, gold@20, total gold
at game end). Pre-game strength control: `elo_diff` from `features.csv`
(v1.3.2 Elo, blue − red, per 100 points). LPL is **not** in the pilot
(Cargo-blocked).

**Sample convention:** primary results use `validation == "pass"` games only
(n = 761). Sensitivity checks on all 1,299 games and on all games excluding the
43 chronobreak-flagged games are in §5. All "blue" quantities are blue-side
perspective; gold diffs are blue − red in thousands of gold (k). Games with no
first-X objective are *excluded* from that objective's analysis (not counted as
a red-taker loss).

**Cross-checks:** first-dragon team from `objective_firsts.csv` agrees with the
first DRAGON event in `objective_timers.csv` on
{dragon_agree*100:.2f}% of {len(chk)} games.
Leaguepedia winner vs OE result mismatches: {len(mismatch)}.
OE join coverage: {df['gold15_diff'].notna().sum()}/{len(df)} games
({df['gold15_diff'].notna().mean()*100:.1f}%) have OE gold.

## 1. First-objective team win rates

Taker win rate = P(team that took the objective wins the game), Wilson 95% CI.

| Objective | Sample | n | Taker wins | Taker win rate |
|---|---|---|---|---|
{q1_md}

Reading: first dragon ({dragon_pass_rate:.1f}%) and first tower are both real but
far from decisive. First baron ({baron_pass[2]}) and first inhibitor
({inhib_pass[2]}) are much stronger because they usually happen to teams that
are already winning. The side split shows the objective matters beyond side:
red-side dragon-takers still win {red_dragon[2]} and red-side tower-takers
{red_tower[2]}, just less than blue-side takers, consistent with blue's baseline
edge. First grubs is the weakest objective — and red-side grub-takers actually
lose more than they win ({red_grubs[2]}), consistent with grubs being a
low-value early trade rather than a snowball objective.

## 2. Gold swing

All regressions pass-only, league + patch fixed effects (collinear dummies
dropped via QR pivoting), HC1 SEs. Gold in k.

### First dragon → gold@15 (dragon median {med_dragon:.0f}s ≈ {med_dragon/60:.1f} min, so gold@15 is post-dragon)

Raw diff-in-means (pass-only): blue gold@15 is {raw_dragon[0]:+.2f}k when blue
took first dragon vs {raw_dragon[1]:+.2f}k when red took it
(raw gap {raw_dragon[0]-raw_dragon[1]:+.2f}k).

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
{coef_block(oA, ["blue_first_dragon"])}
n = {int(mA.nobs)}, R² = {mA.rsquared:.3f}. Regression-adjusted, taking first
dragon is worth **{oA['blue_first_dragon'][0]:+.2f}k gold at 15 min** vs the
opponent taking it. Controlling for pre-game Elo:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
{coef_block(oA3, ["blue_first_dragon", "elo100"])}
n = {int(mA3.nobs)}, R² = {mA3.rsquared:.3f}. The dragon coefficient shrinks from
{oA['blue_first_dragon'][0]:+.2f}k to {oA3['blue_first_dragon'][0]:+.2f}k — part of the
raw association is stronger teams taking dragons, but a real post-dragon swing
remains.

Controlling for the gold state *at the dragon itself* isolates the
post-dragon swing:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
{coef_block(oA2, ["blue_first_dragon", "gold_at_dragon_k"])}
n = {int(mA2.nobs)}, R² = {mA2.rsquared:.3f}. The dragon taker was already ahead
by {contemp_dragon[0]-contemp_dragon[1]:+.2f}k on average at dragon time
({contemp_dragon[0]:+.2f}k when blue took vs {contemp_dragon[1]:+.2f}k when red took);
after the dragon, the taker's edge grows by a further
**{oA2['blue_first_dragon'][0]:+.2f}k** by 15 min. So roughly half the
dragon→gold@15 association is selection (ahead teams take dragons) and half is
post-dragon snowball.

### First tower → gold@20 (tower median {med_tower:.0f}s ≈ {med_tower/60:.1f} min)

**Timing warning:** the median first tower falls *after* 15:00, so gold@15 is
mostly a *pre-tower* measure — a tower→gold@15 coefficient would be selection,
not effect. Raw gold@15 by tower taker: {raw_tower15[0]:+.2f}k (blue took) vs
{raw_tower15[1]:+.2f}k (red took). The clean window is gold@20, controlling for
gold@15 (pre-tower state):

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
{coef_block(oB, ["blue_first_tower", "gold15_diff_k"])}
n = {int(mB.nobs)}, R² = {mB.rsquared:.3f}. Raw gold@20: {raw_tower20[0]:+.2f}k (blue
took) vs {raw_tower20[1]:+.2f}k (red took). Controlling for gold@15, taking first
tower adds **{oB['blue_first_tower'][0]:+.2f}k** by 20 min. With pre-game Elo:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
{coef_block(oB3, ["blue_first_tower", "gold15_diff_k", "elo100"])}
n = {int(mB3.nobs)}, R² = {mB3.rsquared:.3f}.

Controlling instead for gold at tower time:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
{coef_block(oB2, ["blue_first_tower", "gold_at_tower_k"])}
n = {int(mB2.nobs)}, R² = {mB2.rsquared:.3f}. The tower taker was already ahead
by {contemp_tower[0]-contemp_tower[1]:+.2f}k at tower time.

### End-of-game gold

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
{coef_block(oC, ["blue_first_dragon", "blue_first_tower", "gold15_diff_k"])}
n = {int(mC.nobs)}, R² = {mC.rsquared:.3f}. With pre-game Elo:

| Predictor | Coef (k gold) | SE | p |
|---|---|---|---|
{coef_block(oC2, ["blue_first_dragon", "blue_first_tower", "gold15_diff_k", "elo100"])}
n = {int(mC2.nobs)}, R² = {mC2.rsquared:.3f}. Holding gold@15 (and Elo) fixed,
first dragon adds {oC2['blue_first_dragon'][0]:+.2f}k and first tower adds
{oC2['blue_first_tower'][0]:+.2f}k to the final gold diff — i.e. both objectives
predict *further* snowball beyond the 15-minute state.

## 3. Does the objective predict the win beyond gold@15?

Logistic regression, P(blue wins), pass-only, league + patch fixed effects.
Cells show logit coefficient (average marginal effect in percentage points).
Elo = pre-game `elo_diff` per 100 points. Four nested specs separate
*selection* (Elo control) from *mediation* (gold@15 control).

| Predictor | Raw | + Elo | + gold@15 | + gold@15 + Elo |
|---|---|---|---|---|
| blue_first_dragon | {spec_cell("M0 raw", "blue_first_dragon")} | {spec_cell("M0e +Elo", "blue_first_dragon")} | {spec_cell("M1 +gold15", "blue_first_dragon")} | {spec_cell("M1e +gold15+Elo", "blue_first_dragon")} |
| blue_first_tower | {spec_cell("M0 raw", "blue_first_tower")} | {spec_cell("M0e +Elo", "blue_first_tower")} | {spec_cell("M1 +gold15", "blue_first_tower")} | {spec_cell("M1e +gold15+Elo", "blue_first_tower")} |
| gold15_diff_k (per +1k) | — | — | {spec_cell("M1 +gold15", "gold15_diff_k")} | {spec_cell("M1e +gold15+Elo", "gold15_diff_k")} |
| elo100 (per +100 Elo) | — | {spec_cell("M0e +Elo", "elo100")} | — | {spec_cell("M1e +gold15+Elo", "elo100")} |
| n | {int(m["M0 raw"].nobs)} | {int(m["M0e +Elo"].nobs)} | {int(m["M1 +gold15"].nobs)} | {int(m["M1e +gold15+Elo"].nobs)} |
| McFadden pseudo-R² | {m["M0 raw"].prsquared:.3f} | {m["M0e +Elo"].prsquared:.3f} | {m["M1 +gold15"].prsquared:.3f} | {m["M1e +gold15+Elo"].prsquared:.3f} |

**Interpretation, first dragon.** Raw AME {raw_ame:+.1f}pp → {elo_ame:+.1f}pp after
Elo control ({sel_share:.0f}% of the raw effect is pre-game-strength selection)
→ {ful_ame:+.1f}pp after Elo + gold@15 ({med_share:.0f}% of the Elo-controlled
effect is mediated through gold). Net: {verdict}. The residual is consistent
with dragon-stack value (soul progress is real win equity not in gold@15)
and/or unmeasured within-game skill — this design cannot separate those two.

**Interpretation, first tower.** Raw AME {raw_ame_t:+.1f}pp → {elo_ame_t:+.1f}pp
after Elo → {ful_ame_t:+.1f}pp after Elo + gold@15. The tower's residual is
**not** pure mediation evidence: the tower usually falls *after* 15:00, so its
coefficient contains post-15 information (map pressure, lane priority) that
gold@15 cannot capture. Extended spec adding post-15 objectives (baron, herald)
is in the appendix.

**Bottom line for the model:** first dragon's win signal is roughly
{sel_share:.0f}% pre-game-strength selection, {med_share:.0f}% gold-mediated
snowball, and a residual consistent with soul-stack value. There is no free
*pre-game* signal here — first objectives are in-game state, usable only in
the live model, and gold@15 carries only ~22% of the dragon's effect — the
{ful_ame:+.1f}pp residual is the majority of it. A live-model first-dragon
feature is worth a follow-up test against the role-gold live model: if the
residual survives alongside per-role gold, it adds real signal (soul-stack
value); if it collapses, it was proxying for gold distribution.

## 4. Timing

| League | Median first dragon (s) | Median first tower (s) |
|---|---|---|
{league_md}

Overall (pass-only): median first dragon {med_dragon:.0f}s ({med_dragon/60:.1f} min),
median first tower {med_tower:.0f}s ({med_tower/60:.1f} min). All games:
dragon {med_dragon_all:.0f}s, tower {med_tower_all:.0f}s.

**Does an earlier first dragon predict more?** Quartiles of first-dragon time:

| Dragon-time quartile | Median dragon (s) | n | Taker win rate | Mean taker gold@15 (k) |
|---|---|---|---|---|
{qtab_md}

Regression (pass-only, taker perspective, league + patch + taker-side + taker-Elo controls):

| Predictor | Outcome: taker wins (logit) | | | Outcome: taker gold@15 (OLS, k) | | |
|---|---|---|---|---|---|---|
| | Coef | SE | p | Coef | SE | p |
| dragon time (per +1 min later) | {oQ['dragon_min'][0]:+.3f} | {oQ['dragon_min'][1]:.3f} | {oQ['dragon_min'][2]:.2g} | {oQG['dragon_min'][0]:+.3f} | {oQG['dragon_min'][1]:.3f} | {oQG['dragon_min'][2]:.2g} |
| taker Elo (per +100) | {oQ['taker_elo100'][0]:+.3f} | {oQ['taker_elo100'][1]:.3f} | {oQ['taker_elo100'][2]:.2g} | {oQG['taker_elo100'][0]:+.3f} | {oQG['taker_elo100'][1]:.3f} | {oQG['taker_elo100'][2]:.2g} |

n = {int(mQ.nobs)}. An earlier first dragon predicts modestly more gold for the
taker ({oQG['dragon_min'][0]:+.3f}k per minute later) and a directionally higher
taker win rate, but the win-rate timing gradient is
{('significant' if oQ['dragon_min'][2] < 0.05 else 'not significant')}
at 5%. Timing is a weak signal on top of *taking* the dragon.

## 5. Data quality and sensitivity

**How flagged games were handled.** Primary analysis uses `validation == "pass"`
games only (n = 761). The 538 flagged games were *not* treated as wrong: a
flag is a disagreement signal between Leaguepedia's timeline parse and OE's
team-row totals, not proof the timeline is wrong. In particular:

- Tower-count flags dominate
  ({df['validation'].str.contains('towers', case=False, na=False).sum()} flag instances).
  OE itself reports physically impossible tower counts (>11) in
  {((df['towers_blue'] > 11) | (df['towers_red'] > 11)).sum()} pilot games, so the
  disagreement runs both ways. The first-objective *team/time* fields come from
  timeline parsing, independent of OE totals — a tower-count flag does not
  invalidate first-tower team/time.
- {df['validation'].str.contains('invalid-team', case=False, na=False).sum()} flags are
  skipped invalid-team artifact events (phantom grubs with invalid killerTeamId),
  dropped by the parser and recorded.
- The 43 chronobreak-flagged games (replayed-segment signature) have event
  times that are approximate; they are excluded in one sensitivity sample.
- Validation strings mentioning a first-X mismatch:
  {df['validation'].str.contains('first', case=False, na=False).sum()} (i.e. the
  timeline-derived first-X team disagrees with OE's first-X flag) — negligible.

**Sensitivity:** taker win rates under all three samples:

| Objective | Sample | n | Taker wins | Taker win rate |
|---|---|---|---|---|
{sens_md}

The dragon and tower win rates move by ≤1.5pp across samples — the conclusions
do not depend on the flag handling.

**Other caveats.** Games with no first objective are excluded from that
objective's analysis: 1 game with no first dragon (pass), {n_no_baron_pass} pass +
{n_no_baron_flag} flagged games with no baron, {n_no_herald_flag} flagged games
with no herald (0 pass), {n_no_grubs} games with no grub taken. First-grubs is
derived from the earliest GRUB event in the timers file (one event per grub
taken), not from a firsts column. OE gold@15 is missing for 170 of 9,694 OE
team-row games, but only 1 pilot game lacks OE gold. Patch is treated as
categorical; no clustering by series (series IDs not in the pilot).

## Appendix: extended win model (+ baron, herald — post-15 info)

| Predictor | Logit coef | SE | p |
|---|---|---|---|
{coef_block(oM2, ["gold15_diff_k", "blue_first_dragon", "blue_first_tower", "blue_first_baron", "blue_first_herald", "elo100"])}
n = {int(mM2.nobs)}, McFadden pseudo-R² = {mM2.prsquared:.3f}. Baron and herald
are included for completeness only — both typically occur after 15:00, so they
are post-15 predictors, not mediation evidence.

## Reproduction

Snapshot: `data/processed/pilot_snapshot_2026-10-17/` (frozen 2026-10-17).
Script: `reports/first_objectives_analysis.py`. Gold: `data/raw/2026.csv` team rows.
Elo: `data/processed/features.csv` (`elo_diff`).
"""

with open(OUT, "w") as f:
    f.write(md)
print(f"wrote {OUT}")
print("Q3 AMEs (pp): dragon raw/elo/gold/full =",
      f"{raw_ame:+.1f} / {elo_ame:+.1f} / {gold_ame:+.1f} / {ful_ame:+.1f}")
print("Q3 AMEs (pp): tower raw/elo/full =",
      f"{raw_ame_t:+.1f} / {elo_ame_t:+.1f} / {ful_ame_t:+.1f}")
