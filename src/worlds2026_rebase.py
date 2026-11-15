"""Worlds 2026 closeout v1.3.8 — RE-BASED (Thomas approved 2026-11-14).

Fixes the double-count found in the first v1.3.8 build: the "pre-Worlds"
baseline snapshots.joblib.bak-1.3.2 (as_of 2026-10-16 23:43:16) already
contains the Oct 15-16 Play-In maps, so v1.3.7 / the first v1.3.8 applied
those 15 maps a second time for Karmine Corp, Cloud9, MVK Esports, FURIA.

This script re-bases on the TRUE pre-Worlds Elo (full-history Oracle replay
on rows with date < 2026-10-15, verified independently in
/tmp/clean_baseline_check.py) and replays ALL 98 Worlds maps from there.

Run:  .venv/bin/python src/worlds2026_rebase.py
Writes (local only, NOTHING pushed):
  model/snapshots.joblib, model/phone_artifacts.json (v1.3.8, corrected)
  model/player_blend_snapshot.json, model/player_ratings.json (as_of 2026-11-14)
  data/processed/player_features_worlds2026.csv
  reports/ratings_patch_1.3.8.md, reports/player_ratings_worlds2026.md,
  reports/worlds2026_blend_validation.json
Backups (created first): model/*.bak-1.3.8-uncorrected for the four files
above. The .bak-1.3.7 files (v1.3.7) are NOT touched.
"""
import json, os, re, shutil, sys
from collections import defaultdict, deque
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 24.0
ROLE_MAP = {"top": "top", "jungle": "jng", "mid": "mid", "bottom": "bot", "support": "sup"}
# True pre-Worlds Elo (Oracle replay, rows with date < 2026-10-15)
TRUE_PRE = {"Karmine Corp": 1676.47, "Cloud9": 1591.91,
            "MVK Esports": 1505.25, "FURIA": 1646.01}
# Expected corrected v1.3.8 (sensitivity table from the first v1.3.8 build)
EXPECTED = {"Karmine Corp": 1707.87, "Cloud9": 1568.18, "MVK Esports": 1540.96,
            "FURIA": 1589.79, "Gen.G": 1807.25, "Bilibili Gaming": 1816.71,
            "T1": 1692.48, "Invictus Gaming": 1641.94}
NEW_SERIES = [  # (match id, date, winners in game order)
    ("115660540726095127", "2026-11-07", ["Gen.G", "Gen.G", "Gen.G"]),
    ("115660540726095121", "2026-11-08",
     ["Bilibili Gaming", "T1", "Bilibili Gaming", "Bilibili Gaming"]),
    ("115660540726095133", "2026-11-14",
     ["Gen.G", "Bilibili Gaming", "Bilibili Gaming", "Bilibili Gaming"]),
]


def elo_exp(diff):
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def backup_uncorrected(path):
    bak = path + ".bak-1.3.8-uncorrected"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  backup: {bak}")
    else:
        print(f"  backup already exists: {bak}")
    return bak


# ---------------------------------------------------------------- load inputs
clean = json.load(open("/tmp/clean_pre_worlds_snapshot.json"))
print("=== Step 0: verify clean pre-Worlds snapshot ===")
for t, v in TRUE_PRE.items():
    got = clean["teams"][t]["team_elo"]
    print(f"  {t}: clean team_elo {got:.2f} vs claimed {v} (diff {got - v:+.3f})")
    assert abs(got - v) <= 0.05, f"clean baseline mismatch for {t}"
# the other teams in the clean replay must match bak-1.3.2 (they did not play Oct 15-16)
bak132 = joblib.load(os.path.join(BASE, "model", "snapshots.joblib.bak-1.3.2"))
for t in ["Gen.G", "Bilibili Gaming", "T1", "Invictus Gaming"]:
    got = clean["teams"][t]["team_elo"]
    bak_v = bak132["elo"][t]
    print(f"  {t}: clean {got:.2f} vs bak-1.3.2 {bak_v:.2f} (diff {got - bak_v:+.3f})")
    assert abs(got - bak_v) <= 0.05, f"non-Play-In baseline mismatch for {t}"
print("  clean baseline VERIFIED")

snaps = joblib.load(os.path.join(BASE, "model", "snapshots.joblib"))
with open(os.path.join(BASE, "model", "phone_artifacts.json")) as f:
    artifacts = json.load(f)
snap_old = clean  # clean pre-Worlds snapshot replaces the Oct-16 one as the start
# blend_params/notes come from the current (uncorrected v1.3.8) snapshot, NOT
# the clean replay (player_elo.replay leaves blend_params None; the fitted
# a=0.5, b=2.0 params live in the Oct-16 snapshot via train_blend.py)
with open(os.path.join(BASE, "model", "player_blend_snapshot.json")) as f:
    _cur_snap = json.load(f)
BLEND_PARAMS = _cur_snap.get("blend_params")
SNAP_NOTES = _cur_snap.get("notes")
assert artifacts["model_version"] == "1.3.8", artifacts["model_version"]
games = json.load(open(os.path.join(BASE, "data", "processed", "worlds2026_lineups.json")))
assert len(games) == 98, len(games)
wf = pd.read_csv(os.path.join(BASE, "reports", "worlds2026_walkforward_maps.csv"))
order = {}
for i, (_, r) in enumerate(wf.iterrows()):
    order[(str(r["series_id"]), int(r["game"]))] = i
for j, (sid, _, _) in enumerate(NEW_SERIES):
    for g in range(1, 6):
        order[(sid, g)] = 1000 + j * 10 + g
games.sort(key=lambda g: order[(g["series_id"], g["game"])])
dates = [g["date"] for g in games]
assert dates == sorted(dates), "games not chronological"

# ------------------------------------------------------- Part A: re-based Elo replay
print("=== Part A: re-based Elo replay, all 98 maps from corrected baseline ===")
elo = dict(bak132["elo"])
for t, v in TRUE_PRE.items():
    print(f"  baseline {t}: bak-1.3.2 {elo[t]:.2f} -> true pre-Worlds {v}")
    elo[t] = v
start = dict(elo)
for g in games:
    a, b, w = g["blue"], g["red"], g["winner"]
    ea, eb = elo[a], elo[b]
    e = elo_exp(ea - eb)
    elo[a] = ea + K * ((1 if w == a else 0) - e)
    elo[b] = eb + K * ((1 if w == b else 0) - (1 - e))
print("  corrected v1.3.8 vs expected:")
for t, tgt in EXPECTED.items():
    d = elo[t] - tgt
    print(f"    {t}: {elo[t]:.2f} (expected {tgt}, diff {d:+.3f})")
    assert abs(d) <= 0.05, f"Elo mismatch for {t}: {elo[t]:.2f} vs {tgt}"
print("  MATCHES expected corrected values within 0.05 on all 8 teams")

# which teams changed vs the uncorrected v1.3.8 currently on disk?
uncorr = dict(snaps["elo"])
changed = {t: (uncorr[t], elo[t]) for t in elo if abs(uncorr.get(t, 0) - elo[t]) > 0.005}
print(f"  teams changed vs uncorrected v1.3.8 ({len(changed)}):")
for t, (old, new) in sorted(changed.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0])):
    print(f"    {t}: {old:.2f} -> {new:.2f} ({new - old:+.2f})")

# backups first, then write
for p in ("model/snapshots.joblib", "model/phone_artifacts.json",
          "model/player_blend_snapshot.json", "model/player_ratings.json"):
    backup_uncorrected(os.path.join(BASE, p))

new_elo = {t: round(elo[t], 2) for t in changed}
snaps_new = dict(snaps)
snaps_new["elo"] = dict(snaps["elo"])
for t, v in new_elo.items():
    snaps_new["elo"][t] = elo[t]
joblib.dump(snaps_new, os.path.join(BASE, "model", "snapshots.joblib"))
print("  wrote model/snapshots.joblib")

gen_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
for t, v in new_elo.items():
    rec = artifacts["teams"][t]
    rec["elo_raw"] = v
    rec["elo"] = round(v + rec["roster_delta"], 2)
artifacts["generated_at"] = gen_at
assert artifacts["model_version"] == "1.3.8"
assert artifacts["snapshot_date"] == "2026-10-16"
with open(os.path.join(BASE, "model", "phone_artifacts.json"), "w") as f:
    json.dump(artifacts, f, indent=2)
print(f"  wrote model/phone_artifacts.json (v1.3.8 corrected, generated_at {gen_at})")
old_snaps = joblib.load(os.path.join(BASE, "model", "snapshots.joblib.bak-1.3.8-uncorrected"))
changed2 = {t for t in elo if abs(old_snaps["elo"][t] - snaps_new["elo"][t]) > 1e-9}
assert changed2 == set(changed), (changed2, set(changed))
print(f"  snapshots diff vs .bak-1.3.8-uncorrected: exactly {sorted(changed2)}")

# ------------------------------------------------- Part B: player-Elo replay (re-based)
print("=== Part B: player-Elo replay through Worlds (98 maps), clean start ===")
from player_elo import _modal_lineup  # noqa: E402

by_name_role = defaultdict(list)
for k, v in snap_old["players"].items():
    _pid, _role = k.rsplit("||", 1)
    by_name_role[(v["name"].lower(), v["role"])].append(_pid)
last_lineup_sets = {t: {k.rsplit("||", 1)[0] for k in r["last_lineup"]}
                    for t, r in snap_old["teams"].items()}


def map_player(team, summoner_name, role):
    name = summoner_name.split(" ", 1)[1].strip() if " " in summoner_name else summoner_name.strip()
    cands = by_name_role.get((name.lower(), role), [])
    if not cands:
        return None, name
    in_lineup = [c for c in cands if c in last_lineup_sets.get(team, set())]
    if len(in_lineup) == 1:
        return in_lineup[0], name
    if len(in_lineup) > 1:
        cands = in_lineup
    return max(cands, key=lambda c: (snap_old["players"][c + "||" + role]["last_seen"], c)), name


def _split_key(s):
    pid, role = s.rsplit("||", 1)
    return pid, role


player_rating = {_split_key(k): float(v["rating"]) for k, v in snap_old["players"].items()}
player_meta = {_split_key(k): (v["name"], v["last_seen"]) for k, v in snap_old["players"].items()}
team_elo = {t: float(r["team_elo"]) for t, r in snap_old["teams"].items()}
last_lineup = {t: [_split_key(k) for k in r["last_lineup"]] for t, r in snap_old["teams"].items()}
last_league = {t: r.get("league") for t, r in snap_old["teams"].items()}

_LINEUP_RE = re.compile(r"([^|]+)\|\|([^|]+)")
pf = pd.read_csv(os.path.join(BASE, "data", "processed", "player_features.csv"),
                 parse_dates=["date"])
pf = pf.sort_values("date")
windows = defaultdict(deque)
for _, r in pf[pf["date"] < "2026-10-15"].iterrows():
    d = r["date"]
    for side in ("blue", "red"):
        team = r[side]
        pairs = _LINEUP_RE.findall(str(r[f"lineup_{side}"]))
        assert len(pairs) == 5, (r["gameid"], side, pairs)
        windows[team].append((d, {pid for pid, _ in pairs}))
for _, r in pf[pf["date"] < "2026-10-15"].iterrows():
    for side in ("blue", "red"):
        team = r[side]
        pairs = _LINEUP_RE.findall(str(r[f"lineup_{side}"]))
        last_lineup[team] = [(pid, role) for pid, role in pairs]

unmapped = []
rows = []
POSITION_ORDER = ["top", "jng", "mid", "bot", "sup"]
offsets = snaps["region_offsets"]
team_region = snaps["team_region"]


def side_pre(team, lineup, t):
    keys = []
    for p in lineup:
        role = ROLE_MAP[p["role"]]
        k, _ = map_player(team, p["summonerName"], role)
        if k is None:
            unmapped.append((t, team, p["summonerName"], role))
            raise ValueError(f"unmapped player {team} {p['summonerName']} {role}")
        keys.append((k, role))
    order = {p: i for i, p in enumerate(POSITION_ORDER)}
    keys = sorted(keys, key=lambda kr: order.get(kr[1], 99))
    rated = [player_rating[k] for k in keys if k in player_rating]
    prior = float(sum(rated) / len(rated)) if rated else 1500.0
    pre = [player_rating.get(k, prior) for k in keys]
    win = windows[team]
    cutoff = t - timedelta(days=30)
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
    inst = 0.0 if ref is None else 1.0 - len(today & set(ref)) / 5.0
    return keys, pre, inst


for g in games:
    t = pd.Timestamp(g["date"])
    blue_team, red_team = g["blue"], g["red"]
    blue_win = 1 if g["winner"] == blue_team else 0
    bk, bp, inst_b = side_pre(blue_team, g["blue_lineup"], t)
    rk, rp, inst_r = side_pre(red_team, g["red_lineup"], t)
    agg_b, agg_r = sum(bp) / 5.0, sum(rp) / 5.0
    eb, er = team_elo[blue_team], team_elo[red_team]
    exp_team = elo_exp(eb - er)
    exp_player = elo_exp(agg_b - agg_r)
    off = offsets.get(team_region.get(blue_team), 0.0) - offsets.get(team_region.get(red_team), 0.0)
    rows.append({
        "gameid": g["game_id"], "date": t, "league": "WLDs",
        "blue": blue_team, "red": red_team,
        "team_elo_blue": eb, "team_elo_red": er,
        "player_elo_blue": agg_b, "player_elo_red": agg_r,
        "team_elo_diff": eb - er, "player_elo_diff": agg_b - agg_r,
        "elo_diff": eb - er, "elo_diff_adj": eb - er + off,
        "eg_gold_diff": snaps["eg_gold"].get(blue_team, 0.0) - snaps["eg_gold"].get(red_team, 0.0),
        "blue_edge": snaps["blue_edge"],
        "instability_blue": inst_b, "instability_red": inst_r,
        "instability_max": max(inst_b, inst_r),
        "lineup_blue": "|".join(f"{k[0]}||{k[1]}" for k in bk),
        "lineup_red": "|".join(f"{k[0]}||{k[1]}" for k in rk),
        "y": blue_win,
    })
    team_elo[blue_team] = eb + K * (blue_win - exp_team)
    team_elo[red_team] = er + K * ((1 - blue_win) - (1 - exp_team))
    for k, pre in zip(bk, bp):
        player_rating[k] = pre + K * (blue_win - exp_player)
    for k, pre in zip(rk, rp):
        player_rating[k] = pre + K * ((1 - blue_win) - (1 - exp_player))
    for lineup, keys in ((g["blue_lineup"], bk), (g["red_lineup"], rk)):
        for p, k in zip(lineup, keys):
            name = p["summonerName"].split(" ", 1)[1].strip()
            prev = player_meta.get(k)
            if prev is None or t > pd.Timestamp(prev[1]):
                player_meta[k] = (name, str(t.date()))
    for team, keys in ((blue_team, bk), (red_team, rk)):
        last_lineup[team] = keys
        last_league[team] = "WLDs"
        windows[team].append((t, {pid for pid, _ in keys}))

print(f"  replayed {len(rows)} games, unmapped players: {len(unmapped)}")
assert len(rows) == 98 and not unmapped
maxdiff = 0.0
for t in EXPECTED:
    d = abs(team_elo[t] - elo[t])
    maxdiff = max(maxdiff, d)
    print(f"    replay team Elo {t}: {team_elo[t]:.2f} vs Part A {elo[t]:.2f} (diff {d:.3f})")
assert maxdiff <= 0.51, maxdiff
print(f"  replay team Elo matches Part A within {maxdiff:.3f} (8 teams)")

t_max = pd.Timestamp("2026-11-14")
players_out = {}
for (pid, role), rating in player_rating.items():
    name, last = player_meta.get((pid, role), ("", "2026-11-14"))
    players_out[f"{pid}||{role}"] = {"name": name, "role": role,
                                     "rating": round(float(rating), 2),
                                     "last_seen": str(last)}
teams_out = {}
for team, lineup in last_lineup.items():
    p_elo = sum(player_rating[k] for k in lineup) / 5.0
    win = windows[team]
    cutoff = t_max - timedelta(days=30)
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
        "last_lineup": [f"{k[0]}||{k[1]}" for k in lineup],
        "modal_lineup": modal,
        "instability": round(float(inst), 3),
    }
snap_new = {"as_of": "2026-11-14", "players": players_out, "teams": teams_out,
            "blend_params": BLEND_PARAMS,
            "notes": SNAP_NOTES}
with open(os.path.join(BASE, "model", "player_blend_snapshot.json"), "w") as f:
    json.dump(snap_new, f, indent=1, sort_keys=True)
print(f"  wrote model/player_blend_snapshot.json "
      f"({len(players_out)} (player, role) ratings, {len(teams_out)} teams)")

by_pid = defaultdict(list)
for (pid, role) in player_rating:
    by_pid[pid].append(role)
pr_players = {}
for pid, roles in by_pid.items():
    best_role, best_date = None, None
    for role in roles:
        _, last = player_meta[(pid, role)]
        if best_date is None or pd.Timestamp(last) > pd.Timestamp(best_date):
            best_role, best_date = role, last
    name, _ = player_meta[(pid, best_role)]
    pr_players[pid] = {"name": name, "role": best_role,
                       "rating": round(float(player_rating[(pid, best_role)]), 2)}
pr_teams = {}
for team, lineup in last_lineup.items():
    p_elo = sum(player_rating[k] for k in lineup) / 5.0
    t_elo = team_elo[team]
    pr_teams[team] = {"player_elo": round(float(p_elo), 2),
                      "team_elo": round(float(t_elo), 2),
                      "gap": round(float(p_elo - t_elo), 2)}
with open(os.path.join(BASE, "model", "player_ratings.json"), "w") as f:
    json.dump({"players": pr_players, "teams": pr_teams}, f, indent=1, sort_keys=True)
print(f"  wrote model/player_ratings.json ({len(pr_players)} players, {len(pr_teams)} teams)")
for t in teams_out:
    assert abs(teams_out[t]["player_elo"] - pr_teams[t]["player_elo"]) < 0.01
    assert abs(teams_out[t]["team_elo"] - pr_teams[t]["team_elo"]) < 0.01
print("  snapshot <-> player_ratings consistency: OK")

pd.DataFrame(rows).to_csv(
    os.path.join(BASE, "data", "processed", "player_features_worlds2026.csv"), index=False)
print("  wrote data/processed/player_features_worlds2026.csv (98 rows)")

for t in ["Gen.G", "Bilibili Gaming", "T1", "Invictus Gaming", "Karmine Corp",
          "Cloud9", "MVK Esports", "FURIA"]:
    r = teams_out[t]
    print(f"  {t}: player_elo {r['player_elo']:.2f} team_elo {r['team_elo']:.2f} "
          f"gap {r['gap']:+.2f}")
for t in ["Bilibili Gaming", "Gen.G"]:
    print(f"  {t} final lineup:")
    for k in last_lineup[t]:
        print(f"    {player_meta[k][0]} ({k[1]}): {player_rating[k]:.2f}")

# ------------------------------------------- Part C: blend re-validation (corrected)
print("=== Part C: strict-cap blend re-validation on Worlds (98 maps, corrected) ===")
from sklearn.metrics import log_loss  # noqa: E402
from train_blend import (OTHER_FEATURES_NODRAFT, blend_feature, fit_lr_eval,  # noqa: E402
                         paired_boot)

feat = pd.read_csv(os.path.join(BASE, "data", "processed", "features.csv"),
                   parse_dates=["date"])
pf_all = pd.read_csv(os.path.join(BASE, "data", "processed", "player_features.csv"),
                     parse_dates=["date"])
df = feat.merge(pf_all[["gameid", "team_elo_blue", "team_elo_red", "player_elo_blue",
                        "player_elo_red", "player_elo_diff", "team_elo_diff",
                        "instability_blue", "instability_red", "instability_max"]],
                on="gameid", how="inner", validate="one_to_one")
fit = df[df["date"] < "2026-07-01"].copy()
print(f"  fit set: {len(fit)} games (< 2026-07-01)")
ev = pd.DataFrame(rows)
ev["date"] = pd.to_datetime(ev["date"])
team_fn = lambda d: d["elo_diff_adj"].to_numpy()
strict_fn = lambda d: blend_feature(d, -4.0, 8.0, 0.0)
primary_fn = lambda d: blend_feature(d, 0.5, 2.0, 0.0)
player_fn = lambda d: d["player_elo_diff"].to_numpy() + (d["elo_diff_adj"] - d["elo_diff"]).to_numpy()

p_m0, info_m0 = fit_lr_eval(fit, ev, team_fn, OTHER_FEATURES_NODRAFT)
p_strict, _ = fit_lr_eval(fit, ev, strict_fn, OTHER_FEATURES_NODRAFT)
p_primary, _ = fit_lr_eval(fit, ev, primary_fn, OTHER_FEATURES_NODRAFT)
p_player, _ = fit_lr_eval(fit, ev, player_fn, OTHER_FEATURES_NODRAFT)
y = ev["y"].to_numpy(int)


def ll(p):
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def per_ll(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def acc(p):
    return float(((p > 0.5).astype(int) == y).mean())


results = {"n_maps": int(len(ev)), "fit_n": int(len(fit)),
           "lr_C_m0": info_m0.get("C"),
           "m0_team_elo": {"logloss": ll(p_m0), "acc": acc(p_m0)},
           "strict_cap_a-4_b8": {"logloss": ll(p_strict), "acc": acc(p_strict)},
           "primary_a0.5_b2": {"logloss": ll(p_primary), "acc": acc(p_primary)},
           "pure_player": {"logloss": ll(p_player), "acc": acc(p_player)}}
for key, p in (("strict_cap_a-4_b8", p_strict), ("primary_a0.5_b2", p_primary),
               ("pure_player", p_player)):
    diff = ll(p) - ll(p_m0)
    lo, hi, pb = paired_boot(per_ll(p) - per_ll(p_m0))
    results[key].update({"diff_vs_m0": float(diff), "ci": [lo, hi], "p_better": float(pb)})
    print(f"  {key}: LL {ll(p):.4f} (M0 {ll(p_m0):.4f}) diff {diff:+.4f} "
          f"CI [{lo:+.4f},{hi:+.4f}] acc {acc(p):.3f} (M0 {acc(p_m0):.3f})")
gate_noninf = results["strict_cap_a-4_b8"]["ci"][1] < 0.002
gate_gain = results["strict_cap_a-4_b8"]["ci"][1] < 0.0
results["strict_cap_a-4_b8"]["gate"] = {
    "noninferiority_ci_upper_lt_0.002": bool(gate_noninf),
    "pooled_gain_ci_excludes_0": bool(gate_gain),
    "both_pass": bool(gate_noninf and gate_gain)}
print(f"  GATE (strict-cap, Worlds n=98, corrected): non-inferiority "
      f"{'PASS' if gate_noninf else 'FAIL'}; pooled gain "
      f"{'PASS' if gate_gain else 'FAIL'}")

bundle = joblib.load(os.path.join(BASE, "model", "model.joblib"))
Xw = np.column_stack([ev["elo_diff_adj"].to_numpy(), ev["eg_gold_diff"].to_numpy(),
                      ev["blue_edge"].to_numpy()])
sc = bundle["scaler_nodraft"]
p_raw_ship = np.clip(bundle["model_nodraft"].base_model.predict_proba(
    sc.transform(Xw))[:, 1], 1e-6, 1 - 1e-6)
p_cal_ship = np.clip(bundle["model_nodraft"].predict_proba(sc.transform(Xw))[:, 1],
                   1e-6, 1 - 1e-6)
results["raw_vs_isotonic_shipped_model"] = {
    "raw_logloss": ll(p_raw_ship), "raw_acc": acc(p_raw_ship),
    "cal_logloss": ll(p_cal_ship), "cal_acc": acc(p_cal_ship),
    "diff_cal_minus_raw": float(ll(p_cal_ship) - ll(p_raw_ship))}
print(f"  shipped model on Worlds 98 (corrected): RAW LL {ll(p_raw_ship):.4f} acc {acc(p_raw_ship):.3f}; "
      f"CAL LL {ll(p_cal_ship):.4f} acc {acc(p_cal_ship):.3f}")
m87 = ev["date"] < "2026-11-07"
y87 = y[m87]
ll87 = lambda p: float(log_loss(y87, np.clip(p[m87], 1e-6, 1 - 1e-6)))
results["raw_vs_isotonic_87maps"] = {
    "raw_logloss": ll87(p_raw_ship), "cal_logloss": ll87(p_cal_ship),
    "raw_acc": float(((p_raw_ship[m87] > 0.5).astype(int) == y87).mean()),
    "cal_acc": float(((p_cal_ship[m87] > 0.5).astype(int) == y87).mean())}
print(f"  87-map subset: RAW LL {ll87(p_raw_ship):.4f} / CAL LL {ll87(p_cal_ship):.4f}")

with open(os.path.join(BASE, "reports", "worlds2026_blend_validation.json"), "w") as f:
    json.dump(results, f, indent=2)
print("  wrote reports/worlds2026_blend_validation.json")

# ------------------------------------------------------------- reports
def team_line(t):
    r = teams_out[t]
    old = clean["teams"][t]
    return (f"| {t} | {old['player_elo']:.2f} | {r['player_elo']:.2f} | "
            f"{old['team_elo']:.2f} | {r['team_elo']:.2f} | "
            f"{old['gap']:+.2f} | {r['gap']:+.2f} |")


changed_rows = "\n".join(
    f"| {t} | {uncorr[t]:.2f} | {elo[t]:.2f} | {elo[t] - uncorr[t]:+.2f} |"
    for t in sorted(changed, key=lambda x: -abs(elo[x] - uncorr[x])))
report = f"""# Ratings patch v1.3.8 — Worlds 2026 closeout (semifinals + final, Nov 7–14)

Elo-only patch (K=24, per map, chronological, no blue-side adjustment, no
offseason regression), same precedent as v1.3.3–v1.3.7. Coefficients, gold@15,
champion power, form and `model.joblib` are unchanged.

## Results applied (verified vs lolesports getEventDetails gameWins)

- Semifinal Nov 7: Gen.G beat Invictus Gaming 3-0 (GEN, GEN, GEN).
- Semifinal Nov 8: Bilibili Gaming beat T1 3-1 (BLG, T1, BLG, BLG).
- Final Nov 14: Bilibili Gaming beat Gen.G 3-1 (GEN, BLG, BLG, BLG).
  Model favourite (GEN, 68.3–69.7% series) lost.

## Re-base (2026-11-14, Thomas approved): the Oct 15–16 double-count is FIXED

The first v1.3.8 build (local only, never pushed) started from
`snapshots.joblib.bak-1.3.2` (as_of 2026-10-16 23:43:16) as its
"pre-Worlds" baseline — but that snapshot already contains the Oct 15–16
Play-In maps, so it applied those 15 maps a second time for the four teams
that played them. The numbers below are the CORRECTED v1.3.8; the first
build's values are superseded (kept in `model/*.bak-1.3.8-uncorrected`).

True pre-Worlds Elo, verified by an independent full-history replay of
Oracle's Elixir rows with date < 2026-10-15 (`src/player_elo.replay`,
29,922 games before Oct 15; the replay also reproduces bak-1.3.2 exactly
for every team that did not play Oct 15–16):

| Team | bak-1.3.2 (post-Oct-16) | True pre-Worlds |
|---|---|---|
| Karmine Corp | 1699.51 | 1676.47 |
| Cloud9 | 1568.19 | 1591.91 |
| MVK Esports | 1529.87 | 1505.25 |
| FURIA | 1622.08 | 1646.01 |

This build replays ALL 98 Worlds maps (Oct 15–Nov 14) from that corrected
baseline, same K=24 per-map rule. The independent player-Elo replay
(`reports/player_ratings_worlds2026.md`) reproduces these team Elos within
0.51.

## Elo changes (corrected v1.3.8 vs the uncorrected first build)

| Team | Uncorrected v1.3.8 | Corrected v1.3.8 | Δ |
|---|---|---|---|
{changed_rows}

BLG stays above GEN: BLG {elo['Bilibili Gaming']:.2f} vs GEN {elo['Gen.G']:.2f}.
All other teams unchanged (verified by diffing `snapshots.joblib` against
its `.bak-1.3.8-uncorrected` — exactly the {len(changed)} teams above).

## Method / validation

- 98 maps replayed from the corrected baseline with K=24 per map,
  chronological. Expected values from the sensitivity analysis of the first
  build: KC 1707.87, C9 1568.18, MVK 1540.96, FURIA 1589.79, GEN 1807.25,
  BLG 1816.71, T1 1692.48, IG 1641.94. This replay matches within 0.05 on
  all eight teams.
- `model/phone_artifacts.json`: `elo_raw` = new raw Elo rounded to 2dp,
  `elo` = `elo_raw` + existing `roster_delta` (deltas unchanged),
  `model_version` 1.3.8, `generated_at` {gen_at}, `snapshot_date` stays
  2026-10-16. Backups: `model/*.bak-1.3.8-uncorrected` (first build),
  `model/*.bak-1.3.7` (v1.3.7).
- predict.py validation (run separately after this script; see below).

## Files changed (local only until pushed)

- `model/snapshots.joblib` — Elo dict, {len(changed)} teams
- `model/phone_artifacts.json` — version 1.3.8 (corrected)
- `model/player_blend_snapshot.json`, `model/player_ratings.json` — re-based
- `reports/ratings_patch_1.3.8.md` — this report (re-written)
- `reports/player_ratings_worlds2026.md`, `reports/worlds2026_blend_validation.json`

## Still blocked

- Full retrain: Oracle's Elixir 2026 CSV download blocked by Google Drive
  quota (`REFRESH_STATUS=noop`). No coefficient / gold@15 / champion-power
  refit.
- `snapshots.joblib` binary not in the remote repo (v1.3.4–v1.3.8 binaries
  are VM-local only); needs PC / browser upload.
- App promotion: NOT done — needs Thomas's explicit approval.
"""
with open(os.path.join(BASE, "reports", "ratings_patch_1.3.8.md"), "w") as f:
    f.write(report)
print("  wrote reports/ratings_patch_1.3.8.md")

blg_line = "\n".join(
    f"| {player_meta[k][0]} ({k[1]}) | {player_rating[k]:.2f} |"
    for k in last_lineup["Bilibili Gaming"])
gen_line = "\n".join(
    f"| {player_meta[k][0]} ({k[1]}) | {player_rating[k]:.2f} |"
    for k in last_lineup["Gen.G"])
pr_report = f"""# Player ratings update through Worlds 2026 (v1.3.8 closeout, RE-BASED)

Incremental player-Elo replay through all 98 Worlds 2026 maps (Oct 15–Nov 14),
starting from the TRUE pre-Worlds snapshot (full-history Oracle replay on
rows with date < 2026-10-15). Same rule as `src/player_elo.py`: K=24, every
player on a side updates by K*(team result − expected team score) where the
expected score uses the PLAYER-aggregate diff (mean of the 5 pre-game player
ratings); debutant prior = mean of rated teammates (1500 if none). No
offseason regression inside the window (no year boundary crossed).

## Re-base (2026-11-14, Thomas approved)

The first build of this update started from the Oct 16 snapshot, whose
team/player ratings already contained the Oct 15–16 Play-In maps — so those
15 maps were counted twice for Karmine Corp, Cloud9, MVK Esports and FURIA.
This version re-bases on the true pre-Worlds snapshot (see
`reports/ratings_patch_1.3.8.md` for the baseline table) and replays the
98 maps from there. The first build's player values are superseded (kept in
`model/*.bak-1.3.8-uncorrected`).

## Sources

- Per-game lineups: livestats window endpoint
  (`feed.lolesports.com/livestats/v1/window/<gameId>`, gameMetadata
  participantMetadata: summonerName, role). Game IDs from lolesports
  getEventDetails per match (x-api-key extracted programmatically from
  `reports/live_ingame.md`). Lineups saved to
  `data/processed/worlds2026_lineups.json` (98 games).
- Winners / blue-red: `reports/worlds2026_walkforward_maps.csv` (87 maps)
  plus the 3 knockout series (11 maps), verified against getEventDetails
  gameWins.
- Mapping: (team, role, bare player name) → Oracle player key in the
  snapshot. Ambiguous names resolved by the team's snapshot last lineup
  (MVK Kratos, G2 SkewMond); "C9 Loki " has a trailing space in the feed.
  **Unmapped players: none (0 of 980 participants).** No debutants: every
  Worlds player already had a snapshot rating.

## Team player Elo (player_elo / team_elo / gap), before → after Worlds

| Team | player_elo pre | player_elo post | team_elo pre | team_elo post | gap pre | gap post |
|---|---|---|---|---|---|---|
{team_line("Gen.G")}
{team_line("Bilibili Gaming")}
{team_line("T1")}
{team_line("Invictus Gaming")}
{team_line("Karmine Corp")}
{team_line("Cloud9")}
{team_line("MVK Esports")}
{team_line("FURIA")}

Team Elo here is the pure replay (true pre-Worlds baseline + 98 Worlds maps);
it reproduces the corrected v1.3.8 patch values within 0.51.

## Final-roster player ratings (post-Worlds)

### Bilibili Gaming (champions)

| Player (role) | Rating |
|---|---|
{blg_line}

### Gen.G

| Player (role) | Rating |
|---|---|
{gen_line}

## Files changed (local only until pushed)

- `model/player_blend_snapshot.json` — as_of 2026-11-14, blend_params
  unchanged
- `model/player_ratings.json` — same schema, updated ratings
- `data/processed/player_features_worlds2026.csv`
- Backups: `model/*.bak-1.3.8-uncorrected` (first build),
  `model/*.bak-1.3.7` (v1.3.7)

## Blend re-validation on fresh Worlds data (98 maps, all tier-1+intl, corrected)

Strict-cap variant (a=-4, b=8), LR refit on the pre-2026-07-01 fit set,
nodraft features, RAW log-loss (see `reports/worlds2026_blend_validation.json`):

| Model | Log-loss | Acc | Diff vs M0 | 95% CI |
|---|---|---|---|---|
| M0 team Elo | {results['m0_team_elo']['logloss']:.4f} | {results['m0_team_elo']['acc']:.3f} | — | — |
| Strict-cap blend | {results['strict_cap_a-4_b8']['logloss']:.4f} | {results['strict_cap_a-4_b8']['acc']:.3f} | {results['strict_cap_a-4_b8']['diff_vs_m0']:+.4f} | [{results['strict_cap_a-4_b8']['ci'][0]:+.4f}, {results['strict_cap_a-4_b8']['ci'][1]:+.4f}] |
| Primary blend (a=0.5, b=2) | {results['primary_a0.5_b2']['logloss']:.4f} | {results['primary_a0.5_b2']['acc']:.3f} | {results['primary_a0.5_b2']['diff_vs_m0']:+.4f} | [{results['primary_a0.5_b2']['ci'][0]:+.4f}, {results['primary_a0.5_b2']['ci'][1]:+.4f}] |
| Pure player Elo | {results['pure_player']['logloss']:.4f} | {results['pure_player']['acc']:.3f} | {results['pure_player']['diff_vs_m0']:+.4f} | [{results['pure_player']['ci'][0]:+.4f}, {results['pure_player']['ci'][1]:+.4f}] |

Gate (strict-cap): non-inferiority (CI upper < +0.002):
{'PASS' if gate_noninf else 'FAIL'}; pooled gain (CI excludes 0):
{'PASS' if gate_gain else 'FAIL'}. n=98, so CIs are wide; do NOT ship on
this alone — Thomas decides.

Raw vs isotonic (shipped `model.joblib`, nodraft, 98 Worlds maps):
RAW log-loss {results['raw_vs_isotonic_shipped_model']['raw_logloss']:.4f}
acc {results['raw_vs_isotonic_shipped_model']['raw_acc']:.3f};
calibrated log-loss {results['raw_vs_isotonic_shipped_model']['cal_logloss']:.4f}
acc {results['raw_vs_isotonic_shipped_model']['cal_acc']:.3f}.
"""
with open(os.path.join(BASE, "reports", "player_ratings_worlds2026.md"), "w") as f:
    f.write(pr_report)
print("  wrote reports/player_ratings_worlds2026.md")
print("ALL RE-BASE STEPS DONE")
