"""Predict League of Legends match winners (Worlds 2026 ready).

Pre-draft (no draft info yet):
    python predict.py --blue T1 --red BLG
    python predict.py --blue T1 --red BLG --best-of 5

Draft-adjusted:
    python predict.py --blue T1 --red BLG \\
        --blue-picks "Azir,Vi,Ahri,Kai'Sa,Rakan" \\
        --red-picks "Renekton,Sejuani,Orianna,Ziggs,Nautilus" \\
        --blue-bans "Kalista,Azir,Yone" --red-bans "Vi,Rakan,Neeko"

Batch from CSV (columns: blue,red,best_of,label):
    python predict.py --schedule data/worlds2026_schedule.csv

Works on Windows / macOS / Linux. Team tags are matched case-insensitively;
use --list-teams to see every accepted alias.
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np

from model_utils import IsotonicCalibrated  # noqa: F401  (needed to unpickle model.joblib)
from region_offsets import load_offsets

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "model")

# tag -> exact Oracle's Elixir teamname (verified against the data)
ALIASES = {
    "gen": "Gen.G", "geng": "Gen.G", "gen.g": "Gen.G",
    "hle": "Hanwha Life Esports", "hanwha": "Hanwha Life Esports",
    "hanwha life": "Hanwha Life Esports", "hanwha life esports": "Hanwha Life Esports",
    "t1": "T1",
    "dk": "Dplus Kia", "dplus": "Dplus Kia", "dplus kia": "Dplus Kia",
    "al": "Anyone's Legend", "anyone's legend": "Anyone's Legend",
    "anyones legend": "Anyone's Legend",
    "blg": "Bilibili Gaming", "bilibili": "Bilibili Gaming",
    "bilibili gaming": "Bilibili Gaming",
    "tes": "Top Esports", "top esports": "Top Esports", "top": "Top Esports",
    "ig": "Invictus Gaming", "invictus": "Invictus Gaming",
    "invictus gaming": "Invictus Gaming",
    "g2": "G2 Esports", "g2 esports": "G2 Esports",
    "mkoi": "Movistar KOI", "movistar koi": "Movistar KOI", "koi": "Movistar KOI",
    "mkoy": "Movistar KOI",
    "kc": "Karmine Corp", "karmine": "Karmine Corp", "karmine corp": "Karmine Corp",
    "kcorp": "Karmine Corp",
    "tl": "Team Liquid", "tla": "Team Liquid", "tlaw": "Team Liquid",
    "team liquid": "Team Liquid", "team liquid alienware": "Team Liquid",
    "lyon": "LYON",
    "c9": "Cloud9", "cloud9": "Cloud9", "cloud 9": "Cloud9",
    "tsw": "Team Secret Whales", "team secret whales": "Team Secret Whales",
    "secret whales": "Team Secret Whales",
    "cfo": "CTBC Flying Oyster", "ctbc flying oyster": "CTBC Flying Oyster",
    "flying oyster": "CTBC Flying Oyster",
    "mvk": "MVK Esports", "mvk esports": "MVK Esports",
    "los": "LØS", "l0s": "LØS", "lØs": "LØS",
    "fur": "FURIA", "furia": "FURIA",
}


def resolve_team(tag):
    key = tag.strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Unknown team '{tag}'. Run --list-teams for valid tags.")
    return ALIASES[key]


def load_bundle():
    bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    snaps = joblib.load(os.path.join(MODEL_DIR, "snapshots.joblib"))
    # case-insensitive champion lookup; per-patch tables (patch-aware power)
    patch_tables = {
        p: {k.lower(): v for k, v in tbl.items()}
        for p, tbl in snaps.get("champ_power_by_patch", {}).items()
    }
    snaps["patch_tables"] = patch_tables
    cur = snaps.get("current_patch")
    champ_power = patch_tables.get(cur, {})
    if not champ_power:  # very old snapshots without per-patch tables
        champ_power = {k.lower(): v for k, v in snaps["champ_power"].items()}
    # team_champ_comfort keys are "team||Champion" (original case) but lookups
    # use lowercase champion keys (see draft_features) -- normalize so lookups
    # hit, matching how the model was trained. (Without this, every lookup
    # missed and fell back to the team's form rate.)
    snaps["team_champ_comfort"] = {
        f"{t}||{c.lower()}": v
        for k, v in snaps["team_champ_comfort"].items()
        for t, c in (k.rsplit("||", 1),)
    }
    # region offsets: snapshots carry them (build_features.py); older
    # snapshots fall back to model/region_offsets.json (canonical file).
    if "region_offsets" not in snaps or "team_region" not in snaps:
        _off, _tr = load_offsets()
        snaps.setdefault("region_offsets", _off)
        snaps.setdefault("team_region", _tr)
    return bundle, snaps, champ_power


def champ_table_for(snaps, patch):
    """Lowercase champion-power table for a patch (default: current patch)."""
    tables = snaps.get("patch_tables", {})
    if patch is None:
        patch = snaps.get("current_patch")
    if patch not in tables:
        avail = ", ".join(sorted(tables))
        raise ValueError(f"Unknown patch '{patch}'. Available: {avail}")
    return tables[patch], patch


def norm_champ(name, champ_power):
    key = name.strip().lower()
    if key in champ_power:
        return key
    # fuzzy: try without apostrophes/spaces
    alt = key.replace("'", "").replace(" ", "")
    for k in champ_power:
        if k.replace("'", "").replace(" ", "") == alt:
            return k
    return None


def parse_champs(s, champ_power, what):
    if not s:
        return []
    out, unknown = [], []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        key = norm_champ(tok, champ_power)
        if key is None:
            unknown.append(tok)
        else:
            out.append(key)
    if unknown:
        print(f"  warning: unknown {what} ignored: {', '.join(unknown)}", file=sys.stderr)
    return out


def draft_features(blue, red, blue_picks, red_picks, blue_bans, red_bans, snaps, champ_power):
    fb = snaps["form"].get(blue, 0.5)
    fr = snaps["form"].get(red, 0.5)
    tc = snaps["team_champ_comfort"]

    def power(champs):
        return sum(champ_power.get(c, 0.5) for c in champs)

    def comfort(team, champs, f):
        return sum(tc.get(f"{team}||{c}", f) for c in champs)

    d_power = power(blue_picks) - power(red_picks)
    d_comfort = comfort(blue, blue_picks, fb) - comfort(red, red_picks, fr)
    ban_edge = power(blue_bans) - power(red_bans)
    return {"d_power": d_power, "d_comfort": d_comfort, "ban_edge": ban_edge}


def predict_game(blue, red, draft, bundle, snaps, champ_power):
    # region-strength adjustment (src/region_offsets.py): the model trains
    # on elo_diff_adj; unknown region -> offset 0.
    elo_diff = snaps["elo"].get(blue, 1500.0) - snaps["elo"].get(red, 1500.0)
    offsets = snaps.get("region_offsets") or {}
    team_region = snaps.get("team_region") or {}
    reg_b = team_region.get(blue)
    reg_r = team_region.get(red)
    off_b = offsets.get(reg_b, 0.0) if reg_b else 0.0
    off_r = offsets.get(reg_r, 0.0) if reg_r else 0.0
    feats = {
        "elo_diff": elo_diff,
        "elo_diff_adj": elo_diff + off_b - off_r,
        "region_adj": off_b - off_r,
        "off_b": off_b,
        "off_r": off_r,
        "region_blue": reg_b,
        "region_red": reg_r,
        "eg_gold_diff": snaps["eg_gold"].get(blue, 0.0) - snaps["eg_gold"].get(red, 0.0),
        "blue_edge": snaps["blue_edge"],
    }
    if draft is None:
        model = bundle["model_nodraft"]
        scaler = bundle["scaler_nodraft"]
        names = bundle["features_nodraft"]
    else:
        feats.update(draft)
        model = bundle["model_full"]
        scaler = bundle["scaler_full"]
        names = bundle["features_full"]
    X = scaler.transform(np.array([[feats[n] for n in names]]))
    p = float(model.predict_proba(X)[0, 1])
    return p, feats


def series_prob(p_game, best_of):
    """P(blue wins a best-of-N series). Assumes independent games with fixed p."""
    need = best_of // 2 + 1
    return sum(math.comb(best_of, k) * p_game ** k * (1 - p_game) ** (best_of - k)
               for k in range(need, best_of + 1))


def show_prediction(blue_tag, red_tag, best_of, draft_args, bundle, snaps, champ_power,
                    label=None, patch=None):
    blue = resolve_team(blue_tag)
    red = resolve_team(red_tag)
    champ_power, patch_used = champ_table_for(snaps, patch)
    draft = None
    if draft_args:
        bp = parse_champs(draft_args.get("blue_picks"), champ_power, "blue picks")
        rp = parse_champs(draft_args.get("red_picks"), champ_power, "red picks")
        bb = parse_champs(draft_args.get("blue_bans"), champ_power, "blue bans")
        rb = parse_champs(draft_args.get("red_bans"), champ_power, "red bans")
        draft = draft_features(blue, red, bp, rp, bb, rb, snaps, champ_power)

    p_base, feats_base = predict_game(blue, red, None, bundle, snaps, champ_power)
    lines = []
    if label:
        lines.append(f"### {label}")
    lines.append(f"{blue} (blue) vs {red} (red)  [patch {patch_used}]")
    lines.append(f"  Elo: {snaps['elo'].get(blue, 1500):.0f} vs {snaps['elo'].get(red, 1500):.0f} "
                 f"(diff {feats_base['elo_diff']:+.0f})")
    if feats_base["region_adj"]:
        lines.append(f"  Region adjustment: {feats_base['region_adj']:+.0f} "
                     f"({feats_base['region_blue']} {feats_base.get('off_b', 0):+.0f} vs "
                     f"{feats_base['region_red']} {feats_base.get('off_r', 0):+.0f}) "
                     f"-> adjusted Elo diff {feats_base['elo_diff_adj']:+.0f}")
    lines.append(f"  Pre-draft:  P({blue} wins game) = {p_base:.1%}")
    if draft is not None:
        p_adj, feats = predict_game(blue, red, draft, bundle, snaps, champ_power)
        lines.append(f"  Draft-adjusted: P({blue} wins game) = {p_adj:.1%} "
                     f"({p_adj - p_base:+.1%} vs pre-draft)")
        lines.append(f"  Draft edges (blue - red): power {feats['d_power']:+.2f}, "
                     f"comfort {feats['d_comfort']:+.2f}, bans {feats['ban_edge']:+.2f}")
        winner = blue if (feats["d_power"] + feats["d_comfort"] + feats["ban_edge"]) > 0 else red
        lines.append(f"  Draft winner on raw edges: {winner}")
        p_game = p_adj
    else:
        p_game = p_base
    if best_of and best_of > 1:
        ps = series_prob(p_game, best_of)
        lines.append(f"  P({blue} wins Bo{best_of} series) = {ps:.1%} "
                     f"(independent-games assumption)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="LoL Worlds 2026 match predictor")
    ap.add_argument("--blue", help="blue-side team tag")
    ap.add_argument("--red", help="red-side team tag")
    ap.add_argument("--best-of", type=int, default=1, choices=[1, 3, 5])
    ap.add_argument("--blue-picks", default=None)
    ap.add_argument("--red-picks", default=None)
    ap.add_argument("--blue-bans", default=None)
    ap.add_argument("--red-bans", default=None)
    ap.add_argument("--schedule", default=None, help="CSV with blue,red,best_of,label columns")
    ap.add_argument("--patch", default=None,
                    help="patch for champion power (default: current patch)")
    ap.add_argument("--list-patches", action="store_true",
                    help="list patches with champion-power tables")
    ap.add_argument("--list-teams", action="store_true")
    args = ap.parse_args(argv)

    bundle, snaps, champ_power = load_bundle()

    if args.list_patches:
        print(f"current patch: {snaps.get('current_patch')}")
        for p in sorted(snaps.get("patch_tables", {})):
            print(f"  {p} ({len(snaps['patch_tables'][p])} champions)")
        return

    if args.list_teams:
        seen = sorted(set(ALIASES.values()))
        print("Worlds 2026 teams (accepted tags in parentheses):")
        for t in seen:
            tags = sorted(k for k, v in ALIASES.items() if v == t)
            print(f"  {t}  ({', '.join(tags)})")
        return

    bundle, snaps, champ_power = load_bundle()
    print(f"# model snapshots as of {snaps['as_of'][:10]}")

    draft_args = None
    if args.blue_picks or args.red_picks or args.blue_bans or args.red_bans:
        draft_args = {"blue_picks": args.blue_picks, "red_picks": args.red_picks,
                      "blue_bans": args.blue_bans, "red_bans": args.red_bans}

    if args.schedule:
        with open(args.schedule, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            bo = int(r.get("best_of") or 1)
            print(show_prediction(r["blue"], r["red"], bo, draft_args,
                                  bundle, snaps, champ_power,
                                  label=r.get("label"), patch=args.patch))
            print()
    elif args.blue and args.red:
        print(show_prediction(args.blue, args.red, args.best_of, draft_args,
                              bundle, snaps, champ_power, patch=args.patch))
    else:
        ap.error("provide --blue and --red, or --schedule")


if __name__ == "__main__":
    main()
