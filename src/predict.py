"""Predict League of Legends match winners (Worlds 2026 ready).

Pre-draft (no draft info yet):
    python predict.py --blue T1 --red BLG
    python predict.py --blue T1 --red BLG --best-of 5

Tier-conditional player/team Elo blend (offline validation only; the
shipped model is team Elo -- see reports/tier_conditional_blend.md):
    python predict.py --blue T1 --red BLG --elo-blend
    python predict.py --blue T1 --red BLG --elo-blend \\
        --blue-lineup "Zeus,Oner,Faker,Gumayusi,Keria"
Requires model/model_blend.joblib (built by src/train_blend.py) and
model/player_blend_snapshot.json (built by src/build_player_features.py).
Lineups default to each team's last-seen five; --blue-lineup/--red-lineup
override with exactly 5 player names each (matched case-insensitively).

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
import json
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


def load_blend():
    """Load the blend bundle + player snapshot for --elo-blend."""
    blend_path = os.path.join(MODEL_DIR, "model_blend.joblib")
    snap_path = os.path.join(MODEL_DIR, "player_blend_snapshot.json")
    if not os.path.exists(blend_path):
        raise SystemExit("model_blend.joblib not found: run src/train_blend.py "
                         "to fit the blend and build the bundle")
    if not os.path.exists(snap_path):
        raise SystemExit("player_blend_snapshot.json not found: run "
                         "src/build_player_features.py to build the snapshot")
    bundle_blend = joblib.load(blend_path)
    with open(snap_path) as f:
        snap = json.load(f)
    return bundle_blend, snap


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def resolve_lineup(team, lineup_names, snap):
    """Resolve a team's five to snapshot player keys.

    lineup_names=None -> the team's last-seen five from the snapshot.
    Otherwise exactly 5 names, matched case-insensitively against snapshot
    player names; a name matching several (player, role) entries resolves
    to the most recently seen one."""
    teams = snap["teams"]
    if team not in teams:
        raise ValueError(f"Unknown team '{team}' in the blend snapshot")
    if lineup_names is None:
        return list(teams[team]["last_lineup"])
    names = [n.strip() for n in lineup_names.split(",") if n.strip()]
    if len(names) != 5:
        raise ValueError(f"--lineup for {team} must have exactly 5 names, "
                         f"got {len(names)}")
    by_name = {}
    for key, rec in snap["players"].items():
        by_name.setdefault(rec["name"].lower(), []).append((rec["last_seen"], key))
    out = []
    for n in names:
        cands = by_name.get(n.lower())
        if not cands:
            raise ValueError(f"Player '{n}' not found in the blend snapshot")
        out.append(max(cands)[1])
    return out


def blend_side(team, lineup_names, bundle_blend, snap, team_elo):
    """Per-side blended rating for --elo-blend.

    lineup = last-seen five (or --lineup override); instability = 1 - overlap
    of that five with the team's trailing-30-day modal lineup (from the
    snapshot for the last-seen five; recomputed for an override);
    w = sigmoid(a + b*instability + tier_c*is_tier1), where is_tier1 is 1
    when the team's last game was a tier-1 or international league;
    blended = w*player + (1-w)*team. team_elo comes from the LIVE snapshot
    (snapshots.joblib, which carries Elo patches); the blend snapshot's
    own team_elo is the pure replay and would be stale after a patch."""
    keys = resolve_lineup(team, lineup_names, snap)
    players = snap["players"]
    ratings = [players[k]["rating"] for k in keys]
    player_elo = sum(ratings) / 5.0
    if lineup_names is None:
        inst = snap["teams"][team]["instability"]
    else:
        modal = set(snap["teams"][team]["modal_lineup"])
        today = {k.rsplit("||", 1)[0] for k in keys}
        inst = 1.0 - len(today & modal) / 5.0
    a = bundle_blend["blend_a"]
    b = bundle_blend["blend_b"]
    tier_c = bundle_blend.get("blend_tier_c", 0.0)
    is_t1 = 1.0 if snap["teams"][team].get("league") in (
        "LCK", "LPL", "LEC", "LCS", "LTA", "LTA N", "LTA S", "LCP",
        "MSI", "WLDs", "EWC") else 0.0
    w = _sigmoid(a + b * inst + tier_c * is_t1)
    return {"team": team, "lineup": keys, "player_elo": player_elo,
            "team_elo": team_elo, "instability": inst, "w": w,
            "blended": w * player_elo + (1 - w) * team_elo}


def predict_game_blend(blue, red, draft, bundle_blend, snap, snaps,
                       blue_lineup=None, red_lineup=None):
    """Pre-draft/draft prediction with the blend feature (mirrors predict_game).

    The blend bundle's feature list names elo_diff_blend; it is fed the
    blended diff + the same region-offset diff the shipped model uses.
    All other features (eg_gold, blue_edge, draft) come from snapshots."""
    sb = blend_side(blue, blue_lineup, bundle_blend, snap,
                    snaps["elo"].get(blue, 1500.0))
    sr = blend_side(red, red_lineup, bundle_blend, snap,
                    snaps["elo"].get(red, 1500.0))
    offsets = snaps.get("region_offsets") or {}
    team_region = snaps.get("team_region") or {}
    reg_b, reg_r = team_region.get(blue), team_region.get(red)
    off_b = offsets.get(reg_b, 0.0) if reg_b else 0.0
    off_r = offsets.get(reg_r, 0.0) if reg_r else 0.0
    elo_diff_blend = sb["blended"] - sr["blended"] + off_b - off_r
    feats = {
        "elo_diff_blend": elo_diff_blend,
        "elo_diff": sb["team_elo"] - sr["team_elo"],
        "elo_diff_adj": sb["team_elo"] - sr["team_elo"] + off_b - off_r,
        "region_adj": off_b - off_r,
        "eg_gold_diff": snaps["eg_gold"].get(blue, 0.0) - snaps["eg_gold"].get(red, 0.0),
        "blue_edge": snaps["blue_edge"],
        "blend_blue": sb, "blend_red": sr,
    }
    if draft is None:
        model = bundle_blend["model_nodraft"]
        scaler = bundle_blend["scaler_nodraft"]
        names = bundle_blend["features_nodraft"]
    else:
        feats.update(draft)
        model = bundle_blend["model_full"]
        scaler = bundle_blend["scaler_full"]
        names = bundle_blend["features_full"]
    X = scaler.transform(np.array([[feats[n] for n in names]]))
    p = float(model.predict_proba(X)[0, 1])
    return p, feats


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
                    label=None, patch=None, blend=None):
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

    if blend is not None:
        bundle_blend, snap, lineups = blend
        p_base, feats_base = predict_game_blend(
            blue, red, None, bundle_blend, snap, snaps,
            blue_lineup=lineups.get("blue"), red_lineup=lineups.get("red"))
    else:
        p_base, feats_base = predict_game(blue, red, None, bundle, snaps, champ_power)
    lines = []
    if label:
        lines.append(f"### {label}")
    lines.append(f"{blue} (blue) vs {red} (red)  [patch {patch_used}]")
    if blend is not None:
        for side, rec in (("blue", feats_base["blend_blue"]),
                          ("red", feats_base["blend_red"])):
            names = [snap["players"][k]["name"] for k in rec["lineup"]]
            lines.append(f"  {side} lineup ({', '.join(names)}): "
                         f"player Elo {rec['player_elo']:.0f}, "
                         f"team Elo {rec['team_elo']:.0f}, "
                         f"instability {rec['instability']:.2f} -> "
                         f"w={rec['w']:.2f}, blended {rec['blended']:.0f}")
        lines.append(f"  Blended Elo diff: {feats_base['elo_diff_blend']:+.0f} "
                     f"(team-only diff {feats_base['elo_diff']:+.0f}, "
                     f"region adj {feats_base['region_adj']:+.0f})")
    else:
        lines.append(f"  Elo: {snaps['elo'].get(blue, 1500):.0f} vs {snaps['elo'].get(red, 1500):.0f} "
                     f"(diff {feats_base['elo_diff']:+.0f})")
        if feats_base["region_adj"]:
            lines.append(f"  Region adjustment: {feats_base['region_adj']:+.0f} "
                         f"({feats_base['region_blue']} {feats_base.get('off_b', 0):+.0f} vs "
                         f"{feats_base['region_red']} {feats_base.get('off_r', 0):+.0f}) "
                         f"-> adjusted Elo diff {feats_base['elo_diff_adj']:+.0f}")
    lines.append(f"  Pre-draft:  P({blue} wins game) = {p_base:.1%}")
    if draft is not None:
        if blend is not None:
            p_adj, feats = predict_game_blend(
                blue, red, draft, blend[0], blend[1], snaps,
                blue_lineup=blend[2].get("blue"), red_lineup=blend[2].get("red"))
        else:
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
    ap.add_argument("--elo-blend", action="store_true",
                    help="use the tier-conditional player/team Elo blend "
                         "(offline validation only; shipped model is team Elo)")
    ap.add_argument("--blend-a", type=float, default=None,
                    help="with --elo-blend: override the fitted a "
                         "(default: bundle value; e.g. -4 for the strict-cap "
                         "sensitivity variant, see tier_conditional_blend.md)")
    ap.add_argument("--blend-b", type=float, default=None,
                    help="with --elo-blend: override the fitted b (e.g. 8)")
    ap.add_argument("--blend-tier-c", type=float, default=None,
                    help="with --elo-blend: override the fitted tier_c")
    ap.add_argument("--blue-lineup", default=None,
                    help="with --elo-blend: 5 comma-separated player names for blue")
    ap.add_argument("--red-lineup", default=None,
                    help="with --elo-blend: 5 comma-separated player names for red")
    ap.add_argument("--list-patches", action="store_true",
                    help="list patches with champion-power tables")
    ap.add_argument("--list-teams", action="store_true")
    args = ap.parse_args(argv)

    bundle, snaps, champ_power = load_bundle()
    blend = None
    if args.elo_blend:
        bundle_blend, snap = load_blend()
        if args.blend_a is not None:
            bundle_blend = dict(bundle_blend)
            bundle_blend["blend_a"] = args.blend_a
        if args.blend_b is not None:
            bundle_blend = dict(bundle_blend)
            bundle_blend["blend_b"] = args.blend_b
        if args.blend_tier_c is not None:
            bundle_blend = dict(bundle_blend)
            bundle_blend["blend_tier_c"] = args.blend_tier_c
        blend = (bundle_blend, snap,
                 {"blue": args.blue_lineup, "red": args.red_lineup})
        print(f"# blend: w = sigmoid({bundle_blend['blend_a']:.3f} + "
              f"{bundle_blend['blend_b']:.3f} * instability + "
              f"{bundle_blend.get('blend_tier_c', 0.0):.3f} * is_tier1) "
              f"[snapshot as of {snap['as_of'][:10]}]")

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
                                  label=r.get("label"), patch=args.patch,
                                  blend=blend))
            print()
    elif args.blue and args.red:
        print(show_prediction(args.blue, args.red, args.best_of, draft_args,
                              bundle, snaps, champ_power, patch=args.patch,
                              blend=blend))
    else:
        ap.error("provide --blue and --red, or --schedule")


if __name__ == "__main__":
    main()
