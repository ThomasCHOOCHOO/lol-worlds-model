"""Worlds 2026 Match Predictor — interactive Streamlit app.

Run:
    streamlit run app.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd
import streamlit as st

from predict import (
    ALIASES,
    draft_features,
    load_bundle,
    predict_game,
    resolve_team,
    series_prob,
)

st.set_page_config(page_title="Worlds 2026 Predictor", layout="wide")


@st.cache_resource
def get_bundle():
    bundle, snaps, champ_power = load_bundle()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "metrics.json")) as f:
        metrics = json.load(f)
    return bundle, snaps, champ_power, metrics


bundle, snaps, champ_power, metrics = get_bundle()

TEAMS = sorted(set(ALIASES.values()))
CHAMPS = sorted(snaps["champ_power"].keys())  # display-case names
CHAMP_LOOKUP = {c.lower(): c for c in CHAMPS}


def champ_key(display_name):
    """Display name -> lowercase lookup key used by champ_power."""
    return display_name.strip().lower()


def team_prob(blue_tag, red_tag, blue_picks=None, red_picks=None, blue_bans=None, red_bans=None):
    """Returns (p_game, p_base, draft_dict_or_None, feats)."""
    blue = resolve_team(blue_tag)
    red = resolve_team(red_tag)
    draft = None
    if blue_picks and red_picks and len(blue_picks) == 5 and len(red_picks) == 5:
        draft = draft_features(
            blue, red,
            [champ_key(c) for c in blue_picks], [champ_key(c) for c in red_picks],
            [champ_key(c) for c in (blue_bans or [])], [champ_key(c) for c in (red_bans or [])],
            snaps, champ_power,
        )
    p_base, feats_base = predict_game(blue, red, None, bundle, snaps, champ_power)
    if draft is not None:
        p_game, feats = predict_game(blue, red, draft, bundle, snaps, champ_power)
    else:
        p_game, feats = p_base, feats_base
    return p_game, p_base, draft, feats, blue, red


def fair_game_prob(a_tag, b_tag):
    """Side-neutral pre-draft game prob for a vs b (averages both side assignments)."""
    p_ab, _, _, _, _, _ = team_prob(a_tag, b_tag)
    p_ba, _, _, _, _, _ = team_prob(b_tag, a_tag)
    return (p_ab + (1 - p_ba)) / 2


st.sidebar.title("Worlds 2026 Predictor")
st.sidebar.caption(f"Model snapshots as of {snaps['as_of'][:10]}")
st.sidebar.caption("Logistic regression + isotonic calibration, trained on 2024–2026 pro games (Oracle's Elixir). Re-run `build_features.py` + `train.py` as Worlds games are played to keep ratings fresh.")

tab_predict, tab_ratings, tab_playin, tab_backtest = st.tabs(
    ["Match predictor", "Team ratings", "Play-In", "Backtest"]
)

# ---------------- Match predictor ----------------
with tab_predict:
    st.header("Match predictor")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        blue_tag = st.selectbox("Blue side", TEAMS, index=TEAMS.index("Gen.G"))
    with c2:
        red_tag = st.selectbox("Red side", TEAMS, index=TEAMS.index("T1"))
    with c3:
        best_of = st.radio("Series length", [1, 3, 5], format_func=lambda n: f"Bo{n}" if n > 1 else "Single game",
                           horizontal=True)

    with st.expander("Draft (optional — reprices after champ select)", expanded=False):
        d1, d2 = st.columns(2)
        blue_picks, red_picks = [], []
        with d1:
            st.subheader(f"{blue_tag} picks (blue)")
            for i in range(5):
                blue_picks.append(st.selectbox(f"Blue pick {i+1}", ["—"] + CHAMPS, key=f"bp{i}"))
            blue_bans = st.multiselect("Blue bans", CHAMPS, key="bb", max_selections=5)
        with d2:
            st.subheader(f"{red_tag} picks (red)")
            for i in range(5):
                red_picks.append(st.selectbox(f"Red pick {i+1}", ["—"] + CHAMPS, key=f"rp{i}"))
            red_bans = st.multiselect("Red bans", CHAMPS, key="rb", max_selections=5)

    if st.button("Predict", type="primary"):
        if blue_tag == red_tag:
            st.warning("Pick two different teams.")
            st.stop()
        bp = [c for c in blue_picks if c != "—"]
        rp = [c for c in red_picks if c != "—"]
        has_draft = len(bp) == 5 and len(rp) == 5
        p_game, p_base, draft, feats, blue, red = team_prob(
            blue_tag, red_tag,
            blue_picks=bp or None, red_picks=rp or None,
            blue_bans=blue_bans, red_bans=red_bans,
        )
        st.subheader(f"{blue} (blue) vs {red} (red)")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"P({blue} wins game)", f"{p_game:.1%}")
        if has_draft:
            m2.metric("Pre-draft prob", f"{p_base:.1%}", delta=f"{p_game - p_base:+.1%} draft")
        else:
            m2.metric("Pre-draft prob", f"{p_base:.1%}")
            st.info("Enter all 10 picks to get a draft-adjusted price.")
        if best_of > 1:
            m3.metric(f"P({blue} wins Bo{best_of})", f"{series_prob(p_game, best_of):.1%}")
        else:
            m3.metric("Elo", f"{snaps['elo'].get(blue, 1500):.0f} vs {snaps['elo'].get(red, 1500):.0f}")

        with st.expander("Why — model inputs"):
            st.write(f"Elo diff (blue − red): {feats['elo_diff']:+.0f}")
            st.write(f"Form diff (trailing-20 win rate): {feats['form_diff']:+.3f}")
            st.write(f"Blue-side edge: {feats['blue_edge']:.1%}")
            if has_draft and draft is not None:
                st.write(f"Draft power edge: {feats['d_power']:+.2f}")
                st.write(f"Draft comfort edge: {feats['d_comfort']:+.2f}")
                st.write(f"Ban edge: {feats['ban_edge']:+.2f}")
                raw = feats["d_power"] + feats["d_comfort"] + feats["ban_edge"]
                st.write(f"Draft winner on raw edges: {blue if raw > 0 else red}")
        st.caption("Series odds assume independent games with fixed per-game probability.")

# ---------------- Ratings ----------------
with tab_ratings:
    st.header("Team power ratings")
    rows = []
    for t in TEAMS:
        elo = snaps["elo"].get(t, 1500.0)
        p_avg = 1 / (1 + 10 ** (-(elo - 1500) / 400))
        rows.append({"Team": t, "Elo": round(elo),
                     "P(beat avg team)": f"{p_avg:.1%}",
                     "Form (last 20)": f"{snaps['form'].get(t, 0.5):.1%}"})
    df = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    st.dataframe(df, use_container_width=True)
    st.caption("Elo: 1500 start, K=24, 40% regression to 1500 each offseason. "
               "Note: teams dominating weak regions (TSW, LØS) are Elo-inflated — cross-region games are sparse.")

# ---------------- Play-In ----------------
with tab_playin:
    st.header("Play-In simulator")
    st.caption("4 teams, double-elim Bo5, one advances. Draw is random (after the CBLOL final, Oct 10) — every pairing priced side-neutral.")
    seed_choice = st.radio("CBLOL #2 seed", ["LØS", "FURIA"], horizontal=True)
    seed_team = "LØS" if seed_choice == "LØS" else "FURIA"
    four = ["Cloud9", "Karmine Corp", "MVK Esports", seed_team]
    rows = []
    for i in range(len(four)):
        for j in range(i + 1, len(four)):
            a, b = four[i], four[j]
            p = fair_game_prob(a, b)
            rows.append({"Matchup": f"{a} vs {b}",
                         f"P({a} wins game)": f"{p:.1%}",
                         f"P({a} wins Bo5)": f"{series_prob(p, 5):.1%}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Custom series")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        ca = st.selectbox("Team A", TEAMS, key="ca")
    with cc2:
        cb = st.selectbox("Team B", TEAMS, index=1, key="cb")
    with cc3:
        cbo = st.selectbox("Best of", [1, 3, 5], index=2, key="cbo")
    if ca != cb:
        p = fair_game_prob(ca, cb)
        if cbo == 1:
            st.metric(f"P({ca} beats {cb})", f"{p:.1%}")
        else:
            st.metric(f"P({ca} wins Bo{cbo} vs {cb})", f"{series_prob(p, cbo):.1%}")
    st.caption("Side-neutral: averages both blue/red assignments. Series odds assume independent games.")

# ---------------- Backtest ----------------
with tab_backtest:
    st.header("Backtest — honest numbers")
    m = metrics["full"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2026 holdout accuracy", f"{m['accuracy']:.1%}")
    c2.metric("Log-loss", f"{m['log_loss']:.3f}")
    c3.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
    c4.metric("Holdout games", f"{m['n']:,}")
    st.caption(f"Base rate (blue wins): {m['base_rate']:.1%}")
    cal = pd.DataFrame(m["calibration"])[["mean_pred", "actual_rate", "n"]]
    cal.columns = ["Mean predicted", "Actual win rate", "Games"]
    st.subheader("Calibration (predicted vs actual, by decile)")
    st.dataframe((cal * 1).round(3), use_container_width=True)
    st.subheader("Out-of-sample tournament backtests")
    st.write("- **MSI 2026:** 67.6% accuracy (n=71)")
    st.write("- **Worlds 2025:** 56.2% accuracy (n=96) — weak; small sample, flag it")
    st.write("Draft features add a small real edge (log-loss 0.630 vs 0.633 without draft).")
    st.subheader("Limitations")
    st.write(
        "- ~64% game-level accuracy is realistic for pre-match LoL models; draft is unknowable pre-match and decides a lot.\n"
        "- Worlds patch meta shifts: champion power scores lag the current patch by design (trailing window).\n"
        "- Cross-region games are sparse, so inter-region Elos (esp. LCP/CBLOL teams) are noisy.\n"
        "- Bo1 Swiss games are high-variance; series odds compound the independence assumption."
    )
