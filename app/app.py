"""PitWall dashboard.  Run from the repo root:  streamlit run app/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from model import load_all  # noqa: E402
from predict import plan_curves, stint_recommendation  # noqa: E402

st.set_page_config(page_title="PitWall", layout="wide")
BLUE, ORANGE, GREEN, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#9a988f", "#1f1f1f"
COMP_COLOR = {"SOFT": BLUE, "MEDIUM": ORANGE, "HARD": GREEN}
DEMO = "2025_02_NOR_1"     # default view: Norris, Chinese GP, first stint
LAYOUT = dict(template="simple_white", height=420, margin=dict(l=10, r=10, t=50, b=10),
              font=dict(family="Arial", size=13),
              legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
              hovermode="x unified")

ctx = load_all()
feat, stints = ctx["feat"], ctx["stints"]

st.title("PitWall")
st.caption("Predicting Formula 1 tire degradation to recommend pit stop timing. "
           "EM 538 Team 9. Models trained on 2023 and 2024; everything below is the unseen 2025 season.")

tab1, tab2, tab3 = st.tabs(["Pit window (2025 replay)", "Plan a stint", "Model results"])

# ------------------------------------------------------------------ tab 1
with tab1:
    test = feat[(feat["split"] == "test") & feat["in_forecast"]]
    usable = stints[stints["stint_id"].isin(test["stint_id"].unique())]
    c1, c2, c3, c4 = st.columns(4)
    races = usable.sort_values("race_id")[["race_id", "event_name"]].drop_duplicates()
    demo_race, demo_driver = DEMO.split("_")[0] + "_" + DEMO.split("_")[1], DEMO.split("_")[2]
    race_ids = list(races["race_id"])
    race = c1.selectbox("Race", race_ids, index=race_ids.index(demo_race) if demo_race in race_ids else 0,
                        format_func=lambda r: races.set_index("race_id").loc[r, "event_name"])
    drivers = sorted(usable[usable["race_id"] == race]["driver"].unique())
    driver = c2.selectbox("Driver", drivers, index=drivers.index(demo_driver) if demo_driver in drivers else 0)
    opts = usable[(usable["race_id"] == race) & (usable["driver"] == driver)].sort_values("stint")
    sid = c3.selectbox("Stint", opts["stint_id"], format_func=lambda s: (
        f"Stint {int(opts.set_index('stint_id').loc[s, 'stint'])}: "
        f"{opts.set_index('stint_id').loc[s, 'compound'].title()}"))
    tB = ctx["tuned"]["B"]
    best_b = min(tB, key=lambda m: tB[m]["cv_val_mae"])
    models = [best_b] + [m for m in ["Ridge", "Random forest", "Gradient boosting", "Linear baseline"] if m != best_b]
    if ctx["lstm_fn"] is not None:
        models.insert(2, "LSTM")
    model_name = c4.selectbox("Model for current tires", models)
    info = opts.set_index("stint_id").loc[sid]
    default_next = info["next_compound"] if isinstance(info["next_compound"], str) else "HARD"
    nxt = st.radio("Next compound", ["SOFT", "MEDIUM", "HARD"], horizontal=True,
                   index=["SOFT", "MEDIUM", "HARD"].index(default_next),
                   format_func=str.title)

    r = stint_recommendation(ctx, sid, model_name, nxt)
    win = r["table"][r["table"]["in_window"]]["pit_lap"]
    pl = ctx["pit_loss"]
    loss = pl[pl["circuit"] == r["info"]["circuit"]]["pit_loss_s"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision point", f"Lap {int(r['state']['lap_number'])}")
    m2.metric("Recommended pit lap", f"Lap {r['best']}")
    m3.metric("Pit window (within 1 s)", f"Laps {win.min()} to {win.max()}")
    m4.metric("Team actually pitted", f"Lap {r['actual_pit']}" if r["actual_pit"] else "No stop",
              delta=(("same lap as ours" if r["best"] == r["actual_pit"] else
                      f"{abs(r['best'] - r['actual_pit'])} laps {'earlier' if r['actual_pit'] < r['best'] else 'later'} than ours")
                     if r["actual_pit"] else None),
              delta_color="off")

    obs = r["laps_obs"]
    seen = obs.iloc[:5]
    later = obs.iloc[5:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=later["lap_number"], y=later["deg_s"], mode="markers", name="What actually happened",
                             marker=dict(color=GREY, size=7, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=seen["lap_number"], y=seen["deg_s"], mode="markers", name="Laps seen so far",
                             marker=dict(color=INK, size=8)))
    fig.add_trace(go.Scatter(x=r["old_curve"]["lap"], y=r["old_curve"]["loss"], mode="lines",
                             name=f"Current {r['info']['compound'].title()}s ({model_name})",
                             line=dict(color=COMP_COLOR[r["info"]["compound"]], width=3)))
    if len(r["new_curve"]):
        fig.add_trace(go.Scatter(x=r["new_curve"]["lap"], y=r["new_curve"]["loss"], mode="lines",
                                 name=f"New {nxt.title()}s after lap {r['best']}",
                                 line=dict(color=COMP_COLOR[nxt], width=3, dash="dash")))
    fig.add_vrect(x0=win.min() - 0.5, x1=win.max() + 0.5, fillcolor="#e8eef7", line_width=0, layer="below")
    fig.add_vline(x=r["best"], line=dict(color=INK, width=1.5))
    if r["actual_pit"]:
        fig.add_vline(x=r["actual_pit"], line=dict(color=GREY, width=1.5, dash="dot"),
                      annotation_text="actual stop", annotation_position="bottom right")
    fig.add_annotation(x=r["best"], y=1, yref="paper", text="recommended", showarrow=False,
                       xanchor="left", yanchor="top", xshift=4, font=dict(size=12))
    fig.update_layout(**LAYOUT, title="Lap time loss vs. stint start (s)",
                      xaxis_title="Race lap", yaxis_title="seconds")
    st.plotly_chart(fig, use_container_width=True)

    t = r["table"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t["pit_lap"], y=t["cost_s"] - t["cost_s"].min(), mode="lines+markers",
                              name="Extra time vs best choice", line=dict(color=BLUE, width=2.5),
                              marker=dict(size=5)))
    fig2.add_vrect(x0=win.min() - 0.5, x1=win.max() + 0.5, fillcolor="#e8eef7", line_width=0, layer="below")
    if r["actual_pit"]:
        fig2.add_vline(x=r["actual_pit"], line=dict(color=GREY, width=1.5, dash="dot"))
    fig2.update_layout(**{**LAYOUT, "height": 300}, title="Time lost by pitting on each lap, compared with the best lap (s)",
                       xaxis_title="Pit at the end of lap", yaxis_title="seconds", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Pit lane time at {r['info']['circuit']} is about "
               f"{float(loss.iloc[0]):.0f} s (2023 to 2024 median). It is the same whichever lap you stop on, "
               f"so it does not move the recommendation. Compound pace offset "
               f"{r['info']['compound'].title()} to {nxt.title()}: {r['offset']:+.2f} s per lap. "
               "Future laps assume today's gap, track temperature and position, since that is all "
               "the pit wall knows at the decision point.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown("Before a stint starts, only conditions are known (Setup A). Pick a track and "
                "conditions to see the expected loss curve for each compound.")
    c1, c2, c3, c4 = st.columns(4)
    circuit = c1.selectbox("Circuit", sorted(feat["circuit"].unique()))
    team = c2.selectbox("Team", sorted(feat["team"].unique()))
    plan_model = c3.selectbox("Model", ["Ridge", "Gradient boosting", "Random forest", "Linear baseline"])
    n = c4.slider("Stint length (laps)", 10, 40, 25)
    pc = plan_curves(ctx, circuit, n, team, plan_model)
    fig3 = go.Figure()
    for comp, g in pc.groupby("compound"):
        fig3.add_trace(go.Scatter(x=g["stint_lap"], y=g["loss"], mode="lines", name=comp.title(),
                                  line=dict(color=COMP_COLOR[comp], width=3)))
    fig3.update_layout(**LAYOUT, title=f"Expected lap time loss on a new set ({plan_model}, Setup A)",
                       xaxis_title="Laps on the set", yaxis_title="seconds")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Conditions alone explain only about 20% of the variance on 2025 (R² 0.2), "
               "which is why the pit window tab waits until it has seen the first laps on the set. "
               "Tree models can treat two compounds the same at a given track when the data "
               "doesn't separate them; the linear models always keep a separate wear rate per compound.")

# ------------------------------------------------------------------ tab 3
with tab3:
    tr = ctx["test_results"]
    for setup, label in [("B", "Setup B: conditions + first 5 laps of the stint"),
                         ("A", "Setup A: conditions only")]:
        d = tr[(tr["setup"] == setup) & ~tr["model"].str.startswith("LSTM (")].copy()
        lstm_rows = tr[(tr["setup"] == setup) & tr["model"].str.startswith("LSTM (")]
        if len(lstm_rows):
            mean_row = {"setup": setup, "model": "LSTM (mean of 3 seeds)",
                        **lstm_rows.drop(columns=["setup", "model"]).mean().to_dict()}
            d = pd.concat([d, pd.DataFrame([mean_row])], ignore_index=True)
        d = d[["model", "train_mae", "test_mae", "test_rmse", "test_r2", "test_mae_vs_baseline_pct"]]
        d.columns = ["Model", "Train MAE (s)", "2025 test MAE (s)", "RMSE (s)", "R²", "vs baseline (%)"]
        st.subheader(label)
        st.dataframe(d.sort_values("2025 test MAE (s)").round(3), hide_index=True, use_container_width=True)
    st.subheader("Pit window backtest: 2025 green flag stops")
    s = ctx["strategy_summary"].rename(columns={
        "method": "Method", "stints": "Stops", "within_3_laps_pct": "Within 3 laps (%)",
        "within_5_laps_pct": "Within 5 laps (%)", "mean_abs_error_laps": "Mean error (laps)",
        "median_error_laps": "Median signed error (laps)"})
    st.dataframe(s.round(1), hide_index=True, use_container_width=True)
