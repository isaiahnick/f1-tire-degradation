"""PitWall dashboard.  Run from the repo root:  streamlit run app/app.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from model import load_all  # noqa: E402
from predict import best_setup, decision_laps, plan_curves, stint_recommendation, typical_team_call  # noqa: E402

st.set_page_config(page_title="PitWall", layout="wide")

# Red is PitWall (same red as the slides), blue the new tires, black and grey the real laps.
RED, BLUE, INK, MUTED, GRID, ACTUAL = "#CC0000", "#1F5FA8", "#1F1F1F", "#6B6B6B", "#ECECEC", "#9A9A9A"
WINDOW_FILL, PAST_FILL = "rgba(204,0,0,0.11)", "#EFEFEF"
COMP = {"SOFT": "#D7261E", "MEDIUM": "#C98A00", "HARD": "#5C5C5C"}   # Pirelli colors (hard is white, so grey here)
DEMO = "2025_09_VER_2"          # default view: Verstappen, Spanish GP, stint 2
PLOT_CFG = {"displayModeBar": False}
FONT = 14

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
.pw-tile {background: #F5F5F5; border-radius: 10px; padding: 12px 18px 14px; min-height: 124px;}
.pw-lbl {font-size: 0.95rem !important; color: #6B6B6B; line-height: 1.2;}
.pw-val {font-size: 2.5rem !important; font-weight: 700; line-height: 1.1; margin: 4px 0 8px; color: #1F1F1F;}
.pw-red {color: #CC0000;}
.pw-sub {font-size: 0.95rem !important; color: #444; line-height: 1.2;}
.pw-chip {display: inline-block; background: #E2E2E2; border-radius: 999px; padding: 3px 11px;
          font-size: 0.9rem !important; color: #1F1F1F; line-height: 1.2;}
.pw-tip {display: inline-block; width: 17px; height: 17px; border-radius: 50%; background: #DADADA; color: #555;
         font-size: 11px !important; text-align: center; line-height: 17px; cursor: help; margin-left: 4px;}
.pw-head {font-size: 1.1rem !important; font-weight: 650; margin: 0;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def html(s: str, where=st):
    where.markdown(s, unsafe_allow_html=True)


def tile(label: str, value: str, sub: str = "", chip: str = "", red: bool = False, tip: str = "") -> str:
    t = f'<span class="pw-tip" title="{tip}">?</span>' if tip else ""
    return (f'<div class="pw-tile"><div class="pw-lbl">{label}{t}</div>'
            f'<div class="pw-val{" pw-red" if red else ""}">{value}</div>'
            f'<div class="pw-sub">{sub}{f"<span class=pw-chip>{chip}</span>" if chip else ""}</div></div>')


def vs_pitwall(lap: int, best: int) -> str:
    if lap == best:
        return "Same lap as PitWall"
    n = abs(lap - best)
    return f"{n} lap{'s' if n != 1 else ''} {'earlier' if lap < best else 'later'}"


def style(fig: go.Figure, height: int, xtitle: str, ytitle: str, top: int = 20, hover: str = "x unified"):
    fig.update_layout(template="simple_white", height=height, margin=dict(l=10, r=20, t=top, b=10),
                      font=dict(family="Arial", size=FONT, color=INK), hovermode=hover, showlegend=False,
                      hoverlabel=dict(bgcolor="white", font_size=FONT))
    fig.update_xaxes(title=dict(text=xtitle, font=dict(size=FONT, color=MUTED)), showgrid=False,
                     linecolor="#BDBDBD", tickcolor="#BDBDBD")
    fig.update_yaxes(title=dict(text=ytitle, font=dict(size=FONT, color=MUTED)), showgrid=True, gridcolor=GRID,
                     zeroline=True, zerolinecolor="#BDBDBD", showline=False, ticks="")
    return fig


def label(fig, x, y, text, color, **kw):
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color=color, size=FONT), **kw)


def event_lines(fig: go.Figure, events: list, x_lo: float, x_hi: float):
    """Each event gets a vertical line that runs up into its own label row above the plot,
    so labels never collide even when the laps are close."""
    for i, (x, text, color, dash) in enumerate(events):
        y = [1.03, 1.11, 1.19][i]
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=y, xref="x", yref="paper",
                      line=dict(color=color, width=2.5 if i == 0 else 1.5, dash=dash))
        right = (x - x_lo) / max(x_hi - x_lo, 1) > 0.72
        label(fig, x, y, text, color, yref="paper", yanchor="middle",
              xanchor="right" if right else "left", xshift=-6 if right else 6)


# ------------------------------------------------------------------ data
ctx = load_all()
feat, stints = ctx["feat"], ctx["stints"]
FINAL_B, FINAL_A = best_setup(ctx, "B"), best_setup(ctx, "A")
WINDOW_S = float(ctx["cfg"]["strategy"]["window_s"])

st.title("PitWall")
st.caption("F1 tire wear forecasts and pit stop timing. EM 538 Team 9. Trained on 2022 to 2024, replaying 2025.")

tab1, tab2, tab3 = st.tabs(["Pit stop replay (2025)", "Plan a stint", "Model results"])

# ------------------------------------------------------------------ tab 1: replay
with tab1:
    test = feat[(feat["split"] == "test") & feat["in_forecast"]]
    usable = stints[stints["stint_id"].isin(test["stint_id"].unique())]
    races = (usable.sort_values("race_id")[["race_id", "event_name"]].drop_duplicates()
             .set_index("race_id")["event_name"])
    demo_race, demo_driver = "_".join(DEMO.split("_")[:2]), DEMO.split("_")[2]

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1.4, 0.7, 1.4, 1.5, 0.9])
        rids = list(races.index)
        race = c1.selectbox("Race", rids, index=rids.index(demo_race) if demo_race in rids else 0,
                            format_func=lambda r: races[r])
        drivers = sorted(usable.loc[usable["race_id"] == race, "driver"].unique())
        driver = c2.selectbox("Driver", drivers, index=drivers.index(demo_driver) if demo_driver in drivers else 0)
        opts = usable[(usable["race_id"] == race) & (usable["driver"] == driver)].sort_values("stint")
        opts = opts.set_index("stint_id")
        sids = list(opts.index)
        sid = c3.selectbox("Stint", sids, index=sids.index(DEMO) if DEMO in sids else 0, format_func=lambda s: (
            f"Stint {int(opts.loc[s, 'stint'])}: {opts.loc[s, 'compound'].title()}s, "
            f"laps {int(opts.loc[s, 'first_lap'])} to {int(opts.loc[s, 'last_lap'])}"))
        models = ([FINAL_B] + (["LSTM"] if ctx["lstm_fn"] is not None else [])
                  + [m for m in ["Ridge", "Gradient boosting", "Random forest", "Linear baseline"] if m != FINAL_B])
        model_name = c4.selectbox("Tire model", models,
                                  format_func=lambda m: f"{m} (final)" if m == FINAL_B else m)
        info = opts.loc[sid]
        default_next = info["next_compound"] if isinstance(info["next_compound"], str) else "HARD"
        nxt = c5.selectbox("Next tires", ["SOFT", "MEDIUM", "HARD"], format_func=str.title,
                           index=["SOFT", "MEDIUM", "HARD"].index(default_next))
        dl = decision_laps(ctx, sid)
        dec = (st.select_slider("Current lap", options=dl, value=dl[0], format_func=lambda x: f"Lap {x}",
                                help="PitWall only sees laps up to this point. Drag right to replay the stint.")
               if len(dl) > 1 else dl[0])

    t0 = time.perf_counter()
    r = stint_recommendation(ctx, sid, model_name, nxt, dec)
    ms = 1000 * (time.perf_counter() - t0)
    team = typical_team_call(ctx, sid, nxt)
    tbl = r["table"]
    win = tbl.loc[tbl["in_window"], "pit_lap"]
    now, first, best, actual = int(r["state"]["lap_number"]), int(info["first_lap"]), r["best"], r["actual_pit"]
    cur, new = info["compound"].title() + "s", nxt.title() + "s"

    # 1. the answer, as four numbers
    k = st.columns(4, gap="medium")
    html(tile("PitWall recommends", f"Lap {best}", f"Window: laps {win.min()} to {win.max()}", red=True), k[0])
    html(tile("Team actually stopped", f"Lap {actual}" if actual else "No stop",
              chip=vs_pitwall(actual, best) if actual else "", sub="" if actual else "Ran to the finish"), k[1])
    html(tile("Typical team call", f"Lap {team}" if team else "n/a", chip=vs_pitwall(team, best) if team else "",
              tip="What teams usually do in this situation: a second model trained on 2022 to 2024 stops, "
                  "using race context and the tire forecast."), k[2])
    html(tile("Time to compute", f"{ms:.0f} ms", "Goal: under 1 s"), k[3])
    st.write("")

    # 2. the two charts
    obs = r["laps_obs"]
    seen, later = obs.iloc[:r["d"]], obs.iloc[r["d"]:]
    oc, nc = r["old_curve"], r["new_curve"]
    x_lo, x_hi = first - 1, r["segment_end"] + 0.5
    x_label_room = 7
    events = [(best, f"<b>PitWall: lap {best}</b>  (window {win.min()} to {win.max()})", RED, "solid")]
    if actual:
        events.append((actual, f"Team stopped: lap {actual}", INK, "dash"))
    if team:
        events.append((team, f"Typical team call: lap {team}", MUTED, "dot"))

    g1, g2 = st.columns([3, 2], gap="large")
    with g1:
        html('<p class="pw-head">Tire wear: seen and forecast</p>')
        fig = go.Figure()
        fig.add_vrect(x0=first - 0.5, x1=now + 0.5, fillcolor=PAST_FILL, line_width=0, layer="below")
        fig.add_vrect(x0=win.min() - 0.5, x1=win.max() + 0.5, fillcolor=WINDOW_FILL, line_width=0, layer="below")
        label(fig, first - 0.5, 0.99, "Seen so far", MUTED, yref="paper", xanchor="left", yanchor="top", xshift=6)
        fig.add_trace(go.Scatter(x=later["lap_number"], y=later["deg_s"], mode="markers", name="Actual",
                                 marker=dict(color="white", size=8, line=dict(color=ACTUAL, width=1.5)),
                                 hovertemplate="Actual: %{y:.2f} s<extra></extra>"))
        fig.add_trace(go.Scatter(x=seen["lap_number"], y=seen["deg_s"], mode="markers", name="Seen",
                                 marker=dict(color=INK, size=9), hovertemplate="Seen: %{y:.2f} s<extra></extra>"))
        fig.add_trace(go.Scatter(x=oc["lap"], y=oc["loss"], mode="lines", name="Stay out",
                                 line=dict(color=RED, width=3.5), hovertemplate="Stay out: %{y:.2f} s<extra></extra>"))
        ends = [(oc["lap"].iloc[-1], oc["loss"].iloc[-1], "Stay out", RED)]
        if len(nc):
            fig.add_trace(go.Scatter(x=nc["lap"], y=nc["loss"], mode="lines", name="New",
                                     line=dict(color=BLUE, width=3.5, dash="dash"),
                                     hovertemplate=f"New {new}: %{{y:.2f}} s<extra></extra>"))
            ends.append((nc["lap"].iloc[-1], nc["loss"].iloc[-1], f"New {new}", BLUE))
        span = max(float(obs["deg_s"].max()), float(oc["loss"].max())) - min(float(obs["deg_s"].min()), 0.0)
        close = len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 0.08 * max(span, 0.5)
        top_end = max(range(len(ends)), key=lambda i: ends[i][1])
        for i, (x, y, text, color) in enumerate(ends):
            label(fig, x, y, f"<b>{text}</b>", color, xanchor="left", xshift=8, yanchor="middle",
                  yshift=(10 if i == top_end else -10) if close else 0)
        if len(later):
            label(fig, later["lap_number"].iloc[-1], later["deg_s"].iloc[-1], "Actual (not used)", MUTED,
                  xanchor="left", yanchor="bottom", xshift=4, yshift=8, bgcolor="rgba(255,255,255,0.9)")
        event_lines(fig, events, x_lo, x_hi)
        style(fig, 480, "Race lap", "Seconds lost per lap", top=95)
        fig.update_xaxes(range=[x_lo, x_hi + x_label_room])
        st.plotly_chart(fig, theme=None, config=PLOT_CFG, width="stretch")

    with g2:
        html('<p class="pw-head">Total time lost, by stop lap</p>')
        extra = tbl["cost_s"] - tbl["cost_s"].min()
        cap = max(2.5, min(float(extra.max()) * 1.05, 6.0))
        f2 = go.Figure()
        f2.add_vrect(x0=win.min() - 0.5, x1=win.max() + 0.5, fillcolor=WINDOW_FILL, line_width=0, layer="below")
        f2.add_hline(y=WINDOW_S, line=dict(color=RED, width=1, dash="dot"))
        label(f2, best + 0.6, WINDOW_S, f"{WINDOW_S:.0f} s cutoff", RED, xanchor="left", yanchor="bottom",
              bgcolor="rgba(255,255,255,0.9)")
        f2.add_trace(go.Scatter(x=tbl["pit_lap"], y=extra, mode="lines+markers",
                                line=dict(color="#7A7A7A", width=2.5), marker=dict(size=6, color="#7A7A7A"),
                                hovertemplate="Stop on lap %{x}: +%{y:.2f} s<extra></extra>"))
        f2.add_trace(go.Scatter(x=[best], y=[0], mode="markers", marker=dict(size=15, color=RED),
                                hovertemplate="Best: lap %{x}<extra></extra>"))
        lo2, hi2 = tbl["pit_lap"].min() - 1, tbl["pit_lap"].max() + 1
        event_lines(f2, events, lo2, hi2)
        style(f2, 480, "Stop lap", "Extra seconds vs. best stop", top=95, hover="closest")
        f2.update_xaxes(range=[lo2, hi2])
        f2.update_yaxes(range=[-0.3, cap])
        st.plotly_chart(f2, theme=None, config=PLOT_CFG, width="stretch")

    with st.expander("How PitWall picks the lap"):
        pl = ctx["pit_loss"]
        pit_s = pl.loc[pl["circuit"] == info["circuit"], "pit_loss_s"]
        st.markdown(
            f"- **Forecast:** the {model_name} model predicts the time lost on each remaining lap "
            f"if they stay out (red), and on new {new} after a stop (blue).\n"
            "- **Update:** after every lap, the forecast is shifted by its average error over the last 3 laps.\n"
            "- **Choose:** for each stop lap, add the time lost before and after the stop. "
            f"The lowest total is the pick; anything within {WINDOW_S:.0f} s is the window.\n"
            + (f"- **Pit lane:** about {float(pit_s.iloc[0]):.0f} s here, the same on every lap, "
               "so it doesn't move the pick." if len(pit_s) else ""))

# ------------------------------------------------------------------ tab 2: plan
with tab2:
    circuits = sorted(feat["circuit"].unique())
    demo_circuit = stints.loc[stints["stint_id"] == DEMO, "circuit"]
    teams = sorted(feat["team"].unique())
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
        circuit = c1.selectbox("Circuit", circuits, index=circuits.index(demo_circuit.iloc[0])
                               if len(demo_circuit) and demo_circuit.iloc[0] in circuits else 0)
        team_sel = c2.selectbox("Team", teams,
                                index=teams.index("Red Bull Racing") if "Red Bull Racing" in teams else 0)
        plan_models = ["Ridge", "Linear baseline", "Gradient boosting", "Random forest"]
        plan_model = c3.selectbox("Model", [m for m in plan_models if m in ctx["A"]])
        n = c4.slider("Stint length (laps)", 10, 40, 25)
    pc = plan_curves(ctx, circuit, n, team_sel, plan_model)
    html('<p class="pw-head">Expected tire wear on a new set, before the stint starts</p>')
    f3 = go.Figure()
    ends = []
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        g = pc[pc["compound"] == comp]
        if not len(g):
            continue
        f3.add_trace(go.Scatter(x=g["stint_lap"], y=g["loss"], mode="lines", name=comp.title(),
                                line=dict(color=COMP[comp], width=3.5),
                                hovertemplate=f"{comp.title()}: %{{y:.2f}} s<extra></extra>"))
        ends.append([g["stint_lap"].iloc[-1], g["loss"].iloc[-1], comp])
    gap = max(float(pc["loss"].max() - pc["loss"].min()), 0.1) * 0.06     # keep end labels apart
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
    for x, y, comp in ends:
        label(f3, x, y, f"<b>{comp.title()}</b>", COMP[comp], xanchor="left", xshift=10)
    style(f3, 460, "Laps on the set", "Seconds lost per lap")
    f3.update_layout(margin=dict(r=90))
    st.plotly_chart(f3, theme=None, config=PLOT_CFG, width="stretch")
    a_pct = ctx["test_results"].set_index(["setup", "model"]).loc[("A", plan_model), "test_mae_vs_baseline_pct"]
    st.caption(f"Conditions only: {a_pct:.0f}% better than the baseline on 2025. "
               "Seeing the first laps of a stint is what makes the forecast useful.")

# ------------------------------------------------------------------ tab 3: results
with tab3:
    tr = ctx["test_results"]
    us = ctx["update_summary"]
    b = tr[tr["setup"] == "B"].set_index("model")
    fin, base = b.loc[FINAL_B], b.loc["Linear baseline"]
    ss = ctx["strategy_summary"].set_index("method")
    tire = [mth for mth in ss.index if mth.startswith(FINAL_B)]
    k = st.columns(4, gap="medium")
    html(tile("Lower error than baseline", f"{fin['test_mae_vs_baseline_pct']:.0f}%", "Goal: 15%", red=True), k[0])
    html(tile("Average miss per lap", f"{fin['test_mae']:.2f} s", f"Baseline: {base['test_mae']:.2f} s"), k[1])
    html(tile("Lower error with lap-by-lap updates", f"{us['test_improvement_after_lap5_pct']:.0f}%",
              f"{us['test_updated_mae_after_lap5']:.2f} vs. {us['test_frozen_mae_after_lap5']:.2f} s"), k[2])
    if tire:
        html(tile("Pit lap within 3 laps of the team's", f"{ss.loc[tire[0], 'within_3_laps_pct']:.0f}%",
                  "Goal: 60%"), k[3])
    st.write("")

    r1, r2 = st.columns(2, gap="large")
    with r1:
        html('<p class="pw-head">2025 error by model</p>')
        lstm = b[b.index.str.startswith("LSTM (")]
        rows = b[~b.index.str.startswith("LSTM (") & ~b.index.str.startswith("Rule") & (b.index != "Mean (dummy)")]
        mae = rows["test_mae"].copy()
        if len(lstm):
            mae["LSTM"] = lstm["test_mae"].mean()
        mae = mae.sort_values(ascending=False)
        colors = [RED if m == FINAL_B else ("#5C5C5C" if m == "Linear baseline" else "#BDBDBD") for m in mae.index]
        f4 = go.Figure(go.Bar(x=mae.values, y=[f"<b>{m}</b>" if m == FINAL_B else m for m in mae.index],
                              orientation="h", marker_color=colors, text=[f"{v:.2f} s" for v in mae.values],
                              textposition="outside", cliponaxis=False,
                              hovertemplate="%{y}: %{x:.3f} s<extra></extra>"))
        style(f4, 360, "Average miss per lap (s)", "", hover="closest")
        f4.update_xaxes(range=[0, mae.max() * 1.18], showgrid=True, gridcolor=GRID)
        f4.update_yaxes(showgrid=False, zeroline=False)
        st.plotly_chart(f4, theme=None, config=PLOT_CFG, width="stretch")
    with r2:
        html('<p class="pw-head">Updating the forecast every lap</p>')
        ub = ctx["update_by_decision"]
        ub = ub[ub["data"] == "test_2025"]
        names = {"lap 5": "5", "6-9": "6 to 9", "10-14": "10 to 14", "15-19": "15 to 19", "20+": "20+"}
        xs = [names.get(v, v) for v in ub["bucket"]]
        f5 = go.Figure()
        f5.add_trace(go.Scatter(x=xs, y=ub["frozen_mae"], mode="lines+markers", line=dict(color="#9A9A9A", width=3.5),
                                marker=dict(size=8), hovertemplate="Frozen: %{y:.2f} s<extra></extra>"))
        f5.add_trace(go.Scatter(x=xs, y=ub["updated_mae"], mode="lines+markers", line=dict(color=RED, width=3.5),
                                marker=dict(size=8), hovertemplate="Updated: %{y:.2f} s<extra></extra>"))
        label(f5, xs[-1], ub["frozen_mae"].iloc[-1], "<b>Frozen after lap 5</b>", "#7A7A7A",
              xanchor="right", yanchor="bottom", yshift=10)
        label(f5, xs[-1], ub["updated_mae"].iloc[-1], "<b>Updated every lap</b>", RED,
              xanchor="right", yanchor="top", yshift=-10)
        style(f5, 360, "Laps seen on the set", "Average miss per lap (s)", hover="closest")
        f5.update_yaxes(range=[0.3, max(ub["frozen_mae"].max(), 0.6) * 1.08])
        st.plotly_chart(f5, theme=None, config=PLOT_CFG, width="stretch")

    html('<p class="pw-head">Pit stop timing vs. real 2025 stops</p>')
    names = {"Midpoint (no model)": "Middle of the stint (reference)",
             "Learned layer (tire model + race context)": "Team-stop model: race context + tire model",
             "Imitation only (race context, no tire model)": "Team-stop model: race context"}
    order = [o for o in ["Midpoint (no model)", "Learned layer (tire model + race context)",
                         "Imitation only (race context, no tire model)"] + tire if o in ss.index]
    vals = ss.loc[order, "within_3_laps_pct"]
    f6 = go.Figure(go.Bar(x=vals.values, y=[names.get(o, "<b>Tire model only (PitWall)</b>") for o in order],
                          orientation="h", marker_color=[RED if o in tire else "#BDBDBD" for o in order],
                          text=[f"{v:.0f}%" for v in vals.values], textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    f6.add_vline(x=60, line=dict(color=INK, width=1.5, dash="dash"))
    label(f6, 60, 1, "Goal: 60%", INK, yref="paper", xanchor="left", xshift=6, yanchor="top")
    style(f6, 270, "Within 3 laps of the real stop (%)", "", hover="closest")
    f6.update_xaxes(range=[0, 75], showgrid=True, gridcolor=GRID)
    f6.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(f6, theme=None, config=PLOT_CFG, width="stretch")
    st.caption("Tire model: when is it cheapest to stop. Team-stop model: when do teams actually stop.")

    with st.expander("Full results tables"):
        for setup, lab in [("B", "Setup B: conditions + first 5 laps of the stint"), ("A", "Setup A: conditions only")]:
            d = tr[tr["setup"] == setup][["model", "train_mae", "test_mae", "test_rmse", "test_r2",
                                          "test_mae_vs_baseline_pct"]].copy()
            d.columns = ["Model", "Train MAE (s)", "2025 test MAE (s)", "RMSE (s)", "R²", "vs. baseline (%)"]
            st.markdown(f"**{lab}**")
            st.dataframe(d.sort_values("2025 test MAE (s)").round(3), hide_index=True, width="stretch")
        st.markdown("**Forecast updating by decision point (2025)**")
        u = ub[["bucket", "frozen_mae", "updated_mae", "laps"]].copy()
        u.columns = ["Laps seen on the set", "Frozen MAE (s)", "Updated MAE (s)", "Laps scored"]
        st.dataframe(u.round(3), hide_index=True, width="stretch")
        st.markdown("**Pit stop backtest (2025)**")
        s = ctx["strategy_summary"].rename(columns={
            "method": "Method", "stints": "Stops", "within_3_laps_pct": "Within 3 laps (%)",
            "within_5_laps_pct": "Within 5 laps (%)", "mean_abs_error_laps": "Mean error (laps)",
            "median_error_laps": "Median signed error (laps)", "actual_in_window_pct": "Actual stop in window (%)",
            "median_window_laps": "Window width (laps)"})
        s = s.drop(columns=[c for c in ["monotone", "smooth", "rolling", "update"] if c in s])
        st.dataframe(s.round(1), hide_index=True, width="stretch")
