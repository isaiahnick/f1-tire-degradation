"""Plotly figures for the PitWall dashboard. No Streamlit calls in here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Red is PitWall (the same red as the slides), blue the new tires, black what really
# happened, grey what teams usually do. Text stays in ink; the marks carry the color.
RED, BLUE, INK, MUTED, FAINT, GRID = "#CC0000", "#1F5FA8", "#1A1A1A", "#666666", "#9A9A9A", "#ECECEC"
WINDOW, BAR_GREY = "rgba(60,60,60,0.13)", "#C9C9C9"
FONT = '"Source Sans 3", "Source Sans Pro", "Source Sans", Arial, sans-serif'
CONFIG = {"displayModeBar": False, "responsive": True}
HORIZON_MID = [3, 8, 13, 18, 25]      # middle of the laps-ahead buckets in the error tables


def _base(fig: go.Figure, height: int, top: int = 10, right: int = 20) -> go.Figure:
    fig.update_layout(template="simple_white", height=height, margin=dict(l=8, r=right, t=top, b=8),
                      font=dict(family=FONT, size=14, color=INK), showlegend=False,
                      hoverlabel=dict(bgcolor="white", bordercolor="#D0D0D0", font=dict(family=FONT, size=14)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, linecolor="#C8C8C8", tickcolor="#C8C8C8", title_font=dict(size=14, color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, showline=False, ticks="", zeroline=False,
                     title_font=dict(size=14, color=MUTED))
    return fig


def _note(fig, x, y, text, color=MUTED, **kw):
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color=color, size=13), **kw)


def typical_miss(ctx, model: str, updated: bool, laps_ahead: np.ndarray) -> np.ndarray:
    """Average 2025 miss at each distance ahead, read off the saved error tables."""
    if updated and model == ctx["update_summary"]["model"]:
        t = ctx["update_by_horizon"]
        vals = t.loc[t["data"] == "test_2025", "updated_mae"].to_numpy()
    else:
        t = ctx["error_by_horizon"]
        col = model if model in t.columns else ctx["update_summary"]["model"]
        vals = t[col].to_numpy()
    return np.interp(laps_ahead, HORIZON_MID, vals)


# ------------------------------------------------------------------ pit wall view
def pitwall_figure(ctx, s: dict, team_call: int | None, hindsight: bool, labels: dict) -> go.Figure:
    """Top: tire wear seen so far and forecast. Bottom: the pit calls, one row each,
    on the same lap axis so they can be compared straight down the page."""
    stop_coming = s["has_next"]
    rows = []
    if stop_coming:
        rows = ["PitWall window"] + (["Teams usually"] if team_call else []) \
            + (["Actually stopped"] if hindsight and s["actual_pit"] else [])
    two = bool(rows)
    fig = make_subplots(rows=2 if two else 1, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.76, 0.24] if two else None)
    now, best = s["now"], s["best"]
    seen, later, old, new = s["seen"], s["later"], s["old_curve"], s["new_curve"]
    x_lo, x_hi = s["first_lap"] - 0.6, s["segment_end"] + 0.6

    # ---- wear panel
    if len(old):
        miss = typical_miss(ctx, s["model"], s["n_updates"] > 0, (old["lap"] - now).to_numpy())
        fig.add_trace(go.Scatter(x=pd.concat([old["lap"], old["lap"][::-1]]),
                                 y=np.concatenate([old["loss"] + miss, (old["loss"] - miss)[::-1]]),
                                 fill="toself", fillcolor="rgba(204,0,0,0.09)", line=dict(width=0),
                                 hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=old["lap"], y=old["loss"], mode="lines", line=dict(color=RED, width=2.5),
                                 hovertemplate=f"Forecast, {labels['cur']}: %{{y:+.2f}} s<extra></extra>"),
                      row=1, col=1)
    if stop_coming and len(new):
        fig.add_trace(go.Scatter(x=new["lap"], y=new["loss"], mode="lines",
                                 line=dict(color=BLUE, width=2.5, dash="dash"),
                                 hovertemplate=f"Forecast, new {labels['new']}: %{{y:+.2f}} s<extra></extra>"),
                      row=1, col=1)
    if hindsight and len(later):
        fig.add_trace(go.Scatter(x=later["lap_number"], y=later["deg_s"], mode="markers",
                                 marker=dict(color="white", size=9, line=dict(color=FAINT, width=1.8)),
                                 hovertemplate="What happened: %{y:+.2f} s<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=seen["lap_number"], y=seen["deg_s"], mode="markers",
                             marker=dict(color=INK, size=10, line=dict(color="white", width=2)),
                             hovertemplate="Lap seen: %{y:+.2f} s<extra></extra>"), row=1, col=1)

    ys = [seen["deg_s"].max(), seen["deg_s"].min()]
    if len(old):
        ys += [float((old["loss"] + miss).max()), float((old["loss"] - miss).min())]
    if hindsight and len(later):
        ys += [later["deg_s"].max(), later["deg_s"].min()]
    y_lo, y_hi = min(min(ys), 0) - 0.15, max(max(ys), 1.0) + 0.25
    fig.update_yaxes(range=[y_lo, y_hi], title_text="Seconds per lap slower than on fresh tires",
                     tickformat="+.1f", row=1, col=1)

    if stop_coming:      # the window, behind everything in the wear panel
        fig.add_shape(type="rect", x0=s["win_lo"] - 0.5, x1=s["win_hi"] + 0.5, y0=y_lo, y1=y_hi,
                      fillcolor=WINDOW, line_width=0, layer="below", row=1, col=1)
    fig.add_shape(type="line", x0=x_lo, x1=x_hi, y0=0, y1=0, line=dict(color="#B5B5B5", width=1), layer="below",
                  row=1, col=1)

    _note(fig, x_lo, 0, "Fresh pace", MUTED, xanchor="left", yanchor="bottom", xshift=4, row=1, col=1)

    # "now" marker, and names on the lines themselves
    fig.add_shape(type="line", x0=now + 0.5, x1=now + 0.5, y0=y_lo, y1=y_hi, line=dict(color=INK, width=1),
                  row=1, col=1)
    left = (now - x_lo) / (x_hi - x_lo) > 0.18
    _note(fig, now + 0.5, y_hi, f"<b>Now: lap {now}</b>", INK, xanchor="right" if left else "left",
          yanchor="top", xshift=-6 if left else 6, row=1, col=1)
    if stop_coming:
        mid = (s["win_lo"] + s["win_hi"]) / 2
        far = abs(mid - now) > 3
        if far:
            _note(fig, mid, y_hi, "Pit window", INK, yanchor="top", row=1, col=1)
    if len(old):
        _note(fig, old["lap"].iloc[-1], old["loss"].iloc[-1], f"Stay out on {labels['cur']}", INK,
              xanchor="right", yanchor="bottom", yshift=8, row=1, col=1)
    if stop_coming and len(new) > 2:      # under the line, a little back from its end
        j = int(len(new) * 0.75)
        _note(fig, new["lap"].iloc[j], new["loss"].iloc[j], f"New {labels['new']}", INK,
              xanchor="left", yanchor="top", yshift=-6, row=1, col=1)

    # ---- calls panel
    if two:
        y = {r: i for i, r in enumerate(reversed(rows))}
        fig.add_trace(go.Scatter(x=[s["win_lo"] - 0.4, s["win_hi"] + 0.4], y=[y["PitWall window"]] * 2,
                                 mode="lines", line=dict(color="rgba(204,0,0,0.35)", width=12),
                                 hoverinfo="skip"), row=2, col=1)
        marks = [("PitWall window", best, RED, "circle", f"Cheapest lap to stop: {best}")]
        if "Teams usually" in y:
            marks.append(("Teams usually", team_call, MUTED, "diamond", f"Teams usually stop on lap {team_call}"))
        if "Actually stopped" in y:
            marks.append(("Actually stopped", s["actual_pit"], INK, "square",
                          f"{labels['team']} stopped on lap {s['actual_pit']}"))
        for row, lap, color, symbol, tip in marks:
            right = (lap - x_lo) / (x_hi - x_lo) > 0.86
            fig.add_trace(go.Scatter(x=[lap], y=[y[row]], mode="markers+text", text=[f"<b>Lap {lap}</b>"],
                                     textposition="middle left" if right else "middle right",
                                     textfont=dict(color=INK, size=14),
                                     marker=dict(color=color, size=15, symbol=symbol, line=dict(color="white", width=2)),
                                     hovertemplate=tip + "<extra></extra>", cliponaxis=False), row=2, col=1)
        fig.update_yaxes(tickmode="array", tickvals=list(y.values()), ticktext=list(y.keys()),
                         range=[-0.6, len(rows) - 0.4], showgrid=False, tickfont=dict(size=14, color=INK),
                         row=2, col=1)
    fig.update_xaxes(range=[x_lo, x_hi], dtick=5 if x_hi - x_lo > 24 else 2, row=1, col=1)
    fig.update_xaxes(title_text="Race lap", range=[x_lo, x_hi], dtick=5 if x_hi - x_lo > 24 else 2,
                     row=2 if two else 1, col=1)
    _base(fig, 405 if two else 340, top=6)
    fig.update_layout(hovermode="x unified")
    return fig


def cost_figure(s: dict, cap: float = 6.0) -> go.Figure:
    """Time lost on tire pace for every lap they could stop on, against the best one.
    Laps that lose more than `cap` seconds are left off so the valley stays readable."""
    c = s["cost"]
    c = c[c["extra_s"] <= cap]
    fig = go.Figure(go.Bar(x=c["pit_lap"], y=c["extra_s"], width=0.62,
                           marker=dict(color=[RED if w else BAR_GREY for w in c["in_window"]], line_width=0),
                           hovertemplate="Stop on lap %{x}: +%{y:.1f} s<extra></extra>"))
    fig.add_trace(go.Scatter(x=[s["best"]], y=[0], mode="markers+text", text=[f"<b>Lap {s['best']}</b>"],
                             textposition="top center", textfont=dict(color=INK, size=14),
                             marker=dict(color=RED, size=14, line=dict(color="white", width=2)),
                             hoverinfo="skip", cliponaxis=False))
    _base(fig, 290, top=10)
    fig.update_xaxes(title_text="Lap they stop on", dtick=5 if len(c) > 24 else 2)
    fig.update_yaxes(title_text="Seconds lost vs. the best lap", range=[-0.3, max(2.5, float(c["extra_s"].max()) * 1.1)])
    return fig


# ------------------------------------------------------------------ track record
def hbar(names: list[str], values: list[float], highlight: str, fmt: str, xtitle: str, height: int,
         dark: str | None = None, goal: float | None = None, xmax: float | None = None) -> go.Figure:
    colors = [RED if n == highlight else ("#6B6B6B" if n == dark else BAR_GREY) for n in names]
    xmax = xmax or max(values) * 1.18
    # a value label that would sit on the goal line goes inside its bar instead
    pos = ["inside" if goal is not None and 0 <= goal - v < 0.09 * xmax else "outside" for v in values]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", width=0.58, marker=dict(color=colors, line_width=0),
                           text=[fmt.format(v) for v in values], textposition=pos, insidetextanchor="end",
                           cliponaxis=False, textfont=dict(color=INK, size=14),
                           hovertemplate="%{y}: %{text}<extra></extra>"))
    if goal is not None:
        fig.add_vline(x=goal, line=dict(color=INK, width=1.5))
        _note(fig, goal, 1.0, f"Goal: {fmt.format(goal)}", INK, yref="paper", xanchor="left", xshift=6, yanchor="bottom")
    _base(fig, height, top=24 if goal is not None else 8, right=40)
    fig.update_xaxes(title_text=xtitle, range=[0, xmax], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=14, color=INK), ticksuffix="  ")
    return fig


def update_figure(ub: pd.DataFrame) -> go.Figure:
    names = {"lap 5": "5", "6-9": "6 to 9", "10-14": "10 to 14", "15-19": "15 to 19", "20+": "20+"}
    xs = [names.get(v, v) for v in ub["bucket"]]
    fig = go.Figure()
    for col, color, lab, pos, shift in [("frozen_mae", FAINT, "Forecast frozen after 5 laps", "bottom", 10),
                                        ("updated_mae", RED, "Corrected after every lap", "top", -10)]:
        fig.add_trace(go.Scatter(x=xs, y=ub[col], mode="lines+markers", line=dict(color=color, width=2.5),
                                 marker=dict(size=10, color=color, line=dict(color="white", width=2)),
                                 hovertemplate=lab + ": %{y:.2f} s<extra></extra>"))
        _note(fig, xs[-1], ub[col].iloc[-1], f"{lab}  <b>{ub[col].iloc[-1]:.2f} s</b>", INK, xanchor="right",
              yanchor=pos, yshift=shift)
    _base(fig, 330, top=8)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="Laps already seen on the set", type="category")
    fig.update_yaxes(title_text="Average miss per lap (s)", range=[0, max(ub["frozen_mae"].max(), 0.6) * 1.2])
    return fig
