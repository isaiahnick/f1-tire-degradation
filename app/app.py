"""PitWall dashboard.  Run from the repo root:  streamlit run app/app.py

What it is for: a strategist partway through a stint asks "given how these tires are
wearing, which lap is cheapest to stop on, and how does that compare with what teams
usually do?" There is no live race to plug into, so the dashboard replays 2025 races
the models never saw and stands the user on the pit wall at any lap.

Files: model.py loads the saved models and tables, predict.py turns a selection into
a forecast and a call, charts.py draws the figures, style.css holds the styling.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

import charts  # noqa: E402
from model import load_all  # noqa: E402
from predict import best_setup, decision_laps, snapshot, typical_team_call  # noqa: E402

st.set_page_config(page_title="PitWall", page_icon=":checkered_flag:", layout="wide")
st.html(f"<style>{(HERE / 'style.css').read_text()}</style>")

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
DEMO = "2025_09_VER_2"                       # opens on Verstappen, Spanish GP, stint 2
EXAMPLES = [("Spain, Verstappen", "2025_09_VER_2"), ("Austria, Hamilton", "2025_11_HAM_1"),
            ("Saudi Arabia, Russell", "2025_05_RUS_1")]
PLAY_S = 0.9                                 # seconds per lap when the replay is playing
DRIVERS = {"VER": "Verstappen", "NOR": "Norris", "PIA": "Piastri", "LEC": "Leclerc", "HAM": "Hamilton",
           "RUS": "Russell", "ANT": "Antonelli", "ALO": "Alonso", "STR": "Stroll", "GAS": "Gasly",
           "DOO": "Doohan", "COL": "Colapinto", "OCO": "Ocon", "BEA": "Bearman", "HUL": "Hulkenberg",
           "BOR": "Bortoleto", "TSU": "Tsunoda", "LAW": "Lawson", "HAD": "Hadjar", "ALB": "Albon", "SAI": "Sainz"}
TEAM_COLOR = {"Red Bull Racing": "#3671C6", "Ferrari": "#E8002D", "Mercedes": "#27C4AE", "McLaren": "#FF8000",
              "Aston Martin": "#229971", "Alpine": "#FF87BC", "Williams": "#64C4FF", "Racing Bulls": "#6692FF",
              "Kick Sauber": "#52E252", "Haas F1 Team": "#9C9FA2"}


# ------------------------------------------------------------------ data (all cached)
ctx = load_all()
FINAL_B = best_setup(ctx, "B")
WINDOW_S = float(ctx["cfg"]["strategy"]["window_s"])


@st.cache_data(show_spinner=False)
def catalog():
    """2025 stints that ended in a pit stop and have at least one lap PitWall can call."""
    c = load_all()
    feat, s = c["feat"], c["stints"]
    seen = feat.loc[(feat["split"] == "test") & feat["in_forecast"], "stint_id"].unique()
    s = s[s["stint_id"].isin(seen) & s["next_last_lap"].notna() & s["pit_in"]
          & s["next_compound"].isin(COMPOUNDS)].copy()
    s["laps"] = [decision_laps(c, sid) for sid in s["stint_id"]]
    return s[s["laps"].map(len) > 0].sort_values(["race_id", "driver", "stint"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=5000)
def snap(stint_id: str, model: str, nxt: str, lap: int) -> dict:
    return snapshot(load_all(), stint_id, model, nxt, lap)


@st.cache_data(show_spinner=False)
def team_call(stint_id: str, nxt: str):
    return typical_team_call(load_all(), stint_id, nxt)


@st.cache_data(show_spinner=False)
def call_speed_ms() -> float:
    """Median time for one fresh recommendation, measured on this machine."""
    c = load_all()
    lap = decision_laps(c, DEMO)[0]
    return float(np.median([snapshot(c, DEMO, FINAL_B, "MEDIUM", lap)["ms"] for _ in range(5)]))


cat = catalog()
by_id = cat.set_index("stint_id")


# ------------------------------------------------------------------ small helpers
def plural(compound: str) -> str:
    return compound.title() + "s"


def laps_txt(n: int) -> str:
    return f"{n} lap{'' if n == 1 else 's'}"


def versus(lap: int, best: int) -> tuple[str, str]:
    if lap == best:
        return "same lap", "same"
    return f"{laps_txt(abs(lap - best))} {'earlier' if lap < best else 'later'}", ""


def model_label(m: str) -> str:
    return "Final model (GB + Ridge)" if m == FINAL_B else m


def goto(stint_id: str):
    row = by_id.loc[stint_id]
    st.session_state.update(race=row["race_id"], driver=row["driver"], stint=stint_id, playing=False)


def set_lap(key: str, lap: int):
    st.session_state[key] = lap
    st.session_state["playing"] = False


def toggle_play(key: str, laps: list[int]):
    playing = not st.session_state.get("playing", False)
    if playing and st.session_state.get(key, laps[0]) == laps[-1]:
        st.session_state[key] = laps[0]            # at the end already: start over
    st.session_state["playing"] = playing


def race_strip(all_stints, sid: str, s: dict) -> str:
    """The driver's whole race as one bar: a block per set of tires, a marker for now,
    and PitWall's window underneath."""
    total = s["total_laps"]
    pct = lambda lap: f"{100 * lap / total:.2f}%"      # noqa: E731
    blocks = ""
    for r in all_stints.itertuples():
        comp = r.compound if r.compound in COMPOUNDS else "OTHER"
        n = int(r.last_lap - r.first_lap + 1)
        text = f"{plural(r.compound)}, laps {int(r.first_lap)} to {int(r.last_lap)}" if n / total > 0.2 else (
            plural(r.compound) if n / total > 0.09 else "")
        blocks += (f'<div class="pw-stint {comp}{" on" if r.stint_id == sid else ""}" title="Stint {int(r.stint)}: '
                   f'{plural(r.compound)}, laps {int(r.first_lap)} to {int(r.last_lap)}" '
                   f'style="left:{pct(r.first_lap - 1)};width:{pct(n)}">{text}</div>')
    ticks = "".join(f'<span class="pw-tick" style="left:{pct(t - 0.5)}">{t}</span>'
                    for t in range(10, total, 10) if abs(t - s["now"]) > 2)
    return (f'<div class="pw-strip-wrap"><div class="pw-strip">{blocks}'
            f'<div class="pw-now" style="left:{pct(s["now"])}"></div></div>'
            f'<div class="pw-rail"><div class="pw-win" style="left:{pct(s["win_lo"] - 1)};'
            f'width:{pct(s["win_hi"] - s["win_lo"] + 1)}"></div>'
            f'<div class="pw-best" style="left:{pct(s["best"] - 0.5)}" title="PitWall: lap {s["best"]}"></div>'
            f'{ticks}<span class="pw-flag" style="left:{pct(s["now"])}">Lap {s["now"]}</span></div></div>')


def call_card(s: dict, new: str) -> str:
    now, best = s["now"], s["best"]
    if s["status"] == "pit":
        status, hero, sub = "Pit this lap", "Pit now", f"The tires say stop on lap {best}, then {new}"
    elif s["status"] == "open":
        status = "Window open"
        hero, sub = f"Pit on lap {best}", f"in {laps_txt(best - now)}, then {new}"
    else:
        status = f"Stay out, window opens in {laps_txt(s['win_lo'] - now)}"
        hero, sub = f"Pit on lap {best}", f"in {laps_txt(best - now)}, then {new}"
    return (f'<div class="pw-call {s["status"]}"><div class="pw-status"><i></i>{status}</div>'
            f'<div class="pw-hero">{hero}</div><div class="pw-sub">{sub}<br>'
            f'<span>Window: laps {s["win_lo"]} to {s["win_hi"]}</span></div></div>')


def compare_rows(s: dict, team: int | None, hindsight: bool) -> str:
    rows = [("circle", "PitWall", "Cheapest lap on tire wear alone", s["best"], None)]
    if team:
        rows.append(("diamond", "Teams usually", f"Learned from {ctx['n_team_stops']:,} real stops, 2022 to 2024",
                     team, versus(team, s["best"])))
    if hindsight and s["actual_pit"]:
        rows.append(("square", f"{s['team']} actually", "What happened in the race", s["actual_pit"],
                     versus(s["actual_pit"], s["best"])))
    out = ""
    for mark, name, small, lap, diff in rows:
        chip = f'<span class="df {diff[1]}">{diff[0]}</span>' if diff else '<span class="df" style="visibility:hidden">.</span>'
        out += (f'<div class="pw-row"><span class="mk {mark}"></span><span class="nm">{name}<small>{small}</small></span>'
                f'{chip}<span class="lap">Lap {lap}</span></div>')
    return f'<div class="pw-rows">{out}</div>'


def stat(label: str, value: float, unit: str = "s a lap") -> str:
    return f'<div class="pw-stat"><div class="l">{label}</div><div class="v">{value:+.1f}<small>{unit}</small></div></div>'


# ------------------------------------------------------------------ header
st.html('<div class="pw-brand"><div class="pw-logo">Pit<span>Wall</span></div>'
        '<div class="pw-tag">Stand on the pit wall at any lap of a 2025 race: how the tires are wearing, '
        'and when to stop.</div></div>')
models = [FINAL_B] + (["LSTM"] if ctx["lstm_fn"] is not None else []) \
    + [m for m in ["Ridge", "Gradient boosting", "Random forest", "Linear baseline"] if m != FINAL_B]

tab_replay, tab_record = st.tabs(["Race replay", "Track record"])

# ------------------------------------------------------------------ tab 1: race replay
with tab_replay:
    ss = st.session_state
    if "race" not in ss:
        goto(DEMO if DEMO in by_id.index else cat["stint_id"].iloc[0])
    races = cat.drop_duplicates("race_id").set_index("race_id")["event_name"]

    c1, c2, c3, c4, c5 = st.columns([1.35, 1.45, 1.75, 0.8, 0.8], vertical_alignment="bottom")
    race = c1.selectbox("Race", list(races.index), format_func=lambda r: races[r], key="race")
    in_race = cat[cat["race_id"] == race]
    drivers = sorted(in_race["driver"].unique(), key=lambda d: DRIVERS.get(d, d))
    if ss.get("driver") not in drivers:
        ss["driver"] = drivers[0]
    team_of = in_race.drop_duplicates("driver").set_index("driver")["team"]
    driver = c2.selectbox("Driver", drivers, key="driver",
                          format_func=lambda d: f"{DRIVERS.get(d, d)} ({team_of[d]})")
    mine = in_race[in_race["driver"] == driver]
    sids = list(mine["stint_id"])
    if ss.get("stint") not in sids:
        ss["stint"] = sids[0]

    def stint_label(x):
        r = by_id.loc[x]
        return f"Stint {int(r['stint'])}: {plural(r['compound'])}, laps {int(r['first_lap'])} to {int(r['last_lap'])}"

    sid = c3.selectbox("Stint", sids, format_func=stint_label, key="stint",
                       help="Stints that ended in a pit stop, so there is a real call to compare with.")
    info = by_id.loc[sid]
    with c4.popover("Examples", width="stretch"):
        for label, sid_ in EXAMPLES:
            if sid_ in by_id.index:
                st.button(label, on_click=goto, args=(sid_,), width="stretch", key=f"ex_{sid_}")
        st.button("Surprise me", on_click=lambda: goto(random.choice(list(by_id.index))), width="stretch",
                  help="A random driver and stint from 2025")
    picked = ss.get("model", FINAL_B)
    with c5.popover("Model" if picked == FINAL_B else picked, width="stretch", key="model_pop"):
        model_name = st.radio("Model that forecasts the wear", models, format_func=model_label, key="model")
        st.caption("Trained on 2022 to 2024. Every race in the replay is from 2025, which no model saw.")

    # ---- lap control: step, play, or drag
    laps = list(info["laps"])
    lap_key = f"lap::{sid}"
    if ss.get("_play_sid") != sid:                       # a new stint stops the replay
        ss["_play_sid"], ss["playing"] = sid, False
    if "_goto_lap" in ss:
        k, v = ss.pop("_goto_lap")
        if k == lap_key:
            ss[k] = v
    if ss.get(lap_key) not in laps:
        ss[lap_key] = laps[0]
    b1, b2, b3, b4, b5, b6 = st.columns([0.42, 0.95, 0.42, 3.3, 2.05, 1.6], vertical_alignment="bottom")
    i = laps.index(ss[lap_key])
    b1.button("◀", on_click=set_lap, args=(lap_key, laps[max(i - 1, 0)]), disabled=i == 0,
              width="stretch", help="Back one lap", key="back")
    b2.button("Pause" if ss.get("playing") else "Play", on_click=toggle_play, args=(lap_key, laps),
              icon=":material/pause:" if ss.get("playing") else ":material/play_arrow:",
              width="stretch", type="primary", disabled=len(laps) < 2, help="Replay the stint lap by lap", key="play")
    b3.button("▶", on_click=set_lap, args=(lap_key, laps[min(i + 1, len(laps) - 1)]), disabled=i == len(laps) - 1,
              width="stretch", help="Forward one lap", key="fwd")
    if len(laps) > 1:
        lap = b4.select_slider("Current lap", options=laps, key=lap_key, format_func=lambda x: f"Lap {x}",
                               help="PitWall only sees laps up to this one. It makes its first call "
                                    "after 5 clean laps on a set.")
    else:
        lap = laps[0]
        b4.caption(f"Lap {lap} is the only lap in this stint where PitWall has enough laps to make a call.")

    nxt = b5.radio("New tires at the stop", COMPOUNDS, format_func=str.title, horizontal=True, key=f"next::{sid}",
                   index=COMPOUNDS.index(info["next_compound"]),
                   help=f"Starts on what {info['team']} really fitted. Change it to see how the call moves.")
    hindsight = b6.toggle("What happened next", value=True, key="hindsight",
                          help="Off: only what the pit wall knew on this lap. On: the laps and the stop that followed.")

    s = snap(sid, model_name, nxt, int(lap))
    team = team_call(sid, nxt)
    cur, new = plural(s["compound"]), plural(nxt)

    # ---- where we are
    gap = lambda g: "clear" if g >= 9.95 else f"{g:.1f} s"      # noqa: E731
    st.html(
        f'<div class="pw-sit"><h2>{s["event"]} <em>Lap {s["now"]} of {s["total_laps"]}</em></h2>'
        f'<span class="chip"><span class="sw" style="background:{TEAM_COLOR.get(s["team"], "#999")}"></span>'
        f'{DRIVERS.get(s["driver"], s["driver"])}<b>P{s["position"]}</b></span>'
        f'<span class="chip"><span class="tire {s["compound"]}"></span>{cur}<b>{s["tyre_age"]} laps old</b></span>'
        f'<span class="chip">Car ahead<b>{gap(s["gap_ahead"])}</b></span>'
        f'<span class="chip">Car behind<b>{gap(s["gap_behind"])}</b></span></div>'
        + race_strip(ctx["stints"][(ctx["stints"]["race_id"] == race) & (ctx["stints"]["driver"] == driver)]
                     .sort_values("first_lap"), sid, s))

    # ---- the call (left) and the evidence (right)
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        stats = stat("Losing now", s["wear_now"]) + stat(f"By lap {s['best']}", s["wear_at_best"])
        if s["fresh_after"] is not None:
            stats += stat(f"New {new}", s["fresh_after"])
        st.html(call_card(s, new) + compare_rows(s, team, hindsight)
                + f'<div class="pw-stats">{stats}</div>'
                + '<div class="pw-why">PitWall reads tire wear only. Teams also weigh track position, undercuts '
                  'and safety cars, so the calls can differ.</div>')
    with right:
        key = ('<span><i class="k-dot"></i>Laps seen</span>'
               + ('<span><i class="k-ring"></i>What happened next</span>' if hindsight and len(s["later"]) else "")
               + '<span><i class="k-line"></i>Forecast: stay out</span><span><i class="k-band"></i>Typical miss</span>'
               + (f'<span><i class="k-dash"></i>Forecast: new {new}</span>' if len(s["new_curve"]) else "")
               + '<span><i class="k-win"></i>Pit window</span>')
        st.html(f'<p class="pw-h">Tire wear on these {cur}, and where the calls land</p><div class="pw-key">{key}</div>')
        st.plotly_chart(charts.pitwall_figure(ctx, s, team, hindsight, {"cur": cur, "new": new, "team": s["team"]}),
                        theme=None, config=charts.CONFIG, width="stretch", key="pitwall_fig")
        foot = f"{model_label(s['model'])}, computed in {s['ms']:.0f} ms."
        if s["n_updates"]:
            b = s["bias"]
            foot += (f" Forecast moved <b>{'up' if b > 0 else 'down'} {abs(b):.2f} s</b> after checking it against "
                     f"the last {min(s['n_updates'], ctx['update_summary']['k'])} laps.")
        if hindsight and s["miss"] is not None:
            foot += f" Over the laps that followed it was off by <b>{s['miss']:.2f} s a lap</b>."
        st.html(f'<div class="pw-foot">{foot}</div>')

    with st.expander(f"Why lap {s['best']}?"):
        e1, e2 = st.columns([1.5, 1], gap="large")
        with e1:
            st.html('<p class="pw-h">Seconds lost on tire pace for each lap they could stop on</p>'
                    f'<div class="pw-key"><span><i class="k-sq" style="background:{charts.RED}"></i>Within '
                    f'{WINDOW_S:.0f} s of the best lap: the pit window</span>'
                    f'<span><i class="k-sq" style="background:{charts.BAR_GREY}"></i>Other laps</span></div>')
            st.plotly_chart(charts.cost_figure(s), theme=None, config=charts.CONFIG, width="stretch",
                            key="cost_fig")
        with e2:
            pl = ctx["pit_loss"]
            lane = pl.loc[pl["circuit"] == s["circuit"], "pit_loss_s"]
            st.html(
                '<ol class="pw-steps">'
                f'<li><b>Forecast.</b> Time lost each lap if they stay out on the {cur}, and each lap on new {new}.</li>'
                '<li><b>Add it up.</b> For every lap they could stop on: time lost before the stop plus time lost after it.</li>'
                f'<li><b>Pick.</b> The lowest total is the call. Every lap within {WINDOW_S:.0f} s of it is in the window.</li>'
                + (f'<li><b>Pit lane.</b> About {float(lane.iloc[0]):.0f} s here on any lap, so it does not move the call.</li>'
                   if len(lane) else "") + "</ol>")

    if ss.get("playing"):                                  # advance the replay one lap and run again
        time.sleep(PLAY_S)
        if i < len(laps) - 1:
            ss["_goto_lap"] = (lap_key, laps[i + 1])
        else:
            ss["playing"] = False
        st.rerun()

# ------------------------------------------------------------------ tab 2: track record
with tab_record:
    tr, us = ctx["test_results"], ctx["update_summary"]
    b = tr[tr["setup"] == "B"].set_index("model")
    fin, base = b.loc[FINAL_B], b.loc["Linear baseline"]
    ss_ = ctx["strategy_summary"].set_index("method")
    tire = [m for m in ss_.index if m.startswith(FINAL_B)]
    habit = "Imitation only (race context, no tire model)"
    pit_pct = float(ss_.loc[tire[0], "within_3_laps_pct"]) if tire else float("nan")
    n_stops = int(ss_["stints"].iloc[0])
    met = '<span class="badge met">&#10003; Goal met</span>'
    miss = '<span class="badge miss">&#10007; Goal not met</span>'
    st.html(
        '<div class="pw-sit"><h2>Can you trust it? <em>Scored on 2025, a season no model saw</em></h2></div>'
        '<div class="pw-goals">'
        f'<div class="pw-goal"><div class="q">Tire wear forecast vs. a simple baseline</div>'
        f'<div class="v">{fin["test_mae_vs_baseline_pct"]:.0f}% <span style="font-size:1.2rem;font-weight:600">lower error</span>{met}</div>'
        f'<div class="d">Off by {fin["test_mae"]:.2f} s a lap on average; the baseline by {base["test_mae"]:.2f} s. Goal: 15%.</div></div>'
        f'<div class="pw-goal"><div class="q">Pit call within 3 laps of the team\'s real stop</div>'
        f'<div class="v">{pit_pct:.0f}%{miss}</div>'
        f'<div class="d">Goal: 60% of {n_stops} stops. A model of team habits alone gets '
        f'{float(ss_.loc[habit, "within_3_laps_pct"]):.0f}%.</div></div>'
        f'<div class="pw-goal"><div class="q">Time to make one call</div>'
        f'<div class="v">{call_speed_ms():.0f} ms{met}</div>'
        f'<div class="d">Measured on this machine just now. Goal: a few seconds.</div></div></div>')

    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.html('<p class="pw-h">Average miss per lap, by model</p>'
                '<div class="pw-key"><span>Given the first 5 laps on a set, forecast the rest. Lower is better.</span></div>')
        lstm = b[b.index.str.startswith("LSTM (")]
        rows = b[~b.index.str.startswith("LSTM (") & ~b.index.str.startswith("Rule") & (b.index != "Mean (dummy)")]
        mae = rows["test_mae"].copy()
        if len(lstm):
            mae["LSTM"] = lstm["test_mae"].mean()
        mae = mae.sort_values(ascending=False)
        names = [f"<b>{model_label(m)}</b>" if m == FINAL_B else m for m in mae.index]
        st.plotly_chart(charts.hbar(names, list(mae.values), f"<b>{model_label(FINAL_B)}</b>", "{:.2f} s",
                                    "Seconds per lap", 330, dark="Linear baseline"),
                        theme=None, config=charts.CONFIG, width="stretch", key="mae_fig")
    with r2:
        ub = ctx["update_by_decision"]
        ub = ub[ub["data"] == "test_2025"]
        st.html('<p class="pw-h">The forecast stays accurate as the stint goes on</p>'
                f'<div class="pw-key"><span>After each lap it is shifted by its average error on the last {us["k"]} laps: '
                f'{us["test_improvement_after_lap5_pct"]:.0f}% lower error after lap 5.</span></div>')
        st.plotly_chart(charts.update_figure(ub), theme=None, config=charts.CONFIG, width="stretch",
                        key="update_fig")

    r3, r4 = st.columns(2, gap="large")
    with r3:
        st.html('<p class="pw-h">Why the pit call misses: teams stop on more than tire wear</p>'
                '<div class="pw-key"><span>Share of 2025 stops where the call landed within 3 laps of the real one.</span></div>')
        order = [o for o in ["Midpoint (no model)", "Learned layer (tire model + race context)", habit] + tire
                 if o in ss_.index]
        label = {"Midpoint (no model)": "Middle of the stint (no model)",
                 "Learned layer (tire model + race context)": "Team habits + tire wear", habit: "Team habits"}
        names = [label.get(o, "<b>PitWall: tire wear alone</b>") for o in order]
        st.plotly_chart(charts.hbar(names, [float(ss_.loc[o, "within_3_laps_pct"]) for o in order],
                                    "<b>PitWall: tire wear alone</b>", "{:.0f}%", "Stops within 3 laps (%)", 270,
                                    goal=60, xmax=78),
                        theme=None, config=charts.CONFIG, width="stretch", key="pit_fig")
        st.caption("Team habits: a second model that learned when teams stop from position, gaps and laps left. "
                   "Adding tire wear to it changes nothing, so PitWall shows both calls and leaves the decision "
                   "to the strategist.")
    with r4:
        st.html('<p class="pw-h">Where the forecast is harder</p>'
                '<div class="pw-key"><span>Average miss per lap by compound. Hards run the longest stints.</span></div>')
        ec = ctx["error_by_compound"].set_index("compound").reindex(COMPOUNDS[::-1])
        st.plotly_chart(charts.hbar([plural(c) for c in ec.index], list(ec[FINAL_B]), "", "{:.2f} s",
                                    "Seconds per lap", 230),
                        theme=None, config=charts.CONFIG, width="stretch", key="comp_fig")
        ev = ctx["error_by_event"].sort_values(FINAL_B)
        short = lambda d: ", ".join(d["event_name"].str.replace("Grand Prix", "GP"))     # noqa: E731
        st.caption(f"Easiest races: {short(ev.head(3))}. Hardest: {short(ev.tail(3)[::-1])}.")

    with st.expander("Full results tables"):
        for setup, lab in [("B", "With the first 5 laps of the stint (Setup B)"), ("A", "Conditions only (Setup A)")]:
            d = tr[tr["setup"] == setup][["model", "train_mae", "test_mae", "test_rmse", "test_r2",
                                          "test_mae_vs_baseline_pct"]].copy()
            d.columns = ["Model", "Train MAE (s)", "2025 test MAE (s)", "RMSE (s)", "R²", "vs. baseline (%)"]
            st.markdown(f"**{lab}**")
            st.dataframe(d.sort_values("2025 test MAE (s)").round(3), hide_index=True, width="stretch")
        st.markdown("**Forecast error by how many laps have been seen (2025)**")
        u = ub[["bucket", "frozen_mae", "updated_mae", "laps"]].copy()
        u.columns = ["Laps seen on the set", "Frozen MAE (s)", "Updated MAE (s)", "Laps scored"]
        st.dataframe(u.round(3), hide_index=True, width="stretch")
        st.markdown("**Pit stop backtest (2025)**")
        t = ctx["strategy_summary"].rename(columns={
            "method": "Method", "stints": "Stops", "within_3_laps_pct": "Within 3 laps (%)",
            "within_5_laps_pct": "Within 5 laps (%)", "mean_abs_error_laps": "Mean error (laps)",
            "median_error_laps": "Median signed error (laps)", "actual_in_window_pct": "Actual stop in window (%)",
            "median_window_laps": "Window width (laps)"})
        t = t.drop(columns=[c for c in ["monotone", "smooth", "rolling", "update"] if c in t])
        st.dataframe(t.round(1), hide_index=True, width="stretch")
