"""Turns a user's selection into predictions and a pit recommendation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.features import COMPOUNDS
from pitwall.models import feature_lists
from pitwall.strategy import curves, layer_features, layer_predict, new_tire_rows, recommend
from pitwall.update import recent_bias


def best_setup(ctx, setup: str) -> str:
    t = ctx["tuned"][setup]
    return min(t, key=lambda m: t[m]["cv_val_mae"])


def decision_laps(ctx, stint_id: str) -> list[int]:
    """Laps the pit wall can decide on: from the end of the 5-lap window to the last clean lap."""
    g = ctx["feat"][ctx["feat"]["stint_id"] == stint_id].sort_values("lap_number")
    H = ctx["cfg"]["target"]["history_laps"]
    return [int(x) for x in g["lap_number"].iloc[H - 1:]]


def _setup(ctx, stint_id: str, next_compound: str):
    feat = ctx["feat"]
    g = feat[feat["stint_id"] == stint_id].sort_values("lap_number")
    info = ctx["stints"][ctx["stints"]["stint_id"] == stint_id].iloc[0]
    has_next = pd.notna(info["next_last_lap"])
    seg_end = int(info["next_last_lap"]) if has_next else int(info["total_laps"])
    same = has_next and info["next_compound"] == next_compound
    life = info["next_tyre_life_start"] if same else 1
    fresh = info["next_fresh"] if same else True
    off = ctx["offsets"]
    o = off[(off["compound_old"] == info["compound"]) & (off["compound_new"] == next_compound)]["offset_s"]
    return g, info, seg_end, life, fresh, (float(o.iloc[0]) if len(o) else 0.0)


def stint_recommendation(ctx, stint_id: str, model_name: str, next_compound: str, decision_lap: int | None = None):
    cfg, opts = ctx["fcfg"], ctx["options"]
    H = ctx["cfg"]["target"]["history_laps"]
    g, info, seg_end, life, fresh, offset = _setup(ctx, stint_id, next_compound)
    laps = list(g["lap_number"].astype(int))
    d = laps.index(decision_lap) + 1 if decision_lap in laps else H
    state = g.iloc[d - 1]
    model_A = ctx["A"]["Linear baseline"] if model_name == "Linear baseline" else ctx["A"][best_setup(ctx, "A")]
    lstm_fn = ctx["lstm_fn"](state) if model_name == "LSTM" else None
    model_B = None if lstm_fn else ctx["B"][model_name]
    bias, n_seen = 0.0, 0
    if model_B is not None:          # online error correction (stage `update`), only acts after lap 5
        num, cat, inter = feature_lists(cfg, "B")
        resid = np.where(g["in_forecast"], g["deg_s"] - model_B.predict(g[num + cat + inter]), np.nan)
        us = ctx["update_summary"]
        bias = recent_bias(resid[:d], us["k"], us["shrink"])
        n_seen = int(np.sum(~np.isnan(resid[:d])))
    prac = ctx["prac"].get((state["year"], state["event_name"], next_compound))
    kw = dict(next_compound=next_compound, next_life_start=life, next_fresh=fresh, model_B=model_B,
              model_A=model_A, cfg=cfg, offset_s=offset, lstm_fn=lstm_fn, monotone=opts["monotone"],
              smooth=opts["smooth"], bias=bias, prac=prac)
    table, best = recommend(state, seg_end, window_s=ctx["cfg"]["strategy"]["window_s"], **kw)
    now = int(state["lap_number"])
    d_old, _ = curves(state, seg_end, **kw)
    # new-tire curve as it would run after the recommended lap
    na, ca, ia = feature_lists(cfg, "A")
    new = new_tire_rows(state, best, seg_end - best, next_compound, life, fresh, prac)
    new_pred = model_A.predict(new[na + ca + ia]) if len(new) else np.array([])
    if len(new_pred) and opts["smooth"] and len(new_pred) >= 3:
        x = np.arange(len(new_pred))
        new_pred = np.polyval(np.polyfit(x, new_pred, 1), x)
    if len(new_pred) and opts["monotone"]:
        new_pred = np.maximum.accumulate(new_pred)
    return {
        "state": state, "info": info, "laps_obs": g, "d": d, "segment_end": seg_end, "table": table,
        "best": best, "offset": offset, "bias": bias, "n_seen": n_seen,
        "old_curve": pd.DataFrame({"lap": np.arange(now + 1, seg_end), "loss": d_old}),
        "new_curve": pd.DataFrame({"lap": new["lap_number"] if len(new) else [], "loss": new_pred + offset}),
        "actual_pit": int(info["last_lap"]) if bool(info["pit_in"]) else None,
    }


def typical_team_call(ctx, stint_id: str, next_compound: str) -> int | None:
    """Learned strategy layer: where teams usually stop, given the tire model's lap-5
    window and race context. Needs a real next stint (segment end)."""
    lay = ctx.get("layer")
    if lay is None:
        return None
    g, info, seg_end, life, fresh, offset = _setup(ctx, stint_id, next_compound)
    if pd.isna(info["next_last_lap"]):
        return None
    H = ctx["cfg"]["target"]["history_laps"]
    s = g.iloc[H - 1]
    bB, bA = best_setup(ctx, "B"), best_setup(ctx, "A")
    t, p = recommend(s, seg_end, next_compound, life, fresh, ctx["B"][bB], ctx["A"][bA], ctx["fcfg"],
                     offset_s=offset, window_s=ctx["cfg"]["strategy"]["window_s"],
                     monotone=ctx["options"]["monotone"], smooth=ctx["options"]["smooth"],
                     prac=ctx["prac"].get((s["year"], s["event_name"], next_compound)))
    w = t.loc[t["in_window"], "pit_lap"]
    row = pd.DataFrame([{
        "stint": s["stint"], "position": s["position"], "gap_ahead_s": s["gap_ahead_s"],
        "gap_behind_s": ctx["gaps"].get((s["race_id"], s["driver"], int(s["lap_number"])), 10.0),
        "laps_remaining": s["laps_remaining"], "early_deg_slope": s["early_deg_slope"],
        "early_deg_mean": s["early_deg_mean"], "decision_lap": int(s["lap_number"]), "segment_end": seg_end,
        "circuit": info["circuit"], "compound": info["compound"], "next_compound": next_compound,
        "rec_m": p, "rec_m_win_lo": int(w.min()), "rec_m_win_hi": int(w.max())}])
    with_tm, m = lay["layers"]["Learned layer (tire model + race context)"]
    return int(layer_predict(m, row, layer_features(row, "m", lay["pit_loss"], with_tm))[0])


def plan_curves(ctx, circuit: str, n_laps: int, team: str, model_name: str) -> pd.DataFrame:
    """Setup A what-if: fresh-tire loss curve for each compound under chosen conditions."""
    cfg = ctx["fcfg"]
    na, ca, ia = feature_lists(cfg, "A")
    base = ctx["feat"][(ctx["feat"]["circuit"] == circuit)].sort_values("year").iloc[-1].copy()
    base["team"] = team
    base["laps_remaining"] = base["laps_remaining"] + base["lap_number"] - 1
    base["lap_number"], base["stint"] = 1, 1
    base["position"], base["gap_ahead_s"] = 10, 2.0
    out = []
    model = ctx["A"][model_name]
    for comp in COMPOUNDS:
        prac = ctx["prac"].get((base["year"], base["event_name"], comp))
        rows = new_tire_rows(base, 0, n_laps, comp, 1, True, prac)
        out.append(pd.DataFrame({"compound": comp, "stint_lap": rows["stint_lap"],
                                 "loss": model.predict(rows[na + ca + ia])}))
    return pd.concat(out)
