"""Turns a user's selection into predictions and a pit recommendation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.features import COMPOUNDS
from pitwall.models import feature_lists
from pitwall.strategy import decision_state, new_tire_rows, recommend


def best_setup_a(ctx) -> str:
    t = ctx["tuned"]["A"]
    return min(t, key=lambda m: t[m]["cv_val_mae"])


def stint_recommendation(ctx, stint_id: str, model_name: str, next_compound: str):
    feat, cfg = ctx["feat"], ctx["fcfg"]
    g = feat[feat["stint_id"] == stint_id].sort_values("lap_number")
    state = decision_state(g, ctx["cfg"]["target"]["history_laps"])
    info = ctx["stints"][ctx["stints"]["stint_id"] == stint_id].iloc[0]
    has_next = pd.notna(info["next_last_lap"])
    seg_end = int(info["next_last_lap"]) if has_next else int(info["total_laps"])
    life = info["next_tyre_life_start"] if has_next and info["next_compound"] == next_compound else 1
    fresh = info["next_fresh"] if has_next and info["next_compound"] == next_compound else True
    off = ctx["offsets"]
    o = off[(off["compound_old"] == info["compound"]) & (off["compound_new"] == next_compound)]["offset_s"]
    offset = float(o.iloc[0]) if len(o) else 0.0
    model_A = ctx["A"]["Linear baseline"] if model_name == "Linear baseline" else ctx["A"][best_setup_a(ctx)]
    lstm_fn = ctx["lstm_fn"](state) if model_name == "LSTM" else None
    model_B = None if lstm_fn else ctx["B"][model_name]
    mono = bool(ctx["strategy_summary"]["monotone"].iloc[0])
    table, best = recommend(state, seg_end, next_compound, life, fresh, model_B, model_A, cfg,
                            offset_s=offset, lstm_fn=lstm_fn, monotone=mono)
    # curves for plotting: current tires to segment end, new tires after the recommended lap
    num, cat, inter = feature_lists(cfg, "B")
    from pitwall.strategy import old_tire_rows
    laps = np.arange(int(state["lap_number"]) + 1, seg_end + 1)
    old = old_tire_rows(state, laps)
    old_pred = lstm_fn(old) if lstm_fn else model_B.predict(old[num + cat + inter])
    if mono:
        old_pred = np.maximum.accumulate(np.concatenate([[state["deg_s"]], old_pred]))[1:]
    na, ca, ia = feature_lists(cfg, "A")
    new = new_tire_rows(state, best, seg_end - best, next_compound, life, fresh)
    new_pred = model_A.predict(new[na + ca + ia]) if len(new) else np.array([])
    if mono and len(new_pred):
        new_pred = np.maximum.accumulate(new_pred)
    new_pred = new_pred + offset
    return {
        "state": state, "info": info, "laps_obs": g, "segment_end": seg_end, "table": table,
        "best": best, "offset": offset,
        "old_curve": pd.DataFrame({"lap": laps, "loss": old_pred}),
        "new_curve": pd.DataFrame({"lap": new["lap_number"] if len(new) else [], "loss": new_pred}),
        "actual_pit": int(info["last_lap"]) if bool(info["pit_in"]) else None,
    }


def plan_curves(ctx, circuit: str, n_laps: int, team: str, model_name: str) -> pd.DataFrame:
    """Setup A what-if: fresh-tire loss curve for each compound under chosen conditions."""
    cfg = ctx["fcfg"]
    na, ca, ia = feature_lists(cfg, "A")
    base = ctx["feat"][(ctx["feat"]["circuit"] == circuit)].iloc[0].copy()
    base["team"] = team
    base["laps_remaining"] = base["laps_remaining"] + base["lap_number"] - 1
    base["lap_number"], base["stint"] = 1, 1
    base["position"], base["gap_ahead_s"] = 10, 2.0
    out = []
    model = ctx["A"][model_name]
    for comp in COMPOUNDS:
        rows = new_tire_rows(base, 0, n_laps, comp, 1, True)
        out.append(pd.DataFrame({"compound": comp, "stint_lap": rows["stint_lap"],
                                 "loss": model.predict(rows[na + ca + ia])}))
    return pd.concat(out)
