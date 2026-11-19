"""Setup C: history from ALL laps seen so far on the current set, at any decision lap.

For a stint with clean target laps 1..n (in order) and a decision after the d-th
observed lap (d >= 5), the history features use only laps 1..d:
  obs_n, hist_mean_last5, hist_slope_all, hist_slope_last8, hist_last, hist_fit_now
and each later lap gets laps_ahead_c = its stint_lap - stint_lap of lap d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import H

HIST_C = ["obs_n", "hist_mean_last5", "hist_slope_all", "hist_slope_last8", "hist_last",
          "hist_fit_now", "early_deg_mean", "early_deg_slope", "laps_ahead_c"]
INTER_C = ["slope_all_x_ahead"]


def _fit(x, y):
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0, float(np.mean(y))
    s, b = np.polyfit(x, y, 1)
    return float(s), float(b)


def history_at(x: np.ndarray, y: np.ndarray) -> dict:
    s_all, b_all = _fit(x, y)
    s8, _ = _fit(x[-8:], y[-8:])
    return {"obs_n": len(x), "hist_mean_last5": float(y[-5:].mean()), "hist_slope_all": s_all,
            "hist_slope_last8": s8, "hist_last": float(y[-1]), "hist_fit_now": s_all * x[-1] + b_all,
            "last_obs_stint_lap": float(x[-1])}


def history_table(f: pd.DataFrame, step: int = 1) -> pd.DataFrame:
    """One row per (stint, decision d) for d = H, H+step, ..., n-1."""
    out = []
    for sid, g in f.sort_values(["stint_id", "lap_number"]).groupby("stint_id", sort=False):
        x, y, laps = g["stint_lap"].to_numpy(float), g["deg_s"].to_numpy(float), g["lap_number"].to_numpy()
        for d in range(H, len(g), step):
            out.append({"stint_id": sid, "d": d, "decision_lap": int(laps[d - 1]), **history_at(x[:d], y[:d])})
    return pd.DataFrame(out)


def c_rows(f: pd.DataFrame, step: int = 2, max_ahead: int = 40) -> pd.DataFrame:
    """Training/eval rows: every later lap of the stint, for every decision d."""
    ht = history_table(f, step)
    f = f.sort_values(["stint_id", "lap_number"]).copy()
    f["clean_pos"] = f.groupby("stint_id").cumcount() + 1          # 1-based order among target laps
    m = f.merge(ht, on="stint_id")
    m = m[m["clean_pos"] > m["d"]].copy()
    m["laps_ahead_c"] = m["stint_lap"] - m["last_obs_stint_lap"]
    m = m[m["laps_ahead_c"] <= max_ahead]
    m["slope_all_x_ahead"] = m["hist_slope_all"] * m["laps_ahead_c"]
    return m.reset_index(drop=True)
