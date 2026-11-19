"""Stage 8: keep the forecast honest as the stint goes on (online error correction).

Setup B looks at the first 5 laps on a set and then forecasts the rest. In a
real race the pit wall keeps watching, so after every lap it knows how wrong the
forecast has been on the laps it has already seen. This stage tests the simplest
way to use that: add the average of the model's last k errors to every future
prediction, shrunk toward zero when only a few errors have been seen:

    bias = mean(last k residuals) * n / (n + shrink)

Nothing is refit, so there is no new leakage risk: at decision lap d the bias
only uses laps 1..d. k and shrink are chosen by grouped CV on the training
seasons; 2025 is scored once with the saved final model.

Outputs (reports/metrics):
  update_cv_grid.csv        CV MAE for each (k, shrink)
  update_by_decision.csv    frozen vs updated MAE by decision point, CV and 2025
  update_by_horizon.csv     same, by laps ahead of the decision
  update_summary.json
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .models import feature_lists

TARGET = "deg_s"
D_BINS = [4, 5, 9, 14, 19, 99]
D_LABELS = ["lap 5", "6-9", "10-14", "15-19", "20+"]
H_BINS = [0, 5, 10, 15, 20, 99]
H_LABELS = ["1-5", "6-10", "11-15", "16-20", "21+"]


def recent_bias(resid_seen: np.ndarray, k: int, shrink: float) -> float:
    """Mean of the last k residuals on laps already seen, shrunk toward 0."""
    r = resid_seen[~np.isnan(resid_seen)][-k:]
    return float(r.mean() * len(r) / (len(r) + shrink)) if len(r) else 0.0


def decision_rows(df: pd.DataFrame, pred: np.ndarray, history_laps: int, step: int,
                  grid: list[tuple[int, float]]) -> pd.DataFrame:
    """For every stint and decision point d (every `step` laps from d = history_laps),
    every later lap with its frozen prediction and the corrected one for each (k, shrink)."""
    df = df.assign(_pred=pred).sort_values(["stint_id", "lap_number"])
    out = []
    for sid, g in df.groupby("stint_id", sort=False):
        y, p = g[TARGET].to_numpy(float), g["_pred"].to_numpy(float)
        resid = np.where(g["in_forecast"].to_numpy(), y - p, np.nan)
        lap = g["stint_lap"].to_numpy(float)
        for d in range(history_laps, len(g), step):
            row = {"stint_id": sid, "cv_fold": g["cv_fold"].iloc[0], "d": d}
            fut = slice(d, len(g))
            block = pd.DataFrame({"y": y[fut], "frozen": p[fut], "ahead": lap[fut] - lap[d - 1]})
            for k, s in grid:
                block[f"upd_k{k}_s{s:g}"] = p[fut] + recent_bias(resid[:d], k, s)
            out.append(block.assign(**row))
    return pd.concat(out, ignore_index=True)


def by_bucket(rows: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    rows = rows.assign(d_bucket=pd.cut(rows["d"], D_BINS, labels=D_LABELS),
                       h_bucket=pd.cut(rows["ahead"], H_BINS, labels=H_LABELS))
    out = []
    for key, name in [("d_bucket", "decision"), ("h_bucket", "horizon")]:
        g = rows.groupby(key, observed=True)
        out.append(pd.DataFrame({"view": name, "bucket": list(g.groups.keys()),
                                 "frozen_mae": g.apply(lambda x: (x["y"] - x["frozen"]).abs().mean(), include_groups=False).to_numpy(),
                                 "updated_mae": g.apply(lambda x: (x["y"] - x[col]).abs().mean(), include_groups=False).to_numpy(),
                                 "laps": g.size().to_numpy(), "data": label}))
    return pd.concat(out, ignore_index=True)


def run(cfg: dict) -> dict:
    from .strategy import json_best
    from .test import tuned_models
    from .tune import final_config
    out = cfg["paths"]["metrics"]
    fcfg = final_config(cfg)
    u = cfg["update"]
    grid = [(k, s) for k in u["k_grid"] for s in u["shrink_grid"]]
    H = cfg["target"]["history_laps"]
    feat = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    best = json_best(cfg, "B")
    num, cat, inter = feature_lists(fcfg, "B")
    cols = num + cat + inter

    # grouped CV: fit B on the other folds, roll through the held-out races
    train = feat[feat["split"] == "train"]
    parts = []
    for fold in sorted(train["cv_fold"].unique()):
        tr = train[(train["cv_fold"] != fold) & train["in_forecast"]]
        va = train[train["cv_fold"] == fold]
        m = clone(tuned_models(fcfg, "B")[best]).fit(tr[cols], tr[TARGET])
        pred = np.where(va["in_forecast"], m.predict(va[cols]), np.nan)
        parts.append(decision_rows(va, pred, H, u["step"], grid))
        print(f"update: fold {fold} done", flush=True)
    cvr = pd.concat(parts, ignore_index=True)
    later = cvr[cvr["d"] > H]                    # at d = H nothing has been seen yet
    gr = pd.DataFrame([{"k": k, "shrink": s,
                        "cv_mae_all": later.groupby("cv_fold").apply(lambda x: (x["y"] - x[f"upd_k{k}_s{s:g}"]).abs().mean(), include_groups=False).mean()}
                       for k, s in grid])
    gr.loc[len(gr)] = {"k": 0, "shrink": np.nan,
                       "cv_mae_all": later.groupby("cv_fold").apply(lambda x: (x["y"] - x["frozen"]).abs().mean(), include_groups=False).mean()}
    gr.to_csv(out / "update_cv_grid.csv", index=False)
    k, s = gr.iloc[gr.iloc[:-1]["cv_mae_all"].idxmin()][["k", "shrink"]]
    k, s = int(k), float(s)
    col = f"upd_k{k}_s{s:g}"

    # 2025, once, with the saved final model
    B = joblib.load(cfg["paths"]["models"] / "final_B.joblib")["models"][best]
    te = feat[feat["split"] == "test"]
    pred = np.where(te["in_forecast"], B.predict(te[cols]), np.nan)
    ter = decision_rows(te, pred, H, u["step"], [(k, s)])

    tab = pd.concat([by_bucket(cvr, col, "cv"), by_bucket(ter, col, "test_2025")], ignore_index=True)
    tab[tab["view"] == "decision"].drop(columns="view").to_csv(out / "update_by_decision.csv", index=False)
    tab[tab["view"] == "horizon"].drop(columns="view").to_csv(out / "update_by_horizon.csv", index=False)

    def mae(r, c):
        return float((r["y"] - r[c]).abs().mean())
    tl = ter[ter["d"] > H]
    summary = {"model": best, "k": k, "shrink": s, "step": u["step"],
               "cv_frozen_mae_after_lap5": round(mae(later, "frozen"), 4),
               "cv_updated_mae_after_lap5": round(mae(later, col), 4),
               "test_frozen_mae_after_lap5": round(mae(tl, "frozen"), 4),
               "test_updated_mae_after_lap5": round(mae(tl, col), 4),
               "test_frozen_mae_all_decisions": round(mae(ter, "frozen"), 4),
               "test_updated_mae_all_decisions": round(mae(ter, col), 4)}
    summary["test_improvement_after_lap5_pct"] = round(
        100 * (1 - summary["test_updated_mae_after_lap5"] / summary["test_frozen_mae_after_lap5"]), 1)
    (out / "update_summary.json").write_text(json.dumps(summary, indent=2))
    print(gr.round(4).to_string(index=False))
    print(tab.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary
