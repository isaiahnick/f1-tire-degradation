"""Shared pieces for the post-presentation improvement experiments.

Rules: every lap-model comparison uses the same grouped 5-fold CV on 2023-24
(the `cv_fold` column). 2022 rows, when used, only ever join the TRAINING side
of a fold. 2025 is never read here.
"""
from __future__ import annotations

import copy
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pitwall.config import load_config
from pitwall.models import BASELINE_INTERACTIONS, HISTORY_INTERACTIONS
from pitwall.tune import final_config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG = final_config(load_config(ROOT / "configs" / "experiments.yaml"))
SEED = 42
H = CFG["target"]["history_laps"]
DIRTY_AIR_S = 1.0


# ------------------------------------------------------------------ data
def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def load(with_2022: bool = True) -> pd.DataFrame:
    """2022-2024 feature rows (2025 dropped on read) plus the candidate new features."""
    f = pd.read_parquet(ROOT / "data_exp" / "processed" / "features.parquet")
    f = f[f["year"] <= 2024 if with_2022 else f["year"].between(2023, 2024)].copy()
    f = add_new_features(f)
    return f.reset_index(drop=True)


def add_new_features(f: pd.DataFrame) -> pd.DataFrame:
    laps = pd.read_parquet(ROOT / "data_exp" / "interim" / "laps_clean.parquet",
                           columns=["race_id", "stint_id", "lap_number", "lap_time_fc_s", "gap_ahead_s"])
    laps = laps.sort_values(["stint_id", "lap_number"])
    laps["clean_idx"] = laps.groupby("stint_id").cumcount()

    # 1) car pace vs field: stint reference laps vs the field median on the same laps
    field = laps.groupby(["race_id", "lap_number"])["lap_time_fc_s"].median().rename("field_s")
    ref = laps[laps["clean_idx"] < CFG["target"]["reference_laps"]].join(field, on=["race_id", "lap_number"])
    pace = (ref["lap_time_fc_s"] - ref["field_s"]).groupby(ref["stint_id"]).mean().rename("pace_vs_field")
    f = f.join(pace, on="stint_id")

    # 2) dirty air in the observed window (first H target laps)
    lo, hi = CFG["target"]["reference_laps"], CFG["target"]["reference_laps"] + H
    win = laps[(laps["clean_idx"] >= lo) & (laps["clean_idx"] < hi)]
    da = pd.DataFrame({
        "dirty_air_share": (win["gap_ahead_s"] < DIRTY_AIR_S).groupby(win["stint_id"]).mean(),
        "window_gap_mean": win["gap_ahead_s"].clip(upper=10).groupby(win["stint_id"]).mean(),
    })
    f = f.join(da, on="stint_id")

    # 3) practice long-run wear slope for this weekend
    lr = pd.read_parquet(ROOT / "data_exp" / "processed" / "practice_long_runs.parquet")
    lr["ev"] = lr["event_name"].map(_norm)
    f["ev"] = f["event_name"].map(_norm)
    field_slope = lr.groupby(["year", "ev", "compound"])["slope"].median().rename("prac_slope_field")
    team_slope = lr.groupby(["year", "ev", "team", "compound"])["slope"].median().rename("prac_slope_team")
    f = f.join(field_slope, on=["year", "ev", "compound"]).join(team_slope, on=["year", "ev", "team", "compound"])
    f["prac_slope_team"] = f["prac_slope_team"].fillna(f["prac_slope_field"])
    # fill gaps (no long run on that compound, wet practice) with the compound median
    comp_med = lr.groupby("compound")["slope"].median()
    for c in ["prac_slope_field", "prac_slope_team"]:
        f[c + "_missing"] = f[c].isna().astype(int)
        f[c] = f[c].fillna(f["compound"].map(comp_med))
    f["stint_lap_x_prac"] = f["stint_lap"] * f["prac_slope_field"]
    return f.drop(columns="ev")


# ------------------------------------------------------------------ models
TUNED = json.loads((ROOT / "reports" / "metrics" / "tuned_params.json").read_text())


def feature_sets(setup: str, extra: list[str] | None = None, extra_inter: list[str] | None = None):
    num = list(CFG["features"]["numeric"])
    cat = list(CFG["features"]["categorical"])
    inter = list(BASELINE_INTERACTIONS)
    if setup == "B":
        num += list(CFG["features"]["history"])
        inter += HISTORY_INTERACTIONS
    num += list(extra or [])
    inter += list(extra_inter or [])
    return num, cat, inter


def onehot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make(name: str, setup: str, num, cat, inter, **override):
    p = {k: v for k, v in TUNED[setup][name]["params"].items()} if name in TUNED[setup] else {}
    p.update(override)
    if name == "Ridge":
        prep = ColumnTransformer([("num", StandardScaler(), num + inter), ("cat", onehot(), cat)])
        return Pipeline([("prep", prep), ("model", Ridge(alpha=p.get("alpha", 1000)))])
    prep = ColumnTransformer([("num", "passthrough", num), ("cat", onehot(), cat)])
    if name == "Random forest":
        return Pipeline([("prep", prep), ("model", RandomForestRegressor(
            n_estimators=p.get("n_estimators", 200), min_samples_leaf=p.get("min_samples_leaf", 20),
            max_features=p.get("max_features", 0.33), n_jobs=-1, random_state=SEED))])
    gb_kw = dict(learning_rate=p.get("learning_rate", 0.03), max_iter=p.get("max_iter", 300),
                 max_leaf_nodes=p.get("max_leaf_nodes", 15), min_samples_leaf=p.get("min_samples_leaf", 40),
                 early_stopping=False, random_state=SEED)
    gb_kw.update({k: v for k, v in override.items() if k in ("loss", "quantile", "monotonic_cst")})
    return Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(**gb_kw))])


def rows(f: pd.DataFrame, setup: str) -> pd.DataFrame:
    return f[f["in_forecast"]] if setup == "B" else f


def cv(f: pd.DataFrame, setup: str, pipe, cols: list[str], use_2022: bool = False,
       target: str = "deg_s", weight_2022: float = 1.0, label: str = "") -> dict:
    """Grouped CV on 2023-24 folds. Returns per-fold val MAE, train MAE, OOF preds."""
    d = rows(f, setup)
    main = d[d["year"].between(2023, 2024)]
    extra = d[d["year"] == 2022] if use_2022 else d.iloc[:0]
    oof = pd.Series(np.nan, index=main.index)
    va_mae, tr_mae = [], []
    t0 = time.time()
    for k in range(CFG["cv"]["n_splits"]):
        tr = pd.concat([main[main["cv_fold"] != k], extra])
        va = main[main["cv_fold"] == k]
        w = np.where(tr["year"] == 2022, weight_2022, 1.0)
        m = clone(pipe)
        m.fit(tr[cols], tr[target], model__sample_weight=w)
        oof.loc[va.index] = m.predict(va[cols])
        va_mae.append(np.mean(np.abs(va["deg_s"] - oof.loc[va.index])))
        trm = tr[tr["year"] >= 2023]
        tr_mae.append(np.mean(np.abs(trm["deg_s"] - m.predict(trm[cols]))))
    res = {"label": label, "setup": setup, "val_mae": float(np.mean(va_mae)),
           "val_mae_folds": [round(v, 4) for v in va_mae], "train_mae": float(np.mean(tr_mae)),
           "secs": round(time.time() - t0, 1), "oof": oof}
    print(f"  [{setup}] {label:45s} val {res['val_mae']:.4f}  train {res['train_mae']:.4f}  "
          f"folds {res['val_mae_folds']}  ({res['secs']}s)", flush=True)
    return res


def table(results: list[dict], base_label: str) -> pd.DataFrame:
    t = pd.DataFrame([{k: v for k, v in r.items() if k != "oof"} for r in results])
    out = []
    for setup, g in t.groupby("setup", sort=False):
        b = g[g["label"] == base_label]
        if len(b):
            bv = b["val_mae"].iloc[0]
            bf = np.array(b["val_mae_folds"].iloc[0])
            g = g.assign(delta_pct=100 * (g["val_mae"] - bv) / bv,
                         folds_better=[int((np.array(x) < bf).sum()) for x in g["val_mae_folds"]])
        out.append(g)
    return pd.concat(out, ignore_index=True)
