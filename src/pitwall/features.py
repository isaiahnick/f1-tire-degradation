"""Stage 3: build the target and the model-ready feature table.

Input:  data/interim/laps_clean.parquet
Output: data/processed/features.parquet  one row per target lap, with
          - target `deg_s`, the feature columns from the config, ids
          - `split` (train / test by season) and `cv_fold` (GroupKFold by race)
        reports/metrics/dataset_summary.json

Target: fuel-corrected lap time minus the stint's reference pace, where the
reference is the median of the stint's first `reference_laps` clean laps. Those
reference laps are used to define the target and are not themselves target rows.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def build_target(df: pd.DataFrame, reference_laps: int, min_clean_laps: int) -> pd.DataFrame:
    df = df.sort_values(["stint_id", "lap_number"]).copy()
    df["clean_idx"] = df.groupby("stint_id").cumcount()          # 0-based clean lap order
    df["n_clean"] = df.groupby("stint_id")["lap_number"].transform("size")
    ref = (df[df["clean_idx"] < reference_laps]
           .groupby("stint_id")["lap_time_fc_s"].median().rename("ref_pace_s"))
    df = df.join(ref, on="stint_id")
    df = df[(df["n_clean"] >= min_clean_laps) & (df["clean_idx"] >= reference_laps)].copy()
    df["deg_s"] = df["lap_time_fc_s"] - df["ref_pace_s"]
    return df


def add_history(df: pd.DataFrame, reference_laps: int, history_laps: int) -> pd.DataFrame:
    """Setup B features: what the pit wall has already seen on this set of tires.

    The first `history_laps` target laps of each stint form the observed window.
    From them we take the mean loss and the wear slope (s/lap). Rows after the
    window are the forecast rows (`in_forecast`); `laps_ahead` is how far past the
    window each forecast lap is. Window laps never appear as forecast rows.
    """
    df = df.copy()
    lo, hi = reference_laps, reference_laps + history_laps
    win = df[(df["clean_idx"] >= lo) & (df["clean_idx"] < hi)]

    def slope(g: pd.DataFrame) -> float:
        x = g["stint_lap"].to_numpy(float)
        return float(np.polyfit(x, g["deg_s"].to_numpy(float), 1)[0]) if np.ptp(x) > 0 else np.nan

    stats = pd.DataFrame({
        "early_deg_mean": win.groupby("stint_id")["deg_s"].mean(),
        "early_deg_slope": win.groupby("stint_id")[["stint_lap", "deg_s"]].apply(slope),
        "window_end_lap": win.groupby("stint_id")["stint_lap"].max(),
        "window_n": win.groupby("stint_id").size(),
    })
    df = df.join(stats, on="stint_id")
    df["laps_ahead"] = df["stint_lap"] - df["window_end_lap"]
    df["in_forecast"] = (df["clean_idx"] >= hi) & (df["window_n"] == history_laps) \
        & df["early_deg_slope"].notna()
    return df.drop(columns=["window_end_lap", "window_n"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["laps_remaining"] = df["total_laps"] - df["lap_number"]
    df["race_progress"] = df["lap_number"] / df["total_laps"]
    df["fresh_tyre"] = df["fresh_tyre"].fillna(False).astype(int)
    # explicit interactions for the linear models (trees find these on their own)
    for comp in COMPOUNDS:
        df[f"stint_lap_x_{comp.lower()}"] = df["stint_lap"] * (df["compound"] == comp)
    df["stint_lap_x_track_temp"] = df["stint_lap"] * df["track_temp"]
    df["early_slope_x_ahead"] = df["early_deg_slope"] * df["laps_ahead"]
    return df


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["split"] = np.where(df["year"].isin(cfg["train_seasons"]), "train",
                           np.where(df["year"].isin(cfg["test_seasons"]), "test", "unused"))
    df["cv_fold"] = -1
    tr = df.index[df["split"] == "train"]
    gkf = GroupKFold(n_splits=cfg["cv"]["n_splits"])
    for fold, (_, va) in enumerate(gkf.split(df.loc[tr], groups=df.loc[tr, "race_id"])):
        df.loc[tr[va], "cv_fold"] = fold
    return df


def run(cfg: dict) -> pd.DataFrame:
    t = cfg["target"]
    laps = pd.read_parquet(cfg["paths"]["interim"] / "laps_clean.parquet")
    df = build_target(laps, t["reference_laps"], t["min_clean_laps"])
    df = add_history(df, t["reference_laps"], t["history_laps"])
    df = add_features(df)
    df = assign_splits(df, cfg)

    f = cfg["features"]
    ids = ["race_id", "stint_id", "year", "round", "event_name", "driver", "lap_number",
           "lap_time_s", "lap_time_fc_s", "ref_pace_s", "split", "cv_fold"]
    inter = [c for c in df.columns if c.startswith(("stint_lap_x_", "early_slope_x_"))]
    cols = ids + f["numeric"] + f["categorical"] + inter + f["history"] + ["in_forecast", "deg_s"]
    out = df[cols].reset_index(drop=True)
    out.to_parquet(cfg["paths"]["processed"] / "features.parquet", index=False)

    summary = {
        "rows": len(out),
        "stints": int(out["stint_id"].nunique()),
        "races": int(out["race_id"].nunique()),
        "by_split": out.groupby("split").agg(rows=("deg_s", "size"),
                                             races=("race_id", "nunique"),
                                             stints=("stint_id", "nunique")).to_dict("index"),
        "rows_per_fold": out[out["split"] == "train"]["cv_fold"].value_counts().sort_index().to_dict(),
        "n_numeric": len(f["numeric"]), "n_categorical": len(f["categorical"]),
        "n_interactions": len(inter),
        "target_train": out.loc[out["split"] == "train", "deg_s"].describe().round(3).to_dict(),
        "forecast_rows_by_split": out[out["in_forecast"]].groupby("split").size().to_dict(),
        "forecast_stints_by_split": out[out["in_forecast"]].groupby("split")["stint_id"].nunique().to_dict(),
        "missing_after_features": int(out[f["numeric"] + f["categorical"]].isna().sum().sum()),
    }
    (cfg["paths"]["metrics"] / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"features: {summary['rows']:,} rows, {summary['stints']:,} stints, "
          f"{summary['races']} races; split {summary['by_split']}")
    return out
