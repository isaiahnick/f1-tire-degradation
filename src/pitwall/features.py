"""Stage 3: build the target and the model-ready feature table.

Input:  data/interim/laps_clean.parquet
Output: data/processed/features.parquet  one row per target lap, with
          - target `deg_s`, the feature columns from the config, ids
          - `split` (train / test by season) and `cv_fold` (GroupKFold by race)
        reports/metrics/dataset_summary.json

        data/processed/practice_wear.csv  practice long-run wear per race and compound

Target: fuel-corrected lap time minus the stint's reference pace, where the
reference is the median of the stint's first `reference_laps` clean laps. Those
reference laps are used to define the target and are not themselves target rows.

Features added after the midpoint (each one only uses information a pit wall has
at that moment):
  prac_slope_field  wear slope (s/lap) from Friday/Saturday practice long runs,
                    field median for this race and compound (known before the race)
  pace_vs_field     the stint's reference laps vs the field median on the same laps
  dirty_air_share,  traffic during the observed window (Setup B only)
  window_gap_mean
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .ingest import load_practice

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def practice_wear(prac: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Wear slope per practice long run, then the field median per (race, compound).

    A long run is one set with enough green, non-pit laps. Laps far from the run's
    median (push laps, cool-down laps) are dropped. The slope is a Theil-Sen
    estimate (median of pairwise slopes), robust to the odd traffic lap, plus the
    fuel burned over the run, which would otherwise make the tires look better.
    """
    c = cfg["practice"]
    sc = "|".join(cfg["cleaning"]["sc_track_codes"])
    ok = (prac["lap_time_s"].notna() & prac["compound"].isin(COMPOUNDS) & prac["pit_in_s"].isna()
          & prac["pit_out_s"].isna() & ~prac["track_status"].fillna("").str.contains(sc)
          & ~prac["deleted"].fillna(False).astype(bool))
    runs = []
    for (year, ev, sess, drv, stint), g in prac[ok].groupby(["year", "event_name", "session", "driver", "stint"]):
        med = g["lap_time_s"].median()
        g = g[g["lap_time_s"].between(c["keep_band"][0] * med, c["keep_band"][1] * med)]
        x, y = g["tyre_life"].to_numpy(float), g["lap_time_s"].to_numpy(float)
        if len(g) < c["min_laps"] or np.ptp(x) < c["min_span_laps"]:
            continue
        i, j = np.triu_indices(len(x), 1)
        keep = x[j] != x[i]
        slope = float(np.median((y[j] - y[i])[keep] / (x[j] - x[i])[keep])) + c["fuel_s_per_lap"]
        runs.append({"year": year, "event_name": ev, "compound": g["compound"].iloc[0], "slope": slope})
    runs = pd.DataFrame(runs)
    runs = runs[runs["slope"].between(*c["slope_range"])]
    return (runs.groupby(["year", "event_name", "compound"])["slope"]
            .agg(prac_slope_field="median", prac_runs="size").reset_index())


def add_practice(df: pd.DataFrame, wear: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join practice wear; races/compounds with no long run get the compound median
    from the training seasons (wet practice, or nobody ran that compound)."""
    df = df.merge(wear[["year", "event_name", "compound", "prac_slope_field"]],
                  on=["year", "event_name", "compound"], how="left")
    tr = wear[wear["year"].isin(cfg["train_seasons"])]
    df["prac_slope_field"] = df["prac_slope_field"].fillna(
        df["compound"].map(tr.groupby("compound")["prac_slope_field"].median()))
    return df


def add_pace_vs_field(laps: pd.DataFrame, reference_laps: int) -> pd.DataFrame:
    """Car pace: the stint's reference laps minus the field median on those same laps."""
    laps = laps.sort_values(["stint_id", "lap_number"]).copy()
    field = laps.groupby(["race_id", "lap_number"])["lap_time_fc_s"].median().rename("field_s")
    first = laps[laps.groupby("stint_id").cumcount() < reference_laps].join(field, on=["race_id", "lap_number"])
    pace = (first["lap_time_fc_s"] - first["field_s"]).groupby(first["stint_id"]).mean().rename("pace_vs_field")
    return laps.join(pace, on="stint_id")


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


def add_history(df: pd.DataFrame, reference_laps: int, history_laps: int,
                dirty_air_s: float = 1.0) -> pd.DataFrame:
    """Setup B features: what the pit wall has already seen on this set of tires.

    The first `history_laps` target laps of each stint form the observed window.
    From them we take the mean loss, the wear slope (s/lap) and how much of the
    window was spent in dirty air (within `dirty_air_s` of the car ahead). Rows after the
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
        "dirty_air_share": (win["gap_ahead_s"] < dirty_air_s).groupby(win["stint_id"]).mean(),
        "window_gap_mean": win.groupby("stint_id")["gap_ahead_s"].mean(),
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
    df["stint_lap_x_prac"] = df["stint_lap"] * df["prac_slope_field"]
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
    laps = add_pace_vs_field(laps, t["reference_laps"])
    wear = practice_wear(load_practice(cfg), cfg)
    wear.to_csv(cfg["paths"]["processed"] / "practice_wear.csv", index=False)
    laps = add_practice(laps, wear, cfg)
    df = build_target(laps, t["reference_laps"], t["min_clean_laps"])
    df = add_history(df, t["reference_laps"], t["history_laps"], cfg["features"]["dirty_air_s"])
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
