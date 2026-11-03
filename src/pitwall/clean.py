"""Stage 2: clean laps and fuel-correct lap times.

Input:  data/raw/*.parquet (canonical schema)
Output: data/interim/laps_clean.parquet   clean, representative green-flag laps
        reports/metrics/cleaning_funnel.csv   laps removed at each step
        reports/metrics/dropped_races.csv     wet races removed

Each filter is applied in order and logged, so the report can show exactly how
many laps each rule removed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .ingest import load_raw

KEY = ["year", "round", "driver"]


def add_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_id"] = df["year"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)
    df["stint_id"] = (df["race_id"] + "_" + df["driver"].astype(str) + "_"
                      + df["stint"].fillna(-1).astype(int).astype(str))
    return df


def add_gap_ahead(df: pd.DataFrame, cap_s: float = 10.0) -> pd.DataFrame:
    """Gap in seconds to the car ahead in race order when this lap started.

    Computed on ALL laps before any filtering, so a car ahead that is about to pit
    still counts as traffic. The leader gets the cap (clean air).
    """
    df = df.sort_values(["race_id", "lap_number", "lap_start_s"]).copy()
    gap = df.groupby(["race_id", "lap_number"])["lap_start_s"].diff()
    df["gap_ahead_s"] = gap.fillna(cap_s).clip(lower=0, upper=cap_s)
    return df


def add_stint_context(df: pd.DataFrame) -> pd.DataFrame:
    """Tire life when the set was fitted, from the raw stint (before filtering)."""
    df = df.copy()
    df["tyre_life_start"] = df.groupby("stint_id")["tyre_life"].transform("min")
    df["stint_lap"] = df["tyre_life"] - df["tyre_life_start"] + 1
    return df


def fuel_correct(df: pd.DataFrame, start_kg: float, s_per_kg: float) -> pd.DataFrame:
    """Remove the fuel-weight effect so lap time changes reflect tire wear.

    Fuel is assumed to start at `start_kg` and burn linearly to zero at the
    scheduled race distance. We use the average fuel load during the lap.
    """
    df = df.copy()
    frac_left = 1 - (df["lap_number"] - 0.5) / df["total_laps"]
    df["fuel_kg"] = (start_kg * frac_left).clip(lower=0)
    df["lap_time_fc_s"] = df["lap_time_s"] - s_per_kg * df["fuel_kg"]
    return df


def flag_stint_outliers(df: pd.DataFrame, window: int, k: float, min_s: float) -> pd.Series:
    """Robust outliers: residual from a rolling median inside each stint.

    A lap is an outlier if |residual| > k * MAD-sigma of that race's residuals and
    also bigger than `min_s` seconds. Catches mistakes, traffic and lock-ups that
    the 107% rule misses.
    """
    order = df.index
    df = df.sort_values(["stint_id", "lap_number"])
    med = (df.groupby("stint_id")["lap_time_fc_s"]
             .transform(lambda s: s.rolling(window, center=True, min_periods=3).median()))
    resid = df["lap_time_fc_s"] - med
    mad = resid.abs().groupby(df["race_id"]).transform("median") * 1.4826
    out = (resid.abs() > k * mad) & (resid.abs() > min_s)
    return out.reindex(order).fillna(False)


def estimate_fuel_effect(df: pd.DataFrame) -> dict:
    """Data check on the fuel assumption (train seasons only).

    Within each race-driver, regress raw lap time on fuel load plus a wear rate per
    compound. Fuel and tire age are only separable across a driver's different
    stints, which is why the race-driver (not stint) is the fixed effect.
    Returns the pooled estimate and the spread of per-race estimates in s/kg.
    """
    g = df["race_id"] + "_" + df["driver"].astype(str)
    X = pd.DataFrame({"fuel_kg": df["fuel_kg"].astype(float),
                      "tyre_life_start": df["tyre_life_start"].astype(float)})
    for comp in ["SOFT", "MEDIUM"]:
        X[f"is_{comp}"] = (df["compound"] == comp).astype(float)
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        X[f"age_{comp}"] = (df["stint_lap"] * (df["compound"] == comp)).astype(float)
    y = df["lap_time_s"].astype(float)
    Xd = X - X.groupby(g.values).transform("mean")
    yd = y - y.groupby(g.values).transform("mean")
    pooled = np.linalg.lstsq(Xd.values, yd.values, rcond=None)[0]
    per_race = [np.linalg.lstsq(Xd.loc[i].values, yd.loc[i].values, rcond=None)[0][0]
                for i in df.groupby("race_id").groups.values()]
    coefs = dict(zip(X.columns, pooled.round(4)))
    return {"pooled_s_per_kg": float(coefs["fuel_kg"]),
            "per_race_median_s_per_kg": float(np.median(per_race)),
            "per_race_iqr": [float(np.percentile(per_race, 25)), float(np.percentile(per_race, 75))],
            "n_races": len(per_race),
            "wear_rate_s_per_lap": {k[4:]: float(v) for k, v in coefs.items() if k.startswith("age_")}}


def run(cfg: dict) -> pd.DataFrame:
    c = cfg["cleaning"]
    df = add_ids(load_raw(cfg))
    df = add_gap_ahead(df)
    df = add_stint_context(df)
    df = fuel_correct(df, cfg["fuel"]["start_kg"], cfg["fuel"]["s_per_kg"])

    funnel = [("raw laps", len(df), df["race_id"].nunique())]

    def step(name: str, keep: pd.Series) -> None:
        nonlocal df
        df = df[keep.fillna(False).astype(bool)].copy()
        funnel.append((name, len(df), df["race_id"].nunique()))

    # 1. wet races: too many inter/wet laps to treat as a dry race
    wet_share = df.groupby("race_id")["compound"].apply(lambda s: s.isin(c["wet_compounds"]).mean())
    wet_races = wet_share[wet_share > c["wet_race_lap_share"]]
    dropped = (df[df["race_id"].isin(wet_races.index)]
               .groupby("race_id")[["year", "event_name"]].first()
               .assign(wet_lap_share=wet_races.round(3)))
    step("remove wet races", ~df["race_id"].isin(wet_races.index))
    step("excluded events", ~df["event_name"].isin(c.get("exclude_events", [])))

    # 2. missing values in fields the target needs
    step("missing lap time / tire info",
         df[["lap_time_s", "compound", "stint", "tyre_life"]].notna().all(axis=1))
    # 3. only slick compounds
    step("non-dry compound", df["compound"].isin(c["dry_compounds"]))
    # 4. standing start lap
    step("lap 1", df["lap_number"] > 1)
    # 5. in-laps and out-laps include pit lane time
    step("pit in / out laps", df["pit_in_s"].isna() & df["pit_out_s"].isna())
    # 6. safety car, VSC, red flag at any point in the lap
    sc = df["track_status"].fillna("").apply(lambda s: any(code in s for code in c["sc_track_codes"]))
    step("safety car / VSC / red flag", ~sc)
    # 7. rain on track during the lap
    step("rainfall during lap", ~df["rainfall"].fillna(False).astype(bool))
    # 8. FastF1 timing quality flags
    step("inaccurate / deleted / generated",
         df["is_accurate"].fillna(False).astype(bool)
         & ~df["deleted"].fillna(False).astype(bool)
         & ~df["fastf1_generated"].fillna(False).astype(bool))
    # 9. 107% rule against the race's fastest remaining lap, on fuel-corrected time
    #    (raw times would unfairly cut heavy-fuel early laps)
    fastest = df.groupby("race_id")["lap_time_fc_s"].transform("min")
    step("slower than 107% of fastest", df["lap_time_fc_s"] <= c["quicklap_threshold"] * fastest)
    # 10. within-stint robust outliers on fuel-corrected time
    out = flag_stint_outliers(df, c["outlier_window"], c["outlier_mad_k"], c["outlier_min_s"])
    step("within-stint outliers", ~out)

    # weather: a handful of gaps; fill inside the race only (never across races)
    wcols = ["air_temp", "track_temp", "humidity", "pressure", "wind_speed"]
    n_missing_weather = int(df[wcols].isna().any(axis=1).sum())
    df = df.sort_values(["race_id", "lap_start_s"])
    df[wcols] = df.groupby("race_id")[wcols].transform(lambda s: s.ffill().bfill())

    out_dir = cfg["paths"]["metrics"]
    f = pd.DataFrame(funnel, columns=["step", "laps_remaining", "races"])
    f["laps_removed"] = (-f["laps_remaining"].diff()).fillna(0).astype(int)
    f.to_csv(out_dir / "cleaning_funnel.csv", index=False)
    dropped.to_csv(out_dir / "dropped_races.csv")
    (out_dir / "weather_filled.txt").write_text(f"{n_missing_weather}\n")
    fuel = estimate_fuel_effect(df[df["year"].isin(cfg["train_seasons"])])
    (out_dir / "fuel_check.json").write_text(json.dumps(fuel, indent=2))
    print(f"clean: fuel effect estimated from data = {fuel['pooled_s_per_kg']:.4f} s/kg "
          f"(config assumes {cfg['fuel']['s_per_kg']})")

    df = df.sort_values(["race_id", "driver", "lap_number"]).reset_index(drop=True)
    df.to_parquet(cfg["paths"]["interim"] / "laps_clean.parquet", index=False)
    print(f.to_string(index=False))
    print(f"clean: dropped wet races {list(dropped['event_name'] + ' ' + dropped['year'].astype(str))}")
    return df
