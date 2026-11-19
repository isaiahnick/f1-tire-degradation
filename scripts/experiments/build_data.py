"""Experiment data: 2022 races (extra training data) and practice long runs.

Writes into data_exp/ (never touches data/): raw 2022 parquet files, then runs
the normal clean + features stages over 2022-2025 with configs/experiments.yaml.
2022 rows get split "unused", so the 2023-24 CV folds and 2025 test are unchanged.

Practice: for every race weekend, long runs from FP1-FP3 (>= 6 consecutive green
laps on one set, fuel effect ignored inside a run since it is short) give a
wear slope (s/lap) per (race, compound) and per (race, team, compound).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pitwall import clean, features
from pitwall.config import load_config
from pitwall.ingest.archive import KEYMAP, _clean_value, load_race
from pitwall.ingest.schema import TEAM_ALIASES, conform

ARCH = Path("/home/claude/ext/arch")
cfg = load_config("configs/experiments.yaml")


def ingest_2022():
    races = [load_race(p.parent, 2022) for p in sorted((ARCH / "2022").glob("*/Race/session_laptimes.json"))]
    races.sort(key=lambda d: d["_start_date"].iloc[0])
    for rnd, df in enumerate(races, start=1):
        df["round"] = rnd
        conform(df.drop(columns="_start_date")).to_parquet(cfg["paths"]["raw"] / f"2022_{rnd:02d}.parquet", index=False)
    print("2022 races:", len(races))


def practice_long_runs():
    rows = []
    for year in [2022, 2023, 2024, 2025]:
        for p in sorted((ARCH / str(year)).glob("*/Practice */session_laptimes.json")):
            d = json.loads(p.read_text())
            df = pd.DataFrame({KEYMAP[k]: [_clean_value(x) for x in d[k]] for k in KEYMAP if k in d})
            df["event_name"] = p.parent.parent.name
            df["session"] = p.parent.name
            df["year"] = year
            rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    for c in ["lap_time_s", "tyre_life", "stint", "lap_number", "pit_in_s", "pit_out_s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["team"] = df["team"].replace(TEAM_ALIASES)
    ok = df["lap_time_s"].notna() & df["compound"].isin(["SOFT", "MEDIUM", "HARD"]) \
        & df["pit_in_s"].isna() & df["pit_out_s"].isna() \
        & ~df["track_status"].fillna("").astype(str).str.contains("4|5|6|7") \
        & ~df["deleted"].fillna(False).astype(bool)
    df = df[ok].copy()
    out = []
    for key, g in df.groupby(["year", "event_name", "session", "driver", "stint"]):
        g = g.sort_values("lap_number")
        # long run: laps within 107% of the run's median (drops cool-down / push laps)
        med = g["lap_time_s"].median()
        g = g[(g["lap_time_s"] < 1.04 * med) & (g["lap_time_s"] > 0.97 * med)]
        if len(g) < 6:
            continue
        x, y = g["tyre_life"].to_numpy(float), g["lap_time_s"].to_numpy(float)
        if np.ptp(x) < 5:
            continue
        # robust slope: Theil-Sen style median of pairwise slopes
        i, j = np.triu_indices(len(x), 1)
        s = (y[j] - y[i]) / np.where(x[j] != x[i], x[j] - x[i], np.nan)
        slope = np.nanmedian(s)
        # fuel burns over the run (~0.06 s/lap faster); add it back so slope = tire wear
        out.append({"year": key[0], "event_name": key[1], "session": key[2], "driver": key[3],
                    "team": g["team"].iloc[0], "compound": g["compound"].iloc[0], "laps": len(g),
                    "slope": slope + 0.06})
    lr = pd.DataFrame(out)
    lr = lr[lr["slope"].between(-0.3, 0.6)]
    lr.to_parquet("data_exp/processed/practice_long_runs.parquet", index=False)
    print("practice long runs:", len(lr), lr.groupby("year").size().to_dict())
    print(lr.groupby("compound")["slope"].describe().round(3))


if __name__ == "__main__":
    if "--practice-only" not in sys.argv:
        ingest_2022()
        clean.run(cfg)
        features.run(cfg)
    practice_long_runs()
