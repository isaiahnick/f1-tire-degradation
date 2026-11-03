"""Canonical raw lap schema.

Both loaders (FastF1 API and the TracingInsights archive) must return exactly these
columns, one row per driver lap, times in seconds. Everything downstream only
depends on this schema, so the source can be swapped without touching cleaning or
modeling code.
"""
from __future__ import annotations

import pandas as pd

# column -> dtype
RAW_SCHEMA: dict[str, str] = {
    # race identity
    "year": "int64",
    "round": "int64",
    "event_name": "string",
    "circuit": "string",
    "total_laps": "int64",          # scheduled race laps
    # driver / car
    "driver": "string",
    "driver_number": "string",
    "team": "string",
    # lap timing (FastF1 Laps)
    "lap_number": "int64",
    "lap_time_s": "float64",        # LapTime
    "lap_end_s": "float64",         # Time (session time at end of lap)
    "lap_start_s": "float64",       # LapStartTime
    "lap_start_date": "datetime64[ns]",
    "s1_s": "float64",
    "s2_s": "float64",
    "s3_s": "float64",
    "pit_in_s": "float64",          # PitInTime (NaN if no pit entry this lap)
    "pit_out_s": "float64",         # PitOutTime (NaN if no pit exit this lap)
    "position": "float64",
    "track_status": "string",       # concatenated status codes seen during lap
    "deleted": "boolean",
    "fastf1_generated": "boolean",
    "is_accurate": "boolean",
    # tires
    "stint": "float64",
    "compound": "string",
    "tyre_life": "float64",
    "fresh_tyre": "boolean",
    # speed traps
    "speed_i1": "float64",
    "speed_i2": "float64",
    "speed_fl": "float64",
    "speed_st": "float64",
    # weather (one sample per lap, FastF1 Laps.get_weather_data rule)
    "air_temp": "float64",
    "track_temp": "float64",
    "humidity": "float64",
    "pressure": "float64",
    "rainfall": "boolean",
    "wind_direction": "float64",
    "wind_speed": "float64",
    # provenance
    "source": "string",
}

TEAM_ALIASES = {
    # same constructor, renamed between 2023 and 2025
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Alfa Romeo": "Kick Sauber",
    "Sauber": "Kick Sauber",
}


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns, cast dtypes, and fail loudly on missing columns."""
    missing = set(RAW_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"raw laps missing columns: {sorted(missing)}")
    out = df[list(RAW_SCHEMA)].copy()
    for col, dtype in RAW_SCHEMA.items():
        if dtype.startswith("datetime"):
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dtype in ("float64", "int64"):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(dtype)
        else:
            out[col] = out[col].astype(dtype)
    out["team"] = out["team"].replace(TEAM_ALIASES)
    return out.sort_values(["driver", "lap_number"]).reset_index(drop=True)
