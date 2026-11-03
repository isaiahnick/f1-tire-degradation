"""Loader for the TracingInsights GitHub archive.

TracingInsights (https://github.com/TracingInsights) runs FastF1 after every session
and commits the output as JSON, one repo per season. `Race/session_laptimes.json`
is FastF1's `session.laps` with `laps.get_weather_data()` joined on, using short
column names. We map those names back to the FastF1 names in our schema.

Only race-level files are fetched (sparse checkout), about 9 MB per season.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import conform

SPARSE_PATTERNS = [
    "/*/Race/session_laptimes.json",
    "/*/Race/session_info.json",
    "/*/Race/weather.json",
    "/*/Race/rcm.json",
    "/*/Race/drivers.json",
]

# archive key -> canonical column
KEYMAP = {
    "time": "lap_time_s", "lap": "lap_number", "compound": "compound", "stint": "stint",
    "s1": "s1_s", "s2": "s2_s", "s3": "s3_s", "life": "tyre_life", "pos": "position",
    "status": "track_status", "sesT": "lap_end_s", "drv": "driver", "dNum": "driver_number",
    "pout": "pit_out_s", "pin": "pit_in_s", "vi1": "speed_i1", "vi2": "speed_i2",
    "vfl": "speed_fl", "vst": "speed_st", "fresh": "fresh_tyre", "team": "team",
    "lST": "lap_start_s", "lSD": "lap_start_date", "del": "deleted",
    "ff1G": "fastf1_generated", "iacc": "is_accurate", "wAT": "air_temp",
    "wTT": "track_temp", "wH": "humidity", "wP": "pressure", "wR": "rainfall",
    "wWD": "wind_direction", "wWS": "wind_speed",
}


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def fetch_season(year: int, url: str, commit: str, dest: Path) -> Path:
    """Sparse, shallow checkout of one season at a pinned commit (idempotent)."""
    repo = dest / f"tracinginsights_{year}"
    head = repo / ".git" / "HEAD"
    if head.exists():
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
        if res.stdout.strip() == commit:
            return repo
    repo.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        _git(["init", "-q"], repo)
        _git(["remote", "add", "origin", url], repo)
    _git(["sparse-checkout", "init", "--no-cone"], repo)
    (repo / ".git" / "info" / "sparse-checkout").write_text("\n".join(SPARSE_PATTERNS) + "\n")
    _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], repo)
    _git(["checkout", "-q", "FETCH_HEAD"], repo)
    return repo


def _clean_value(v):
    return np.nan if v in ("None", None, "") else v


def load_race(race_dir: Path, year: int) -> pd.DataFrame:
    laps = json.loads((race_dir / "session_laptimes.json").read_text())
    info = json.loads((race_dir / "session_info.json").read_text())
    meeting = info["session_info"]["Meeting"]

    df = pd.DataFrame({KEYMAP[k]: [_clean_value(x) for x in laps[k]] for k in KEYMAP})
    df["lap_start_date"] = pd.to_datetime(df["lap_start_date"], errors="coerce")
    df["year"] = year
    df["event_name"] = meeting["Name"]
    df["circuit"] = meeting["Circuit"]["ShortName"]
    df["total_laps"] = int(info["total_laps"])
    df["source"] = "archive"
    df["_start_date"] = info["session_info"]["StartDate"]
    return df


def load_season(year: int, cfg: dict) -> list[pd.DataFrame]:
    arch = cfg["ingest"]["archive"]
    repo = fetch_season(
        year, arch["url"].format(year=year), arch["commits"][year], cfg["paths"]["external"]
    )
    races = [load_race(p.parent, year) for p in sorted(repo.glob("*/Race/session_laptimes.json"))]
    # round number = calendar order (matches FastF1 RoundNumber for championship races)
    races.sort(key=lambda d: d["_start_date"].iloc[0])
    out = []
    for rnd, df in enumerate(races, start=1):
        df["round"] = rnd
        out.append(conform(df.drop(columns="_start_date")))
    return out
