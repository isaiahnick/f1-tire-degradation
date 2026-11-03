"""Stage 1: pull every race for the configured seasons into data/raw.

Output: data/raw/{year}_{round:02d}.parquet, one file per race, canonical schema.
"""
from __future__ import annotations

import pandas as pd

from . import archive, fastf1_source

SOURCES = {"archive": archive, "fastf1": fastf1_source}


def run(cfg: dict, source: str | None = None) -> pd.DataFrame:
    source = source or cfg["ingest"]["source"]
    loader = SOURCES[source]
    raw_dir = cfg["paths"]["raw"]
    rows = []
    for year in cfg["seasons"]:
        for df in loader.load_season(year, cfg):
            rnd = int(df["round"].iloc[0])
            df.to_parquet(raw_dir / f"{year}_{rnd:02d}.parquet", index=False)
            rows.append({"year": year, "round": rnd, "event_name": df["event_name"].iloc[0],
                         "circuit": df["circuit"].iloc[0], "laps": len(df), "source": source})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(raw_dir / "manifest.csv", index=False)
    print(f"ingest: {len(manifest)} races, {manifest['laps'].sum():,} laps -> {raw_dir}")
    return manifest


def load_raw(cfg: dict, seasons: list[int] | None = None) -> pd.DataFrame:
    seasons = seasons or cfg["seasons"]
    files = sorted(p for y in seasons for p in cfg["paths"]["raw"].glob(f"{y}_*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
