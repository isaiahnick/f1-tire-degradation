"""Helpers for teammates (EDA notebook, LSTM, dashboard) to read the shared tables.

Use these instead of re-deriving splits, so every model is trained and scored on
the same rows and the same CV folds.

    from pitwall.data import load_features, stint_sequences
    train = load_features("train")              # 2023-2024, has cv_fold 0..4
    seqs = stint_sequences(train)               # one dict per stint for the LSTM
"""
from __future__ import annotations

import pandas as pd

from .config import load_config


def load_features(split: str | None = "train", cfg: dict | None = None) -> pd.DataFrame:
    """split = "train", "test", or None for everything. Only open "test" for the
    final evaluation, after model choices are frozen."""
    cfg = cfg or load_config()
    df = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    return df if split is None else df[df["split"] == split].reset_index(drop=True)


def stint_sequences(df: pd.DataFrame, cols: list[str] | None = None) -> list[dict]:
    """Group target laps into ordered per-stint sequences for sequence models."""
    cols = cols or ["stint_lap", "tyre_life", "track_temp", "air_temp", "laps_remaining",
                    "gap_ahead_s"]
    out = []
    for sid, g in df.sort_values(["stint_id", "lap_number"]).groupby("stint_id", sort=False):
        out.append({"stint_id": sid, "race_id": g["race_id"].iloc[0],
                    "cv_fold": int(g["cv_fold"].iloc[0]), "compound": g["compound"].iloc[0],
                    "circuit": g["circuit"].iloc[0], "x": g[cols].to_numpy(float),
                    "y": g["deg_s"].to_numpy(float), "in_forecast": g["in_forecast"].to_numpy()})
    return out
