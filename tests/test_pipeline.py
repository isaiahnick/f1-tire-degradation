"""Unit tests for the parts most likely to break silently: schema, fuel
correction, the target definition, the history window, and the splits."""
import numpy as np
import pandas as pd
import pytest

from pitwall import clean, features
from pitwall.config import load_config
from pitwall.ingest import archive
from pitwall.ingest.schema import RAW_SCHEMA, conform


def test_archive_keymap_covers_schema():
    derived = {"year", "round", "event_name", "circuit", "total_laps", "source"}
    assert set(archive.KEYMAP.values()) | derived == set(RAW_SCHEMA)


def test_conform_rejects_missing_columns():
    with pytest.raises(ValueError):
        conform(pd.DataFrame({"year": [2024]}))


def test_fuel_correction_removes_linear_fuel_effect():
    # a car with zero tire wear gets faster only because fuel burns off
    total = 50
    lap = np.arange(1, total + 1)
    fuel = 110 * (1 - (lap - 0.5) / total)
    df = pd.DataFrame({"lap_number": lap, "total_laps": total, "lap_time_s": 90 + 0.03 * fuel})
    out = clean.fuel_correct(df, 110, 0.03)
    assert np.allclose(out["lap_time_fc_s"], 90)


def _stint(n=12, slope=0.1):
    lap = np.arange(2, 2 + n)
    return pd.DataFrame({"stint_id": "s1", "lap_number": lap, "stint_lap": lap - 1,
                         "lap_time_fc_s": 80 + slope * (lap - 1)})


def test_target_excludes_reference_laps_and_is_relative():
    out = features.build_target(_stint(), reference_laps=3, min_clean_laps=6)
    assert len(out) == 12 - 3                      # reference laps are not target rows
    assert out["clean_idx"].min() == 3
    # reference = median of laps with stint_lap 1..3 -> 80.2; first target lap = 80.4
    assert out["deg_s"].iloc[0] == pytest.approx(0.2)


def test_short_stints_dropped():
    assert features.build_target(_stint(n=5), 3, 6).empty


def test_history_window_does_not_overlap_forecast_rows():
    t = features.build_target(_stint(n=14), 3, 6)
    h = features.add_history(t, reference_laps=3, history_laps=5)
    fc = h[h["in_forecast"]]
    assert fc["clean_idx"].min() == 8               # 3 reference + 5 window laps
    assert (fc["laps_ahead"] >= 1).all()
    assert h["early_deg_slope"].iloc[0] == pytest.approx(0.1)


def test_splits_group_by_race_and_lock_test_season():
    cfg = load_config()
    races = [f"{y}_{r:02d}" for y in (2023, 2024, 2025) for r in range(1, 9)]
    df = pd.DataFrame({"race_id": np.repeat(races, 10)})
    df["year"] = df["race_id"].str[:4].astype(int)
    out = features.assign_splits(df, cfg)
    assert (out.loc[out["year"] == 2025, "split"] == "test").all()
    assert (out.loc[out["split"] == "test", "cv_fold"] == -1).all()
    assert out[out["split"] == "train"].groupby("race_id")["cv_fold"].nunique().max() == 1
