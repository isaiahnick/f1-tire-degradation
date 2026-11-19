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
                         "lap_time_fc_s": 80 + slope * (lap - 1),
                         "gap_ahead_s": np.where(lap % 2 == 0, 0.5, 3.0)})


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
    # dirty air is measured on the 5 window laps only
    win_laps = t[(t["clean_idx"] >= 3) & (t["clean_idx"] < 8)]["lap_number"]
    assert h["dirty_air_share"].iloc[0] == pytest.approx((win_laps % 2 == 0).mean())


def test_splits_group_by_race_and_lock_test_season():
    cfg = load_config()
    races = [f"{y}_{r:02d}" for y in (2023, 2024, 2025) for r in range(1, 9)]
    df = pd.DataFrame({"race_id": np.repeat(races, 10)})
    df["year"] = df["race_id"].str[:4].astype(int)
    out = features.assign_splits(df, cfg)
    assert (out.loc[out["year"] == 2025, "split"] == "test").all()
    assert (out.loc[out["split"] == "test", "cv_fold"] == -1).all()
    assert out[out["split"] == "train"].groupby("race_id")["cv_fold"].nunique().max() == 1


# ---------------------------------------------------------------- final stages
class _Const:
    """Stand-in model: loss grows linearly with laps on the set."""

    def __init__(self, rate):
        self.rate = rate

    def predict(self, X):
        return self.rate * X["stint_lap"].to_numpy(float)


def _state():
    return pd.Series({"lap_number": 10, "stint_lap": 8.0, "tyre_life": 8.0, "tyre_life_start": 1.0,
                      "stint": 1.0, "fresh_tyre": 1, "laps_remaining": 40, "race_progress": 0.2,
                      "gap_ahead_s": 2.0, "position": 5.0, "track_temp": 40.0, "air_temp": 25.0,
                      "humidity": 50.0, "compound": "MEDIUM", "circuit": "X", "team": "Y",
                      "early_deg_mean": 0.3, "early_deg_slope": 0.05, "stint_id": "s1", "total_laps": 50,
                      "laps_ahead": 0.0, "deg_s": 0.6, "prac_slope_field": 0.1, "dirty_air_share": 0.2,
                      "window_gap_mean": 2.5, "pace_vs_field": 0.1})


def test_recommend_matches_brute_force():
    from pitwall.strategy import recommend
    cfg = load_config()
    s = _state()
    old, new = _Const(0.08), _Const(0.04)
    table, best = recommend(s, segment_end=40, next_compound="HARD", next_life_start=1,
                            next_fresh=True, model_B=old, model_A=new, cfg=cfg)
    # brute force: old set runs laps 11..p at stint_lap 9.., new set runs laps p+1..40
    def cost(p):
        o = sum(0.08 * (8 + (lap - 10)) for lap in range(11, p + 1))
        n = sum(0.04 * j for j in range(1, 40 - p + 1))
        return o + n
    best_cost = min(cost(p) for p in range(10, 40))
    assert cost(best) == pytest.approx(best_cost)           # ties are allowed
    assert table["cost_s"].min() == pytest.approx(best_cost)


def test_old_tire_rows_only_use_decision_state():
    from pitwall.strategy import old_tire_rows
    rows = old_tire_rows(_state(), np.array([11, 12, 13]))
    assert list(rows["stint_lap"]) == [9, 10, 11]
    assert list(rows["laps_ahead"]) == [1, 2, 3]
    assert (rows["gap_ahead_s"] == 2.0).all()          # future gap = gap known now
    assert list(rows["laps_remaining"]) == [39, 38, 37]


def test_online_correction_only_uses_laps_already_seen():
    from pitwall.update import recent_bias
    r = np.array([np.nan] * 5 + [0.4, 0.2, 0.6, -5.0])     # window laps have no residual
    assert recent_bias(r[:5], k=3, shrink=0) == 0.0          # nothing seen yet -> no correction
    assert recent_bias(r[:8], k=3, shrink=0) == pytest.approx(0.4)   # the -5.0 lap is in the future
    assert recent_bias(r[:8], k=3, shrink=3) == pytest.approx(0.2)   # shrunk: 0.4 * 3 / 6


def test_practice_wear_recovers_a_known_slope():
    cfg = load_config()
    life = np.arange(1, 13)
    prac = pd.DataFrame({"year": 2024, "event_name": "X", "session": "Practice 2", "driver": "AAA",
                         "stint": 2, "compound": "MEDIUM", "tyre_life": life,
                         "lap_time_s": 90 + 0.1 * life - 0.06 * life,       # wear 0.1, fuel -0.06 s/lap
                         "pit_in_s": np.nan, "pit_out_s": np.nan, "track_status": "1", "deleted": False})
    w = features.practice_wear(prac, cfg)
    assert w["prac_slope_field"].iloc[0] == pytest.approx(0.1)
