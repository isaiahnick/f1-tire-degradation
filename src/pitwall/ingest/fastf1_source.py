"""Loader that calls the FastF1 API directly.

Use this on a machine that can reach livetiming.formula1.com. It returns the same
canonical schema as the archive loader. FastF1 caches every session under
data/external/fastf1_cache, so each race is downloaded once.
"""
from __future__ import annotations

import pandas as pd

from .schema import conform


def _sec(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds()


def load_race(year: int, rnd: int) -> pd.DataFrame:
    import fastf1

    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=True)
    laps = session.laps.reset_index(drop=True)
    w = laps.get_weather_data().reset_index(drop=True)

    df = pd.DataFrame({
        "year": year,
        "round": int(session.event["RoundNumber"]),
        "event_name": session.event["EventName"],
        "circuit": session.session_info["Meeting"]["Circuit"]["ShortName"],
        "total_laps": int(session.total_laps),
        "driver": laps["Driver"],
        "driver_number": laps["DriverNumber"],
        "team": laps["Team"],
        "lap_number": laps["LapNumber"],
        "lap_time_s": _sec(laps["LapTime"]),
        "lap_end_s": _sec(laps["Time"]),
        "lap_start_s": _sec(laps["LapStartTime"]),
        "lap_start_date": laps["LapStartDate"],
        "s1_s": _sec(laps["Sector1Time"]),
        "s2_s": _sec(laps["Sector2Time"]),
        "s3_s": _sec(laps["Sector3Time"]),
        "pit_in_s": _sec(laps["PitInTime"]),
        "pit_out_s": _sec(laps["PitOutTime"]),
        "position": laps["Position"],
        "track_status": laps["TrackStatus"],
        "deleted": laps["Deleted"],
        "fastf1_generated": laps["FastF1Generated"],
        "is_accurate": laps["IsAccurate"],
        "stint": laps["Stint"],
        "compound": laps["Compound"],
        "tyre_life": laps["TyreLife"],
        "fresh_tyre": laps["FreshTyre"],
        "speed_i1": laps["SpeedI1"],
        "speed_i2": laps["SpeedI2"],
        "speed_fl": laps["SpeedFL"],
        "speed_st": laps["SpeedST"],
        "air_temp": w["AirTemp"],
        "track_temp": w["TrackTemp"],
        "humidity": w["Humidity"],
        "pressure": w["Pressure"],
        "rainfall": w["Rainfall"],
        "wind_direction": w["WindDirection"],
        "wind_speed": w["WindSpeed"],
        "source": "fastf1",
    })
    return conform(df)


def load_season(year: int, cfg: dict) -> list[pd.DataFrame]:
    import fastf1

    cache = cfg["paths"]["external"] / "fastf1_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return [load_race(year, int(r)) for r in schedule["RoundNumber"] if r > 0]
