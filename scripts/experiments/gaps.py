"""Gap to the car behind at each lap (undercut threat), from raw laps."""
import pandas as pd

from pitwall.clean import add_ids


def gap_behind(raw: pd.DataFrame, cap_s: float = 10.0) -> pd.DataFrame:
    df = add_ids(raw).sort_values(["race_id", "lap_number", "lap_start_s"]).copy()
    nxt = df.groupby(["race_id", "lap_number"])["lap_start_s"].shift(-1)
    df["gap_behind_s"] = (nxt - df["lap_start_s"]).fillna(cap_s).clip(lower=0, upper=cap_s)
    return df[["race_id", "driver", "lap_number", "gap_behind_s"]]
