"""Stage 8: pit window recommender and its backtest on real 2025 stops.

The decision: a car is 5 laps past its reference laps on the current set (the
Setup B decision point). The team has already chosen how many stops to make, so
the question is where to split the current segment, which runs from now to the
end of the NEXT stint. For every candidate pit lap p:

    cost(p) = sum over laps now..p of predicted loss on the current tires (Setup B)
            + sum over laps after p of predicted loss on the new set (Setup A,
              fresh tire of the compound actually fitted next)
            + (laps on the new set) x compound pace offset (old -> new)

Pit lane time is the same for every p, so it does not move the best lap; it is
reported for context. Only information available at the decision point is used:
future laps reuse the current gap, track temperature and position.

The recommended lap is argmin cost; the pit window is every lap within
`window_s` seconds of the best. Backtest: 2025 stints that ended in a green-flag
stop, compared with the lap the team actually pitted.

Outputs (reports/metrics): pit_loss_by_circuit.csv, compound_offsets.csv,
strategy_backtest.csv (one row per stint), strategy_summary.csv
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .clean import add_ids
from .features import COMPOUNDS
from .ingest import load_raw
from .models import feature_lists

SC = ("4", "5", "6", "7")


# ---------------------------------------------------------------- race facts from raw laps
def stint_table(raw: pd.DataFrame) -> pd.DataFrame:
    raw = add_ids(raw)
    g = raw.sort_values("lap_number").groupby(["race_id", "driver", "stint"])
    st = g.agg(first_lap=("lap_number", "min"), last_lap=("lap_number", "max"),
               compound=("compound", "first"), tyre_life_start=("tyre_life", "min"),
               fresh=("fresh_tyre", "first"), team=("team", "first"),
               total_laps=("total_laps", "first"), event_name=("event_name", "first"),
               circuit=("circuit", "first"), year=("year", "first"),
               pit_in=("pit_in_s", lambda s: s.notna().iloc[-1])).reset_index()
    status = raw.set_index(["race_id", "driver", "lap_number"])["track_status"].fillna("")
    st["in_lap_status"] = [status.get((r, d, l), "") for r, d, l in
                           zip(st["race_id"], st["driver"], st["last_lap"])]
    st["out_lap_status"] = [status.get((r, d, l + 1), "") for r, d, l in
                            zip(st["race_id"], st["driver"], st["last_lap"])]
    st["green_stop"] = st["pit_in"] & ~st["in_lap_status"].str.contains("|".join(SC)) \
        & ~st["out_lap_status"].str.contains("|".join(SC))
    nxt = st[["race_id", "driver", "stint", "compound", "last_lap", "tyre_life_start", "fresh"]].copy()
    nxt["stint"] -= 1
    st = st.merge(nxt.rename(columns={"compound": "next_compound", "last_lap": "next_last_lap",
                                      "tyre_life_start": "next_tyre_life_start", "fresh": "next_fresh"}),
                  on=["race_id", "driver", "stint"], how="left")
    st["stint_id"] = (st["race_id"] + "_" + st["driver"].astype(str) + "_"
                      + st["stint"].fillna(-1).astype(int).astype(str))
    return st


def pit_loss_by_circuit(raw: pd.DataFrame) -> pd.DataFrame:
    """Green-flag pit loss: in-lap + out-lap minus two typical laps nearby (s)."""
    raw = add_ids(raw).sort_values(["race_id", "driver", "lap_number"])
    ok = raw["pit_in_s"].isna() & raw["pit_out_s"].isna() & (raw["lap_number"] > 1) \
        & ~raw["track_status"].fillna("").str.contains("|".join(SC))
    rows = []
    for (rid, drv), g in raw.groupby(["race_id", "driver"]):
        g = g.set_index("lap_number")
        for lap in g.index[g["pit_in_s"].notna()]:
            if lap + 1 not in g.index:
                continue
            both = g.loc[[lap, lap + 1]]
            if both["track_status"].fillna("").str.contains("|".join(SC)).any():
                continue
            near = g[(abs(g.index - lap) <= 6) & ok.loc[g["lap_time_s"].index].to_numpy()]["lap_time_s"]
            if len(near) >= 4 and both["lap_time_s"].notna().all():
                rows.append({"circuit": g["circuit"].iloc[0], "race_id": rid,
                             "loss_s": both["lap_time_s"].sum() - 2 * near.median()})
    d = pd.DataFrame(rows)
    d = d[(d["loss_s"] > 10) & (d["loss_s"] < 45)]
    return d.groupby("circuit")["loss_s"].agg(pit_loss_s="median", stops="size").reset_index()


def compound_offsets(feat: pd.DataFrame) -> pd.DataFrame:
    """Median change in fuel-corrected reference pace from one stint to the next,
    by (old, new) compound. Positive = new set starts slower."""
    st = feat.groupby("stint_id").agg(race_id=("race_id", "first"), driver=("driver", "first"),
                                      stint=("stint", "first"), compound=("compound", "first"),
                                      ref=("ref_pace_s", "first")).reset_index()
    nxt = st.assign(stint=st["stint"] - 1)[["race_id", "driver", "stint", "compound", "ref"]]
    pair = st.merge(nxt, on=["race_id", "driver", "stint"], suffixes=("_old", "_new"))
    pair["delta"] = pair["ref_new"] - pair["ref_old"]
    pair = pair[pair["delta"].abs() < 5]
    return (pair.groupby(["compound_old", "compound_new"])["delta"]
            .agg(offset_s="median", n="size").reset_index())


# ---------------------------------------------------------------- predictions for one decision
def decision_state(feat_stint: pd.DataFrame, history_laps: int = 5) -> pd.Series | None:
    """The last observed lap of the window: the state the pit wall sees."""
    g = feat_stint.sort_values("lap_number")
    if len(g) <= history_laps:
        return None
    return g.iloc[history_laps - 1]


def old_tire_rows(state: pd.Series, laps: np.ndarray) -> pd.DataFrame:
    """Feature rows for future laps on the current set, from the decision state only."""
    k = laps - state["lap_number"]
    df = pd.DataFrame([state] * len(laps)).reset_index(drop=True)
    df["lap_number"] = laps
    df["stint_lap"] = state["stint_lap"] + k
    df["tyre_life"] = state["tyre_life"] + k
    df["laps_remaining"] = state["total_laps"] - laps if "total_laps" in state else state["laps_remaining"] - k
    df["race_progress"] = laps / (state["laps_remaining"] + state["lap_number"])
    df["laps_ahead"] = k
    for comp in COMPOUNDS:
        df[f"stint_lap_x_{comp.lower()}"] = df["stint_lap"] * (df["compound"] == comp)
    df["stint_lap_x_track_temp"] = df["stint_lap"] * df["track_temp"]
    df["early_slope_x_ahead"] = df["early_deg_slope"] * df["laps_ahead"]
    return df


def new_tire_rows(state: pd.Series, pit_lap: int, n_laps: int, compound: str,
                  tyre_life_start: float, fresh: bool) -> pd.DataFrame:
    j = np.arange(1, n_laps + 1)
    laps = pit_lap + j
    total = state["laps_remaining"] + state["lap_number"]
    df = pd.DataFrame([state] * n_laps).reset_index(drop=True)
    df["compound"] = compound
    df["lap_number"] = laps
    df["stint_lap"] = j
    df["tyre_life_start"] = tyre_life_start
    df["tyre_life"] = tyre_life_start + j - 1
    df["fresh_tyre"] = int(bool(fresh))
    df["stint"] = state["stint"] + 1
    df["laps_remaining"] = total - laps
    df["race_progress"] = laps / total
    for comp in COMPOUNDS:
        df[f"stint_lap_x_{comp.lower()}"] = df["stint_lap"] * (df["compound"] == comp)
    df["stint_lap_x_track_temp"] = df["stint_lap"] * df["track_temp"]
    return df


def recommend(state: pd.Series, segment_end: int, next_compound: str, next_life_start: float,
              next_fresh: bool, model_B, model_A, cfg: dict, offset_s: float = 0.0,
              window_s: float = 1.0, lstm_fn=None, monotone: bool = False) -> tuple[pd.DataFrame, int]:
    """Cost of pitting at the end of each lap from now until the segment end.

    model_B / model_A are fitted pipelines; lstm_fn(old_rows) can replace model_B.
    monotone=True applies a physical constraint: a tire set never gets faster as it
    ages, so each predicted curve is replaced by its running maximum.
    Returns (table of cost by pit lap, recommended pit lap).
    """
    now = int(state["lap_number"])
    num_b, cat_b, inter_b = feature_lists(cfg, "B")
    num_a, cat_a, inter_a = feature_lists(cfg, "A")
    old_laps = np.arange(now + 1, segment_end)              # can pit at the end of any of these
    old = old_tire_rows(state, old_laps)
    d_old = lstm_fn(old) if lstm_fn else model_B.predict(old[num_b + cat_b + inter_b])
    if monotone:
        d_old = np.maximum.accumulate(np.concatenate([[state["deg_s"]], d_old]))[1:]
    cum_old = np.concatenate([[0.0], np.cumsum(d_old)])     # loss through pit lap p
    # new-tire curve for the longest possible stint; reuse prefixes for shorter ones
    longest = segment_end - now
    new = new_tire_rows(state, now, longest, next_compound, next_life_start, next_fresh)
    d_new = model_A.predict(new[num_a + cat_a + inter_a])
    if monotone:
        d_new = np.maximum.accumulate(d_new)
    d_new = d_new + offset_s
    cum_new = np.concatenate([[0.0], np.cumsum(d_new)])
    rows = []
    for i, p in enumerate(np.arange(now, segment_end)):
        rows.append({"pit_lap": int(p), "old_tire_s": cum_old[i],
                     "new_tire_s": cum_new[segment_end - p], "cost_s": cum_old[i] + cum_new[segment_end - p]})
    t = pd.DataFrame(rows)
    best = int(t.loc[t["cost_s"].idxmin(), "pit_lap"])
    t["in_window"] = t["cost_s"] <= t["cost_s"].min() + window_s
    return t, best


# ---------------------------------------------------------------- backtest
def backtest(cfg: dict, fcfg: dict, feat_eval: pd.DataFrame, raw_eval: pd.DataFrame, methods: list,
             off: dict, monotone: bool) -> pd.DataFrame:
    """methods: list of (label, model_B or None, model_A, lstm_fn_for or None)."""
    st = stint_table(raw_eval)
    st = st[st["green_stop"] & st["next_compound"].isin(COMPOUNDS) & st["next_last_lap"].notna()]
    ev = feat_eval.merge(st[["stint_id", "total_laps"]], on="stint_id", how="inner")
    rows = []
    for sid, g in ev.groupby("stint_id"):
        s = decision_state(g, cfg["target"]["history_laps"])
        if s is None or not g["in_forecast"].any():
            continue
        info = st[st["stint_id"] == sid].iloc[0]
        end, actual = int(info["next_last_lap"]), int(info["last_lap"])
        if actual <= s["lap_number"]:
            continue
        kw = dict(segment_end=end, next_compound=info["next_compound"],
                  next_life_start=info["next_tyre_life_start"], next_fresh=info["next_fresh"],
                  cfg=fcfg, offset_s=off.get((info["compound"], info["next_compound"]), 0.0),
                  monotone=monotone)
        rec = {}
        for label, mB, mA, fn_for in methods:
            t, p = recommend(s, model_B=mB, model_A=mA, lstm_fn=fn_for(s) if fn_for else None, **kw)
            rec[label] = p
            w = t.loc[t["in_window"], "pit_lap"]
            rec[f"{label}_win_lo"], rec[f"{label}_win_hi"] = int(w.min()), int(w.max())
            act = t.loc[t["pit_lap"] == actual, "cost_s"]
            rec[f"{label}_actual_extra_s"] = float(act.iloc[0] - t["cost_s"].min()) if len(act) else np.nan
        rows.append({"stint_id": sid, "event_name": info["event_name"], "driver": info["driver"],
                     "compound": info["compound"], "next_compound": info["next_compound"],
                     "decision_lap": int(s["lap_number"]), "segment_end": end, "actual_pit_lap": actual,
                     **{f"rec_{k}": v for k, v in rec.items()},
                     "rec_Midpoint": int(round((s["lap_number"] + end) / 2))})
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame, labels: list) -> pd.DataFrame:
    summ = []
    for k in labels + ["Midpoint"]:
        err = bt[f"rec_{k}"] - bt["actual_pit_lap"]
        row = {"method": k if k != "Midpoint" else "Midpoint (no model)", "stints": len(bt),
               "within_3_laps_pct": 100 * (err.abs() <= 3).mean(),
               "within_5_laps_pct": 100 * (err.abs() <= 5).mean(),
               "mean_abs_error_laps": err.abs().mean(), "median_error_laps": err.median()}
        if k != "Midpoint":
            lo, hi = bt[f"rec_{k}_win_lo"], bt[f"rec_{k}_win_hi"]
            row["actual_in_window_pct"] = 100 * ((bt["actual_pit_lap"] >= lo) & (bt["actual_pit_lap"] <= hi)).mean()
            row["median_window_laps"] = float((hi - lo + 1).median())
            row["median_actual_extra_s"] = float(bt[f"rec_{k}_actual_extra_s"].median())
        summ.append(row)
    return pd.DataFrame(summ)


def validate_on_2024(cfg: dict, fcfg: dict, feat: pd.DataFrame, best_A: str, best_B: str) -> pd.DataFrame:
    """Choose the monotone constraint WITHOUT touching 2025: fit on 2023, backtest 2024."""
    from sklearn.base import clone

    from .test import tuned_models
    tr = feat[feat["year"] == 2023]
    ev = feat[feat["year"] == 2024]
    off_df = compound_offsets(tr)
    off = {(a, b): o for a, b, o in zip(off_df["compound_old"], off_df["compound_new"], off_df["offset_s"])}
    fitted = {}
    for setup in ["A", "B"]:
        num, cat, inter = feature_lists(fcfg, setup)
        rows = tr[tr["in_forecast"]] if setup == "B" else tr
        models = tuned_models(fcfg, setup)
        for name in {best_A if setup == "A" else best_B, "Linear baseline"}:
            fitted[(setup, name)] = clone(models[name]).fit(rows[num + cat + inter], rows["deg_s"])
    methods = [(best_B, fitted[("B", best_B)], fitted[("A", best_A)], None),
               ("Linear baseline", fitted[("B", "Linear baseline")], fitted[("A", "Linear baseline")], None)]
    raw = load_raw(cfg, [2024])
    out = []
    for mono in [False, True]:
        bt = backtest(cfg, fcfg, ev, raw, methods, off, mono)
        out.append(summarize(bt, [best_B, "Linear baseline"]).assign(monotone=mono))
    return pd.concat(out, ignore_index=True)


def run(cfg: dict) -> pd.DataFrame:
    from .tune import final_config
    out = cfg["paths"]["metrics"]
    fcfg = final_config(cfg)
    feat = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    raw_tr = load_raw(cfg, cfg["train_seasons"])
    raw_te = load_raw(cfg, cfg["test_seasons"])

    pit_loss = pit_loss_by_circuit(raw_tr)
    pit_loss.to_csv(out / "pit_loss_by_circuit.csv", index=False)
    offsets = compound_offsets(feat[feat["split"] == "train"])
    offsets.to_csv(out / "compound_offsets.csv", index=False)
    off = {(a, b): o for a, b, o in zip(offsets["compound_old"], offsets["compound_new"], offsets["offset_s"])}
    best_A, best_B = json_best(cfg, "A"), json_best(cfg, "B")

    # 1) pick the monotone setting on 2024 (models fit on 2023 only)
    val = validate_on_2024(cfg, fcfg, feat, best_A, best_B)
    val.to_csv(out / "strategy_validation_2024.csv", index=False)
    v = val[val["method"] == best_B].set_index("monotone")["within_3_laps_pct"]
    monotone = bool(v[True] > v[False])
    print(val.round(2).to_string(index=False))
    print(f"strategy: monotone constraint {'ON' if monotone else 'OFF'} (chosen on 2024)")

    # 2) one run on 2025 with the final models
    A = joblib.load(cfg["paths"]["models"] / "final_A.joblib")["models"]
    B = joblib.load(cfg["paths"]["models"] / "final_B.joblib")["models"]
    methods = [(best_B, B[best_B], A[best_A], None),
               ("LSTM", None, A[best_A], lstm_predictor(cfg, fcfg, feat)),
               ("Linear baseline", B["Linear baseline"], A["Linear baseline"], None)]
    bt = backtest(cfg, fcfg, feat[feat["split"] == "test"], raw_te, methods, off, monotone)
    bt.to_csv(out / "strategy_backtest.csv", index=False)
    summ = summarize(bt, [best_B, "LSTM", "Linear baseline"]).assign(monotone=monotone)
    summ.to_csv(out / "strategy_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))
    return summ


def json_best(cfg: dict, setup: str) -> str:
    """Best tuned tree/linear model for a setup by CV MAE (used for the new-tire curve)."""
    import json
    tuned = json.loads((cfg["paths"]["metrics"] / "tuned_params.json").read_text())[setup]
    return min(tuned, key=lambda m: tuned[m]["cv_val_mae"])


def lstm_predictor(cfg: dict, fcfg: dict, feat: pd.DataFrame):
    """Returns f(state) -> (rows -> LSTM predictions), using the saved LSTM."""
    import torch

    from . import lstm
    ck = torch.load(cfg["paths"]["models"] / "lstm_B.pt", weights_only=False)
    m = lstm.LSTMRegressor(fcfg, **ck["hp"])
    m.prep = lstm.Prep()
    m.prep.__dict__.update(ck["prep"])
    wins = lstm.windows(feat, fcfg)
    n_seq = len(lstm.seq_features(fcfg))
    n_head = len(lstm.head_numeric(fcfg)) + sum(len(v) for v in m.prep.cats.values())
    m.net = lstm.make_net(n_seq, n_head, ck["hp"]["hidden"], ck["hp"]["dense"], ck["hp"]["dropout"])
    m.net.load_state_dict(ck["state"])

    def for_state(state):
        return lambda rows: m.predict(rows, wins)
    return for_state
