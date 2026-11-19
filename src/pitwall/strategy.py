"""Stage 9: pit window recommender and its backtest on real 2025 stops.

The decision: a car has done its reference laps plus 5 more on the current set
(the Setup B decision point). The team has already chosen how many stops to make,
so the question is where to split the current segment, which runs from now to
the end of the NEXT stint. For every candidate pit lap p:

    cost(p) = sum over laps now..p of predicted loss on the current tires (Setup B)
            + sum over laps after p of predicted loss on the new set (Setup A,
              fresh tire of the compound actually fitted next)
            + (laps on the new set) x compound pace offset (old -> new)

Pit lane time is the same for every p, so it does not move the best lap; it is
reported for context. Only information available at the decision point is used:
future laps reuse the current gap, track temperature and position.

Decision options (chosen on a 2024 backtest with models fit on 2022-2023):
  monotone  loss curves never improve as a set ages (running maximum)
  smooth    each predicted curve replaced by its straight-line fit
  rolling   re-decide after every observed lap; pit when the best lap is the next one
  update    (rolling) add the online error correction from stage `update`

Learned strategy layer: a small boosting model that predicts where in the segment
the team will stop, from the tire model's window plus race context (gap behind =
undercut threat, position, stint, laps left, pit lane loss). It is compared with
the same model WITHOUT the tire model's output (pure imitation of team habits).
It is trained on out-of-fold recommendations for the training seasons.

Outputs (reports/metrics): pit_loss_by_circuit.csv, compound_offsets.csv,
strategy_validation_2024.csv, strategy_backtest.csv (one row per stint),
strategy_summary.csv, strategy_layer_importance.csv
"""
from __future__ import annotations

import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from .clean import add_ids
from .features import COMPOUNDS
from .ingest import load_raw
from .models import feature_lists
from .update import recent_bias

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


def gap_behind(raw: pd.DataFrame, cap_s: float = 10.0) -> dict:
    """(race, driver, lap) -> gap to the car behind at the start of the lap (s)."""
    df = add_ids(raw).sort_values(["race_id", "lap_number", "lap_start_s"])
    nxt = df.groupby(["race_id", "lap_number"])["lap_start_s"].shift(-1)
    gb = (nxt - df["lap_start_s"]).fillna(cap_s).clip(lower=0, upper=cap_s)
    return dict(zip(zip(df["race_id"], df["driver"], df["lap_number"]), gb))


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


def offsets_dict(off_df: pd.DataFrame) -> dict:
    return {(a, b): o for a, b, o in zip(off_df["compound_old"], off_df["compound_new"], off_df["offset_s"])}


def practice_lookup(cfg: dict) -> dict:
    """(year, event, compound) -> practice wear slope, for the NEW set's compound."""
    w = pd.read_csv(cfg["paths"]["processed"] / "practice_wear.csv")
    return {(y, e, c): s for y, e, c, s in zip(w["year"], w["event_name"], w["compound"], w["prac_slope_field"])}


# ---------------------------------------------------------------- predictions for one decision
def decision_state(feat_stint: pd.DataFrame, history_laps: int = 5) -> pd.Series | None:
    """The last observed lap of the window: the state the pit wall sees."""
    g = feat_stint.sort_values("lap_number")
    if len(g) <= history_laps:
        return None
    return g.iloc[history_laps - 1]


def _interactions(df: pd.DataFrame) -> pd.DataFrame:
    for comp in COMPOUNDS:
        df[f"stint_lap_x_{comp.lower()}"] = df["stint_lap"] * (df["compound"] == comp)
    df["stint_lap_x_track_temp"] = df["stint_lap"] * df["track_temp"]
    if "prac_slope_field" in df:
        df["stint_lap_x_prac"] = df["stint_lap"] * df["prac_slope_field"]
    return df


def old_tire_rows(state: pd.Series, laps: np.ndarray) -> pd.DataFrame:
    """Feature rows for future laps on the current set, from the decision state only.
    `laps_ahead` counts from the end of the first 5-lap window, as in training."""
    k = laps - state["lap_number"]
    df = pd.DataFrame([state] * len(laps)).reset_index(drop=True)
    df["lap_number"] = laps
    df["stint_lap"] = state["stint_lap"] + k
    df["tyre_life"] = state["tyre_life"] + k
    df["laps_remaining"] = state["laps_remaining"] - k
    df["race_progress"] = laps / (state["laps_remaining"] + state["lap_number"])
    df["laps_ahead"] = state["laps_ahead"] + k
    df = _interactions(df)
    df["early_slope_x_ahead"] = df["early_deg_slope"] * df["laps_ahead"]
    return df


def new_tire_rows(state: pd.Series, pit_lap: int, n_laps: int, compound: str,
                  tyre_life_start: float, fresh: bool, prac: float | None = None) -> pd.DataFrame:
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
    if prac is not None:
        df["prac_slope_field"] = prac
    return _interactions(df)


def _line(y: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return y
    x = np.arange(len(y))
    s, b = np.polyfit(x, y, 1)
    return s * x + b


def curves(state: pd.Series, segment_end: int, next_compound: str, next_life_start: float,
           next_fresh: bool, model_B, model_A, cfg: dict, offset_s: float = 0.0, lstm_fn=None,
           monotone: bool = False, smooth: bool = False, bias: float = 0.0, prac: float | None = None):
    """Predicted loss per lap on the current set (laps now+1 .. end-1) and on the new set
    (1 .. end-now laps), after the chosen options. Returns (old, new incl. offset)."""
    now = int(state["lap_number"])
    num_b, cat_b, inter_b = feature_lists(cfg, "B")
    num_a, cat_a, inter_a = feature_lists(cfg, "A")
    old = old_tire_rows(state, np.arange(now + 1, segment_end))
    d_old = (lstm_fn(old) if lstm_fn else model_B.predict(old[num_b + cat_b + inter_b])) + bias
    new = new_tire_rows(state, now, segment_end - now, next_compound, next_life_start, next_fresh, prac)
    d_new = model_A.predict(new[num_a + cat_a + inter_a])
    if smooth:
        d_old, d_new = _line(d_old), _line(d_new)
    if monotone:
        d_old = np.maximum.accumulate(np.concatenate([[state["deg_s"]], d_old]))[1:]
        d_new = np.maximum.accumulate(d_new)
    return d_old, d_new + offset_s


def recommend(state: pd.Series, segment_end: int, next_compound: str, next_life_start: float,
              next_fresh: bool, model_B, model_A, cfg: dict, offset_s: float = 0.0,
              window_s: float = 1.0, lstm_fn=None, monotone: bool = False, smooth: bool = False,
              bias: float = 0.0, prac: float | None = None) -> tuple[pd.DataFrame, int]:
    """Cost of pitting at the end of each lap from now until the segment end.
    Returns (table of cost by pit lap, recommended pit lap)."""
    now = int(state["lap_number"])
    d_old, d_new = curves(state, segment_end, next_compound, next_life_start, next_fresh, model_B,
                          model_A, cfg, offset_s, lstm_fn, monotone, smooth, bias, prac)
    cum_old = np.concatenate([[0.0], np.cumsum(d_old)])     # loss through pit lap p
    cum_new = np.concatenate([[0.0], np.cumsum(d_new)])     # loss over the laps after p
    p = np.arange(now, segment_end)
    t = pd.DataFrame({"pit_lap": p, "old_tire_s": cum_old[p - now], "new_tire_s": cum_new[segment_end - p]})
    t["cost_s"] = t["old_tire_s"] + t["new_tire_s"]
    best = int(t.loc[t["cost_s"].idxmin(), "pit_lap"])
    t["in_window"] = t["cost_s"] <= t["cost_s"].min() + window_s
    return t, best


def rolling_recommend(g: pd.DataFrame, segment_end: int, kw: dict, model_B, cols_B: list[str],
                      history_laps: int, update: tuple[int, float] | None = None, lstm_fn_for=None):
    """Re-decide after every observed lap from the decision point on. The team pits
    at the end of the next lap as soon as that is the best lap. If that never
    happens before the data ends, the last recommendation stands.
    Returns (pit lap, table at the final decision, decision lap of that table)."""
    g = g.sort_values("lap_number")
    resid = np.where(g["in_forecast"], g["deg_s"] - model_B.predict(g[cols_B]), np.nan) \
        if (update and model_B is not None) else None
    last = None
    for d in range(history_laps, len(g) + 1):
        s = g.iloc[d - 1]
        now = int(s["lap_number"])
        if now >= segment_end - 1:
            break
        bias = recent_bias(resid[:d], *update) if resid is not None else 0.0
        t, best = recommend(s, segment_end, model_B=model_B, bias=bias,
                            lstm_fn=lstm_fn_for(s) if lstm_fn_for else None, **kw)
        last = (max(best, now + 1), t, now)
        if best <= now + 1:
            break
    return last


# ---------------------------------------------------------------- backtest
def eval_stints(raw: pd.DataFrame) -> pd.DataFrame:
    st = stint_table(raw)
    return st[st["green_stop"] & st["next_compound"].isin(COMPOUNDS) & st["next_last_lap"].notna()
              & st["next_tyre_life_start"].notna()]


def backtest(cfg: dict, fcfg: dict, feat_eval: pd.DataFrame, st: pd.DataFrame, methods: list,
             off: dict, opts: dict, prac: dict, gaps: dict | None = None,
             update: tuple[int, float] | None = None) -> pd.DataFrame:
    """methods: list of (label, model_B or None, model_A, lstm_fn_for or None)."""
    H = cfg["target"]["history_laps"]
    ev = feat_eval[feat_eval["stint_id"].isin(st["stint_id"])]
    info_by = st.set_index("stint_id")
    cols_B = sum(feature_lists(fcfg, "B"), [])
    rows = []
    for sid, g in ev.groupby("stint_id"):
        g = g.sort_values("lap_number")
        s = decision_state(g, H)
        if s is None or not g["in_forecast"].any():
            continue
        info = info_by.loc[sid]
        end, actual = int(info["next_last_lap"]), int(info["last_lap"])
        if actual <= s["lap_number"]:
            continue
        kw = dict(segment_end=end, next_compound=info["next_compound"],
                  next_life_start=info["next_tyre_life_start"], next_fresh=info["next_fresh"],
                  model_A=None, cfg=fcfg, offset_s=off.get((info["compound"], info["next_compound"]), 0.0),
                  window_s=cfg["strategy"]["window_s"], monotone=opts["monotone"], smooth=opts["smooth"],
                  prac=prac.get((s["year"], s["event_name"], info["next_compound"])))
        rec = {}
        for label, mB, mA, fn_for in methods:
            kw["model_A"] = mA
            if opts["rolling"]:
                kw_r = {k: v for k, v in kw.items() if k != "segment_end"}
                p, t, _ = rolling_recommend(g, end, kw_r, mB, cols_B, H,
                                            update if opts.get("update") else None, fn_for)
            else:
                t, p = recommend(s, model_B=mB, lstm_fn=fn_for(s) if fn_for else None, **kw)
            rec[label] = p
            w = t.loc[t["in_window"], "pit_lap"]
            rec[f"{label}_win_lo"], rec[f"{label}_win_hi"] = int(w.min()), int(w.max())
        rows.append({"stint_id": sid, "race_id": s["race_id"], "event_name": info["event_name"],
                     "circuit": info["circuit"], "driver": info["driver"],
                     "compound": info["compound"], "next_compound": info["next_compound"],
                     "stint": s["stint"], "position": s["position"], "gap_ahead_s": s["gap_ahead_s"],
                     "gap_behind_s": (gaps or {}).get((s["race_id"], s["driver"], int(s["lap_number"])), 10.0),
                     "laps_remaining": s["laps_remaining"], "early_deg_mean": s["early_deg_mean"],
                     "early_deg_slope": s["early_deg_slope"],
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
        if f"rec_{k}_win_lo" in bt:
            lo, hi = bt[f"rec_{k}_win_lo"], bt[f"rec_{k}_win_hi"]
            row["actual_in_window_pct"] = 100 * ((bt["actual_pit_lap"] >= lo) & (bt["actual_pit_lap"] <= hi)).mean()
            row["median_window_laps"] = float((hi - lo + 1).median())
        summ.append(row)
    return pd.DataFrame(summ)


def fit_lap_models(fcfg: dict, train: pd.DataFrame, names: dict) -> dict:
    """Fit the chosen Setup A and B models (by name) on `train`."""
    from .test import tuned_models
    out = {}
    for setup, setup_names in names.items():
        num, cat, inter = feature_lists(fcfg, setup)
        rows = train[train["in_forecast"]] if setup == "B" else train
        models = tuned_models(fcfg, setup)
        for n in setup_names:
            out[(setup, n)] = clone(models[n]).fit(rows[num + cat + inter], rows["deg_s"])
    return out


def choose_options(cfg: dict, fcfg: dict, feat: pd.DataFrame, best_A: str, best_B: str,
                   prac: dict, update: tuple[int, float]) -> tuple[dict, pd.DataFrame]:
    """Pick the decision options WITHOUT touching 2025: fit on seasons before 2024,
    backtest every option combination on 2024, keep the best within-3-laps rate."""
    val_year = max(cfg["train_seasons"])
    tr, ev = feat[feat["year"] < val_year], feat[feat["year"] == val_year]
    off = offsets_dict(compound_offsets(tr))
    m = fit_lap_models(fcfg, tr, {"A": [best_A], "B": [best_B]})
    methods = [(best_B, m[("B", best_B)], m[("A", best_A)], None)]
    st = eval_stints(load_raw(cfg, [val_year]))
    g = cfg["strategy"]["options_grid"]
    out = []
    for mono, smooth, rolling, upd in itertools.product(g["monotone"], g["smooth"], g["rolling"], g["update"]):
        if upd and not rolling:
            continue
        opts = {"monotone": mono, "smooth": smooth, "rolling": rolling, "update": upd}
        bt = backtest(cfg, fcfg, ev, st, methods, off, opts, prac, update=update)
        s = summarize(bt, [best_B]).assign(**opts)
        out.append(s)
        print(f"strategy[{val_year}] {opts}: within 3 laps {s['within_3_laps_pct'].iloc[0]:.1f}%", flush=True)
    val = pd.concat(out, ignore_index=True)
    v = val[val["method"] == best_B].sort_values(["within_3_laps_pct", "actual_in_window_pct"], ascending=False)
    best = {k: bool(v.iloc[0][k]) for k in ["monotone", "smooth", "rolling", "update"]}
    return best, val


# ---------------------------------------------------------------- learned strategy layer
CTX_NUM = ["stint", "position", "gap_ahead_s", "gap_behind_s", "laps_remaining", "early_deg_slope",
           "early_deg_mean", "seg_len", "pit_loss_s"]
TIRE_MODEL = ["rec_frac", "win_lo_frac", "win_hi_frac"]


def layer_features(bt: pd.DataFrame, label: str, pit_loss: pd.Series, with_tire_model: bool) -> pd.DataFrame:
    X = bt.copy()
    X["seg_len"] = X["segment_end"] - X["decision_lap"]
    X["pit_loss_s"] = X["circuit"].map(pit_loss).fillna(pit_loss.median())
    cols = list(CTX_NUM)
    for c in COMPOUNDS:
        X[f"cur_{c.lower()}"] = (X["compound"] == c).astype(int)
        X[f"next_{c.lower()}"] = (X["next_compound"] == c).astype(int)
        cols += [f"cur_{c.lower()}", f"next_{c.lower()}"]
    if with_tire_model:
        X["rec_frac"] = (X[f"rec_{label}"] - X["decision_lap"]) / X["seg_len"]
        X["win_lo_frac"] = (X[f"rec_{label}_win_lo"] - X["decision_lap"]) / X["seg_len"]
        X["win_hi_frac"] = (X[f"rec_{label}_win_hi"] - X["decision_lap"]) / X["seg_len"]
        cols += TIRE_MODEL
    return X[cols]


def layer_model() -> HistGradientBoostingRegressor:
    # small and heavily regularized: about 1,000 training stints
    return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=200,
                                         max_leaf_nodes=7, min_samples_leaf=20, random_state=42)


def layer_target(bt: pd.DataFrame) -> pd.Series:
    """Where in the segment the team stopped: 0 = right away, 1 = at the segment end."""
    return (bt["actual_pit_lap"] - bt["decision_lap"]) / (bt["segment_end"] - bt["decision_lap"])


def layer_predict(m, bt: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    frac = np.clip(m.predict(X), 0, 1)
    return np.round(bt["decision_lap"] + frac * (bt["segment_end"] - bt["decision_lap"])).astype(int).to_numpy()


def oof_recommendations(cfg: dict, fcfg: dict, feat: pd.DataFrame, raw: pd.DataFrame, best_A: str,
                        best_B: str, opts: dict, prac: dict, gaps: dict) -> pd.DataFrame:
    """Single-decision recommendations for every training-season stint, each from lap
    models that never saw that stint's race (grouped folds by race)."""
    tr = feat[feat["split"] == "train"]
    st = eval_stints(raw)
    single = {**opts, "rolling": False, "update": False}
    parts = []
    for fold in sorted(tr["cv_fold"].unique()):
        fit_on, ev = tr[tr["cv_fold"] != fold], tr[tr["cv_fold"] == fold]
        m = fit_lap_models(fcfg, fit_on, {"A": [best_A], "B": [best_B]})
        off = offsets_dict(compound_offsets(fit_on))
        parts.append(backtest(cfg, fcfg, ev, st, [(best_B, m[("B", best_B)], m[("A", best_A)], None)],
                              off, single, prac, gaps))
        print(f"strategy: out-of-fold recommendations, fold {fold}: {len(parts[-1])} stints", flush=True)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- run
def run(cfg: dict) -> pd.DataFrame:
    from .tune import final_config
    out = cfg["paths"]["metrics"]
    fcfg = final_config(cfg)
    feat = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    raw_tr = load_raw(cfg, cfg["train_seasons"])
    raw_te = load_raw(cfg, cfg["test_seasons"])
    prac = practice_lookup(cfg)
    gaps = gap_behind(pd.concat([raw_tr, raw_te]))
    upd = json.loads((out / "update_summary.json").read_text())
    update = (upd["k"], upd["shrink"])

    pit_loss = pit_loss_by_circuit(raw_tr)
    pit_loss.to_csv(out / "pit_loss_by_circuit.csv", index=False)
    pit_loss = pit_loss.set_index("circuit")["pit_loss_s"]
    offsets = compound_offsets(feat[feat["split"] == "train"])
    offsets.to_csv(out / "compound_offsets.csv", index=False)
    off = offsets_dict(offsets)
    best_A, best_B = json_best(cfg, "A"), json_best(cfg, "B")

    # 1) decision options, chosen on the last training season
    opts, val = choose_options(cfg, fcfg, feat, best_A, best_B, prac, update)
    val.to_csv(out / "strategy_validation_2024.csv", index=False)
    (out / "strategy_options.json").write_text(json.dumps({**opts, "update_k": update[0],
                                                          "update_shrink": update[1]}, indent=2))
    print(f"strategy: chosen options {opts}")

    # 2) learned layer, trained on out-of-fold recommendations for the training seasons
    bt_tr = oof_recommendations(cfg, fcfg, feat, raw_tr, best_A, best_B, opts, prac, gaps)
    bt_tr.to_csv(out / "strategy_oof_train.csv", index=False)
    y = layer_target(bt_tr)
    layers = {}
    for with_tm, name in [(True, "Learned layer (tire model + race context)"),
                          (False, "Imitation only (race context, no tire model)")]:
        layers[name] = (with_tm, layer_model().fit(layer_features(bt_tr, best_B, pit_loss, with_tm), y))

    # 3) one run on 2025 with the final models
    A = joblib.load(cfg["paths"]["models"] / "final_A.joblib")["models"]
    B = joblib.load(cfg["paths"]["models"] / "final_B.joblib")["models"]
    st_te = eval_stints(raw_te)
    test = feat[feat["split"] == "test"]
    single = {**opts, "rolling": False, "update": False}
    single_label = f"{best_B} (one decision at lap 5)"
    methods = [(single_label, B[best_B], A[best_A], None),
               ("Linear baseline", B["Linear baseline"], A["Linear baseline"], None)]
    labels = [single_label, "Linear baseline"]
    try:
        methods.insert(1, ("LSTM (one decision at lap 5)", None, A[best_A], lstm_predictor(cfg, fcfg, feat)))
        labels.insert(1, "LSTM (one decision at lap 5)")
    except (ImportError, FileNotFoundError):
        print("strategy: LSTM not available, skipped")
    bt1 = backtest(cfg, fcfg, test, st_te, methods, off, single, prac, gaps)
    bt = bt1
    if opts["rolling"]:
        roll_label = f"{best_B} (re-decided every lap)"
        bt2 = backtest(cfg, fcfg, test, st_te, [(roll_label, B[best_B], A[best_A], None)], off, opts, prac,
                       update=update)
        bt = bt1.merge(bt2[["stint_id"] + [c for c in bt2 if c.startswith(f"rec_{roll_label}")]], on="stint_id")
        labels.insert(1, roll_label)
    for name, (with_tm, m) in layers.items():
        bt[f"rec_{name}"] = layer_predict(m, bt, layer_features(bt, single_label, pit_loss, with_tm))
        labels.append(name)
    bt.to_csv(out / "strategy_backtest.csv", index=False)
    summ = summarize(bt, labels)
    for k, v in opts.items():
        summ[k] = v
    summ.to_csv(out / "strategy_summary.csv", index=False)

    # which inputs the learned layer leans on (permutation importance on 2025)
    with_tm, m = layers["Learned layer (tire model + race context)"]
    X = layer_features(bt, single_label, pit_loss, True)
    pi = permutation_importance(m, X, layer_target(bt), scoring="neg_mean_absolute_error",
                                n_repeats=10, random_state=42)
    pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean, "sd": pi.importances_std}) \
        .sort_values("importance", ascending=False).to_csv(out / "strategy_layer_importance.csv", index=False)
    joblib.dump({"layers": layers, "pit_loss": pit_loss, "label": best_B},
                cfg["paths"]["models"] / "strategy_layer.joblib")
    print(summ.round(2).to_string(index=False))
    return summ


def json_best(cfg: dict, setup: str) -> str:
    """Best tuned model for a setup by CV MAE (used for the pit window)."""
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
