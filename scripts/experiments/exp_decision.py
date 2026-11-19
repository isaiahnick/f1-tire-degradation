"""Experiment 1: the decision layer. Fit on 2023 (optionally + 2022), backtest
on 2024 green-flag stops. 2025 is never used.

Variants
  P0 production   one decision at lap 5 of the window, B + A curves, monotone
  P1 smoothed     P0, but each predicted curve is replaced by a straight-line fit
  P2 rolling B    re-decide after every clean lap; B frozen; pit when best lap = next lap
  P3 rolling B+bias  same, B corrected by its recent errors on laps already seen
  P4 rolling C    same, old-tire curve from the Setup C (all laps seen) model
  P5 better A     P0 with the new-tire model given practice wear + 2022 data
  L  learned strategy layer: a small model trained on 2023 (out-of-fold) that
     corrects P0's lap using race context (gap behind, position, stint, ...);
     compared with a pure imitation model that never sees the tire curves.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from common import H, OUT, feature_sets, load, make
from gaps import gap_behind
from hist import HIST_C, INTER_C, c_rows, history_at
from pitwall.config import load_config
from pitwall.features import COMPOUNDS
from pitwall.ingest import load_raw
from pitwall.strategy import compound_offsets, new_tire_rows, old_tire_rows, pit_loss_by_circuit, stint_table

RAW_CFG = load_config()
USE_2022_FOR_A = "--no2022" not in sys.argv

numB, cat, interB = feature_sets("B")
numA, _, interA = feature_sets("A")
numAp, _, interAp = feature_sets("A", ["prac_slope_field"], ["stint_lap_x_prac"])
numC, interC = numA + HIST_C, interA + INTER_C
colsB, colsA, colsAp, colsC = numB + cat + interB, numA + cat + interA, numAp + cat + interAp, numC + cat + interC

F = load(with_2022=True)
PRAC = F.groupby(["year", "event_name", "compound"])["prac_slope_field"].first()


def fit_models(tr: pd.DataFrame, tr22: pd.DataFrame) -> dict:
    m = {}
    b = tr[tr["in_forecast"]]
    m["B"] = make("Gradient boosting", "B", numB, cat, interB).fit(b[colsB], b["deg_s"])
    m["A"] = make("Gradient boosting", "A", numA, cat, interA).fit(tr[colsA], tr["deg_s"])
    a = pd.concat([tr, tr22]) if USE_2022_FOR_A else tr
    m["A+"] = make("Gradient boosting", "A", numAp, cat, interAp).fit(a[colsAp], a["deg_s"])
    c = c_rows(tr, step=2)
    m["C"] = make("Gradient boosting", "B", numC, cat, interC).fit(c[colsC], c["deg_s"])
    m["LinB"] = make("Ridge", "B", numB, cat, interB)  # placeholder, not used
    return m


def line(y: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return y
    x = np.arange(len(y))
    s, b = np.polyfit(x, y, 1)
    return s * x + b


def costs(now, end, d_old, d_new, start_deg, offset, monotone=True, smooth=False):
    if smooth:
        d_old, d_new = line(d_old), line(d_new)
    if monotone:
        d_old = np.maximum.accumulate(np.concatenate([[start_deg], d_old]))[1:]
        d_new = np.maximum.accumulate(d_new)
    d_new = d_new + offset
    cum_old = np.concatenate([[0.0], np.cumsum(d_old)])
    cum_new = np.concatenate([[0.0], np.cumsum(d_new)])
    p = np.arange(now, end)
    c = cum_old[p - now] + cum_new[end - p]
    return p, c


def new_curve(models, key, state, now, end, nxt):
    rows = new_tire_rows(state, now, end - now, nxt["next_compound"], nxt["next_tyre_life_start"], nxt["next_fresh"])
    if key == "A+":
        rows["prac_slope_field"] = PRAC.get((state["year"], state["event_name"], nxt["next_compound"]),
                                            state["prac_slope_field"])
        rows["stint_lap_x_prac"] = rows["stint_lap"] * rows["prac_slope_field"]
        return models["A+"].predict(rows[colsAp])
    return models["A"].predict(rows[colsA])


def old_curve(models, kind, state, now, end, hist=None, bias=0.0):
    laps = np.arange(now + 1, end)
    rows = old_tire_rows(state, laps)
    k = laps - now
    if kind == "C":
        for c, v in hist.items():
            rows[c] = v
        rows["laps_ahead_c"] = k
        rows["slope_all_x_ahead"] = rows["hist_slope_all"] * rows["laps_ahead_c"]
        return models["C"].predict(rows[colsC])
    rows["laps_ahead"] = state["laps_ahead"] + k               # B only knows the first window
    rows["early_slope_x_ahead"] = rows["early_deg_slope"] * rows["laps_ahead"]
    return models["B"].predict(rows[colsB]) + bias


def eval_stints(feat: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    st = stint_table(raw)
    st = st[st["green_stop"] & st["next_compound"].isin(COMPOUNDS) & st["next_last_lap"].notna()]
    return st


def backtest(models, feat, st, off, gb, variants):
    out = []
    feat = feat[feat["stint_id"].isin(st["stint_id"])]
    for sid, g in feat.sort_values("lap_number").groupby("stint_id"):
        if len(g) <= H or not g["in_forecast"].any():
            continue
        info = st[st["stint_id"] == sid].iloc[0]
        end, actual = int(info["next_last_lap"]), int(info["last_lap"])
        s0 = g.iloc[H - 1]
        if actual <= s0["lap_number"]:
            continue
        o = off.get((info["compound"], info["next_compound"]), 0.0)
        rec = {"stint_id": sid, "event_name": info["event_name"], "driver": info["driver"], "circuit": info["circuit"],
               "compound": info["compound"], "next_compound": info["next_compound"], "stint": s0["stint"],
               "decision_lap": int(s0["lap_number"]), "segment_end": end, "actual_pit_lap": actual,
               "position": s0["position"], "gap_ahead_s": s0["gap_ahead_s"], "laps_remaining": s0["laps_remaining"],
               "early_deg_slope": s0["early_deg_slope"], "early_deg_mean": s0["early_deg_mean"],
               "gap_behind_s": gb.get((s0["race_id"], s0["driver"], int(s0["lap_number"])), 10.0),
               "rec_Midpoint": int(round((s0["lap_number"] + end) / 2))}
        now = int(s0["lap_number"])
        newA = new_curve(models, "A", s0, now, end, info)
        oldB = old_curve(models, "B", s0, now, end)
        for name, (d_new, smooth) in {"P0 production": (newA, False), "P1 smoothed": (newA, True),
                                      "P5 better A": (new_curve(models, "A+", s0, now, end, info), False)}.items():
            if name not in variants:
                continue
            p, c = costs(now, end, oldB, d_new, s0["deg_s"], o, smooth=smooth)
            rec[f"rec_{name}"] = int(p[np.argmin(c)])
            w = p[c <= c.min() + 1.0]
            rec[f"rec_{name}_win_lo"], rec[f"rec_{name}_win_hi"] = int(w.min()), int(w.max())
        # rolling re-decision
        y, x = g["deg_s"].to_numpy(float), g["stint_lap"].to_numpy(float)
        resB = np.where(g["in_forecast"], y - models["B"].predict(g[colsB]), np.nan)
        for name, kind in [("P2 rolling B", "B"), ("P3 rolling B+bias", "Bb"), ("P4 rolling C", "C")]:
            if name not in variants:
                continue
            decided = None
            for d in range(H, len(g) + 1):
                s = g.iloc[d - 1]
                now = int(s["lap_number"])
                if now >= end - 1:
                    break
                hist = bias = None
                if kind == "C":
                    hist = {k: v for k, v in history_at(x[:d], y[:d]).items() if k != "last_obs_stint_lap"}
                    hist["early_deg_mean"], hist["early_deg_slope"] = s["early_deg_mean"], s["early_deg_slope"]
                seen = resB[:d][~np.isnan(resB[:d])][-5:]
                bias = seen.mean() * len(seen) / (len(seen) + 2) if (kind == "Bb" and len(seen)) else 0.0
                d_old = old_curve(models, "C" if kind == "C" else "B", s, now, end, hist, bias)
                d_new = new_curve(models, "A", s, now, end, info)
                p, c = costs(now, end, d_old, d_new, s["deg_s"], o)
                best = int(p[np.argmin(c)])
                w = p[c <= c.min() + 1.0]
                decided = (max(best, now + 1), int(w.min()), int(w.max()))
                if best <= now + 1:
                    break
            if decided is None:
                continue
            rec[f"rec_{name}"], rec[f"rec_{name}_win_lo"], rec[f"rec_{name}_win_hi"] = decided
        out.append(rec)
    return pd.DataFrame(out)


def summarize(bt: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for k in labels:
        ok = bt[f"rec_{k}"].notna()
        err = bt.loc[ok, f"rec_{k}"] - bt.loc[ok, "actual_pit_lap"]
        r = {"method": k, "stints": int(ok.sum()), "within_3_laps_pct": 100 * (err.abs() <= 3).mean(),
             "within_5_laps_pct": 100 * (err.abs() <= 5).mean(), "mean_abs_error_laps": err.abs().mean(),
             "median_error_laps": err.median()}
        if f"rec_{k}_win_lo" in bt:
            lo, hi, a = bt.loc[ok, f"rec_{k}_win_lo"], bt.loc[ok, f"rec_{k}_win_hi"], bt.loc[ok, "actual_pit_lap"]
            r["actual_in_window_pct"] = 100 * ((a >= lo) & (a <= hi)).mean()
            r["median_window_laps"] = float((hi - lo + 1).median())
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ learned strategy layer
CTX_NUM = ["stint", "position", "gap_ahead_s", "gap_behind_s", "laps_remaining", "early_deg_slope",
           "early_deg_mean", "seg_len", "pit_loss_s"]


def ctx(bt: pd.DataFrame, pit_loss: pd.Series, with_model: bool) -> pd.DataFrame:
    X = bt.copy()
    X["seg_len"] = X["segment_end"] - X["decision_lap"]
    X["pit_loss_s"] = X["circuit"].map(pit_loss).fillna(pit_loss.median())
    for c in COMPOUNDS:
        X[f"cur_{c}"] = (X["compound"] == c).astype(int)
        X[f"next_{c}"] = (X["next_compound"] == c).astype(int)
    cols = CTX_NUM + [f"cur_{c}" for c in COMPOUNDS] + [f"next_{c}" for c in COMPOUNDS]
    if with_model:
        p0 = "rec_P0 production"
        X["rec_frac"] = (X[p0] - X["decision_lap"]) / X["seg_len"]
        X["win_lo_frac"] = (X[f"{p0}_win_lo"] - X["decision_lap"]) / X["seg_len"]
        X["win_hi_frac"] = (X[f"{p0}_win_hi"] - X["decision_lap"]) / X["seg_len"]
        cols += ["rec_frac", "win_lo_frac", "win_hi_frac"]
    return X[cols]


def strategy_model():
    return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=200,
                                         max_leaf_nodes=7, min_samples_leaf=20, random_state=42)


def main():
    raw23, raw24 = load_raw(RAW_CFG, [2023]), load_raw(RAW_CFG, [2024])
    raw22 = pd.concat([pd.read_parquet(p) for p in sorted((RAW_CFG["paths"]["raw"].parent.parent / "data_exp" / "raw").glob("2022_*.parquet"))])
    gb = gap_behind(pd.concat([raw23, raw24])).set_index(["race_id", "driver", "lap_number"])["gap_behind_s"].to_dict()
    tr, ev, tr22 = F[F["year"] == 2023], F[F["year"] == 2024], F[F["year"] == 2022]
    off_df = compound_offsets(tr)
    off = {(a, b): o for a, b, o in zip(off_df["compound_old"], off_df["compound_new"], off_df["offset_s"])}
    pit_loss = pit_loss_by_circuit(raw23).set_index("circuit")["pit_loss_s"]
    st23, st24 = eval_stints(tr, raw23), eval_stints(ev, raw24)
    variants = ["P0 production", "P1 smoothed", "P2 rolling B", "P3 rolling B+bias", "P4 rolling C", "P5 better A"]

    models = fit_models(tr, tr22)
    bt = backtest(models, ev, st24, off, gb, variants)
    bt.to_csv(OUT / "exp_decision_backtest_2024.csv", index=False)
    summ = summarize(bt, variants + ["Midpoint"])

    # learned layer: out-of-fold P0 recommendations on 2023 to train on
    races = tr["race_id"].unique()
    parts = []
    for k, (_, va) in enumerate(GroupKFold(5).split(races, groups=races)):
        va_r = set(races[va])
        m = fit_models(tr[~tr["race_id"].isin(va_r)], tr22)
        parts.append(backtest(m, tr[tr["race_id"].isin(va_r)], st23, off, gb, ["P0 production"]))
        print(f"2023 oof fold {k}: {len(parts[-1])} stints", flush=True)
    bt23 = pd.concat(parts, ignore_index=True)
    bt23.to_csv(OUT / "exp_decision_oof_2023.csv", index=False)
    for with_model, name in [(True, "L learned layer (P0 + context)"), (False, "L imitation only (no tire model)")]:
        y = (bt23["actual_pit_lap"] - bt23["decision_lap"]) / (bt23["segment_end"] - bt23["decision_lap"])
        m = strategy_model().fit(ctx(bt23, pit_loss, with_model), y)
        frac = np.clip(m.predict(ctx(bt, pit_loss, with_model)), 0, 1)
        bt[f"rec_{name}"] = np.round(bt["decision_lap"] + frac * (bt["segment_end"] - bt["decision_lap"])).astype(int)
    summ = pd.concat([summ, summarize(bt, ["L learned layer (P0 + context)", "L imitation only (no tire model)"])],
                     ignore_index=True)
    bt.to_csv(OUT / "exp_decision_backtest_2024.csv", index=False)
    summ.to_csv(OUT / "exp_decision_summary_2024.csv", index=False)
    print(summ.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
