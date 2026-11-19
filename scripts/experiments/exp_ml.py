"""Experiment 5: ML tricks on the tuned Setup B models (and A where it feeds the
pit decision). Monotone constraints, MAE loss, smoothed training target,
model ensembles from out-of-fold predictions, quantile prediction intervals."""
import numpy as np
import pandas as pd

from common import OUT, cv, feature_sets, load, make, table

f = load(with_2022=False)
res, oofs = [], {}

# smoothed target: centred 3- and 5-lap rolling median of deg_s within the stint.
# Only the TRAINING target changes; every score is still against raw deg_s.
f = f.sort_values(["stint_id", "lap_number"])
for w in [3, 5]:
    f[f"deg_s_med{w}"] = f.groupby("stint_id")["deg_s"].transform(
        lambda s: s.rolling(w, center=True, min_periods=1).median())
f = f.sort_index()

MONO = {
    "wear only (stint_lap, tyre_life)": ["stint_lap", "tyre_life"],
    "+ history (early mean/slope)": ["stint_lap", "tyre_life", "early_deg_mean", "early_deg_slope"],
}

for setup in ["B", "A"]:
    num, cat, inter = feature_sets(setup)
    cols = num + cat + inter
    gb = make("Gradient boosting", setup, num, cat, inter)
    r = cv(f, setup, gb, cols, label="GB tuned (base)"); res.append(r); oofs[(setup, "GB")] = r["oof"]
    for name, feats in MONO.items():
        if setup == "A" and "history" in name:
            continue
        p = make("Gradient boosting", setup, num, cat, inter)
        p.named_steps["prep"].set_output(transform="pandas")
        p.set_params(model__monotonic_cst={f"num__{c}": 1 for c in feats})
        res.append(cv(f, setup, p, cols, label=f"GB monotone: {name}"))
    p = make("Gradient boosting", setup, num, cat, inter, loss="absolute_error")
    r = cv(f, setup, p, cols, label="GB loss=absolute_error"); res.append(r); oofs[(setup, "GB_l1")] = r["oof"]
    for w in [3, 5]:
        res.append(cv(f, setup, gb, cols, target=f"deg_s_med{w}", label=f"GB smoothed target ({w}-lap median)"))
    rid = make("Ridge", setup, num, cat, inter)
    r = cv(f, setup, rid, cols, label="Ridge tuned"); res.append(r); oofs[(setup, "Ridge")] = r["oof"]
    for w in [3, 5]:
        res.append(cv(f, setup, rid, cols, target=f"deg_s_med{w}", label=f"Ridge smoothed target ({w}-lap median)"))
    rf = make("Random forest", setup, num, cat, inter)
    r = cv(f, setup, rf, cols, label="RF tuned"); res.append(r); oofs[(setup, "RF")] = r["oof"]

    # quantile intervals (80%): coverage and width on validation folds
    lo = cv(f, setup, make("Gradient boosting", setup, num, cat, inter, loss="quantile", quantile=0.1), cols,
            label="GB quantile 0.10")
    hi = cv(f, setup, make("Gradient boosting", setup, num, cat, inter, loss="quantile", quantile=0.9), cols,
            label="GB quantile 0.90")
    d = f.loc[lo["oof"].index]
    cover = ((d["deg_s"] >= lo["oof"]) & (d["deg_s"] <= hi["oof"])).mean()
    width = (hi["oof"] - lo["oof"]).median()
    q = pd.DataFrame({"y": d["deg_s"], "lo": lo["oof"], "hi": hi["oof"], "h": d["laps_ahead"] if setup == "B" else d["stint_lap"]})
    q["hb"] = pd.cut(q["h"], [0, 5, 10, 15, 20, 99])
    byh = q.groupby("hb", observed=True).apply(lambda g: pd.Series({
        "coverage": ((g.y >= g.lo) & (g.y <= g.hi)).mean(), "median_width_s": (g.hi - g.lo).median(), "n": len(g)}))
    byh.to_csv(OUT / f"exp_quantile_{setup}.csv")
    print(f"  [{setup}] 80% interval: coverage {cover:.3f}, median width {width:.2f} s\n{byh.round(3)}")

# ensembles from OOF predictions (no refits, no extra tuning: equal weights)
lstm = None
try:
    ls = pd.read_parquet(OUT / "oof_lstm_seeds.parquet")
    seeds = [c for c in ls.columns if c.startswith("lstm_")]
    ls["lstm_avg"] = ls[seeds].mean(axis=1)
    rowsB = f[f["in_forecast"]][["stint_id", "lap_number"]]
    lstm = rowsB.merge(ls, on=["stint_id", "lap_number"], how="left").set_index(rowsB.index)
except FileNotFoundError:
    pass
for setup in ["B", "A"]:
    d = f.loc[oofs[(setup, "GB")].index]
    y = d["deg_s"]
    combos = {"GB+Ridge": ["GB", "Ridge"], "GB+RF": ["GB", "RF"], "GB+Ridge+RF": ["GB", "Ridge", "RF"]}
    preds = {k: oofs[(setup, k)] for k in ["GB", "Ridge", "RF"]}
    if setup == "B" and lstm is not None and lstm["lstm_avg"].notna().all():
        for s in seeds:
            preds[s] = lstm[s]
        preds["LSTM (seed avg)"] = lstm["lstm_avg"]
        combos.update({"GB+Ridge+RF+LSTMavg": ["GB", "Ridge", "RF", "LSTM (seed avg)"],
                       "GB+LSTMavg": ["GB", "LSTM (seed avg)"]})
    for k, v in preds.items():
        if k.startswith("lstm_") or k.startswith("LSTM"):
            folds = [float(np.mean(np.abs(y[d.cv_fold == i] - v[d.cv_fold == i]))) for i in range(5)]
            res.append({"label": f"OOF {k}", "setup": setup, "val_mae": float(np.mean(folds)),
                        "val_mae_folds": [round(x, 4) for x in folds], "train_mae": np.nan})
    for name, parts in combos.items():
        p = sum(preds[k] for k in parts) / len(parts)
        folds = [float(np.mean(np.abs(y[d.cv_fold == i] - p[d.cv_fold == i]))) for i in range(5)]
        res.append({"label": f"ensemble {name}", "setup": setup, "val_mae": float(np.mean(folds)),
                    "val_mae_folds": [round(x, 4) for x in folds], "train_mae": np.nan})
        print(f"  [{setup}] ensemble {name:30s} val {np.mean(folds):.4f}")

t = table(res, "GB tuned (base)")
t.to_csv(OUT / "exp_ml.csv", index=False)
print(t[["setup", "label", "val_mae", "train_mae", "delta_pct", "folds_better"]].round(4).to_string(index=False))
