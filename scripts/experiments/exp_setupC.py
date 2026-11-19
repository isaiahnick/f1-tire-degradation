"""Experiment 3: use every lap seen so far (Setup C) instead of the first 5 only.

Rows: for each 2023-24 stint and each decision point d (every 2nd observed lap
from d=5), every later lap of the stint. Same grouped CV folds. Compared on the
SAME rows:
  B frozen      production Setup B GB (only knows the first 5 laps)
  B + bias      B frozen plus the mean of its recent errors on laps already seen
  C GB / Ridge  models trained on Setup C rows with all-laps history features
  C rules       flat at the recent level; extend the all-laps line
"""
import numpy as np
import pandas as pd

from common import OUT, cv, feature_sets, load, make
from hist import HIST_C, INTER_C, c_rows

f = load(with_2022=False)
C = c_rows(f, step=2)
print(f"C rows {len(C):,} from {C.stint_id.nunique()} stints; decision points {C.groupby('stint_id').d.nunique().sum()}")

numB, cat, interB = feature_sets("B")
colsB = numB + cat + interB
numA, _, interA = feature_sets("A")
numC = numA + HIST_C
interC = interA + INTER_C
colsC = numC + cat + interC

res = []
pred = pd.DataFrame(index=C.index)
for k in range(5):
    trB = f[f["in_forecast"] & (f["cv_fold"] != k)]
    trC, va = C[C["cv_fold"] != k], C["cv_fold"] == k
    gbB = make("Gradient boosting", "B", numB, cat, interB).fit(trB[colsB], trB["deg_s"])
    pred.loc[va, "B frozen"] = gbB.predict(C.loc[va, colsB])
    # online bias: B's errors on the observed laps after the first window
    fB = f[f["cv_fold"] == k].copy()
    fB["resB"] = np.where(fB["in_forecast"], fB["deg_s"] - gbB.predict(fB[colsB]), np.nan)
    fB = fB.sort_values(["stint_id", "lap_number"])
    fB["clean_pos"] = fB.groupby("stint_id").cumcount() + 1
    vv = C.loc[va, ["stint_id", "d"]].drop_duplicates()
    bias = {}
    for sid, g in fB.groupby("stint_id"):
        r = g["resB"].to_numpy()
        for d in vv.loc[vv.stint_id == sid, "d"]:
            seen = r[:d][~np.isnan(r[:d])][-5:]
            bias[(sid, d)] = seen.mean() * len(seen) / (len(seen) + 2) if len(seen) else 0.0
    pred.loc[va, "B + bias"] = pred.loc[va, "B frozen"].to_numpy() + np.array(
        [bias[(s, d)] for s, d in zip(C.loc[va, "stint_id"], C.loc[va, "d"])])
    for name in ["Gradient boosting", "Ridge"]:
        m = make(name, "B", numC, cat, interC).fit(trC[colsC], trC["deg_s"])
        pred.loc[va, f"C {name}"] = m.predict(C.loc[va, colsC])
    print(f"fold {k} done", flush=True)
pred["Rule: flat recent level"] = C["hist_mean_last5"]
pred["Rule: extend all-laps line"] = C["hist_fit_now"] + C["hist_slope_all"] * C["laps_ahead_c"]

C["d_bucket"] = pd.cut(C["d"], [4, 5, 9, 14, 19, 99], labels=["d=5", "6-9", "10-14", "15-19", "20+"])
C["h_bucket"] = pd.cut(C["laps_ahead_c"], [0, 5, 10, 15, 20, 99], labels=["1-5", "6-10", "11-15", "16-20", "21+"])
err = pred.sub(C["deg_s"], axis=0).abs()
overall = err.groupby(C["cv_fold"]).mean().mean().rename("val_mae")
by_d = err.groupby(C["d_bucket"], observed=True).mean().assign(n=C.groupby("d_bucket", observed=True).size())
by_h = err.groupby(C["h_bucket"], observed=True).mean().assign(n=C.groupby("h_bucket", observed=True).size())
pd.concat({"overall": overall.to_frame().T}).to_csv(OUT / "exp_setupC_overall.csv")
by_d.to_csv(OUT / "exp_setupC_by_decision.csv")
by_h.to_csv(OUT / "exp_setupC_by_horizon.csv")
print("overall (mean of fold MAEs):\n", overall.round(4).to_string())
print("by decision point:\n", by_d.round(4).to_string())
print("by horizon:\n", by_h.round(4).to_string())
