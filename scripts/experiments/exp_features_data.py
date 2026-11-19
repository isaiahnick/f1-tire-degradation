"""Experiments 2 and 4: new F1 features and adding 2022 to training.

Setup A (pre-stint, feeds the new-tire curve) only gets features known before the
stint: practice long-run wear. Setup B also gets car pace vs field and dirty air
from the observed window.
"""
import pandas as pd

from common import OUT, cv, feature_sets, load, make, table

f = load(with_2022=True)
print("practice coverage (2023-24 rows):",
      1 - f.loc[f.year >= 2023, "prac_slope_field_missing"].mean().round(3))

PRAC = (["prac_slope_field"], ["stint_lap_x_prac"])
PRAC_T = (["prac_slope_field", "prac_slope_team"], ["stint_lap_x_prac"])
PACE = (["pace_vs_field"], [])
AIR = (["dirty_air_share", "window_gap_mean"], [])


def combo(*parts):
    return [c for p in parts for c in p[0]], [c for p in parts for c in p[1]]


variants = {
    "A": {"base": combo(), "+practice (field)": combo(PRAC), "+practice (field+team)": combo(PRAC_T)},
    "B": {"base": combo(), "+pace vs field": combo(PACE), "+dirty air": combo(AIR),
          "+practice (field)": combo(PRAC), "+practice (field+team)": combo(PRAC_T),
          "+all new": combo(PACE, AIR, PRAC_T)},
}
res = []
for setup, vs in variants.items():
    for model in ["Gradient boosting", "Ridge"]:
        for name, (extra, extra_inter) in vs.items():
            num, cat, inter = feature_sets(setup, extra, extra_inter)
            cols = num + cat + inter
            pipe = make(model, setup, num, cat, inter)
            res.append(cv(f, setup, pipe, cols, label=f"{model} | {name}"))
        # 2022 as extra training data (base features, then with practice)
        for name, (extra, extra_inter) in {"base": combo(), "+practice (field)": combo(PRAC)}.items():
            num, cat, inter = feature_sets(setup, extra, extra_inter)
            pipe = make(model, setup, num, cat, inter)
            for w in [1.0, 0.5]:
                res.append(cv(f, setup, pipe, num + cat + inter, use_2022=True, weight_2022=w,
                              label=f"{model} | {name} | +2022 w={w}"))

t = []
for model in ["Gradient boosting", "Ridge"]:
    sub = [r for r in res if r["label"].startswith(model)]
    t.append(table(sub, f"{model} | base"))
t = pd.concat(t, ignore_index=True)
t.to_csv(OUT / "exp_features_data.csv", index=False)
print(t[["setup", "label", "val_mae", "train_mae", "delta_pct", "folds_better"]].round(4).to_string(index=False))
