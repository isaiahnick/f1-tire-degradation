"""Stage 5: hyperparameter tuning and the weather-feature test.

Grid search runs inside the same grouped folds as `train` (PredefinedSplit on
`cv_fold`), so tuning never sees a validation race while fitting. Only 2023 and
2024 rows are used. Because the same folds pick the settings and score them, the
tuned CV numbers are slightly optimistic; the 2025 test set (stage `test`) is
the unbiased check.

Outputs (reports/metrics):
  tuning_results.csv   every grid point: mean train/val MAE across folds
  tuned_params.json    best settings per setup and model (read by `test`)
  weather_ablation.csv tuned models with vs without the weather features
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from .models import build_models, feature_lists
from .train import TARGET, load_train

WEATHER = ["track_temp", "air_temp", "humidity"]

GRIDS = {
    "Ridge": {"model__alpha": [1, 10, 100, 1000, 3000, 10000, 30000]},
    "Decision tree": {"model__max_depth": [2, 3, 4, 5, 6, 8],
                      "model__min_samples_leaf": [20, 50, 100]},
    "Random forest": {"model__n_estimators": [200],
                      "model__min_samples_leaf": [20, 50, 100],
                      "model__max_features": [0.33, 0.6]},
    "Gradient boosting": {"model__learning_rate": [0.03, 0.1],
                          "model__max_leaf_nodes": [7, 15, 31],
                          "model__min_samples_leaf": [40, 100],
                          "model__max_iter": [300]},
}


def rows_for(train: pd.DataFrame, setup: str) -> pd.DataFrame:
    return (train[train["in_forecast"]] if setup == "B" else train).reset_index(drop=True)


def without_weather(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["features"]["numeric"] = [f for f in c["features"]["numeric"] if f not in WEATHER]
    return c


def final_config(cfg: dict) -> dict:
    """Config with the final feature set (weather dropped if features.drop_weather)."""
    return without_weather(cfg) if cfg["features"].get("drop_weather", False) else copy.deepcopy(cfg)


def cv_mae(df: pd.DataFrame, cfg: dict, setup: str, name: str, params: dict) -> tuple[float, float]:
    num, cat, inter = feature_lists(cfg, setup)
    X, y = df[num + cat + inter], df[TARGET].to_numpy()
    pipe = clone(build_models(cfg, setup)[name]).set_params(**params)
    tr_mae, va_mae = [], []
    for fold in sorted(df["cv_fold"].unique()):
        tr, va = (df["cv_fold"] != fold).to_numpy(), (df["cv_fold"] == fold).to_numpy()
        m = clone(pipe).fit(X[tr], y[tr])
        tr_mae.append(mean_absolute_error(y[tr], m.predict(X[tr])))
        va_mae.append(mean_absolute_error(y[va], m.predict(X[va])))
    return float(np.mean(tr_mae)), float(np.mean(va_mae))


def run(cfg: dict) -> dict:
    out = cfg["paths"]["metrics"]
    full_cfg, cfg = cfg, final_config(cfg)          # search on the final feature set
    train = load_train(cfg)
    rows, best = [], {}
    for setup in ["A", "B"]:
        df = rows_for(train, setup)
        num, cat, inter = feature_lists(cfg, setup)
        X, y = df[num + cat + inter], df[TARGET].to_numpy()
        cv = PredefinedSplit(df["cv_fold"].to_numpy())
        best[setup] = {}
        for name, grid in GRIDS.items():
            t0 = time.time()
            gs = GridSearchCV(build_models(cfg, setup)[name], grid, cv=cv,
                              scoring="neg_mean_absolute_error", return_train_score=True,
                              refit=False, n_jobs=1)
            gs.fit(X, y)
            res = pd.DataFrame(gs.cv_results_)
            for _, r in res.iterrows():
                rows.append({"setup": setup, "model": name,
                             "params": json.dumps({k.replace("model__", ""): v for k, v in r["params"].items()}),
                             "train_mae": -r["mean_train_score"], "val_mae": -r["mean_test_score"],
                             "val_mae_std": r["std_test_score"]})
            i = int(np.argmax(res["mean_test_score"]))
            best[setup][name] = {"params": res["params"][i], "val_mae": -res["mean_test_score"][i],
                                 "train_mae": -res["mean_train_score"][i]}
            print(f"tune[{setup}] {name:18s} best val MAE {-res['mean_test_score'][i]:.4f} "
                  f"{res['params'][i]}  ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(out / "tuning_results.csv", index=False)

    # weather test: the same tuned settings with and without the weather features
    ab = []
    for setup in ["A", "B"]:
        df = rows_for(train, setup)
        for name, b in best[setup].items():
            _, with_w = cv_mae(df, full_cfg, setup, name, b["params"])
            tr, without_w = cv_mae(df, without_weather(full_cfg), setup, name, b["params"])
            ab.append({"setup": setup, "model": name, "val_mae_with_weather": with_w,
                       "val_mae_without_weather": without_w, "train_mae_without_weather": tr})
            print(f"weather[{setup}] {name:18s} with {with_w:.4f} without {without_w:.4f}")
    ab = pd.DataFrame(ab)
    ab["change_pct"] = 100 * (ab["val_mae_without_weather"] - ab["val_mae_with_weather"]) / ab["val_mae_with_weather"]
    ab.to_csv(out / "weather_ablation.csv", index=False)

    serial = {s: {m: {"params": {k.replace("model__", ""): v for k, v in b["params"].items()},
                      "cv_val_mae": round(b["val_mae"], 4), "cv_train_mae": round(b["train_mae"], 4)}
                  for m, b in d.items()} for s, d in best.items()}
    (out / "tuned_params.json").write_text(json.dumps(serial, indent=2))
    return serial
