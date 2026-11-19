"""Stage 7: the one-time evaluation on the locked 2025 season.

Every model is refit on all training rows (2022 to 2024) with its tuned settings
(tuned_params.json) and the final feature set, then scored once on 2025. Nothing
here feeds back into model choices; that was settled with cross-validation.

Outputs (reports/metrics):
  test_results.csv         MAE / RMSE / R2 and % vs baseline, Setup A and B
  test_predictions_{A,B}.parquet   2025 predictions for every model
  test_error_by_compound.csv, test_error_by_event.csv, test_error_by_horizon.csv
Saved models (models/): final_A.joblib, final_B.joblib (best tree models + baseline),
  lstm_B.pt (LSTM state + preprocessing)
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import lstm
from .models import ENSEMBLE, build_models, ensemble, feature_lists
from .tune import final_config

TARGET = "deg_s"


def tuned_models(cfg: dict, setup: str) -> dict:
    tuned = json.loads((cfg["paths"]["metrics"] / "tuned_params.json").read_text())[setup]
    models = build_models(cfg, setup)
    for name, t in tuned.items():
        if name != ENSEMBLE:
            models[name] = clone(models[name]).set_params(**{f"model__{k}": v for k, v in t["params"].items()})
    if ENSEMBLE in tuned:
        models[ENSEMBLE] = ensemble(models)
    return models


def metrics(y, p) -> dict:
    return {"mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": r2_score(y, p)}


def run(cfg: dict) -> pd.DataFrame:
    out, model_dir = cfg["paths"]["metrics"], cfg["paths"]["models"]
    fcfg = final_config(cfg)
    df = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    train_all, test_all = df[df["split"] == "train"], df[df["split"] == "test"]
    rows, saved = [], {}

    for setup in ["A", "B"]:
        tr = train_all[train_all["in_forecast"]] if setup == "B" else train_all
        te = test_all[test_all["in_forecast"]] if setup == "B" else test_all
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        num, cat, inter = feature_lists(fcfg, setup)
        X_tr, X_te = tr[num + cat + inter], te[num + cat + inter]
        preds = te[["race_id", "stint_id", "event_name", "driver", "lap_number", "compound",
                    "stint_lap", "laps_ahead", TARGET]].copy()
        fitted = {}
        for name, pipe in tuned_models(fcfg, setup).items():
            m = clone(pipe).fit(X_tr, tr[TARGET])
            preds[name] = m.predict(X_te)
            fitted[name] = m
            rows.append({"setup": setup, "model": name, "n_test": len(te),
                         "train_mae": mean_absolute_error(tr[TARGET], m.predict(X_tr)),
                         **{f"test_{k}": v for k, v in metrics(te[TARGET], preds[name]).items()}})
        if setup == "B" and cfg.get("lstm", {}).get("retrain", True):
            wins = lstm.windows(pd.concat([train_all, test_all]), fcfg)
            hp = json.loads((out / "lstm_summary.json").read_text())["best_hp"]
            net_preds = []
            for seed in [cfg["seed"], cfg["seed"] + 1, cfg["seed"] + 2]:
                m = lstm.LSTMRegressor(fcfg, seed=seed, **hp).fit(tr, wins)
                net_preds.append(m.predict(te, wins))
                rows.append({"setup": setup, "model": f"LSTM (seed {seed})", "n_test": len(te),
                             "train_mae": mean_absolute_error(tr[TARGET], m.predict(tr, wins)),
                             **{f"test_{k}": v for k, v in metrics(te[TARGET], net_preds[-1]).items()}})
                if seed == cfg["seed"]:
                    lstm_model = m
            preds["LSTM"] = net_preds[0]
            import torch
            torch.save({"state": lstm_model.net.state_dict(), "prep": lstm_model.prep.__dict__,
                        "hp": lstm_model.hp}, model_dir / "lstm_B.pt")
        preds.to_parquet(out / f"test_predictions_{setup}.parquet", index=False)
        saved[setup] = fitted

    res = pd.DataFrame(rows)
    for setup in ["A", "B"]:
        base = res.loc[(res["setup"] == setup) & (res["model"] == "Linear baseline"), "test_mae"].iloc[0]
        sel = res["setup"] == setup
        res.loc[sel, "test_mae_vs_baseline_pct"] = 100 * (base - res.loc[sel, "test_mae"]) / base
    res.to_csv(out / "test_results.csv", index=False)

    # final models for the strategy layer and dashboard
    joblib.dump({"cfg": fcfg, "models": saved["A"]}, model_dir / "final_A.joblib")
    joblib.dump({"cfg": fcfg, "models": saved["B"]}, model_dir / "final_B.joblib")

    pb = pd.read_parquet(out / "test_predictions_B.parquet")
    names = [n for n in ["Linear baseline", "Ridge", "Random forest", "Gradient boosting", ENSEMBLE, "LSTM"]
             if n in pb]
    g = pb.groupby("compound")
    pd.DataFrame({n: g[[TARGET, n]].apply(lambda x, n=n: mean_absolute_error(x[TARGET], x[n]))
                  for n in names}).assign(laps=g.size()).to_csv(out / "test_error_by_compound.csv")
    g = pb.groupby("event_name")
    pd.DataFrame({n: g[[TARGET, n]].apply(lambda x, n=n: mean_absolute_error(x[TARGET], x[n]))
                  for n in names}).assign(laps=g.size()).sort_values("Random forest") \
        .to_csv(out / "test_error_by_event.csv")
    g = pb.groupby(pd.cut(pb["laps_ahead"], [0, 5, 10, 15, 20, 60]), observed=True)
    pd.DataFrame({n: g[[TARGET, n]].apply(lambda x, n=n: mean_absolute_error(x[TARGET], x[n]))
                  for n in names}).assign(laps=g.size()).to_csv(out / "test_error_by_horizon.csv")

    show = res[["setup", "model", "train_mae", "test_mae", "test_rmse", "test_r2", "test_mae_vs_baseline_pct"]]
    print(show.round(3).to_string(index=False))
    return res
