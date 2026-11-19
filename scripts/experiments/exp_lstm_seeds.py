"""Experiment 5b: LSTM seed ensemble. Same CV folds, best settings, 3 seeds;
saves each seed's OOF predictions so they can be averaged and blended."""
import json
import time

import pandas as pd

from common import OUT
from pitwall import lstm
from pitwall.config import load_config
from pitwall.train import load_train
from pitwall.tune import final_config

cfg = final_config(load_config())
train = load_train(cfg)
wins = lstm.windows(train, cfg)
rows = train[train["in_forecast"]].reset_index(drop=True)
hp = json.loads((cfg["paths"]["metrics"] / "lstm_summary.json").read_text())["best_hp"]
out = rows[["stint_id", "lap_number", "cv_fold", "deg_s"]].copy()
for seed in [42, 43, 44, 45, 46]:
    t0 = time.time()
    folds, oof = lstm.cross_validate(rows, wins, cfg, seed=seed, **hp)
    out[f"lstm_{seed}"] = oof
    print(f"seed {seed}: val {folds['val_mae'].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    out.to_parquet(OUT / "oof_lstm_seeds.parquet", index=False)
