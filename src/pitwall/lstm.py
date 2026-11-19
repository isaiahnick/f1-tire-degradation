"""Deep learning model: LSTM over the first laps of a stint (Setup B task).

Same information and same rows as Setup B, so the comparison with the tree models
is fair:
  - the LSTM reads the stint's observed window (first `history_laps` target laps)
    as a sequence: lap time loss, laps on the set, gap ahead, laps remaining,
    and track temperature when weather features are in use
  - its last hidden state is joined with the forecast lap's own features
    (the Setup A inputs plus laps past the window) and a small dense head
    predicts the lap time loss for that lap
Training uses MAE (L1) loss, Adam, dropout, and early stopping on a held-out 15%
of the training races (grouped, so no race is in both). Scaling and encoders are
fit on training rows only.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

TARGET = "deg_s"


def _torch():
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return torch


# ---------------------------------------------------------------- data
def seq_features(cfg: dict) -> list[str]:
    cols = [TARGET, "stint_lap", "gap_ahead_s", "laps_remaining"]
    if "track_temp" in cfg["features"]["numeric"]:
        cols.append("track_temp")
    return cols


def head_numeric(cfg: dict) -> list[str]:
    """Static inputs next to the sequence: conditions + the Setup B history features
    (so the LSTM sees the same information as the other Setup B models)."""
    return list(cfg["features"]["numeric"]) + list(cfg["features"]["history"])


def windows(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """stint_id -> (history_laps, n_seq_features) array of the observed window."""
    h = cfg["target"]["history_laps"]
    cols = seq_features(cfg)
    out = {}
    for sid, g in df.sort_values(["stint_id", "lap_number"]).groupby("stint_id", sort=False):
        w = g.iloc[:h]
        if len(w) == h:
            out[sid] = w[cols].to_numpy(np.float32)
    return out


class Prep:
    """Fit-on-train scaling for sequences and head features, one-hot for categories."""

    def fit(self, rows: pd.DataFrame, wins: dict, cfg: dict) -> "Prep":
        self.cfg = cfg
        stack = np.concatenate([wins[s] for s in rows["stint_id"].unique()])
        self.seq_mu, self.seq_sd = stack.mean(0), stack.std(0) + 1e-6
        num = rows[head_numeric(cfg)].astype(float)
        self.num_mu, self.num_sd = num.mean().to_numpy(), num.std().to_numpy() + 1e-6
        self.cats = {c: sorted(rows[c].astype(str).unique()) for c in cfg["features"]["categorical"]}
        return self

    def transform(self, rows: pd.DataFrame, wins: dict):
        seq = np.stack([wins[s] for s in rows["stint_id"]])
        seq = (seq - self.seq_mu) / self.seq_sd
        num = (rows[head_numeric(self.cfg)].astype(float).to_numpy() - self.num_mu) / self.num_sd
        oh = [np.stack([(rows[c].astype(str).to_numpy() == v) for v in vals], 1).astype(np.float32)
              for c, vals in self.cats.items()]
        head = np.concatenate([num.astype(np.float32), *oh], 1)
        return seq.astype(np.float32), np.nan_to_num(head).astype(np.float32)


# ---------------------------------------------------------------- model
def make_net(n_seq: int, n_head: int, hidden: int, dense: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class StintLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden + n_head, dense), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(dense, dense // 2), nn.ReLU(), nn.Linear(dense // 2, 1))

        def forward(self, seq, head):
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([h[-1], head], 1)).squeeze(1)

    return StintLSTM()


DEFAULTS = {"hidden": 32, "dense": 64, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4,
            "batch": 256, "max_epochs": 80, "patience": 8, "val_frac": 0.15}


class LSTMRegressor:
    def __init__(self, cfg: dict, seed: int = 42, **hp):
        self.cfg, self.seed = cfg, seed
        self.hp = {**DEFAULTS, **hp}

    def fit(self, rows: pd.DataFrame, wins: dict):
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        races = rows["race_id"].unique()
        val_races = set(rng.choice(races, max(1, int(len(races) * self.hp["val_frac"])), replace=False))
        is_val = rows["race_id"].isin(val_races).to_numpy()
        fit_rows, val_rows = rows[~is_val], rows[is_val]

        self.prep = Prep().fit(fit_rows, wins, self.cfg)
        s_tr, h_tr = self.prep.transform(fit_rows, wins)
        s_va, h_va = self.prep.transform(val_rows, wins)
        y_tr = torch.tensor(fit_rows[TARGET].to_numpy(np.float32))
        y_va = val_rows[TARGET].to_numpy()
        s_tr, h_tr = torch.tensor(s_tr), torch.tensor(h_tr)

        self.net = make_net(s_tr.shape[2], h_tr.shape[1], self.hp["hidden"], self.hp["dense"],
                            self.hp["dropout"])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hp["lr"],
                               weight_decay=self.hp["weight_decay"])
        loss_fn = torch.nn.L1Loss()
        best, best_state, bad, self.history = np.inf, None, 0, []
        n = len(y_tr)
        for epoch in range(self.hp["max_epochs"]):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.hp["batch"]):
                idx = perm[i:i + self.hp["batch"]]
                opt.zero_grad()
                loss = loss_fn(self.net(s_tr[idx], h_tr[idx]), y_tr[idx])
                loss.backward()
                opt.step()
            val_mae = float(np.mean(np.abs(self._predict_arrays(s_va, h_va) - y_va)))
            tr_mae = float(np.mean(np.abs(self._predict_arrays(s_tr.numpy(), h_tr.numpy()) - y_tr.numpy())))
            self.history.append({"epoch": epoch + 1, "train_mae": tr_mae, "early_stop_val_mae": val_mae})
            if val_mae < best - 1e-4:
                best, best_state, bad = val_mae, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.hp["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.best_epoch = int(np.argmin([h["early_stop_val_mae"] for h in self.history]) + 1)
        return self

    def _predict_arrays(self, seq, head) -> np.ndarray:
        torch = _torch()
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.as_tensor(seq), torch.as_tensor(head)).numpy()

    def predict(self, rows: pd.DataFrame, wins: dict) -> np.ndarray:
        s, h = self.prep.transform(rows, wins)
        return self._predict_arrays(s, h)


# ---------------------------------------------------------------- experiments
SWEEP = [{"hidden": 16, "dropout": 0.2, "lr": 1e-3}, {"hidden": 32, "dropout": 0.2, "lr": 1e-3},
         {"hidden": 64, "dropout": 0.3, "lr": 1e-3}, {"hidden": 64, "dropout": 0.3, "lr": 3e-4},
         {"hidden": 128, "dropout": 0.4, "lr": 3e-4}]


def cross_validate(rows: pd.DataFrame, wins: dict, cfg: dict, seed: int = 42, **hp):
    """Grouped 5-fold CV on Setup B forecast rows. Returns (fold metrics, OOF predictions)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    folds, oof = [], np.full(len(rows), np.nan)
    for fold in sorted(rows["cv_fold"].unique()):
        tr, va = (rows["cv_fold"] != fold).to_numpy(), (rows["cv_fold"] == fold).to_numpy()
        m = LSTMRegressor(cfg, seed=seed, **hp).fit(rows[tr], wins)
        p_tr, p_va = m.predict(rows[tr], wins), m.predict(rows[va], wins)
        y_tr, y_va = rows.loc[tr, TARGET].to_numpy(), rows.loc[va, TARGET].to_numpy()
        oof[va] = p_va
        folds.append({"fold": fold, "seed": seed, **hp, "best_epoch": m.best_epoch,
                      "train_mae": mean_absolute_error(y_tr, p_tr),
                      "val_mae": mean_absolute_error(y_va, p_va),
                      "val_rmse": float(np.sqrt(mean_squared_error(y_va, p_va))),
                      "val_r2": r2_score(y_va, p_va)})
    return pd.DataFrame(folds), oof


def run(cfg: dict) -> pd.DataFrame:
    """Small sweep over network size, then 3 seeds of the best size."""
    import json
    import time

    from .train import load_train
    out = cfg["paths"]["metrics"]
    train = load_train(cfg)
    wins = windows(train, cfg)
    rows = train[train["in_forecast"]].reset_index(drop=True)

    sweep, sweep_oof = [], {}
    if not cfg.get("lstm", {}).get("run_sweep", True):
        hp = json.loads((out / "lstm_summary.json").read_text())["best_hp"]
        f, oof = cross_validate(rows, wins, cfg, seed=cfg["seed"], **hp)
        print(f"lstm: kept {hp} (sweep skipped), seed {cfg['seed']} val MAE {f['val_mae'].mean():.4f}")
        sweep_oof[(hp["hidden"], hp["dropout"], hp["lr"])] = (f, oof)
    for hp in (SWEEP if cfg.get("lstm", {}).get("run_sweep", True) else []):
        t0 = time.time()
        f, oof = cross_validate(rows, wins, cfg, seed=cfg["seed"], **hp)
        sweep.append(f)
        sweep_oof[(hp["hidden"], hp["dropout"], hp["lr"])] = (f, oof)
        print(f"lstm: {hp} val MAE {f['val_mae'].mean():.4f} train MAE {f['train_mae'].mean():.4f} "
              f"({time.time() - t0:.0f}s)")
    if sweep:
        sweep = pd.concat(sweep)
        sweep.to_csv(out / "lstm_sweep.csv", index=False)
        best_hp = (sweep.groupby(["hidden", "dropout", "lr"])["val_mae"].mean().idxmin())
        best_hp = {"hidden": int(best_hp[0]), "dropout": float(best_hp[1]), "lr": float(best_hp[2])}
    else:
        best_hp = hp

    first = sweep_oof[(best_hp["hidden"], best_hp["dropout"], best_hp["lr"])]  # seed 42 already run
    seeds, oofs = [first[0]], [first[1]]
    n_seeds = cfg.get("lstm", {}).get("cv_seeds", 3)
    for seed in [cfg["seed"] + i for i in range(1, n_seeds)]:
        f, oof = cross_validate(rows, wins, cfg, seed=seed, **best_hp)
        seeds.append(f)
        oofs.append(oof)
        print(f"lstm: seed {seed} val MAE {f['val_mae'].mean():.4f}")
    seeds = pd.concat(seeds)
    seeds.to_csv(out / "lstm_cv_folds.csv", index=False)
    keep = ["race_id", "stint_id", "event_name", "driver", "lap_number", "compound",
            "stint_lap", "laps_ahead", "cv_fold", TARGET]
    oof_df = rows[keep].copy()
    oof_df["LSTM"] = oofs[0]
    oof_df.to_parquet(out / "oof_B_lstm.parquet", index=False)
    per_seed = seeds.groupby("seed")[["train_mae", "val_mae", "val_rmse", "val_r2"]].mean()
    summary = {"best_hp": best_hp, "seeds": per_seed.round(4).to_dict("index"),
               "val_mae_mean": round(per_seed["val_mae"].mean(), 4),
               "val_mae_sd_across_seeds": round(per_seed["val_mae"].std(), 4) if len(per_seed) > 1 else None,
               "cv_seeds": int(len(per_seed)),
               "train_mae_mean": round(per_seed["train_mae"].mean(), 4),
               "val_r2_mean": round(per_seed["val_r2"].mean(), 4),
               "median_best_epoch": float(seeds["best_epoch"].median())}
    (out / "lstm_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"lstm: best {best_hp}, val MAE {summary['val_mae_mean']} ± {summary['val_mae_sd_across_seeds']} across seeds")
    return seeds
