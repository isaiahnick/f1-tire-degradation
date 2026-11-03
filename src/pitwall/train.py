"""Stage 4: grouped cross-validation of the baseline and classical models.

Only `split == "train"` rows (2023 and 2024) are read. Folds come from the
`cv_fold` column written by the features stage, so every model (including the
LSTM later) is scored on exactly the same held-out races. The 2025 test season
is not touched here.

Three experiments:
  A     all target laps,      conditions features      (proposal framing)
  A_fc  forecast laps only,   conditions features      (ablation: same rows as B)
  B     forecast laps only,   conditions + stint history

Outputs (reports/metrics), one set per experiment:
  cv_folds_<exp>.csv, cv_summary_<exp>.csv, oof_<exp>.parquet,
  error_by_compound_<exp>.csv, plus permutation_importance.csv, depth_curve.csv,
  error_by_event_A.csv, error_by_horizon_B.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import plots
from .models import MODEL_SETTINGS, build_models, feature_lists

TARGET = "deg_s"
EXPERIMENTS = {"A": ("A", False), "A_fc": ("A", True), "B": ("B", True)}  # exp -> (setup, forecast rows)
KEEP = ["race_id", "stint_id", "event_name", "driver", "lap_number", "compound",
        "stint_lap", "laps_ahead", "cv_fold", TARGET]


def metrics(y, p) -> dict:
    return {"mae": mean_absolute_error(y, p),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": r2_score(y, p)}


def load_train(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    df = df[df["split"] == "train"].reset_index(drop=True)
    assert (df["cv_fold"] >= 0).all(), "train rows without a CV fold"
    return df


def cross_validate(df: pd.DataFrame, cfg: dict, setup: str, tag: str):
    num, cat, inter = feature_lists(cfg, setup)
    X, y = df[num + cat + inter], df[TARGET].to_numpy()
    rows, oof = [], pd.DataFrame(index=df.index)
    for name, pipe in build_models(cfg, setup).items():
        oof[name] = np.nan
        for fold in sorted(df["cv_fold"].unique()):
            tr, va = (df["cv_fold"] != fold).to_numpy(), (df["cv_fold"] == fold).to_numpy()
            t0 = time.time()
            m = clone(pipe).fit(X[tr], y[tr])
            fit_s = time.time() - t0
            p_tr, p_va = m.predict(X[tr]), m.predict(X[va])
            oof.loc[va, name] = p_va
            rows.append({"experiment": tag, "model": name, "fold": fold, "fit_s": fit_s,
                         "n_train": int(tr.sum()), "n_val": int(va.sum()),
                         **{f"train_{k}": v for k, v in metrics(y[tr], p_tr).items()},
                         **{f"val_{k}": v for k, v in metrics(y[va], p_va).items()}})
        mae = np.mean([r["val_mae"] for r in rows if r["model"] == name])
        print(f"train[{tag}]: {name:18s} val MAE {mae:.4f}")
    return pd.DataFrame(rows), oof


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in folds.columns if c.startswith(("train_", "val_"))] + ["fit_s"]
    s = folds.groupby("model", sort=False)[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    base = s.loc["Linear baseline", "val_mae_mean"]
    s["mae_vs_baseline_pct"] = 100 * (base - s["val_mae_mean"]) / base
    s["gap_val_minus_train"] = s["val_mae_mean"] - s["train_mae_mean"]
    s["settings"] = [MODEL_SETTINGS[m] for m in s.index]
    s.insert(0, "experiment", folds["experiment"].iloc[0])
    return s.reset_index()


def importance(df: pd.DataFrame, cfg: dict, setup: str, model_name: str, n_rows: int = 3000):
    """Permutation importance on each validation fold, on raw columns (before encoding)."""
    num, cat, inter = feature_lists(cfg, setup)
    feats = num + cat + inter
    pipe = build_models(cfg, setup)[model_name]
    out = []
    for fold in sorted(df["cv_fold"].unique()):
        tr, va = df["cv_fold"] != fold, df["cv_fold"] == fold
        m = clone(pipe).fit(df.loc[tr, feats], df.loc[tr, TARGET])
        sub = df[va].sample(min(n_rows, int(va.sum())), random_state=cfg["seed"])
        r = permutation_importance(m, sub[feats], sub[TARGET], scoring="neg_mean_absolute_error",
                                   n_repeats=5, random_state=cfg["seed"], n_jobs=-1)
        out.append(pd.DataFrame({"feature": feats, "fold": fold, "mae_increase": r.importances_mean}))
    imp = pd.concat(out).groupby("feature")["mae_increase"].agg(["mean", "std"]).reset_index()
    imp["model"], imp["setup"] = model_name, setup
    # tree pipelines ignore the linear-only interaction columns: drop their zero rows
    return imp[~imp["feature"].isin(inter)].sort_values("mean", ascending=False)


def depth_curve(df: pd.DataFrame, cfg: dict, depths=(2, 4, 6, 8, 10, 12, 16, 20, None)):
    num, cat, inter = feature_lists(cfg, "A")
    X, y = df[num + cat + inter], df[TARGET].to_numpy()
    base = build_models(cfg, "A")["Decision tree"]
    rows = []
    for d in depths:
        pipe = clone(base).set_params(model__max_depth=d, model__min_samples_leaf=1)
        for fold in sorted(df["cv_fold"].unique()):
            tr, va = (df["cv_fold"] != fold).to_numpy(), (df["cv_fold"] == fold).to_numpy()
            m = pipe.fit(X[tr], y[tr])
            rows.append({"max_depth": "none" if d is None else d, "fold": fold,
                         "train_mae": mean_absolute_error(y[tr], m.predict(X[tr])),
                         "val_mae": mean_absolute_error(y[va], m.predict(X[va]))})
    return (pd.DataFrame(rows).groupby("max_depth", sort=False)[["train_mae", "val_mae"]]
            .mean().reset_index())


def mae_by(oof: pd.DataFrame, key, names: list[str]) -> pd.DataFrame:
    g = oof.groupby(key, observed=True)
    out = pd.DataFrame({n: g[[TARGET, n]].apply(lambda x, n=n: mean_absolute_error(x[TARGET], x[n]))
                        for n in names})
    out["laps"] = g.size()
    return out


def variance_check(df: pd.DataFrame) -> dict:
    """How much of the target is stint-specific? In-sample R2 of (a) each stint's own
    mean and (b) each stint's own straight line. An upper bound on what is learnable
    if you knew each stint's level and wear rate exactly."""
    y = df[TARGET].to_numpy(float)
    tot = ((y - y.mean()) ** 2).sum()

    def line(g):
        b = np.polyfit(g["stint_lap"].to_numpy(float), g[TARGET].to_numpy(float), 1)
        return pd.Series(np.polyval(b, g["stint_lap"].to_numpy(float)), index=g.index)

    stint_mean = df.groupby("stint_id")[TARGET].transform("mean").to_numpy()
    stint_line = df.groupby("stint_id", group_keys=False)[["stint_lap", TARGET]].apply(line)
    stint_line = stint_line.reindex(df.index).to_numpy()
    return {"r2_stint_mean": float(1 - ((y - stint_mean) ** 2).sum() / tot),
            "r2_stint_line": float(1 - ((y - stint_line) ** 2).sum() / tot),
            "resid_sd_stint_line_s": float(np.std(y - stint_line))}


def run(cfg: dict) -> pd.DataFrame:
    out = cfg["paths"]["metrics"]
    train = load_train(cfg)
    vc = variance_check(train)
    pd.Series(vc).to_json(out / "variance_check.json", indent=2)
    print(f"train: per-stint line explains R2={vc['r2_stint_line']:.3f} of the target (in-sample)")
    summaries, oofs = [], {}
    for tag, (setup, fc_only) in EXPERIMENTS.items():
        df = train[train["in_forecast"]].reset_index(drop=True) if fc_only else train
        folds, oof = cross_validate(df, cfg, setup, tag)
        s = summarize(folds)
        folds.to_csv(out / f"cv_folds_{tag}.csv", index=False)
        s.to_csv(out / f"cv_summary_{tag}.csv", index=False)
        oof_df = pd.concat([df[KEEP], oof], axis=1)
        oof_df.to_parquet(out / f"oof_{tag}.parquet", index=False)
        mae_by(oof_df, "compound", list(oof.columns)).to_csv(out / f"error_by_compound_{tag}.csv")
        summaries.append(s)
        oofs[tag] = (df, oof_df)

    df_a, oof_a = oofs["A"]
    mae_by(oof_a, "event_name", ["Linear baseline", "Random forest", "Gradient boosting"]) \
        .sort_values("Random forest").to_csv(out / "error_by_event_A.csv")
    df_b, oof_b = oofs["B"]
    bins = pd.cut(oof_b["laps_ahead"], [0, 5, 10, 15, 20, 60])
    mae_by(oof_b.assign(horizon=bins), "horizon",
           ["Linear baseline", "Rule: flat", "Rule: extrapolate", "Ridge", "Gradient boosting"]) \
        .to_csv(out / "error_by_horizon_B.csv")

    imp = pd.concat([importance(df_a, cfg, "A", "Random forest"),
                     importance(df_a, cfg, "A", "Gradient boosting"),
                     importance(df_b, cfg, "B", "Gradient boosting")])
    imp.to_csv(out / "permutation_importance.csv", index=False)
    depth_curve(df_a, cfg).to_csv(out / "depth_curve.csv", index=False)

    allsum = pd.concat(summaries, ignore_index=True)
    allsum.to_csv(out / "cv_summary.csv", index=False)
    plots.make_all(cfg)
    cols = ["experiment", "model", "train_mae_mean", "val_mae_mean", "val_mae_std",
            "val_rmse_mean", "val_r2_mean", "mae_vs_baseline_pct"]
    print(allsum[cols].round(3).to_string(index=False))
    return allsum
