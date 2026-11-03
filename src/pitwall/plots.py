"""Report figures. Reads only files the earlier stages wrote, so it can be rerun alone:
`python -c "from pitwall.config import load_config; from pitwall import plots; plots.make_all(load_config())"`
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # categorical slots 1-3, fixed order
COMPOUND_COLOR = dict(zip(["SOFT", "MEDIUM", "HARD"], SERIES))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 200,
})


def _save(fig, cfg, name):
    fig.tight_layout()
    fig.savefig(cfg["paths"]["figures"] / name, bbox_inches="tight")
    plt.close(fig)


def degradation_curves(cfg):
    df = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    df = df[df["split"] == "train"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for comp, color in COMPOUND_COLOR.items():
        g = df[(df["compound"] == comp)].groupby("stint_lap")["deg_s"]
        s = pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.size()), "n": g.size()})
        s = s[s["n"] >= 40]
        ax.fill_between(s.index, s["mean"] - 1.96 * s["se"], s["mean"] + 1.96 * s["se"],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(s.index, s["mean"], color=color, linewidth=2)
        ax.annotate(comp.title(), (s.index[-1], s["mean"].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8.5)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Laps on the tire set this stint")
    ax.set_ylabel("Lap time loss vs stint start (s)")
    ax.set_title("Average fuel-corrected lap time loss by compound (2023 to 2024, 95% CI)")
    _save(fig, cfg, "fig1_degradation_by_compound.png")


def model_comparison(cfg):
    s = pd.read_csv(cfg["paths"]["metrics"] / "cv_summary.csv")
    panels = [("A", "Setup A: conditions only (all target laps)"),
              ("B", "Setup B: conditions + first 5 laps of the stint (forecast laps)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharex=True)
    for ax, (exp, title) in zip(axes, panels):
        d = s[(s["experiment"] == exp) & (s["model"] != "Rule: extrapolate")].iloc[::-1]
        y = np.arange(len(d))
        ax.hlines(y, d["train_mae_mean"], d["val_mae_mean"], color=GRID, linewidth=2)
        ax.scatter(d["train_mae_mean"], y, s=36, facecolor=SURFACE, edgecolor=SERIES[0],
                   linewidth=1.5, zorder=3, label="Train MAE")
        ax.errorbar(d["val_mae_mean"], y, xerr=d["val_mae_std"], fmt="o", ms=6, color=SERIES[0],
                    ecolor=SERIES[0], elinewidth=1, capsize=0, zorder=4, label="Validation MAE (mean ± sd)")
        base = d.loc[d["model"] == "Linear baseline", "val_mae_mean"].iloc[0]
        ax.axvline(base, color=SERIES[1], linewidth=1, linestyle="--", label="Linear baseline (val)")
        ax.set_yticks(y, d["model"])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("MAE of lap time loss (s), grouped 5-fold CV")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower left", fontsize=7.5, bbox_to_anchor=(0, -0.42), ncol=3)
    _save(fig, cfg, "fig2_model_comparison.png")


def importance_plot(cfg):
    imp = pd.read_csv(cfg["paths"]["metrics"] / "permutation_importance.csv")
    panels = [("A", "Random forest", "Setup A: random forest"),
              ("B", "Gradient boosting", "Setup B: gradient boosting")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, (setup, model, title) in zip(axes, panels):
        d = imp[(imp["setup"] == setup) & (imp["model"] == model)].nlargest(10, "mean").iloc[::-1]
        ax.barh(d["feature"], d["mean"], xerr=d["std"], color=SERIES[0], height=0.6,
                error_kw={"elinewidth": 0.8, "ecolor": INK2})
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Increase in validation MAE when shuffled (s)")
        ax.grid(axis="y", visible=False)
    _save(fig, cfg, "fig3_permutation_importance.png")


def depth_plot(cfg):
    d = pd.read_csv(cfg["paths"]["metrics"] / "depth_curve.csv")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(x, d["train_mae"], color=SERIES[0], linewidth=2, marker="o", ms=4)
    ax.plot(x, d["val_mae"], color=SERIES[1], linewidth=2, marker="o", ms=4)
    ax.annotate("train", (x[-1], d["train_mae"].iloc[-1]), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8.5)
    ax.annotate("validation", (x[-1], d["val_mae"].iloc[-1]), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8.5)
    ax.set_xticks(x, d["max_depth"].astype(str))
    ax.set_xlabel("Decision tree max_depth (min_samples_leaf=1)")
    ax.set_ylabel("MAE (s)")
    ax.set_title("Decision tree: deeper trees memorize races (Setup A)")
    _save(fig, cfg, "fig4_tree_depth_curve.png")


def sample_stints(cfg, n=3):
    """One typical stint per compound: the long stint whose random forest error is the
    median for that compound (not the best case)."""
    oof = pd.read_parquet(cfg["paths"]["metrics"] / "oof_B.parquet")
    oof["abs_err"] = (oof["deg_s"] - oof["Random forest"]).abs()
    long = oof.groupby("stint_id").agg(n=("deg_s", "size"), comp=("compound", "first"),
                                       err=("abs_err", "mean"))
    picks = []
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        c = long[(long["comp"] == comp) & (long["n"] >= 12)].sort_values("err")
        if len(c):
            picks.append(c.index[len(c) // 2])
    fig, axes = plt.subplots(1, len(picks), figsize=(7.6, 2.7), sharey=True)
    for ax, sid in zip(np.atleast_1d(axes), picks):
        g = oof[oof["stint_id"] == sid].sort_values("stint_lap")
        ax.plot(g["stint_lap"], g["deg_s"], "o", ms=3.5, color=INK2, label="Actual")
        ax.plot(g["stint_lap"], g["Linear baseline"], color=SERIES[1], linewidth=2, label="Linear baseline")
        ax.plot(g["stint_lap"], g["Random forest"], color=SERIES[0], linewidth=2, label="Random forest (B)")
        r = g.iloc[0]
        ax.set_title(f"{r['event_name'].replace(' Grand Prix', '')} {sid[:4]}, {r['driver']}, "
                     f"{r['compound'].title()}", fontsize=8.5)
        ax.set_xlabel("Laps on the tire set")
    np.atleast_1d(axes)[0].set_ylabel("Lap time loss (s)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")
    _save(fig, cfg, "fig5_sample_stints.png")


def make_all(cfg):
    degradation_curves(cfg)
    model_comparison(cfg)
    importance_plot(cfg)
    depth_plot(cfg)
    sample_stints(cfg)
    print(f"plots: wrote figures to {cfg['paths']['figures']}")
