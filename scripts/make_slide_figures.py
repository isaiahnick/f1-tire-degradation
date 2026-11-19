"""Figures for the final presentation (reports/figures/slides/).

Plain matplotlib: default color cycle, boxed axes, light grid, legends in a box.
Reads only files the pipeline already wrote (run `make final` first).

    python scripts/make_slide_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"
FEATURES = ROOT / "data" / "processed" / "features.parquet"
OUT = ROOT / "reports" / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight",
})
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COMP = {"SOFT": "tab:red", "MEDIUM": "tab:orange", "HARD": "tab:gray"}
ENS = "Ensemble (GB + Ridge)"
MODEL_ORDER = ["Linear baseline", "Decision tree", "Random forest", "Gradient boosting", "Ridge", "LSTM", ENS]
COLOR = {m: C[i] for i, m in enumerate(MODEL_ORDER)}


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def stint_example():
    """One 2025 race: fuel-corrected lap times with the pit stops marked."""
    raw = pd.read_parquet(ROOT / "data" / "raw" / "2025_09.parquet")
    d = raw[raw["driver"] == "VER"].sort_values("lap_number")
    clean = pd.read_parquet(FEATURES)
    clean = clean[clean["stint_id"].str.startswith("2025_09_VER")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for comp, g in clean.groupby("compound"):
        ax.plot(g["lap_number"], g["lap_time_fc_s"], "o", ms=5, color=COMP[comp], label=comp.title())
    for lap in d.loc[d["pit_in_s"].notna(), "lap_number"]:
        ax.axvline(lap, color="k", ls="--", lw=1)
    ax.plot([], [], "k--", lw=1, label="Pit stop")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Fuel-corrected lap time (s)")
    ax.set_title("2025 Spanish GP, Verstappen (clean laps only)")
    ax.legend(loc="upper right")
    save(fig, "s_stint_example.png")


def degradation():
    df = pd.read_parquet(FEATURES)
    df = df[df["split"] == "train"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        g = df[df["compound"] == comp].groupby("stint_lap")["deg_s"]
        s = pd.DataFrame({"m": g.mean(), "se": g.std() / np.sqrt(g.size()), "n": g.size()})
        s = s[s["n"] >= 40]
        ax.plot(s.index, s["m"], color=COMP[comp], lw=2, label=comp.title())
        ax.fill_between(s.index, s["m"] - 1.96 * s["se"], s["m"] + 1.96 * s["se"], color=COMP[comp], alpha=0.15)
    ax.set_xlabel("Laps on current tire set")
    ax.set_ylabel("Lap time loss vs. stint start (s)")
    ax.set_title("Mean loss by compound, 2022-2024 (95% CI)")
    ax.legend(loc="upper left")
    save(fig, "s_degradation.png")


def train_vs_test():
    r = pd.read_csv(M / "test_results.csv")
    b = r[r["setup"] == "B"].set_index("model")
    lstm = b[b.index.str.startswith("LSTM (")]
    b.loc["LSTM", ["train_mae", "test_mae"]] = lstm[["train_mae", "test_mae"]].mean()
    b = b.loc[MODEL_ORDER]
    x = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - 0.2, b["train_mae"], 0.4, label="Train (2022-2024)", color="tab:blue")
    ax.bar(x + 0.2, b["test_mae"], 0.4, label="Test (2025)", color="tab:orange")
    for xi, (tr, te) in enumerate(zip(b["train_mae"], b["test_mae"])):
        ax.text(xi - 0.2, tr + 0.01, f"{tr:.2f}", ha="center", fontsize=11)
        ax.text(xi + 0.2, te + 0.01, f"{te:.2f}", ha="center", fontsize=11)
    ax.set_xticks(x, ["Linear\nbaseline", "Decision\ntree", "Random\nforest", "Gradient\nboosting", "Ridge",
                      "LSTM\n(3 seeds)", "GB + Ridge\nensemble"])
    ax.set_ylabel("MAE (s)")
    ax.set_ylim(0, 0.8)
    ax.set_title("Setup B: training vs. test MAE")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "s_train_test.png")


def depth_curve():
    d = pd.read_csv(M / "depth_curve.csv")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x, d["train_mae"], "o-", label="Training")
    ax.plot(x, d["val_mae"], "o-", label="Validation (grouped CV)")
    ax.set_xticks(x, d["max_depth"].astype(str))
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("MAE (s)")
    ax.set_title("Decision tree, Setup A")
    ax.legend(loc="lower left")
    save(fig, "s_depth_curve.png")


def horizon():
    d = pd.read_csv(M / "test_error_by_horizon.csv")
    labels = ["1-5", "6-10", "11-15", "16-20", "21+"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in ["Linear baseline", "Random forest", "Gradient boosting", "Ridge", "LSTM", ENS]:
        ax.plot(x, d[m], marker="o", color=COLOR[m], label="GB + Ridge ensemble" if m == ENS else m,
                ls="--" if m == "Linear baseline" else "-")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Laps ahead of the decision point")
    ax.set_ylabel("2025 test MAE (s)")
    ax.set_title("Error by forecast horizon")
    ax.legend(loc="lower right", fontsize=11, ncol=2)
    save(fig, "s_horizon.png")


def importance():
    d = pd.read_csv(M / "final_importance_test.csv")
    nice = {"early_deg_mean": "Early-stint mean loss", "laps_ahead": "Laps past 5-lap window",
            "pace_vs_field": "Car pace vs. field", "early_deg_slope": "Early-stint wear slope",
            "gap_ahead_s": "Gap to car ahead", "position": "Position", "window_gap_mean": "Mean gap in window",
            "circuit": "Circuit", "prac_slope_field": "Practice long-run wear", "compound": "Compound",
            "laps_remaining": "Laps remaining", "stint_lap": "Laps on set", "tyre_life": "Tire age",
            "dirty_air_share": "Dirty air share", "team": "Team"}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = np.arange(len(d))
    ax.barh(y, d["mean"], xerr=d["std"], color="tab:blue", capsize=3)
    ax.set_yticks(y, [nice.get(f, f) for f in d["feature"]])
    ax.set_xlabel("Increase in 2025 MAE when shuffled (s)")
    ax.set_title("Permutation importance (ensemble)")
    ax.grid(axis="y", visible=False)
    save(fig, "s_importance.png")


def update():
    d = pd.read_csv(M / "update_by_decision.csv")
    d = d[d["data"] == "test_2025"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, d["frozen_mae"], "o--", color="tab:gray", label="Forecast fixed after lap 5")
    ax.plot(x, d["updated_mae"], "o-", color="tab:blue", label="Forecast + mean of last 3 errors")
    ax.set_xticks(x, ["5", "6-9", "10-14", "15-19", "20+"])
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Laps observed on the current set at the decision point")
    ax.set_ylabel("2025 test MAE, all later laps (s)")
    ax.set_title("GB + Ridge ensemble, 2025")
    ax.legend(loc="lower left")
    save(fig, "s_update.png")


def pit_backtest():
    s = pd.read_csv(M / "strategy_summary.csv")
    names = {s["method"].iloc[0]: "Cost model, ensemble", "LSTM (one decision at lap 5)": "Cost model, LSTM",
             "Linear baseline": "Cost model, linear baseline",
             "Learned layer (tire model + race context)": "Learned layer",
             "Imitation only (race context, no tire model)": "Learned layer, no cost model",
             "Midpoint (no model)": "Midpoint of segment"}
    s["label"] = s["method"].map(names)
    s = s.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(s))
    ax.barh(y, s["within_3_laps_pct"], color="tab:blue")
    for yi, v in zip(y, s["within_3_laps_pct"]):
        ax.text(v + 0.8, yi, f"{v:.0f}%", va="center", fontsize=13)
    ax.axvline(60, color="k", ls="--", lw=1, label="Objective (60%)")
    ax.set_yticks(y, s["label"])
    ax.set_xlim(0, 70)
    ax.set_xlabel(f"Within 3 laps of actual stop (% of {int(s['stints'].iloc[0])} stops)")
    ax.set_title("Pit stop backtest, 2025 green-flag stops")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "s_pit_backtest.png")


if __name__ == "__main__":
    stint_example()
    degradation()
    train_vs_test()
    depth_curve()
    horizon()
    importance()
    update()
    pit_backtest()
