"""Figures for the final presentation (reports/figures/slides/).

matplotlib with a light, consistent style: no top/right spines, light y grid,
grey for comparison models and NC State red for the final model. Reads only
files the pipeline already wrote (run `make final` first).

    python scripts/make_slide_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"
FEATURES = ROOT / "data" / "processed" / "features.parquet"
OUT = ROOT / "reports" / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

RED, DARK, MID, LIGHT = "#CC0000", "#404040", "#8C8C8C", "#C8C8C8"
COMP = {"SOFT": "#E10600", "MEDIUM": "#E8B400", "HARD": "#7F7F7F"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MID,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E5E5E5", "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "text.color": "#1A1A1A",
    "axes.labelcolor": "#1A1A1A", "xtick.color": DARK, "ytick.color": DARK,
})
ENS = "Ensemble (GB + Ridge)"


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def stint_example():
    """One 2025 race: fuel-corrected lap times by compound, pit stops marked."""
    raw = pd.read_parquet(ROOT / "data" / "raw" / "2025_09.parquet")
    d = raw[raw["driver"] == "VER"].sort_values("lap_number")
    clean = pd.read_parquet(FEATURES)
    clean = clean[clean["stint_id"].str.startswith("2025_09_VER")]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for lap in d.loc[d["pit_in_s"].notna(), "lap_number"]:
        ax.axvline(lap, color=MID, ls=(0, (4, 3)), lw=1.2, zorder=1)
    for comp, g in clean.groupby("compound"):
        ax.scatter(g["lap_number"], g["lap_time_fc_s"], s=42, color=COMP[comp], label=comp.title(), zorder=3,
                   edgecolor="white", linewidth=0.6)
    ax.plot([], [], color=MID, ls=(0, (4, 3)), label="Pit stop")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Fuel-corrected lap time (s)")
    ax.set_ylim(None, ax.get_ylim()[1] + 0.35)
    ax.legend(loc="upper left", ncol=3)
    save(fig, "s_stint_example.png")


def degradation():
    df = pd.read_parquet(FEATURES)
    df = df[df["split"] == "train"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        g = df[df["compound"] == comp].groupby("stint_lap")["deg_s"]
        s = pd.DataFrame({"m": g.mean(), "se": g.std() / np.sqrt(g.size()), "n": g.size()})
        s = s[s["n"] >= 40]
        ax.fill_between(s.index, s["m"] - 1.96 * s["se"], s["m"] + 1.96 * s["se"], color=COMP[comp], alpha=0.15, lw=0)
        ax.plot(s.index, s["m"], color=COMP[comp], lw=2.5, label=comp.title())
    ax.set_xlabel("Laps on current tire set")
    ax.set_ylabel("Lap time loss (s)")
    ax.legend(loc="upper left")
    save(fig, "s_degradation.png")


def train_vs_test():
    r = pd.read_csv(M / "test_results.csv")
    b = r[r["setup"] == "B"].set_index("model")
    lstm = b[b.index.str.startswith("LSTM (")]
    b.loc["LSTM", ["train_mae", "test_mae"]] = lstm[["train_mae", "test_mae"]].mean()
    order = ["Linear baseline", "Decision tree", "Random forest", "Gradient boosting", "Ridge", "LSTM", ENS]
    labels = ["Linear\nbaseline", "Decision\ntree", "Random\nforest", "Gradient\nboosting", "Ridge", "LSTM",
              "GB+Ridge\nensemble"]
    b = b.loc[order]
    x = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - 0.19, b["train_mae"], 0.38, color=LIGHT, label="Train (2022-2024)")
    ax.bar(x + 0.19, b["test_mae"], 0.38, color=[RED if m == ENS else DARK for m in order], label="Test (2025)")
    for xi, te in enumerate(b["test_mae"]):
        ax.text(xi + 0.19, te + 0.012, f"{te:.2f}", ha="center", fontsize=12,
                fontweight="bold" if order[xi] == ENS else "normal", color=RED if order[xi] == ENS else DARK)
    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylabel("MAE (s)")
    ax.set_ylim(0, 0.75)
    ax.legend(loc="upper right")
    save(fig, "s_train_test.png")


def horizon():
    """Baseline vs. final model only, with the MAE value at every point."""
    d = pd.read_csv(M / "test_error_by_horizon.csv")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for col, color, ls, lab, dy in [("Linear baseline", MID, (0, (4, 3)), "Linear baseline", 0.035),
                                     (ENS, RED, "-", "Final model (GB + Ridge)", -0.06)]:
        ax.plot(x, d[col], color=color, ls=ls, lw=2.6, marker="o", label=lab)
        for xi, v in zip(x, d[col]):
            ax.text(xi, v + dy, f"{v:.2f}", ha="center", fontsize=12, color=color,
                    fontweight="bold" if col == ENS else "normal")
    ax.set_xticks(x, ["1-5", "6-10", "11-15", "16-20", "21+"])
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("Laps ahead of the decision point")
    ax.set_ylabel("2025 test MAE (s per lap)")
    ax.legend(loc="lower right")
    save(fig, "s_horizon.png")


def importance():
    d = pd.read_csv(M / "final_importance_test.csv")
    nice = {"early_deg_mean": "Mean loss, first 5 laps", "laps_ahead": "Laps ahead of window",
            "pace_vs_field": "Car pace vs. field", "early_deg_slope": "Wear slope, first 5 laps",
            "gap_ahead_s": "Gap to car ahead", "position": "Position", "window_gap_mean": "Mean gap, first 5 laps",
            "circuit": "Circuit", "prac_slope_field": "Practice long-run wear", "compound": "Compound"}
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    y = np.arange(len(d))
    colors = [RED if i >= len(d) - 3 else MID for i in range(len(d))]
    ax.barh(y, d["mean"], color=colors, height=0.65)
    ax.set_yticks(y, [nice.get(f, f) for f in d["feature"]])
    ax.set_xlabel("Increase in 2025 MAE when shuffled (s)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    save(fig, "s_importance.png")


def update():
    d = pd.read_csv(M / "update_by_decision.csv")
    d = d[d["data"] == "test_2025"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, d["frozen_mae"], color=MID, ls=(0, (4, 3)), lw=2.2, marker="o", label="Forecast fixed after lap 5")
    ax.plot(x, d["updated_mae"], color=RED, lw=2.8, marker="o", label="Updated after every lap")
    for xi, (a, b) in enumerate(zip(d["frozen_mae"], d["updated_mae"])):
        if xi:
            ax.text(xi, a + 0.025, f"{a:.2f}", ha="center", fontsize=12, color=MID)
        ax.text(xi, b - 0.055, f"{b:.2f}", ha="center", fontsize=12, color=RED, fontweight="bold")
    ax.set_xticks(x, ["5", "6-9", "10-14", "15-19", "20+"])
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Laps already seen on the current set")
    ax.set_ylabel("2025 MAE on remaining laps (s)")
    ax.legend(loc="lower left")
    save(fig, "s_update.png")


def pit_backtest():
    """Two questions: when is it cheapest to stop (tire model) vs. when do teams stop."""
    s = pd.read_csv(M / "strategy_summary.csv").set_index("method")
    rows = [("Tire model only", s.iloc[0]["within_3_laps_pct"], DARK),
            ("Team-stop model:\nrace context", s.loc["Imitation only (race context, no tire model)", "within_3_laps_pct"], MID),
            ("Team-stop model:\nrace context + tire model", s.loc["Learned layer (tire model + race context)", "within_3_laps_pct"], MID),
            ("Middle of the stint\n(reference)", s.loc["Midpoint (no model)", "within_3_laps_pct"], LIGHT)][::-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for yi, (_, v, c) in zip(y, rows):
        ax.text(v - 1.0, yi, f"{v:.0f}%", va="center", ha="right", fontsize=14, fontweight="bold",
                color="#1A1A1A" if c == LIGHT else "white")
    ax.axvline(60, color="#1A1A1A", ls=(0, (2, 2)), lw=1.2)
    ax.text(60.8, len(rows) - 0.5, "Goal 60%", fontsize=12, va="center")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 72)
    ax.set_xlabel(f"Within 3 laps of the team's stop (% of {int(s['stints'].iloc[0])}, 2025)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    save(fig, "s_pit_backtest.png")


if __name__ == "__main__":
    stint_example()
    degradation()
    train_vs_test()
    horizon()
    importance()
    update()
    pit_backtest()
