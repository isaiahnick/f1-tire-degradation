"""Report figures for the Team 9 midpoint report.

Reads only files the pipeline already wrote (run `make all` first), so it is fast
and can be rerun any time:

    python scripts/make_figures.py

Writes four PNGs (300 dpi) to reports/figures/report/, numbered like the report:
    fig1_degradation.png      Figure 1  lap time loss by laps on the set, per compound
    fig2_importance.png       Figure 2  permutation importance, Setup A vs Setup B
    fig3_depth_curve.png      Figure 3  decision tree train vs validation MAE by depth
    fig4_sample_stints.png    Figure 4  sample predictions vs actual, one stint per compound
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
FEATURES = ROOT / "data" / "processed" / "features.parquet"
OUT = ROOT / "reports" / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, BG = "#1f1f1f", "#55534e", "#8a8882", "#e6e4de", "white"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b7b0"
COMPOUNDS = {"SOFT": ("Soft", BLUE), "MEDIUM": ("Medium", ORANGE), "HARD": ("Hard", GREEN)}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 8, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

NICE = {
    "stint_lap": "Laps on current set", "tyre_life": "Total tire age",
    "tyre_life_start": "Tire age when fitted", "stint": "Stint number",
    "fresh_tyre": "New set", "track_temp": "Track temperature", "air_temp": "Air temperature",
    "humidity": "Humidity", "laps_remaining": "Laps remaining", "race_progress": "Race progress",
    "gap_ahead_s": "Gap to car ahead", "position": "Position", "compound": "Compound",
    "circuit": "Circuit", "team": "Team", "early_deg_mean": "Early-stint average loss",
    "early_deg_slope": "Early-stint wear trend", "laps_ahead": "Laps past early window",
}


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {OUT / name}")


# ---------------------------------------------------------------- figure 1
def fig1_degradation():
    df = pd.read_parquet(FEATURES)
    df = df[df["split"] == "train"]
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    for comp, (label, color) in COMPOUNDS.items():
        g = df[df["compound"] == comp].groupby("stint_lap")["deg_s"]
        s = pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.size()), "n": g.size()})
        s = s[s["n"] >= 40]
        lo, hi = s["mean"] - 1.96 * s["se"], s["mean"] + 1.96 * s["se"]
        ax.fill_between(s.index, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.plot(s.index, s["mean"], color=color, linewidth=2, solid_capstyle="round")
        ax.text(s.index[-1] + 0.6, s["mean"].iloc[-1], label, color=color, fontsize=9,
                fontweight="bold", va="center")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(3, 47)
    ax.set_xlabel("Laps on the current tire set")
    ax.set_ylabel("Lap time loss vs. stint start (s)")
    ax.text(0, 1.02, "Mean with 95% confidence band, 2023 and 2024 races, fuel corrected",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "fig1_degradation.png")


# ---------------------------------------------------------------- figure 2
def fig2_importance(top=8):
    imp = pd.read_csv(METRICS / "permutation_importance.csv")
    panels = [("A", "Random forest", "Setup A: race conditions only"),
              ("B", "Gradient boosting", "Setup B: conditions + first 5 laps of stint")]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={"wspace": 0.95})
    for ax, (setup, model, title) in zip(axes, panels):
        d = imp[(imp["setup"] == setup) & (imp["model"] == model)]
        # top features by size, keep any negative ones from that set visible
        d = d.reindex(d["mean"].abs().sort_values(ascending=False).index).head(top)
        d = d.sort_values("mean")
        y = np.arange(len(d))
        colors = [BLUE if v > 0 else GREY for v in d["mean"]]
        ax.barh(y, d["mean"], color=colors, height=0.62, zorder=2)
        ax.errorbar(d["mean"], y, xerr=d["std"], fmt="none", ecolor=INK2, elinewidth=0.8,
                    capsize=0, zorder=3)
        ax.axvline(0, color=INK2, linewidth=0.8, zorder=1)
        ax.set_yticks(y, [NICE.get(f, f) for f in d["feature"]])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        ax.set_title(title)
        ax.set_xlabel("Increase in validation MAE (s)")
    axes[0].text(0, -0.27, "Grey bars: shuffling the feature made validation error slightly better.",
                 transform=axes[0].transAxes, fontsize=8, color=MUTED)
    save(fig, "fig2_importance.png")


# ---------------------------------------------------------------- figure 3
def fig3_depth_curve():
    d = pd.read_csv(METRICS / "depth_curve.csv")
    labels = d["max_depth"].astype(str).str.replace("none", "None")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    best = int(d["val_mae"].idxmin())
    ax.axvspan(best + 0.5, x[-1] + 0.4, color="#f4f3ef", zorder=0, linewidth=0)
    ax.text(best + 0.65, 0.89, "deeper trees overfit", ha="left", fontsize=8, color=MUTED, style="italic")
    ax.plot(x, d["train_mae"], color=BLUE, linewidth=2, marker="o", ms=4.5, mec="white", mew=0.8)
    ax.plot(x, d["val_mae"], color=ORANGE, linewidth=2, marker="o", ms=4.5, mec="white", mew=0.8)
    ax.text(x[-1] + 0.25, d["train_mae"].iloc[-1], "Training", color=BLUE, fontweight="bold",
            va="center", fontsize=9)
    ax.text(x[-1] + 0.25, d["val_mae"].iloc[-1], "Validation", color=ORANGE, fontweight="bold",
            va="center", fontsize=9)
    ax.scatter([x[best]], [d["val_mae"][best]], s=70, facecolor="none", edgecolor=INK, linewidth=1.1, zorder=4)
    ax.annotate(f"best validation: depth {labels[best]} ({d['val_mae'][best]:.3f} s)",
                xy=(x[best], d["val_mae"][best]), xytext=(x[best] + 0.65, 0.74),
                fontsize=8, color=INK2, ha="left",
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.4, x[-1] + 0.4)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("Decision tree max depth")
    ax.set_ylabel("MAE (s)")
    save(fig, "fig3_depth_curve.png")


# ---------------------------------------------------------------- figure 4
def fig4_sample_stints():
    oof = pd.read_parquet(METRICS / "oof_B.parquet")
    oof["abs_err"] = (oof["deg_s"] - oof["Random forest"]).abs()
    stints = oof.groupby("stint_id").agg(n=("deg_s", "size"), comp=("compound", "first"),
                                         err=("abs_err", "mean"))
    picks = []
    for comp in COMPOUNDS:  # the long stint with the median RF error: a typical case
        c = stints[(stints["comp"] == comp) & (stints["n"] >= 12)].sort_values("err")
        picks.append(c.index[len(c) // 2])

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, sid in zip(axes, picks):
        g = oof[oof["stint_id"] == sid].sort_values("stint_lap")
        r = g.iloc[0]
        label = COMPOUNDS[r["compound"]][0]
        ax.scatter(g["stint_lap"], g["deg_s"], s=14, color=INK2, zorder=3, label="Actual")
        ax.plot(g["stint_lap"], g["Linear baseline"], color=GREY, linewidth=1.8,
                linestyle=(0, (4, 2)), label="Linear baseline")
        ax.plot(g["stint_lap"], g["Random forest"], color=BLUE, linewidth=2, label="Random forest")
        race = r["event_name"].replace(" Grand Prix", "")
        ax.set_title(f"{race} {sid[:4]}, {r['driver']}", fontsize=9)
        ax.set_xlabel(f"Laps on set ({label.lower()})")
    axes[0].set_ylabel("Lap time loss (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09), fontsize=8.5)
    save(fig, "fig4_sample_stints.png")


if __name__ == "__main__":
    fig1_degradation()
    fig2_importance()
    fig3_depth_curve()
    fig4_sample_stints()