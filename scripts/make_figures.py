"""Report figures for the Team 9 midpoint report.

Reads only files the pipeline already wrote (run `make all` first), so it is fast
and can be rerun any time:

    python scripts/make_figures.py

Midpoint figures (needs `make all`) go to reports/figures/report/; final figures
(needs `make final`, run with --final) go to reports/figures/final/.

Midpoint PNGs (300 dpi), numbered like the midpoint report:
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


# ================================================================ final figures
FINAL = ROOT / "reports" / "figures" / "final"
YELLOW, MAGENTA = "#eda100", "#e87ba4"
MODEL_COLOR = {"Linear baseline": GREY, "Ridge": ORANGE, "Random forest": BLUE,
               "Gradient boosting": GREEN, "LSTM": MAGENTA, "Decision tree": YELLOW}


def save_final(fig, name):
    FINAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL / name)
    plt.close(fig)
    print(f"wrote {FINAL / name}")


def final_results():
    """2025 test MAE (filled) vs training MAE (hollow) for every model, both setups."""
    r = pd.read_csv(METRICS / "test_results.csv")
    lstm = r[r["model"].str.startswith("LSTM (")]
    r = r[~r["model"].str.startswith("LSTM (") & ~r["model"].isin(["Rule: extrapolate", "Mean (dummy)"])]
    r = r.assign(lo=np.nan, hi=np.nan)
    if len(lstm):
        r = pd.concat([r, pd.DataFrame([{"setup": "B", "model": "LSTM", "train_mae": lstm["train_mae"].mean(),
                                         "test_mae": lstm["test_mae"].mean(),
                                         "lo": lstm["test_mae"].min(), "hi": lstm["test_mae"].max()}])])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True, gridspec_kw={"wspace": 0.62})
    for ax, setup, title in [(axes[0], "A", "Setup A: race conditions only"),
                             (axes[1], "B", "Setup B: + first 5 laps of the stint")]:
        d = r[r["setup"] == setup].sort_values("test_mae", ascending=False).reset_index(drop=True)
        y = np.arange(len(d))
        base = d.loc[d["model"] == "Linear baseline", "test_mae"].iloc[0]
        ax.axvline(base, color=GREY, linestyle=(0, (4, 2)), linewidth=1.2, zorder=1)
        ax.hlines(y, d["train_mae"], d["test_mae"], color=GRID, linewidth=2.5, zorder=1)
        ax.scatter(d["train_mae"], y, s=46, facecolor="white", edgecolor=INK2, linewidth=1.3, zorder=3)
        colors = [MODEL_COLOR.get(m, GREY) for m in d["model"]]
        ax.scatter(d["test_mae"], y, s=70, color=colors, zorder=4)
        for k in d.index[d["lo"].notna()]:
            ax.hlines(y[k], d.loc[k, "lo"], d.loc[k, "hi"], color=MAGENTA, linewidth=4, alpha=0.4, zorder=2)
        for yi, (m, v) in enumerate(zip(d["model"], d["test_mae"])):
            pct = 100 * (base - v) / base
            right = max(v, d.loc[yi, "train_mae"], d.loc[yi, "hi"] if pd.notna(d.loc[yi, "hi"]) else 0)
            lab = "baseline" if m == "Linear baseline" else f"{pct:+.0f}%"
            ax.text(right + 0.022, yi, lab, va="center", fontsize=8.5,
                    color=INK if (pct > 0 and m != "Linear baseline") else MUTED,
                    fontweight="bold" if pct > 15 else "normal")
        ax.set_yticks(y, [m.replace("Rule: flat", "Carry early loss forward") for m in d["model"]])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        ax.set_title(title)
        ax.set_xlabel("MAE of lap time loss (s)")
    axes[0].set_xlim(0.25, 0.80)
    fig.text(0.125, -0.07, "Filled dot: 2025 test MAE (unseen season).  Hollow dot: training MAE.  "
             "Dashed line: linear baseline.  LSTM: mean of 3 seeds, shaded bar shows the range.",
             fontsize=8, color=MUTED)
    save_final(fig, "final_results.png")


def final_horizon():
    d = pd.read_csv(METRICS / "test_error_by_horizon.csv")
    labels = ["1-5", "6-10", "11-15", "16-20", "21+"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ends = {}
    for m in ["Linear baseline", "LSTM", "Gradient boosting", "Random forest", "Ridge"]:
        ls = (0, (4, 2)) if m == "Linear baseline" else "-"
        ax.plot(x, d[m], color=MODEL_COLOR[m], linewidth=2.2, linestyle=ls, marker="o", ms=4.5,
                mec="white", mew=0.8)
        ends[m] = d[m].iloc[-1]
    # spread end labels so they never overlap
    order = sorted(ends, key=ends.get)
    ypos, last = {}, -1
    for m in order:
        yv = max(ends[m], last + 0.035)
        ypos[m], last = yv, yv
    for m in order:
        ax.text(x[-1] + 0.12, ypos[m], m, color=MODEL_COLOR[m] if m != "Linear baseline" else INK2,
                fontsize=8.5, fontweight="bold", va="center")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.2, x[-1] + 1.5)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Laps ahead of the decision point")
    ax.set_ylabel("2025 test MAE (s)")
    save_final(fig, "final_horizon.png")


def final_importance(n_rows=4000):
    """Permutation importance of the final Setup B gradient boosting model on 2025 laps."""
    import joblib
    from sklearn.inspection import permutation_importance

    from pitwall.models import feature_lists
    obj = joblib.load(ROOT / "models" / "final_B.joblib")
    cfg, model = obj["cfg"], obj["models"]["Gradient boosting"]
    num, cat, inter = feature_lists(cfg, "B")
    df = pd.read_parquet(FEATURES)
    te = df[(df["split"] == "test") & df["in_forecast"]].sample(n_rows, random_state=42)
    feats = num + cat + inter
    r = permutation_importance(model, te[feats], te["deg_s"], scoring="neg_mean_absolute_error",
                               n_repeats=5, random_state=42, n_jobs=-1)
    d = pd.DataFrame({"feature": feats, "mean": r.importances_mean, "std": r.importances_std})
    d = d[~d["feature"].isin(inter)].nlargest(8, "mean").iloc[::-1]
    d.to_csv(METRICS / "final_importance_test.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    y = np.arange(len(d))
    ax.barh(y, d["mean"], color=BLUE, height=0.62, zorder=2)
    ax.errorbar(d["mean"], y, xerr=d["std"], fmt="none", ecolor=INK2, elinewidth=0.8, zorder=3)
    ax.set_yticks(y, [NICE.get(f, f) for f in d["feature"]])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Increase in 2025 test MAE when shuffled (s)")
    save_final(fig, "final_importance.png")


def final_pit_backtest():
    s = pd.read_csv(METRICS / "strategy_summary.csv")
    s["label"] = s["method"].str.replace(" (no model)", "", regex=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(s))[::-1].astype(float)
    h = 0.36
    ax.barh(y + h / 2, s["within_3_laps_pct"], height=h, color=BLUE, zorder=2,
            label="Recommended lap within 3 laps of the real stop")
    ax.barh(y - h / 2, s["actual_in_window_pct"].fillna(0), height=h, color=ORANGE, zorder=2,
            label="Real stop inside the 1-second pit window")
    for yi, a, b in zip(y, s["within_3_laps_pct"], s["actual_in_window_pct"]):
        ax.text(a + 1, yi + h / 2, f"{a:.0f}%", va="center", fontsize=8.5)
        ax.text((b if pd.notna(b) else 0) + 1, yi - h / 2, f"{b:.0f}%" if pd.notna(b) else "n/a",
                va="center", fontsize=8.5, color=INK if pd.notna(b) else MUTED)
    ax.set_yticks(y, s["label"])
    ax.set_xlim(0, 75)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Share of {int(s['stints'].iloc[0])} green flag stops in 2025 (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=8)
    save_final(fig, "final_pit_backtest.png")


if __name__ == "__main__":
    import sys
    if "--final" in sys.argv:
        sys.path.insert(0, str(ROOT / "src"))
        final_results()
        final_horizon()
        final_importance()
        final_pit_backtest()
    else:
        fig1_degradation()
        fig2_importance()
        fig3_depth_curve()
        fig4_sample_stints()
