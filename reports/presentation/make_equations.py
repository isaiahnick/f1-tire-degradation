"""Render the slide equations with matplotlib mathtext (transparent PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})
EQ = {
    "fuel": r"$t_{fc} = t - 0.03\,m_{fuel}, \qquad m_{fuel} = 110\,\left(1 - \frac{\mathrm{lap} - 0.5}{N_{laps}}\right)$",
    "target": r"$y_{lap} = t_{fc,\,lap} - \mathrm{median}\left(t_{fc,1},\, t_{fc,2},\, t_{fc,3}\right)$",
    "mae": r"$\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i - \hat{y}_i\right|$",
    "rmse": r"$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2}$",
    "r2": r"$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$",
    "cost": r"$C(p) = \sum_{l=d+1}^{p}\hat{y}_B(l) \;+\; \sum_{j=1}^{E-p}\left[\hat{y}_A(j) + \delta\right], \qquad p^{*} = \arg\min_p C(p)$",
    "update": r"$\hat{y}^{\,\prime}(l) = \hat{y}(l) + \frac{1}{3}\sum_{i=d-2}^{d}\left(y_i - \hat{y}_i\right)$",
}
for name, tex in EQ.items():
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, tex, fontsize=22)
    fig.savefig(f"eq/{name}.png", dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(name)
