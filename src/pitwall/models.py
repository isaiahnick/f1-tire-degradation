"""Model definitions. Every model is a scikit-learn Pipeline, so scaling and
encoding are fit inside each CV fold (no leakage from validation races).

Two feature setups:
  A "conditions": what you know before the stint starts (compound, tire age,
     track, weather, fuel, traffic). This is the proposal's framing.
  B "history":    A plus what the pit wall has seen on this set so far (mean loss
     and wear slope over the first laps). This matches the real pit decision and
     is the same information the LSTM gets.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASELINE_INTERACTIONS = ["stint_lap_x_soft", "stint_lap_x_medium", "stint_lap_x_hard"]
HISTORY_INTERACTIONS = ["early_slope_x_ahead"]


class EarlyStintRule(BaseEstimator, RegressorMixin):
    """Rule-of-thumb forecasts from the observed window (no fitting).

    mode="flat":        future loss = mean loss seen so far
    mode="extrapolate": continue the window's straight line
    """

    def __init__(self, mode: str = "flat", history_laps: int = 5):
        self.mode = mode
        self.history_laps = history_laps

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        mean = X["early_deg_mean"].to_numpy(float)
        if self.mode == "flat":
            return mean
        # window mean sits at the window centre, (n-1)/2 laps before its last lap
        dist = X["laps_ahead"].to_numpy(float) + (self.history_laps - 1) / 2
        return mean + X["early_deg_slope"].to_numpy(float) * dist


def _onehot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def feature_lists(cfg: dict, setup: str = "A") -> tuple[list[str], list[str], list[str]]:
    """(numeric for trees, categorical, extra linear-only interaction columns)."""
    num = list(cfg["features"]["numeric"])
    cat = list(cfg["features"]["categorical"])
    inter = BASELINE_INTERACTIONS + ["stint_lap_x_track_temp"]
    if setup == "B":
        num += list(cfg["features"]["history"])
        inter += HISTORY_INTERACTIONS
    return num, cat, inter


def build_models(cfg: dict, setup: str = "A") -> dict[str, Pipeline]:
    num, cat, inter = feature_lists(cfg, setup)
    seed = cfg["seed"]

    baseline_prep = ColumnTransformer([
        ("compound", _onehot(), ["compound"]),
        ("wear", "passthrough", BASELINE_INTERACTIONS),
    ])
    linear_prep = ColumnTransformer([
        ("num", StandardScaler(), num + inter),
        ("cat", _onehot(), cat),
    ])
    tree_prep = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", _onehot(), cat),
    ])

    models = {
        "Mean (dummy)": Pipeline([("prep", baseline_prep), ("model", DummyRegressor())]),
        "Linear baseline": Pipeline([("prep", baseline_prep), ("model", LinearRegression())]),
    }
    if setup == "B":
        h = cfg["target"]["history_laps"]
        models["Rule: flat"] = EarlyStintRule("flat", h)
        models["Rule: extrapolate"] = EarlyStintRule("extrapolate", h)
    models.update({
        "Ridge": Pipeline([("prep", linear_prep), ("model", Ridge(alpha=1.0))]),
        "Decision tree": Pipeline([("prep", tree_prep), ("model", DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=50, random_state=seed))]),
        "Random forest": Pipeline([("prep", tree_prep), ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=20, max_features=0.33, n_jobs=-1,
            random_state=seed))]),
        "Gradient boosting": Pipeline([("prep", tree_prep), ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
            early_stopping=False, random_state=seed))]),
    })
    return models


MODEL_SETTINGS = {
    "Mean (dummy)": "predicts the training mean",
    "Linear baseline": "loss = wear rate x stint lap, one rate per compound",
    "Rule: flat": "future loss = mean loss over first 5 laps",
    "Rule: extrapolate": "extend the first-5-lap straight line",
    "Ridge": "all features + interactions, alpha=1.0",
    "Decision tree": "max_depth=8, min_samples_leaf=50",
    "Random forest": "300 trees, min_samples_leaf=20, max_features=0.33",
    "Gradient boosting": "HistGB, lr=0.05, 400 iters, 31 leaves",
}
