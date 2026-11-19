"""Loads everything the dashboard needs, once (cached by Streamlit)."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from pitwall.config import load_config
from pitwall.strategy import gap_behind, practice_lookup, stint_table
from pitwall.ingest import load_raw


@st.cache_resource
def load_all():
    cfg = load_config()
    m = cfg["paths"]["metrics"]
    A = joblib.load(cfg["paths"]["models"] / "final_A.joblib")
    B = joblib.load(cfg["paths"]["models"] / "final_B.joblib")
    feat = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    raw = load_raw(cfg, cfg["test_seasons"])
    stints = stint_table(raw)
    layer_path = cfg["paths"]["models"] / "strategy_layer.joblib"
    tuned = json.loads((m / "tuned_params.json").read_text())
    lstm_fn = None
    try:
        from pitwall.strategy import lstm_predictor
        lstm_fn = lstm_predictor(cfg, A["cfg"], feat)
    except Exception:          # torch not installed: dashboard still works without the LSTM
        pass
    return {
        "cfg": cfg, "fcfg": A["cfg"], "A": A["models"], "B": B["models"], "feat": feat,
        "stints": stints, "tuned": tuned, "lstm_fn": lstm_fn,
        "options": json.loads((m / "strategy_options.json").read_text()),
        "prac": practice_lookup(cfg), "gaps": gap_behind(raw),
        "layer": joblib.load(layer_path) if layer_path.exists() else None,
        "update_by_decision": pd.read_csv(m / "update_by_decision.csv"),
        "update_summary": json.loads((m / "update_summary.json").read_text()),
        "pit_loss": pd.read_csv(m / "pit_loss_by_circuit.csv"),
        "offsets": pd.read_csv(m / "compound_offsets.csv"),
        "test_results": pd.read_csv(m / "test_results.csv"),
        "strategy_summary": pd.read_csv(m / "strategy_summary.csv"),
        "update_by_horizon": pd.read_csv(m / "update_by_horizon.csv"),
        "n_team_stops": len(pd.read_csv(m / "strategy_oof_train.csv")),   # stops the team-stop model learned from
        "error_by_horizon": pd.read_csv(m / "test_error_by_horizon.csv"),
        "error_by_compound": pd.read_csv(m / "test_error_by_compound.csv"),
        "error_by_event": pd.read_csv(m / "test_error_by_event.csv"),
    }
