"""Loads everything the dashboard needs, once (cached by Streamlit)."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from pitwall.config import load_config
from pitwall.strategy import stint_table
from pitwall.ingest import load_raw


@st.cache_resource
def load_all():
    cfg = load_config()
    m = cfg["paths"]["metrics"]
    A = joblib.load(cfg["paths"]["models"] / "final_A.joblib")
    B = joblib.load(cfg["paths"]["models"] / "final_B.joblib")
    feat = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    stints = stint_table(load_raw(cfg, cfg["test_seasons"]))
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
        "pit_loss": pd.read_csv(m / "pit_loss_by_circuit.csv"),
        "offsets": pd.read_csv(m / "compound_offsets.csv"),
        "test_results": pd.read_csv(m / "test_results.csv"),
        "strategy_summary": pd.read_csv(m / "strategy_summary.csv"),
    }
