# PitWall: Predicting F1 Tire Degradation to Recommend Pit Stop Timing

EM 538 capstone, Team 9 (Isaiah Nick, Anand Vadlamani).

We predict how much slower each lap gets as a tire wears (lap time loss in seconds,
fuel-corrected, relative to the start of the stint), then use that to recommend a pit
window. Data: every dry 2022 to 2025 Grand Prix plus practice sessions, from FastF1.

## Quick start

```bash
conda env create -f environment.yml      # or: pip install -e ".[dev]"
conda activate pitwall
make all                                 # ingest -> clean -> features -> train -> tests
```

Each stage can also be run alone: `python -m pitwall ingest|clean|features|train`.
All settings live in `configs/default.yaml`.

## Final stages (after the midpoint)

```bash
pip install -e ".[dl,app]"     # adds torch, streamlit, plotly (or: conda env update -f environment.yml)
make final                     # tune -> lstm -> evaluate (2025 test) -> update -> strategy backtest
python scripts/make_figures.py --final
make app                       # Streamlit dashboard at http://localhost:8501
```

| Stage | Module | What it does | Output |
|---|---|---|---|
| tune | `pitwall/tune.py` | Grid search inside the same grouped folds; weather-feature test | `tuning_results.csv`, `tuned_params.json`, `weather_ablation.csv` |
| lstm | `pitwall/lstm.py` | PyTorch LSTM over the first 5 laps of a stint (Setup B task), size sweep + 3 seeds | `lstm_sweep.csv`, `lstm_summary.json` |
| evaluate | `pitwall/test.py` | Refits tuned models (and the GB + Ridge ensemble) on 2022-24, scores 2025, saves final models to `models/` | `test_results.csv`, `test_error_by_*.csv` |
| update | `pitwall/update.py` | Online error correction: shifts the Setup B forecast by its mean error on the last k laps seen (k by CV) | `update_*.csv`, `update_summary.json` |
| strategy | `pitwall/strategy.py` | Pit window recommender; decision options chosen on 2024; learned strategy layer (with a race-context-only ablation); backtests 2025 green flag stops | `strategy_*.csv`, `strategy_layer_importance.csv` |

Final feature set: weather (track/air temperature, humidity) is dropped (`features.drop_weather` in the
config) because removing it lowered CV error for 5 of 8 tuned models, including the best one (the
other 3 changed by under 0.4%).

**After the first final run** we tested improvements without touching 2025 (`scripts/experiments/`,
results in `reports/experiments/`) and kept: 2022 as extra training data, practice long-run wear,
dirty air and car pace features, a GB + Ridge ensemble, and the online forecast update. 2025 was
then scored a second time; the first final run is on the `final-presentation` history (commit a081123).

**Pit window:** at the decision point (5 laps past the stint's reference laps), for every candidate
pit lap p, cost(p) = predicted loss on the current set through p (Setup B model) + predicted loss on
the next set after p (Setup A model, the compound actually fitted next) + a compound pace offset.
Pit lane time is the same for every p, so it does not move the answer. The window is every lap
within 1 s of the best.

**Dashboard** (`app/`, the lecture's three-file layout): `model.py` loads the saved models and
tables, `predict.py` turns a selection into curves and a recommendation, `app.py` is the UI. Tabs:
2025 replay with pit window, plan-a-stint (Setup A what-if), and model results.

## Live demo checklist

1. `make app` a few minutes early (the first load reads the data and models, about 30 s).
2. Data ingestion: `python -m pitwall ingest` pulls all 92 races and the practice sessions from the
   pinned archive (about 15 s).
3. Dashboard: 2025 Spanish GP, Verstappen, stint 2 (recommended lap 29, real stop lap 29, typical team
   call lap 30). The blue box explains what the model used and how long inference took. Drag the
   decision-lap slider to re-check the forecast; switch the model to compare calls. Backups: Austria,
   Hamilton, stint 1 (25 vs 26); Saudi Arabia, Russell, stint 1 (19 vs 20).

## Data source

The pipeline has two loaders that return the same canonical table
(`src/pitwall/ingest/schema.py`):

| `ingest.source` | What it does | Needs |
|---|---|---|
| `archive` (default) | Sparse-clones the [TracingInsights](https://github.com/TracingInsights) season repos at pinned commits. Those repos store FastF1's `session.laps` with weather joined, for every session. | git |
| `fastf1` | Calls `fastf1.get_session(year, round, "R").load()` directly and caches to `data/external/fastf1_cache`. | access to livetiming.formula1.com |

`make ingest SOURCE=fastf1` switches to the API. The archive is the default because it
is pinned (every teammate gets byte-identical inputs) and does not depend on the F1
timing servers being up. The canonical raw parquet (`data/raw/`) and the model table
(`data/processed/features.parquet`) are committed so nobody has to re-download.

## Pipeline

| Stage | Module | Output |
|---|---|---|
| ingest | `pitwall/ingest` | `data/raw/{year}_{round}.parquet`, `manifest.csv` |
| clean | `pitwall/clean.py` | `data/interim/laps_clean.parquet`, `cleaning_funnel.csv`, `fuel_check.json` |
| features | `pitwall/features.py` | `data/processed/features.parquet`, `dataset_summary.json` |
| train | `pitwall/train.py`, `models.py` | `reports/metrics/cv_*.csv`, OOF predictions, `reports/figures/` |

**Cleaning** (in order, each step logged): drop wet races (>10% of laps on inter/wet),
excluded events (Monaco), missing lap/tire fields, non-slick laps, lap 1, pit in/out
laps, SC/VSC/red flag laps, laps with rain, FastF1 inaccurate/deleted/generated laps,
laps slower than 107% of the race's fastest (fuel-corrected), and within-stint
outliers (more than 1 s and 5 robust sigma off a rolling median).

**Fuel correction:** 110 kg at the start, burned linearly to zero, 0.03 s per kg. The
clean stage re-estimates this from the 2022 to 2024 data (`fuel_check.json`) as a check.

**Target `deg_s`:** fuel-corrected lap time minus the median of the stint's first 3
clean laps. Those 3 reference laps are not target rows.

**Splits:** train = 2022 to 2024, test = 2025 (locked, not read by `train`). Train rows
carry a `cv_fold` (GroupKFold by race, 5 folds) so every model, including the LSTM,
uses identical folds.

**Feature setups:**
- **A, conditions:** compound, stint lap, tire life, tire life at fitting, stint
  number, fresh set, track/air temp, humidity, laps remaining, race progress, gap to
  car ahead, position, circuit, team.
- **B, history:** A plus the mean loss and wear slope over the first 5 target laps
  of the stint, and laps since that window. Forecast rows are the laps after the window.

## For teammates

```python
from pitwall.data import load_features, stint_sequences
train = load_features("train")          # never open "test" until models are frozen
seqs = stint_sequences(train)           # per-stint arrays + cv_fold for the LSTM
```

Score the LSTM with the same `cv_fold` values and report MAE/RMSE/R2 on
`in_forecast` rows so it is directly comparable to Setup B.

## Layout

```
configs/default.yaml     every tunable setting
src/pitwall/             package (ingest, clean, features, models, train, plots, data)
tests/                   unit tests (pytest)
data/raw, data/processed committed parquet; data/external + data/interim are rebuilt
reports/metrics          CV, tuning, LSTM, 2025 test and backtest results
reports/figures          midpoint (report/) and final (final/) figures
app/                     Streamlit dashboard
models/                  fitted final models (rebuilt by `make final`, not committed)
notebooks/               EDA and LSTM notebooks (import from pitwall, don't copy code)
```
