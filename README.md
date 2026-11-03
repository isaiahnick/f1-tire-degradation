# PitWall: Predicting F1 Tire Degradation to Recommend Pit Stop Timing

EM 538 capstone, Team 9 (Isaiah Nick, Anand Vadlamani).

We predict how much slower each lap gets as a tire wears (lap time loss in seconds,
fuel-corrected, relative to the start of the stint), then use that to recommend a pit
window. Data: every dry 2023 to 2025 Grand Prix, from FastF1.

## Quick start

```bash
conda env create -f environment.yml      # or: pip install -e ".[dev]"
conda activate pitwall
make all                                 # ingest -> clean -> features -> train -> tests
```

Each stage can also be run alone: `python -m pitwall ingest|clean|features|train`.
All settings live in `configs/default.yaml`.

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
clean stage re-estimates this from the 2023 to 2024 data (`fuel_check.json`) as a check.

**Target `deg_s`:** fuel-corrected lap time minus the median of the stint's first 3
clean laps. Those 3 reference laps are not target rows.

**Splits:** train = 2023 and 2024, test = 2025 (locked, not read by `train`). Train rows
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
reports/metrics          CV results, cleaning funnel, data checks
reports/figures          report figures
notebooks/               EDA and LSTM notebooks (import from pitwall, don't copy code)
```
