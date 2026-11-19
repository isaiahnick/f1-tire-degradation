PY ?= python

.PHONY: setup data ingest clean features train test all figures tune lstm evaluate update strategy final app

setup:            ## install the package in editable mode
	$(PY) -m pip install -e ".[dev]"

ingest:           ## pull 2022-2025 races and practice sessions (archive by default; SOURCE=fastf1 to use the API)
	$(PY) -m pitwall ingest $(if $(SOURCE),--source $(SOURCE),)

clean:
	$(PY) -m pitwall clean

features:
	$(PY) -m pitwall features

data: ingest clean features

train:            ## grouped CV on 2022-2024 (2025 test season stays locked)
	$(PY) -m pitwall train

figures:
	$(PY) -c "from pitwall.config import load_config; from pitwall import plots; plots.make_all(load_config())"

test:
	$(PY) -m pytest -q

all: data train test

# ---- final stages (after the midpoint) ----
tune:             ## grid search inside the grouped folds + weather feature test
	$(PY) -m pitwall tune

lstm:             ## LSTM (size sweep if lstm.run_sweep) and 3 seeds, same folds as Setup B (needs torch)
	$(PY) -m pitwall lstm

evaluate:         ## one-time 2025 test evaluation, saves final models to models/
	$(PY) -m pitwall test

update:           ## online error correction of the forecast as laps are seen (CV, then 2025)
	$(PY) -m pitwall update

strategy:         ## pit window options (chosen on 2024), learned layer, 2025 backtest
	$(PY) -m pitwall strategy

final: tune lstm evaluate update strategy

app:              ## launch the dashboard
	streamlit run app/app.py
