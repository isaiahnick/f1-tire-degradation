PY ?= python

.PHONY: setup data ingest clean features train test all figures

setup:            ## install the package in editable mode
	$(PY) -m pip install -e ".[dev]"

ingest:           ## pull 2023-2025 races (archive by default; SOURCE=fastf1 to use the API)
	$(PY) -m pitwall ingest $(if $(SOURCE),--source $(SOURCE),)

clean:
	$(PY) -m pitwall clean

features:
	$(PY) -m pitwall features

data: ingest clean features

train:            ## grouped CV on 2023-2024 (2025 test season stays locked)
	$(PY) -m pitwall train

figures:
	$(PY) -c "from pitwall.config import load_config; from pitwall import plots; plots.make_all(load_config())"

test:
	$(PY) -m pytest -q

all: data train test
