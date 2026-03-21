PYTHON := python3
SRC := src

.PHONY: help fetch-data validate-data build-features run-baseline run-clustering run-regimes run-event-study run-migration generate-figures generate-report run-all test lint clean

help:
	@echo "Available targets:"
	@echo "  fetch-data        Download raw data from FMP API"
	@echo "  validate-data     Check manifests and checksums"
	@echo "  build-features    Compute returns and similarity matrices"
	@echo "  run-baseline      Static correlation clustering"
	@echo "  run-clustering    Full multi-layer dynamic clustering"
	@echo "  run-regimes       Regime detection (HMM + change-point)"
	@echo "  run-event-study   Iran shock event study"
	@echo "  run-migration     Migration metrics (CMI, TDS, bridges)"
	@echo "  generate-figures  All visualizations"
	@echo "  generate-report   Compile research report"
	@echo "  run-all           Full pipeline end-to-end"
	@echo "  test              Run test suite"
	@echo "  lint              Ruff + mypy"
	@echo "  clean             Remove generated outputs"

fetch-data:
	 -m src.pipeline.orchestrator run-step fetch-data

validate-data:
	 -m src.pipeline.orchestrator run-step validate-data

build-features:
	 -m src.pipeline.orchestrator run-step build-features

run-baseline:
	 -m src.pipeline.orchestrator run-step run-baseline

run-clustering:
	 -m src.pipeline.orchestrator run-step run-clustering

run-regimes:
	 -m src.pipeline.orchestrator run-step run-regimes

run-event-study:
	 -m src.pipeline.orchestrator run-step run-event-study

run-migration:
	 -m src.pipeline.orchestrator run-step run-migration

generate-figures:
	 -m src.pipeline.orchestrator run-step generate-figures

generate-report:
	 -m src.pipeline.orchestrator run-step generate-report

run-all:
	 -m src.pipeline.orchestrator run-all

test:
	 -m pytest tests/ -v

lint:
	 -m ruff check 
	 -m mypy  --ignore-missing-imports

clean:
	@echo "Removing generated outputs..."
	@rm -rf outputs/figures/* outputs/tables/* outputs/reports/*
	@echo "Done."
