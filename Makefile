# cpilens pipeline. `make all` rebuilds everything from public sources.
# Stages not built yet print a notice instead of failing silently.
UV := uv run
.PHONY: all data label models eval econ figures paper check test lint smoke

all: data models eval econ figures paper check

data:
	$(UV) cpilens fetch-raw
	$(UV) cpilens fetch-tpso
	$(UV) cpilens weights
	$(UV) cpilens products
	$(UV) cpilens sample-gold

label:
	$(UV) cpilens label --pass 1

models:
	$(UV) cpilens models
	$(UV) cpilens cascade

eval:
	$(UV) cpilens eval

econ:
	$(UV) cpilens econ

figures paper:
	@echo "$@: not built yet"; exit 1

check: lint test

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .
	$(UV) mypy cpilens

test:
	$(UV) pytest -q --cov=cpilens.taxonomy --cov=cpilens.evaluate --cov=cpilens.econ \
		--cov-report=term-missing --cov-fail-under=80

# CI: no raw data; tests that need data/raw are skipped.
smoke: lint test
