# cpilens pipeline. `make all` rebuilds everything from public sources.
# Stages not built yet print a notice instead of failing silently.
UV := uv run
.PHONY: all data label models eval econ agreement figures paper arxiv check test lint smoke

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

agreement:
	$(UV) cpilens agreement

figures:
	for f in figures/fig*.py; do $(UV) python $$f || exit 1; done

paper:
	$(UV) cpilens report
	$(UV) python scripts/check_numbers.py
	cd paper && tectonic -X compile main.tex
	cd paper && tectonic -X compile brief.tex
	$(UV) python scripts/web_paper.py

arxiv: paper
	$(UV) python scripts/arxiv_bundle.py

check: lint test

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .
	$(UV) mypy cpilens
	$(UV) python scripts/check_numbers.py

test:
	$(UV) pytest -q --cov=cpilens.taxonomy --cov=cpilens.evaluate --cov=cpilens.econ \
		--cov-report=term-missing --cov-fail-under=80

# CI: no raw data and no ml group (torch); tests that need them are skipped.
smoke: lint test
