ENV ?= lys-bbb
CONFIG ?= config/animals/TEMPLATE.yml
IHC ?=

.PHONY: env-check test lint run clean

env-check:
	conda run -n $(ENV) python scripts/check_env.py

test:
	conda run -n $(ENV) python -m pytest -q

lint:
	conda run -n $(ENV) ruff check .

# Run one animal: make run CONFIG=config/animals/M07.yml [IHC=work/M07/ihc_A.csv]
run:
	conda run -n $(ENV) python -m src.run_animal --config $(CONFIG) $(if $(IHC),--ihc-csv $(IHC),)

clean:
	rm -rf work/* outputs/* .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
