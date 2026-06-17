ENV ?= lys-bbb
CONFIG ?= config/animals/TEMPLATE.yml
IHC ?=
RUN_ARGS ?=

.PHONY: env-check test lint convert-mri calibrate-ihc run clean

env-check:
	conda run -n $(ENV) python scripts/check_env.py

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n $(ENV) python -m pytest -q

lint:
	conda run -n $(ENV) ruff check .

convert-mri:
	conda run -n $(ENV) python scripts/convert_bruker_t2.py --config $(CONFIG)

calibrate-ihc:
	conda run -n $(ENV) python -m src.ihc.calibrate --config $(CONFIG)

# Run one animal:
#   make run CONFIG=config/animals/M07.yml RUN_ARGS=--no-mask-editor [IHC=work/M07/ihc_A.csv]
run:
	conda run -n $(ENV) python -m src.run_animal --config $(CONFIG) $(if $(IHC),--ihc-csv $(IHC),) $(RUN_ARGS)

clean:
	rm -rf work/* outputs/* .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
