ENV ?= lys-bbb
CONFIG ?= config/animals/TEMPLATE.yml
IHC ?=
RUN_ARGS ?=

.PHONY: env-check test lint convert-mri calibrate-ihc ihc-diagnose ihc-section-qc ihc-threshold-sweeps ihc-threshold-review ihc-threshold-dashboard ihc-threshold-signoff ihc-quantify ihc-one-animal run clean

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

ihc-diagnose:
	conda run -n $(ENV) python scripts/run_ihc_threshold_sweeps.py --config $(CONFIG) --diagnose $(RUN_ARGS)

ihc-section-qc:
	conda run -n $(ENV) python scripts/run_ihc_threshold_sweeps.py --config $(CONFIG) --section-qc $(RUN_ARGS)

ihc-threshold-sweeps:
	conda run -n $(ENV) python scripts/run_ihc_threshold_sweeps.py --config $(CONFIG) $(RUN_ARGS)

ihc-threshold-review:
	conda run -n $(ENV) python scripts/run_ihc_threshold_sweeps.py --config $(CONFIG) --review-thumbnails $(RUN_ARGS)

ihc-threshold-dashboard:
	conda run -n $(ENV) python scripts/build_ihc_threshold_review_dashboard.py --config $(CONFIG) $(RUN_ARGS)

ihc-threshold-signoff:
	conda run -n $(ENV) python scripts/approve_ihc_threshold.py --config $(CONFIG) $(RUN_ARGS)

ihc-quantify:
	conda run -n $(ENV) python scripts/run_ihc_quantification.py --config $(CONFIG) $(RUN_ARGS)

ihc-one-animal:
	conda run -n $(ENV) python scripts/run_ihc_one_animal.py --config $(CONFIG) $(RUN_ARGS)

# Run one animal:
#   make run CONFIG=config/animals/BD_08_5D.yml RUN_ARGS=--no-mask-editor [IHC=work/BD_08_5D/ihc_A.csv]
run:
	conda run -n $(ENV) python -m src.run_animal --config $(CONFIG) $(if $(IHC),--ihc-csv $(IHC),) $(RUN_ARGS)

clean:
	rm -rf work/* outputs/* .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
