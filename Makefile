ENV ?= lys-bbb
CONFIG ?= config/animals/TEMPLATE.yml
IHC ?=
RUN_ARGS ?=
RATLESNET_CONFIG ?= ratlesnetv2_finetune/configs/dataset_template.yml

.PHONY: env-check test lint convert-mri calibrate-ihc ihc-diagnose ihc-threshold-sweeps ratlesnetv2-prepare ratlesnetv2-cloud-plan run clean

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

ihc-threshold-sweeps:
	conda run -n $(ENV) python scripts/run_ihc_threshold_sweeps.py --config $(CONFIG) $(RUN_ARGS)

ratlesnetv2-prepare:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.prepare_dataset --config $(RATLESNET_CONFIG) $(RUN_ARGS)

ratlesnetv2-cloud-plan:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.plan_cloud_run --config $(RATLESNET_CONFIG) $(RUN_ARGS)

# Run one animal:
#   make run CONFIG=config/animals/BD_08_5D.yml RUN_ARGS=--no-mask-editor [IHC=work/BD_08_5D/ihc_A.csv]
run:
	conda run -n $(ENV) python -m src.run_animal --config $(CONFIG) $(if $(IHC),--ihc-csv $(IHC),) $(RUN_ARGS)

clean:
	rm -rf work/* outputs/* .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
