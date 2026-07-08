ENV ?= lys-bbb
CONFIG ?= config/animals/TEMPLATE.yml
IHC ?=
RUN_ARGS ?=
RATLESNET_CONFIG ?= ratlesnetv2_finetune/configs/dataset_template.yml

.PHONY: env-check test lint convert-mri calibrate-ihc ihc-diagnose ihc-threshold-sweeps ratlesnetv2-roiset-to-mask ratlesnetv2-add-source ratlesnetv2-download-external ratlesnetv2-orient-external-lsp ratlesnetv2-flip-external-si ratlesnetv2-prepare-lys-roisets ratlesnetv2-review-lys-masks ratlesnetv2-prepare ratlesnetv2-split-prepared ratlesnetv2-cloud-plan an2023-finetune run clean

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

ratlesnetv2-roiset-to-mask:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.roiset_to_nifti_mask $(RUN_ARGS)

ratlesnetv2-add-source:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.add_source_folder --config $(RATLESNET_CONFIG) $(RUN_ARGS)

ratlesnetv2-download-external:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.download_external_datasets $(RUN_ARGS)

ratlesnetv2-orient-external-lsp:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.orient_external_dataset_lsp $(RUN_ARGS)

ratlesnetv2-flip-external-si:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.flip_external_si_axis $(RUN_ARGS)

ratlesnetv2-prepare-lys-roisets:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.prepare_lys_roiset_dataset $(RUN_ARGS)

ratlesnetv2-review-lys-masks:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.review_lys_masks_itksnap $(RUN_ARGS)

ratlesnetv2-prepare:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.prepare_dataset --config $(RATLESNET_CONFIG) $(RUN_ARGS)

ratlesnetv2-split-prepared:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.split_prepared_dataset $(RUN_ARGS)

ratlesnetv2-cloud-plan:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.plan_cloud_run --config $(RATLESNET_CONFIG) $(RUN_ARGS)

an2023-finetune:
	conda run -n $(ENV) python -m ratlesnetv2_finetune.scripts.finetune_an2023 $(RUN_ARGS)

# Run one animal:
#   make run CONFIG=config/animals/BD_08_5D.yml RUN_ARGS=--no-mask-editor [IHC=work/BD_08_5D/ihc_A.csv]
run:
	conda run -n $(ENV) python -m src.run_animal --config $(CONFIG) $(if $(IHC),--ihc-csv $(IHC),) $(RUN_ARGS)

clean:
	rm -rf work/* outputs/* .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
