# Documentation

This training branch intentionally keeps four detailed references:

1. [ratlesnetv2_lys_v1_kaggle_workflow.md](ratlesnetv2_lys_v1_kaggle_workflow.md)
   — active step-by-step experiment, decision gates, frozen test, and artifact
   packaging.
2. [ratlesnetv2_external_datasets.md](ratlesnetv2_external_datasets.md) — source
   selection, deduplication, geometry, orientation, and provenance for the
   external-mouse comparator.
3. [lys_v2_architecture_comparator_protocol.md](lys_v2_architecture_comparator_protocol.md)
   — the separate post-RatLesNetV2 An et al./nnU-Net v2 OOF comparator. Its
   executable notebook is
   [../notebooks/lys_v2_architecture_comparator_kaggle.ipynb](../notebooks/lys_v2_architecture_comparator_kaggle.ipynb).
4. [ratlesnetv2_mac_inference.md](ratlesnetv2_mac_inference.md) — package the
   frozen five-model ensemble and run draft-mask inference on unlabeled scans
   with Apple MPS or CPU.

Use [../AGENTS.md](../AGENTS.md) for branch rules and [../README.md](../README.md)
for orientation. Preparation utility details live in
[../ratlesnetv2_finetune/README.md](../ratlesnetv2_finetune/README.md).

Older v1/IHC roadmaps and preliminary model notes were removed from this branch
because they conflicted with the active training protocol. Git history retains
them if historical context is ever required.
