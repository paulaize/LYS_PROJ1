"""Milestone 3 STUB — MRI->Allen (AIDAmri/ANTs) and IHC->Allen (ABBA).

Not part of v1. Implement when the atlas milestone is reached; until then these
raise NotImplementedError so callers fail loudly rather than silently skipping.
See AGENTS.md §3 (scope) and docs/development_roadmap.md (Milestone 3).
"""
from __future__ import annotations


def register_mri_to_allen(*args, **kwargs):
    raise NotImplementedError(
        "Milestone 3. Use AIDAmri (https://github.com/Aswendt-Lab/AIDAmri) — "
        "needs FSL + LIP orientation — or ANTs SyN. Feed the lesion mask INTO "
        "registration. Do NOT add this to v1."
    )


def register_ihc_to_allen(*args, **kwargs):
    raise NotImplementedError(
        "Milestone 3. Use ABBA via the QuPath extension (NOT abba_python GUI on "
        "macOS). Drive DeepSlice off the DAPI channel; refine on 5d sections."
    )
