"""Milestone 4 STUB — core / peri / contra compartments.

Not part of v1. Implement when compartments milestone is reached.
See AGENTS.md §7 and docs/development_roadmap.md (Milestone 4).
"""
from __future__ import annotations


def build_compartments(*args, **kwargs):
    """core = corrected lesion mask; peri = dilation ring (config peri_ring_mm)
    minus core; contra = atlas-midline mirror. Returns a label image in Allen
    space (0=other,1=core,2=peri,3=contra-core,4=contra-peri)."""
    raise NotImplementedError(
        "Milestone 4. Needs the atlas (Milestone 3) for the midline + region "
        "tags. Do NOT add to v1. peri_ring_mm is a config TODO."
    )
