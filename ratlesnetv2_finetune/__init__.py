"""Local utilities for preparing LYS T2w/manual-mask data for RatLesNetV2.

This folder is intentionally independent from the v1 MRI/IHC spine. It can read
the same configured NIfTI and corrected mask outputs, but generated training
sets stay under work/ and the cloud training script expects an external
RatLesNetV2 checkout.
"""

