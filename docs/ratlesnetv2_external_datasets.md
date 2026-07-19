# External mouse data for RatLesNetV2

Last reviewed: 2026-07-15

External mouse data are a controlled initialization comparator. The source
checkpoint is selected using external validation only, then fine-tuned on the
same LYS folds/settings as the direct baseline. Only paired LYS OOF evidence
can show whether external adaptation helped. Held-out LYS cases provide the
final performance claim.

## Included sources

| Source | Record | Included native manual data |
|---|---|---|
| `An2022` | <https://zenodo.org/records/6379879> | T2w scans with manual lesion masks |
| `Knab2025` | <https://zenodo.org/records/14709930> | unique cases not already in An2022 |
| `Koch2017` | <https://zenodo.org/records/842677> | native manual scan/mask pairs |

`Mulder2017` is not in the current prepared set. An2022 already contains a
Mulder subset, so adding it later requires explicit deduplication and a new
dataset version.

## Prepared comparator

Archive: `External_Mouse_T2w_manual_LSP_SI_v0.tar.gz`

The manifest contains 426 distinct `animal_id` values:

| Dataset | Shape | Cases |
|---|---:|---:|
| An2022 | `256 x 256 x 32` | 331 |
| Knab2025 | `256 x 256 x 32` | 45 |
| Knab2025 | `192 x 192 x 32` | 35 |
| Koch2017 | `256 x 256 x 32` | 15 |

The active workflow makes an external-only 80/20 split grouped by manifest
`animal_id`. With seed `20260715`, expected train/validation counts are 341/85.
Richer source-animal metadata is not currently available; record that
limitation when interpreting the comparison.

## Inclusion policy

Include only:

- manual labels;
- native-space scan/mask pairs;
- one record per unique image/case;
- binary non-empty lesion masks;
- matching scan/mask shapes and affines.

Exclude automated masks, atlas-space masks, duplicate native/cropped copies,
and known cross-dataset duplicates.

Specific choices:

- An2022: `t2.nii` with `masklesion_manual.nii`.
- Knab2025: native `t2.nii` with `masklesion.nii` only when absent from An2022.
- Koch2017: native `all/dat` pairs; do not mix the cropped copy as an
  independent case.

## Geometry and orientation

| Dataset | Typical spacing (mm) | Role |
|---|---:|---|
| LYS | `0.07 x 0.07 x 0.5` | target domain |
| An2022/Knab2025/Koch2017 | about `0.1 x 0.1 x 0.5` | source adaptation |

The current external version is header-relabelled to `LSP` and then receives
the visually confirmed superior/inferior voxel-array flip. Scan/mask arrays are
flipped together; affines continue to match; mask voxel counts remain
unchanged. Keep representative visual QC and do not introduce resampling as an
unrecorded cleanup step.

Rebuild commands, with paths supplied locally rather than committed:

```bash
make ratlesnetv2-download-external \
  RUN_ARGS="--output-root <external-root> --delete-archives"

make ratlesnetv2-orient-external-lsp \
  RUN_ARGS="--external-root <external-root> --target-axcodes LSP --overwrite"

make ratlesnetv2-flip-external-si \
  RUN_ARGS="--external-root <external-root> \
  --output-source-root <flipped-source-root> \
  --output-manifest <manifest.csv> --all --overwrite"
```

## Required provenance

Keep, where available:

```text
source_dataset, source_record_url, source_case_id, animal_id,
study, timepoint, species, stroke_model, label_source, label_observer,
image_space, shape, spacing_mm, affine_hash, mask_voxels,
lesion_volume_mm3, duplicate_of, qc_flag
```

The portable prepared manifest, orientation/flip manifests, QC report, external
split manifest, seed, and selected external checkpoint are part of the final
artifact trail.
