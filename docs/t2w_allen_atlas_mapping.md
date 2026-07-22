# T2w MRI to Allen CCFv3 atlas mapping

This workflow adds atlas-derived region information downstream of the frozen
RatLesNetV2 ensemble. It does not change lesion inference, probability
thresholding, postprocessing, or the locked-test protocol. Model masks remain
drafts unless a human-reviewed mask is supplied and identified explicitly.

```text
frozen inference manifest
        |
        v
validated 3-D LIP staging + orientation review
        |
        v
AIDAmri v3 preprocessing and MRI-template-to-T2 registration
        |
        v
subject-space Allen annotation + normalized label lookup
        |
        v
native-grid lesion/region overlap + QC overlay
        |
        v
hash-bound human registration approval -> final export
```

## Why AIDAmri is the primary backend

[AIDAmri v3](https://github.com/Aswendt-Lab/AIDAmri) is the primary route for
this first implementation. Its current manual identifies its target as the
Allen Mouse Brain Reference Atlas, CCFv3, uses a custom MRI template for T2w
registration, provides subject-space detailed/parental and hemisphere-split
annotations, and requires visual registration review. Its published validation
includes mouse stroke MRI with anisotropic acquisitions similar to this
project.

Do not silently substitute BrainGlobe's
`perens_stereotaxic_mri_mouse_25um`. Although it has a T2-weighted reference,
[BrainGlobe documents that it is not in the original Allen CCFv3 coordinate
space](https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html#gubra-s-mri-mouse-brain-atlas).
That atlas could support a separate, explicitly named analysis, but it cannot
be presented as this project's Allen CCFv3 bridge.

The AIDAmri repository and Docker image are external dependencies. Pin and
record the exact AIDAmri release and Git revision; do not commit its atlas
volumes, Docker layers, or registered images to this repository. Follow the
[official v3 manual](https://github.com/Aswendt-Lab/AIDAmri/blob/master/Manual.md)
for installation and inspect each installed script's `-h` output before use.

## 1. Stage frozen inference cases

Run the frozen ensemble first. Then stage its manifest under `work/`:

```bash
make atlas-prepare-aidamri RUN_ARGS="\
  --inference-manifest /absolute/path/to/inference_output/inference_manifest.csv \
  --output work/t2w_allen_aidamri_inputs"
```

The staging command:

- accepts native 3-D T2w images and `X x Y x Z x 1` inference inputs;
- writes a 3-D float T2 and binary draft lesion mask without resampling;
- requires matching shape, spacing, and affine;
- requires the header orientation AIDAmri v3 currently specifies (`LIP`);
- refuses header relabeling or orientation guesses;
- writes SHA-256 provenance and one pending orientation review per case; and
- creates `atlas_mapping_manifest.csv`, with the AIDAmri result fields blank.

The LIP header check is necessary but not sufficient. Open every staged T2 in
a header-aware viewer and confirm that anatomy actually matches the header.
Then edit its `orientation_review.json` without changing the recorded hashes:

```json
{
  "orientation_status": "approved",
  "anatomical_orientation_matches_header": true,
  "reviewer": "REVIEWER NAME",
  "review_date": "YYYY-MM-DD"
}
```

Keep the other generated fields in the record. If orientation is wrong, set
the status to `rejected`. Use AIDAmri's reviewed reorientation tool and apply
the identical voxel transform to the lesion mask; never change only the NIfTI
axis labels.

## 2. Register each T2 with AIDAmri v3

### LYS partial-volume gate

The prepared LYS acquisitions contain 18 coronal slices at 0.5 mm spacing.
They are complete acquisitions, but they are partial brain volumes: brain
tissue reaches both anterior-posterior array boundaries. The standard AIDAmri
full-template registration must therefore not be used directly for LYS. In the
first test case it fitted a 13.2 mm AIDAmri template to a 9.0 mm acquired slab,
introduced a large rotation/shear, and left only 48% brain coverage on the
posterior slice. That run is rejected evidence, not a usable atlas result.

Prepare a hash-bound slab review instead:

```bash
make atlas-prepare-partial-slab RUN_ARGS="\
  --case-id CASE_ID \
  --subject-t2 work/.../PROCESSED_BRAIN_EXTRACTED_T2.nii.gz \
  --subject-brain-mask work/.../PROCESSED_BRAIN_EXTRACTED_T2_mask.nii.gz \
  --lesion-mask work/.../lesion_model_draft_native.nii.gz \
  --template-t2 work/external/AIDAmri/lib/NP_template_sc0.nii.gz \
  --output work/.../partial_slab_review"
```

This ranks every fitting template slab by normalized cross-sectional brain
area, then writes candidate montages. The rank is only a geometry aid. A human
must compare anatomical landmarks away from the lesion, select a candidate in
`slab_selection.json`, confirm `anatomical_landmarks_confirmed`, and record
reviewer and date. The future registration stage must reject pending or stale
selections.

After approval, prepare a registration slab. Keep an edge guard equal to half
the acquired through-plane slice thickness so the approved first and last
slice centres are not themselves crop planes. For the current 0.5 mm LYS
acquisition this is 0.25 mm, or five voxels in the 50-um template:

```bash
make atlas-prepare-partial-registration RUN_ARGS="\
  --selection-json work/.../partial_slab_review/slab_selection.json \
  --candidates-csv work/.../partial_slab_review/slab_candidates.csv \
  --subject-t2 work/.../PROCESSED_BRAIN_EXTRACTED_T2.nii.gz \
  --subject-brain-mask work/.../PROCESSED_BRAIN_EXTRACTED_T2_mask.nii.gz \
  --lesion-mask work/.../lesion_model_draft_native.nii.gz \
  --template-t2 work/external/AIDAmri/lib/NP_template_sc0.nii.gz \
  --template-atlas-labels \
    work/external/AIDAmri/lib/annoVolume+2000_rsfMRI.nii.gz \
  --slab-edge-padding-mm 0.25 \
  --output work/.../partial_registration"
```

The command fails if the approved slab is stale or if the requested padding is
not present in the template. Padding does not change the approved slice-centre
mapping; it only retains the acquired slice extent during a rotated resample.
The first unpadded candidate-0088 trial remains rejected diagnostic evidence
because its exact crop caused a diagonal field-of-view cut-off on the edge
slices.

Run the pinned registration container where it remains visible in Docker
Desktop. Replace `CASE_ROOT` and the image tag with the prepared directory and
the locally pinned image:

```bash
docker --context desktop-linux run -dit \
  --name aidamri_partial_CASE \
  --platform linux/amd64 \
  --mount type=bind,source="$(pwd)/work/CASE_ROOT",target=/aida/DATA \
  aidamri:3408ed46-cmakebin /bin/bash

docker --context desktop-linux exec aidamri_partial_CASE \
  /aida/NiftyReg/niftyreg_install/bin/reg_aladin \
  -ref /aida/DATA/subject_t2.nii.gz \
  -flo /aida/DATA/template_slab_t2.nii.gz \
  -rmask /aida/DATA/subject_cost_mask_brain_minus_lesion.nii.gz \
  -fmask /aida/DATA/template_slab_mask.nii.gz \
  -rigOnly -cog \
  -res /aida/DATA/template_slab_rigid_in_subject.nii.gz \
  -aff /aida/DATA/rigid_transform.txt -omp 4

docker --context desktop-linux exec aidamri_partial_CASE \
  /aida/NiftyReg/niftyreg_install/bin/reg_resample \
  -ref /aida/DATA/subject_t2.nii.gz \
  -flo /aida/DATA/template_slab_atlas_labels.nii.gz \
  -trans /aida/DATA/rigid_transform.txt -inter 0 \
  -res /aida/DATA/template_slab_atlas_labels_rigid_in_subject.nii.gz

docker --context desktop-linux exec aidamri_partial_CASE \
  /aida/NiftyReg/niftyreg_install/bin/reg_resample \
  -ref /aida/DATA/subject_t2.nii.gz \
  -flo /aida/DATA/template_slab_mask.nii.gz \
  -trans /aida/DATA/rigid_transform.txt -inter 0 \
  -res /aida/DATA/template_slab_mask_rigid_in_subject.nii.gz
```

Validate the exact-grid labels, lookup coverage, rigid transform, whole-brain
and per-slice support, and render both QC views:

```bash
make atlas-validate-partial-registration RUN_ARGS="\
  --prepared-root work/CASE_ROOT \
  --structures-csv work/.../aidamri_split_parental_structures.csv \
  --aidamri-revision AIDAMRI_GIT_COMMIT \
  --niftyreg-revision NIFTYREG_GIT_COMMIT \
  --container-image-id DOCKER_IMAGE_ID \
  --registration-runtime-seconds ELAPSED_SECONDS"
```

This writes `rigid_registration_validation.json`,
`rigid_registration_qc.png`, `rigid_atlas_mapping_qc.png`, the 18-panel
`rigid_atlas_all_slices_qc.png`, and a hash-bound pending
`rigid_registration_review.json`. It rejects scale, shear, reflection, changed
inputs, grid mismatch, non-integer labels, or missing lookup rows. It does not
approve anatomy automatically. Inspect every panel in the all-slice montage
before changing the review to approved.

The constrained registration will use the selected partial template slab,
rigid initialization, subject/template masks, and a lesion-excluded reference
cost mask. Cost-function masking is motivated by evidence that focal lesions
can drive inappropriate spatial-normalization distortion
([Brett et al., 2001](https://doi.org/10.1006/nimg.2001.0845)). Unrestricted
affine or nonlinear registration must not be promoted merely because an
optimizer finishes: it needs its own deformation checks and anatomical review.

### Current example-case disposition

Paul rejected the padded rigid result for final atlas use on 2026-07-22: it was
much better than the full-template run but still not anatomically accurate
enough. Three bounded refinements were then tested as diagnostics from that
same rigid initialization, without a parameter grid:

| Diagnostic | Runtime | Brain support | Lesion with nonzero label | Result |
| --- | ---: | ---: | ---: | --- |
| padded masked rigid | 5 min 56 s | 90.8% | 85.1% | human-rejected baseline |
| masked affine | 8 min 06 s | 85.3% | 80.7% | automatically rejected; posterior slice lost all support |
| standard F3D | 3 min 00 s | 81.8% | 55.1% | Paul's visually preferred candidate; landmark validation pending |
| F3D2 velocity bi-mask | 81 min 58 s | 90.1% | 80.2% | not advanced; no coverage gain or validated landmark gain |

Paul identified `nonlinear_atlas_all_slices_qc.png`, produced by the standard
F3D transform, as the best of the tested registrations on 2026-07-22. This is
recorded in a hash-bound `standard_f3d_candidate_preference.json`. Candidate
preference does not by itself approve anatomical landmarks or regional export.
The accompanying `standard_f3d_lesion_coverage_qc.png` separates lesion voxels
with a nonzero atlas region from those on atlas label 0 or outside the
intensity-derived template support; these categories must not be collapsed
into a single claim that the lesion is simply outside the atlas.

The F3D2 transform contains no nonpositive Jacobian voxels, but its in-brain
Jacobian spans 0.613 to 1.552. This is reported as deformation evidence, not
judged against an invented acceptance cutoff. Coverage is also a field-of-view
integrity measure, not a proxy for anatomical accuracy. The source validation
JSON files, all-slice montages, and the consolidated
`registration_diagnostic_comparison.json` remain under the example case's
`work/t2w_allen_partial_registration_0088_padded` directory. No method is
accepted and no regional table from this case is a biological result.

The nonlinear validator is intentionally a diagnostic-only target:

```bash
make atlas-validate-partial-nonlinear-diagnostic RUN_ARGS="\
  --prepared-root work/CASE_ROOT \
  --structures-csv work/.../aidamri_split_parental_structures.csv \
  --diagnostic-id standard_f3d \
  --aidamri-revision AIDAMRI_GIT_COMMIT \
  --niftyreg-revision NIFTYREG_GIT_COMMIT \
  --container-image-id DOCKER_IMAGE_ID \
  --registration-runtime-seconds ELAPSED_SECONDS"
```

It hash-checks the rigid initialization and inputs, requires exact output
grids and lookup coverage, rejects nonpositive Jacobians or a nonempty brain
slice with zero template support, and emits a pending anatomical review. It
does not run registration and does not make a nonlinear transform eligible for
mapping automatically.

### Next route after the rejected example

Do not continue selecting affine/F3D settings on this single lesion-bearing
partial volume. First validate Paul's preferred standard F3D candidate using
explicit corresponding anatomical landmarks across the acquired slab and a
frozen landmark-error acceptance policy. The landmark names and correspondence
must be supplied or signed off scientifically; they must not be inferred from
the overlay. If that candidate fails, the next route is a same-protocol,
same-geometry acquisition template from a declared cohort, used as an
intermediate bridge to the Allen atlas. Only one example case is currently
staged. Before implementing the cohort-template route, Paul must provide:

- the additional T2w scans acquired with the same geometry and protocol;
- explicit case IDs and cohort membership;
- which cases, if any, are suitable for template construction; and
- whether control/unlesioned status is scientifically established in metadata.

None of these facts may be inferred from filenames or image appearance. The
bridge must then be validated on multiple held-out cases before any atlas
region statistics are exported.

### Standard full-volume sequence (not for current LYS acquisitions)

Run this step on the staged, writable case directory. From the installed
AIDAmri v3 container, inspect the current interfaces first:

```bash
python preProcessing_T2.py -h
python registration_T2.py -h
```

The official single-file sequence is:

```bash
python preProcessing_T2.py -i /mounted/work/cases/CASE_ID/t2_for_aidamri.nii.gz
python registration_T2.py -i /mounted/work/cases/CASE_ID/PROCESSED_BRAIN_EXTRACTED_T2.nii.gz
```

Use the actual processed filename written by the installed AIDAmri revision;
do not predict it in a wrapper. Inspect the bias-corrected image and brain
extraction before registration. AIDAmri then writes the registered MRI/Allen
templates, transforms, and subject-space annotations alongside the processed
T2.

AIDAmri offers at least two relevant hemisphere-split outputs:

- `*_AnnoSplit_parental.nii.gz`: larger parental regions;
- `*_AnnoSplit.nii.gz`: detailed regions.

The choice is a scientific protocol decision. Do not mix the two annotation
variants within one pooled report. With 0.5 mm through-plane sampling, validate
the coarser and detailed variants separately before deciding which resolution
is defensible for reporting.

For a provisional split-parental run, normalize the two lookup files from the
same pinned AIDAmri checkout:

```bash
make atlas-normalize-aidamri-structures RUN_ARGS="\
  --names-table work/external/AIDAmri/lib/annoVolume+2000_rsfMRI.nii.txt \
  --acronyms-table work/external/AIDAmri/lib/acronym_rsfMRI.txt \
  --aidamri-revision AIDAMRI_GIT_COMMIT \
  --output work/t2w_allen_aidamri_inputs/aidamri_split_parental_structures.csv"
```

The converter requires complete left/right pairs, cross-checks every base label
against the acronym table, records the AIDAmri revision and source hashes, and
writes explicit hemisphere metadata. It is specific to AIDAmri's split-parental
table; do not use it for the detailed annotation variant.

## 3. Complete the explicit mapping manifest

For each row in the generated `atlas_mapping_manifest.csv`, fill:

- `native_atlas_labels`: the chosen AIDAmri subject-space annotation;
- `structures_csv`: a normalized lookup for that exact AIDAmri annotation;
- `atlas_id`, `atlas_version`, `coordinate_space`, and `annotation_variant`;
- `registration_version` and the exact `registration_revision`;
- `registration_qc_json`, once a registration review exists.

`label_interpolation` must remain `nearest_neighbor`. The lookup must contain
the remapped IDs actually present in the selected AIDAmri annotation, not IDs
assumed from an unmodified Allen download. Normalize the AIDAmri label table to
this schema:

```csv
label_id,acronym,name,hemisphere
1,EXAMPLE,Example region,left
```

`hemisphere` is optional. If it is absent or incomplete, the code leaves
hemisphere blank; it never infers hemisphere from array position. Label ID 0
may be omitted and is reported as `outside atlas or unmapped`.

Paths may be absolute or relative to the manifest. Case IDs must remain the
original inference case IDs.

## 4. Produce draft regional results and registration QC

```bash
make atlas-summarize RUN_ARGS="\
  --manifest work/t2w_allen_aidamri_inputs/atlas_mapping_manifest.csv \
  --output work/t2w_allen_mapping_review"
```

For every case this writes:

- `atlas_mapping_qc.png`: six native slices with atlas boundaries and lesion;
- `region_metrics.csv`: all native-grid atlas regions;
- `affected_regions.csv`: regions overlapping the lesion;
- `mapping_summary.json`: hashes, spacing, orientation, coverage, and QC state;
- `registration_review_template.json`: a hash-bound review record.

Volumes are calculated on the native T2 grid using native header spacing. The
atlas annotation is used to assign each native lesion voxel to a region; the
lesion is not resampled to a high-resolution atlas merely to inflate apparent
precision. `atlas_coverage_fraction_of_lesion` reports the fraction of lesion
voxels assigned a nonzero atlas label. No automatic acceptance threshold is
currently claimed.

Review the overlay across the full rostro-caudal coverage, including
lesion-distorted slices. The project does not yet have a signed numerical or
landmark-based registration acceptance threshold, so approval remains an
explicit human decision. Copy each template to a durable review location,
fill `registration_qc_status` with `approved` or `rejected`, and add reviewer,
date, and notes. Do not change its atlas identity or label-image hash.

## 5. Export only reviewed mappings

Point `registration_qc_json` in the manifest to the durable decisions and run
the approval gate into `outputs/`:

```bash
make atlas-summarize RUN_ARGS="\
  --manifest work/t2w_allen_aidamri_inputs/atlas_mapping_manifest.csv \
  --output outputs/t2w_allen_mapping_v1 \
  --require-approved-registration"
```

The final gate requires both the orientation and registration reviews to be
approved and hash-current. It fails on changed images, duplicate cases,
geometry mismatches, non-binary masks, non-integer atlas labels, linear label
interpolation, missing lookup IDs, or stale reviews.

Even after registration approval, a row based on the model ensemble retains
`lesion_mask_status=model_draft`. Atlas mapping approval confirms spatial
alignment only; it does not convert a model prediction into a human-reviewed
lesion mask. Changing the status to `human_reviewed` also requires a separate
`lesion_mask_review_json` whose approved decision is bound to the case ID and
mask SHA-256 and records reviewer and date; editing the CSV status alone fails.

## Remaining scientific validation

Before treating regional values as biological results, Paul must still freeze:

1. the exact AIDAmri v3 release/Git revision and atlas variant;
2. parental versus detailed annotation granularity;
3. the normalized, hemisphere-aware structure lookup;
4. registration review landmarks and acceptance policy; and
5. a representative validation set spanning timepoints and lesion sizes.

None of these choices may be tuned on the locked LYS test to change the lesion
model. Atlas analysis is a separate downstream protocol and should receive its
own version when these decisions are frozen.
