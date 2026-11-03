# Parallel work plan

Proposed ownership for Owen and Sean. Swap names if preferred; keep the boundary fixed.

| Owner | Files | Deliverable |
| --- | --- | --- |
| Owen: preprocessing | IrisLocalization.py, IrisNormalization.py, ImageEnhancement.py | Correct circles, 64x512 normalized/enhanced arrays, cached handoff |
| Sean: recognition/evaluation | FeatureExtraction.py, IrisMatching.py, PerformanceEvaluation.py | 1536 features, Fisher model, three CRRs, verification table/curve |
| Owen: integration lead | IrisRecognition.py | Load data, call stages, cache, run end to end |
| Both | PipelineConfig.py, README.md, SUBMISSION_README.template | Agree choices before changing contracts; document and review |

## Handoff that enables parallel work

Owen exports `outputs/preprocessed.npz` following `HANDOFF.md`. Sean starts with
`python make_demo_handoff.py` while Owen implements preprocessing. This synthetic
fixture validates shapes/interfaces only; it is not iris data and must never be
used for reported recognition performance. Sean owns feature extraction, so Owen
does not need to wait for filters or Fisher implementation.

Stage order: raw image -> circles -> normalized at angle -> enhancement -> handoff
-> crop/filter/statistics -> features -> training-only Fisher -> rotated centers
-> distances -> CRR and FMR/FNMR.

## Before working separately (joint, short interface meeting)

- [ ] Fill UNIs: Owen TODO, Sean TODO; confirm ownership.
- [ ] Read paper sections 3.2-3.4; review constants and HANDOFF.md together.
- [ ] Choose filter frequencies, support, padding, interpolation, and Fisher solver.
- [ ] Confirm positive rotation direction, feature order, and score definitions.
- [ ] Document Fisher fitting/rotation convention and verification comparisons.
- [ ] Confirm conflicting assignment figure references/expected operating points.

## Owen's independent checklist

- [ ] Projection-profile pupil estimate; 120x120 histogram refinement twice.
- [ ] Canny/Hough boundary estimation; separate pupil and iris centers.
- [ ] Paper normalization with initial-angle support; verify coordinate direction.
- [ ] 16x16 background means, bicubic expansion, subtraction, 32x32 equalization.
- [ ] Review overlays/strips on training images, including difficult cases.
- [ ] Export actual training rotations and zero-angle test strips plus provenance.
- [ ] Complete orchestration and dataset/path/error handling.

## Sean's independent checklist

- [ ] Build custom radial-cosine filters; keep unresolved values explicit.
- [ ] Extract channel/block-ordered mean and mean absolute deviation features.
- [ ] Implement rank-aware Fisher solver; at most 107 directions on 108 classes.
- [ ] Average projected training features per class/angle for seven templates.
- [ ] Score L1, squared L2, cosine; minimize across seven templates per class.
- [ ] Implement CRR, genuine/impostor comparisons, threshold sweep, and outputs.
- [ ] Exercise extraction/matching on synthetic fixture; report no real accuracy yet.

## Integration and final review (both)

- [ ] Replace synthetic fixture with real preprocessing cache; verify labels/order.
- [ ] Fit using session 1 only; freeze parameters before final session 2 evaluation.
- [ ] Run all 432 test images; report failures explicitly, do not remove them.
- [ ] Inspect low-scoring examples together; separate algorithm bugs from limitations.
- [ ] Produce all three CRRs, dimension sweep if required, verification table/curve.
- [ ] Cross-review each other's logic and all key parameters.
- [ ] Complete README, actual contribution record, and signed peer evaluation.
- [ ] Clean-environment run; package without data; submit November 9, 11:59 PM.

## Decisions requiring explicit documentation

| Choice | Current state / rationale |
| --- | --- |
| Filter cosine frequencies, kernel support/padding | Unresolved; do not invent paper values |
| Normalization interpolation / clipping | Unresolved; record implementation |
| Enhancement quantization | Unresolved; shared output convention float64 0..255 |
| Fisher output size | Up to 107; effective rank may lower it; 200 is infeasible for standard 108-class LDA |
| Fisher solver | Unresolved; singular raw scatter requires rank-aware handling |
| Rotation/Fisher fitting | Team proposal: fit zero-angle vectors; project/average each training rotation |
| Verification comparison unit | Team proposal: query vs enrolled class's rotation-minimized center score |
| Output references | Assignment figure numbering inconsistent; confirm |

## Experiment log

No recognition results yet. Record revision, parameters, fitted dimension, failure
counts, and all three CRRs. Never present synthetic fixture scores as results.

| Revision | Configuration / dimension | L1 CRR | L2 CRR | Cosine CRR | Failures |
| --- | --- | --- | --- | --- | --- |
| TODO | TODO | Pending | Pending | Pending | Pending |

## Actual contributions

| Member / UNI | Completed work | Review and integration |
| --- | --- | --- |
| Owen / TODO | TODO | TODO |
| Sean / TODO | TODO | TODO |
