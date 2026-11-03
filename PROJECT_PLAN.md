# Shared project plan

## Team

| Member | UNI | Agreed responsibilities |
| --- | --- | --- |
| Owen | TODO | Confirm proposed README split |
| Sean | TODO | Confirm proposed README split |

## Integration checklist

- [ ] Read paper and resolve conflicting figure references.
- [ ] Validate dataset and inspect images from both sessions.
- [ ] Agree on normalized image shape and boundary conventions.
- [ ] Inspect localization overlays and normalized/enhanced images across eyes.
- [ ] Freeze paper-based filter parameters and feature ordering.
- [ ] Fit Fisher projection and class centers on 324 training images only.
- [ ] Evaluate all 432 test images with all three metrics.
- [ ] Produce CRR and verification table/ROC; investigate failures.
- [ ] Complete documentation, contribution form/signatures, clean run, and packaging.

## Paper decisions

Record equations, page/section references, parameter values, and reasons for deviations.

| Component | Paper reference | Chosen implementation/parameters |
| --- | --- | --- |
| Pupil/iris localization | TODO | TODO; alternative method permitted |
| Normalization grid/interpolation | TODO | TODO |
| Enhancement/illumination correction | TODO | TODO |
| Filter family, scales, parameters | TODO | TODO |
| Local statistics/feature ordering | TODO | TODO |
| Rotation compensation | TODO | TODO |
| Fisher rank/numerical handling | TODO | TODO |
| Class centers and metric formulas | TODO | TODO |
| Verification comparisons/operating points | TODO | TODO |

## Experiment log

No results yet. Record configuration and code revision for each run. Do not tune on the test set.

| Date/revision | Configuration | L1 CRR | L2 CRR | Cosine CRR | Notes |
| --- | --- | --- | --- | --- | --- |
| TODO | TODO | Pending | Pending | Pending | TODO |

## Actual contributions

Use this record to complete the supplied evaluation form. Both members sign the final form.

| Member | Completed work | Review/integration |
| --- | --- | --- |
| Owen | TODO | TODO |
| Sean | TODO | TODO |
