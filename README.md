# Iris recognition group project

STATGR 5293, Fall 2026 · Owen and Sean · Due November 9, 11:59 PM on Coursework.

This is a starter template. The algorithm functions deliberately raise `NotImplementedError`; no recognition results have been computed.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bsdtar -xf 'datasets/CASIA Iris Image Database (version 1.0).rar'
python IrisRecognition.py --check-data
```

Extract the archive beside the scripts. Expected layout: `CASIA Iris Image Database (version 1.0)/001/1/001_1_1.bmp`, continuing through eye 108 and sessions 1 and 2. For another location, use `--data-root '/path/to/database'`. The data checker needs only standard Python. After implementation, run `python IrisRecognition.py`.

## Assignment checklist

- [ ] Obtain and read Ma et al., *Personal Identification Based on Iris Texture Analysis*, IEEE TPAMI 25(12), 1519–1533, 2003. The paper is not included here.
- [ ] Reproduce its preprocessing, feature extraction, and matching design. Alternative localization methods are allowed.
- [ ] Train on session 1: 324 images, three per eye.
- [ ] Test on session 2: 432 images, four per eye. Do not randomly split images.
- [ ] Use Fisher linear discriminant reduction and nearest center classification.
- [ ] Report identification CRR for L1, L2, and cosine similarity; each must be at least 80%.
- [ ] Produce verification ROC and results table. Bootstrap/confidence intervals are unnecessary.
- [ ] Explain script logic and ALL key parameters in each module.
- [ ] Document limitations and improvements.
- [ ] Register the two-person group using the course Google form.
- [ ] Complete the supplied peer evaluation and obtain both signatures.
- [ ] Submit runnable code, README, and peer evaluation in one UNI-named ZIP/RAR. Exclude the dataset.

Source: `GroupProject.pdf`. Its figure references conflict: the experiment slide mentions Tables 3/10, Table 4, and Fig. 13; the submission slide mentions Table 3/Fig. 10 and Table 4/Fig. 11. Confirm the intended paper references before finalizing outputs.

## Parallel work split

| Owner | Work | Handoff |
| --- | --- | --- |
| Owen | Localization, normalization, enhancement, main orchestration | Enhanced arrays via `HANDOFF.md` |
| Sean | Features, Fisher model, matching, evaluation | Features, model, score tables, plots |
| Both | Parameter decisions, cross-review, results and submission | `PROJECT_PLAN.md` |

Sean can start immediately using `python make_demo_handoff.py`. This generates
an explicitly synthetic shape fixture, not recognition data. Owen replaces it
with the real preprocessing cache when ready. See `HANDOFF.md` for the exact
schema and `PROJECT_PLAN.md` for independent checklists and integration steps.

## Design contracts

`PipelineConfig.py` centralizes shapes, block sizes, filter envelopes, rotation
angles, feature length, and unresolved choices. Module docstrings now specify
the paper-based operations and interfaces. `DesignChoices` leaves undocumented
filter frequencies and numerical choices unset rather than inventing defaults.

Train features: `(324,7,1536)`; zero-angle test features: `(432,1536)`.
Seven rotated class centers are built from training only. Matching uses L1,
squared L2 (equation 8), and cosine distance, minimizing across rotation centers.
Standard 108-class Fisher LDA permits at most 107 discriminant directions;
using 200 as in the paper is not feasible here. Document the rank-aware solver.

The scaffold remains unimplemented for recognition. Paper reference:
[Ma et al. (2003)](https://www.cse.msu.edu/~rossarun/BiometricsTextBook/Papers/Iris/Ma_IrisTexture_PAMI03.pdf),
sections 3.2–3.4. Team choices about Fisher fitting and verification comparisons
are explicitly separated from paper constants in `PROJECT_PLAN.md`.

## Dataset and submission notes

CASIA-IrisV1 has 756 images from 108 eyes, 320 × 280 grayscale BMPs. Ignore `Thumbs.db`. Pupil regions were replaced with constant-intensity circles, which limits generalization of segmentation results to unedited images. Keep images out of Git and the submission. Follow the acknowledgment and usage terms in `datasets/Note_CASIA-IrisV1.pdf` and `datasets/Agreements.pdf`.

Copy `SUBMISSION_README.template` to `<OWEN_UNI>_<SEAN_UNI>_IrisRecognition.README`, fill every TODO, and include it in `<OWEN_UNI>_<SEAN_UNI>_IrisRecognition.zip`. Run from a clean environment before submission.
