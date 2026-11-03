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

## Suggested work split

This is a proposal to agree on together, not a statement of completed contributions.

| Work | Proposed lead | Reviewer | Status |
| --- | --- | --- | --- |
| Localization, normalization, enhancement | Owen | Sean | Not started |
| Features, Fisher projection, matching | Sean | Owen | Not started |
| Integration and evaluation | Both | Both | Scaffold only |
| Documentation, peer evaluation, packaging | Both | Both | Not started |

Track decisions and actual contributions in `PROJECT_PLAN.md`.

## Module interfaces

Images use `[row, column]` indexing. Circle centers use `(x, y)` = `(column, row)` in original pixels. Feature matrices use `(n_images, n_features)`, labels `(n_images,)`.

| Module | Contract |
| --- | --- |
| IrisRecognition.py | Main orchestration and dataset checks |
| IrisLocalization.py | Grayscale image → pupil and iris circles |
| IrisNormalization.py | Image and circles → polar iris array |
| ImageEnhancement.py | Normalized iris → enhanced iris |
| FeatureExtraction.py | Enhanced iris → fixed-length feature vector |
| IrisMatching.py | Train features/labels → Fisher model and class centers; queries → distances |
| PerformanceEvaluation.py | Labels and distances → CRRs and verification ROC/table |

Choose exact dimensions, filters, rotation handling, and score definitions after reading the paper; these are intentionally not guessed. Fit all learned transforms and class centers on training images only. Keep the test set out of parameter selection.

## Dataset and submission notes

CASIA-IrisV1 has 756 images from 108 eyes, 320 × 280 grayscale BMPs. Ignore `Thumbs.db`. Pupil regions were replaced with constant-intensity circles, which limits generalization of segmentation results to unedited images. Keep images out of Git and the submission. Follow the acknowledgment and usage terms in `datasets/Note_CASIA-IrisV1.pdf` and `datasets/Agreements.pdf`.

Copy `SUBMISSION_README.template` to `<OWEN_UNI>_<SEAN_UNI>_IrisRecognition.README`, fill every TODO, and include it in `<OWEN_UNI>_<SEAN_UNI>_IrisRecognition.zip`. Run from a clean environment before submission.
