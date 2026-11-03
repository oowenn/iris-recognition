# Preprocessing handoff

Producer: Owen. Consumer: Sean. Contract version: 1. Cache with NumPy
`savez_compressed`; load using `np.load(path, allow_pickle=False)`.

| Key | Shape / type | Meaning |
| --- | --- | --- |
| contract_version | scalar integer, 1 | Schema version |
| synthetic | scalar bool | True only for the development fixture |
| train_images | (324,7,64,512), float64 | Enhanced training strips, full normalized height |
| test_images | (432,64,512), float64 | Enhanced test strips at zero angle |
| train_labels | (324,), integer | Eye IDs 1..108, each repeated three times |
| test_labels | (432,), integer | Eye IDs 1..108, each repeated four times |
| train_ids | (324,), Unicode | Original image stems, e.g. 001_1_1 |
| test_ids | (432,), Unicode | Original image stems, e.g. 001_2_1 |
| angles_deg | (7,), integer | [-9,-6,-3,0,3,6,9] |
| provenance | scalar Unicode | JSON string: revision, choices, synthetic flag |

Sort by eye ID, then image index. No object arrays or pickled values. Pixel range
is 0..255, finite values only. Shapes refer to the complete dataset. Center x/y
means column/row; positive normalization angles run counterclockwise in the
original image. Normalize and enhance each rotation separately; do not rotate
Fisher vectors. Sean takes first 48 radial rows before filtering.

Compute features to `train_features` (324,7,1536) and `test_features`
(432,1536). Preserve label/ID order. Feature order is channel, block row, block
column, statistic [mean magnitude, mean absolute deviation]. The zero angle is
index 3. Fit Fisher on zero-angle TRAINING vectors; apply identical learned
centering/projection to all vectors. Build class/angle centers only from training.

The synthetic cache is about 0.7 GB uncompressed in memory. It is for development
and interface checks only, contains no biometrics, and must be rejected by the
final results/reporting path. It does not establish recognition quality. Real
caches contain derived biometric images: keep under ignored outputs/ and exclude
from submission and public distribution. Do not omit failed images to meet shapes;
raise an informative error and retain an explicit failure log for resolution.
