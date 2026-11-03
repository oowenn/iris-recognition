"""Display the iris ROI and the block features returned by both filter channels."""

from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt

# Find the project modules and dataset from this script's location.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from IrisLocalization import iris_localization
from IrisNormalization import iris_normalization
from ImageEnhancement import image_enhancement
from FeatureExtraction import feature_extraction

# 1. Choose training images and create one row of plots per eye.
root = project_root / "CASIA Iris Image Database (version 1.0)"
eyes = ["001", "018", "063"]
roi_rows = 48  # rows nearest the pupil used for feature extraction
fig, axes = plt.subplots(len(eyes), 3, figsize=(15, 8), squeeze=False)

for row_axes, eye in zip(axes, eyes):
    path = root / eye / "1" / f"{eye}_1_1.bmp"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # 2. Run the preprocessing stages to obtain the enhanced iris image.
    pupil, iris = iris_localization(image)
    normalized = iris_normalization(image, pupil, iris)
    enhanced = image_enhancement(normalized)

    # 3. Extract features using the default two filters and 8x8 blocks.
    # The expected output is a (1536,) float32 vector: 768 values per channel.
    features = feature_extraction(enhanced, roi_rows=roi_rows)
    print(
        f"{path.name}: shape={features.shape}, dtype={features.dtype}, "
        f"range={features.min():.3f}..{features.max():.3f}"
    )

    # 4. Show the exact region of the enhanced image used for extraction.
    row_axes[0].imshow(enhanced[:roi_rows], cmap="gray", vmin=0, vmax=255)
    row_axes[0].set_title(f"{eye}: enhanced ROI ({roi_rows}x{enhanced.shape[1]})")
    row_axes[0].axis("off")

    # 5. Separate the returned vector into channels and pairs of block statistics.
    # reshape(2, -1, 2) gives (2 channels, 384 blocks, 2 statistics).
    # This only rearranges the output for plotting; it does not recompute features.
    channel_features = features.reshape(2, -1, 2)
    for channel, ax in enumerate(row_axes[1:]):
        ax.plot(channel_features[channel, :, 0], label="Mean magnitude")
        ax.plot(channel_features[channel, :, 1], label="Mean absolute deviation")
        ax.set_title(f"{eye}: filter channel {channel + 1}")
        ax.set_xlabel("Block index (left to right, then top to bottom)")
        ax.set_ylabel("Feature value")
        ax.legend(fontsize=8)

# 6. Display the ROI and features together. These plots show extraction behavior;
# recognition accuracy will be evaluated after matching is implemented.
plt.tight_layout()
plt.show()
