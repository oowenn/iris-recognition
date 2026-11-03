"""Visually compare normalized iris strips before and after enhancement."""

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

# 1. Choose training images and the block sizes used by image enhancement.
root = project_root / "CASIA Iris Image Database (version 1.0)"
eyes = ["001", "018", "063"]
background_block = 16  # side length for averaging background brightness
histogram_block = 32  # side length for local histogram equalization
fig, axes = plt.subplots(len(eyes), 2, figsize=(12, 6), squeeze=False)

for (before_ax, after_ax), eye in zip(axes, eyes):
    path = root / eye / "1" / f"{eye}_1_1.bmp"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # 2. Locate and normalize the iris to get the input to enhancement.
    pupil, iris = iris_localization(image)
    normalized = iris_normalization(image, pupil, iris)

    # 3. Enhance the strip and print its shape, datatype, and intensity range.
    enhanced = image_enhancement(normalized, background_block, histogram_block)
    print(
        f"{path.name}: shape={enhanced.shape}, dtype={enhanced.dtype}, "
        f"range={enhanced.min()}..{enhanced.max()}"
    )

    # 4. Compare both images using the same 0..255 grayscale display scale.
    # Look for clearer texture; block boundaries and eyelashes may also stand out.
    before_ax.imshow(normalized, cmap="gray", vmin=0, vmax=255)
    before_ax.set_title(f"{eye}: normalized (before)")
    before_ax.axis("off")
    after_ax.imshow(enhanced, cmap="gray", vmin=0, vmax=255)
    after_ax.set_title(f"{eye}: enhanced (after)")
    after_ax.axis("off")

# 5. Display all comparisons in one window.
plt.tight_layout()
plt.show()
