"""Visually compare localized eyes with their normalized iris strips."""

from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Find the project modules and dataset from this script's location.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from IrisLocalization import iris_localization
from IrisNormalization import iris_normalization

# 1. Choose training images and the starting angle for normalization.
# Change eyes to inspect other subjects, or angle_deg to rotate the strip.
root = project_root / "CASIA Iris Image Database (version 1.0)"
eyes = ["001", "018", "063"]
angle_deg = 0
fig, axes = plt.subplots(len(eyes), 2, figsize=(12, 8), squeeze=False)

for (eye_ax, strip_ax), eye in zip(axes, eyes):
    path = root / eye / "1" / f"{eye}_1_1.bmp"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # 2. Locate the boundaries, then unwrap the region between them.
    pupil, iris = iris_localization(image)
    normalized = iris_normalization(image, pupil, iris, angle_deg=angle_deg)
    print(f"{path.name}: shape={normalized.shape}, dtype={normalized.dtype}")

    # 3. Show the source image with the pupil in green and outer iris in red.
    # Inaccurate circles here can explain unwanted regions in the strip.
    eye_ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    for (x, y, radius), color in [(pupil, "lime"), (iris, "red")]:
        eye_ax.add_patch(Circle(
            (x, y), radius, fill=False, color=color, linewidth=1.5
        ))
    eye_ax.set_title(path.name)
    eye_ax.axis("off")

    # 4. Display the normalized strip with the same grayscale intensity scale.
    # Top rows are near the pupil; columns move counterclockwise around the eye.
    strip_ax.imshow(normalized, cmap="gray", vmin=0, vmax=255)
    strip_ax.set_title(f"Normalized iris: {normalized.shape}, angle={angle_deg}°")
    strip_ax.set_xlabel("Angle around iris (columns)")
    strip_ax.set_ylabel("Pupil → outer iris (rows)")

# 5. Display all comparisons in one window.
plt.tight_layout()
plt.show()
