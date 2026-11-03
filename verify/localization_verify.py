from pathlib import Path
from importlib import reload
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Find the project modules and dataset from this script's location.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import IrisLocalization

# 1. Reload the module to pick up your latest saved edits
reload(IrisLocalization)

# 2. Choose a few training images from different eyes
root = project_root / "CASIA Iris Image Database (version 1.0)"
eyes = ["001", "009", "018", "036", "063", "108"]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

for ax, eye in zip(axes.flat, eyes):
    path = root / eye / "1" / f"{eye}_1_1.bmp"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # 3. Run localization and display the original image
    pupil, iris = IrisLocalization.iris_localization(image)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)

    # 4. Draw the pupil in green and outer iris in red
    for (x, y, radius), color in [(pupil, "lime"), (iris, "red")]:
        ax.add_patch(Circle(
            (x, y), radius,
            fill=False, color=color, linewidth=1.5
        ))

    ax.set_title(path.name)
    ax.axis("off")

plt.tight_layout()
plt.show()
