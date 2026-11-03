"""Inspect iris matching on a small subset using the prescribed session split."""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# Find project modules from this script's location.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from IrisRecognition import load_dataset, extract_features, ROTATION_ANGLES
from IrisMatching import iris_matching

# 1. Choose a few eyes while keeping session 1 for training and session 2 for testing.
eyes = ["001", "018", "063"]
n_components = len(eyes) - 1  # 2 Fisher dimensions for 3 eyes; use None for raw features
train_samples, test_samples = load_dataset()
train_samples = [sample for sample in train_samples if sample[1] in eyes]
test_samples = [sample for sample in test_samples if sample[1] in eyes]
print(f"Small subset: {len(train_samples)} training and {len(test_samples)} test images.")
print("These results do not represent the full 108-eye experiment.")

# 2. Extract seven rotated training templates and one unrotated test feature per image.
train_features, train_labels = extract_features(train_samples, ROTATION_ANGLES)
test_features, test_labels = extract_features(test_samples, (0,))
test_features = test_features[:, 0, :]  # remove the test set's single-angle axis

# 3. Fit and match once, obtaining all three test-to-class distance matrices.
classes, distances = iris_matching(
    train_features, train_labels, test_features, n_components=n_components
)
test_rows = np.arange(len(test_samples))
true_columns = np.searchsorted(classes, test_labels)  # classes is sorted by iris_matching
fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)

for ax, (metric, scores) in zip(axes, distances.items()):
    # 4. The smallest distance in each row selects the predicted eye class.
    predicted_columns = scores.argmin(axis=1)
    predictions = classes[predicted_columns]
    correct = predictions == test_labels
    accuracy = 100 * correct.mean()
    print(f"\n{metric}: shape={scores.shape}, {correct.sum()}/{len(correct)} correct ({accuracy:.1f}%)")
    for (path, true_label), prediction in zip(test_samples, predictions):
        print(f"  {path.name}: true={true_label}, predicted={prediction}")

    # 5. Show distances with a red circle for the prediction and a black x for truth.
    # A circle and x in the same cell indicate a correct prediction.
    heatmap = ax.imshow(scores, cmap="Blues", aspect="auto")
    ax.scatter(predicted_columns, test_rows, facecolors="none", edgecolors="red", s=100)
    ax.scatter(true_columns, test_rows, marker="x", color="black", s=35)
    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(test_rows, labels=[path.name for path, _ in test_samples])
    ax.set_xlabel("Candidate eye class")
    ax.set_title(f"{metric}: subset accuracy {accuracy:.1f}%")
    fig.colorbar(heatmap, ax=ax, label="Distance (lower = closer)")

# 6. Display the three measures together; each has its own distance scale.
axes[0].set_ylabel("Session-2 test image")
fig.suptitle(f"{len(eyes)}-eye subset | Red circle: prediction | Black x: true class")
plt.tight_layout()
plt.show()
