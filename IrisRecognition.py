"""Load the session split and connect the required iris recognition stages."""

from pathlib import Path

import cv2
import numpy as np

from IrisLocalization import iris_localization
from IrisNormalization import iris_normalization
from ImageEnhancement import image_enhancement
from FeatureExtraction import feature_extraction
from PerformanceEvaluation import performance_evaluation


# Dataset stays beside the scripts. Session 1 trains; session 2 tests.
DATA_DIR = Path(__file__).resolve().parent / "CASIA Iris Image Database (version 1.0)"
# Initial unwrapping angles, in degrees, for the seven enrolled templates.
ROTATION_ANGLES = (-9, -6, -3, 0, 3, 6, 9)


def load_dataset(data_dir=DATA_DIR):
    """Return train/test lists of (BMP path, eye ID), e.g. (path, '001')."""
    # Folder layout is eye_id/session/image.bmp; no random split is used.
    train = [(p, p.parent.parent.name) for p in sorted(data_dir.glob("*/1/*.bmp"))]
    test = [(p, p.parent.parent.name) for p in sorted(data_dir.glob("*/2/*.bmp"))]
    return train, test


def extract_features(samples, angles):
    """Return features (images, angles, 1536) and one eye label per image.

    samples is a list from load_dataset; angles contains starting angles in
    degrees. Each image is localized once, then processed at each angle.
    """
    features, labels = [], []
    for path, label in samples:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        pupil, iris = iris_localization(image)
        templates = []
        for angle in angles:
            normalized = iris_normalization(image, pupil, iris, angle_deg=angle)
            enhanced = image_enhancement(normalized)
            templates.append(feature_extraction(enhanced))
        features.append(templates)
        labels.append(label)
    return np.asarray(features), np.asarray(labels)


def run_experiment(data_dir=DATA_DIR):
    """Pipeline wiring for use after the algorithm TODOs are implemented."""
    train, test = load_dataset(data_dir)
    train_features, train_labels = extract_features(train, ROTATION_ANGLES)
    test_features, test_labels = extract_features(test, (0,))
    # The matcher compares each unrotated test feature to seven class centers.
    return performance_evaluation(
        train_features, train_labels, test_features[:, 0, :], test_labels
    )


def main():
    """Show the available split while the recognition stages are being built."""
    train, test = load_dataset()
    print(f"Session 1: {len(train)} training images")
    print(f"Session 2: {len(test)} testing images")
    print("Scaffold ready. Implement the TODOs before calling run_experiment().")
    # TODO: replace the preview above with run_experiment() when ready.


if __name__ == "__main__":
    main()
