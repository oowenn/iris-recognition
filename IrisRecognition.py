from pathlib import Path

import cv2
import numpy as np

from IrisLocalization import iris_localization
from IrisNormalization import iris_normalization
from ImageEnhancement import image_enhancement
from FeatureExtraction import feature_extraction
from PerformanceEvaluation import performance_evaluation


# locate dataset
DATA_DIR = Path(__file__).resolve().parent / "CASIA Iris Image Database (version 1.0)"
# starting angles for training templates
ROTATION_ANGLES = (-9, -6, -3, 0, 3, 6, 9)


def load_dataset(data_dir=DATA_DIR):
    """
    Return training and test image paths with their eye Ids.
    """
    # get image paths and eye ID
    train = [(p, p.parent.parent.name) for p in sorted(data_dir.glob("*/1/*.bmp"))]
    test = [(p, p.parent.parent.name) for p in sorted(data_dir.glob("*/2/*.bmp"))]
    return train, test


def extract_features(samples, angles):
    """
    Return features for each image and angle.
    """
    features, labels = [], []
    for path, label in samples:
        # read grayscale image
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # get pupil and iris circles
        pupil, iris = iris_localization(image)
        # normalize, enhance, and extract features for each angle
        templates = []
        for angle in angles:
            normalized = iris_normalization(image, pupil, iris, angle_deg=angle)
            enhanced = image_enhancement(normalized)
            templates.append(feature_extraction(enhanced))
        # store the features and labels for each image
        features.append(templates)
        labels.append(label)
    return np.asarray(features), np.asarray(labels)


def run_experiment(data_dir=DATA_DIR):
    """
    Extract train/test features and evaluate recognition performance.
    """
    train, test = load_dataset(data_dir)
    print(f"Processing {len(train)} training and {len(test)} test images...")

    # extract training features at 7 angles
    train_features, train_labels = extract_features(train, ROTATION_ANGLES)
    # extract test features at 0 degrees
    test_features, test_labels = extract_features(test, (0,))

    # select the 0-degree feature vector for each test image
    return performance_evaluation(
        train_features, train_labels, test_features[:, 0, :], test_labels
    )


def main():
    """Run the experiment and display its tables and plots."""
    run_experiment()


if __name__ == "__main__":
    main()
