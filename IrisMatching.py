"""Fisher dimension reduction and nearest centers (section 3.4)."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def iris_matching(train_features, train_labels, test_features, n_components=107):
    """Return sorted eye IDs and a dictionary of three distance matrices.

    train_features has shape (training images, rotations, features), normally
    (324, 7, 1536). train_labels has one eye ID per training image.
    test_features has shape (test images, features), normally (432, 1536).
    n_components selects the Fisher dimension, at most classes - 1 (107 for
    all 108 eyes); None skips reduction for the original-feature comparison.

    distances['L1'], distances['L2'], and distances['cosine'] each have shape
    (test images, eye classes). Columns follow classes; smaller means closer.
    Cosine distance assumes nonzero feature vectors and class templates.
    """
    # 1. Prepare arrays and identify the eye classes present in training data.
    # Use float64 for the projection and distance calculations.
    train_features = np.asarray(train_features, dtype=np.float64)
    train_labels = np.asarray(train_labels)
    test_features = np.asarray(test_features, dtype=np.float64)
    classes = np.unique(train_labels)
    _, rotations, feature_count = train_features.shape

    # 2. Fit Fisher/LDA once using only training features and labels.
    # Flatten image/rotation axes and repeat each image label for its rotations.
    # None skips this step; otherwise n_components must fit the class count.
    if n_components is not None:
        flat_train = train_features.reshape(-1, feature_count)
        repeated_labels = np.repeat(train_labels, rotations)
        lda = LinearDiscriminantAnalysis(solver="svd", n_components=n_components)
        lda.fit(flat_train, repeated_labels)
        projection = lda.scalings_[:, :n_components]

        # Equation 7 uses features @ W. Apply the same W to training and testing.
        # LDA's SVD transform() also subtracts a mean, which changes cosine scores.
        train_features = train_features @ projection
        test_features = test_features @ projection

    # 3. Average the training images for each eye, keeping rotations separate.
    # centers has shape (classes, rotations, projected or original features).
    centers = []
    for label in classes:
        centers.append(train_features[train_labels == label].mean(axis=0))
    centers = np.asarray(centers)

    # 4. Allocate one test-to-class distance matrix for each measure.
    distances = {
        metric: np.zeros((len(test_features), len(classes)))
        for metric in ("L1", "L2", "cosine")
    }

    # 5. Compare each test image with every eye's seven mean templates.
    # Each distance vector below contains one score per rotation (equation 8).
    for sample_index, feature in enumerate(test_features):
        for class_index, templates in enumerate(centers):
            difference = templates - feature
            l1 = np.abs(difference).sum(axis=1)
            l2 = (difference**2).sum(axis=1)  # squared L2, as printed in the paper
            similarity = (templates @ feature) / (
                np.linalg.norm(templates, axis=1) * np.linalg.norm(feature)
            )
            # Clip small floating-point overshoots of the cosine's [-1, 1] range.
            cosine = 1 - np.clip(similarity, -1, 1)

            # 6. Keep the best rotation independently for each distance measure.
            distances["L1"][sample_index, class_index] = l1.min()
            distances["L2"][sample_index, class_index] = l2.min()
            distances["cosine"][sample_index, class_index] = cosine.min()

    # 7. Return all scores so evaluation can compute both identification and ROC.
    # For a measure, predicted labels = classes[distances[measure].argmin(axis=1)].
    return classes, distances
