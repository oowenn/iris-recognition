import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def iris_matching(train_features, train_labels, test_features, n_components=107):
    """
    Compute L1, L2, and cosine distances between test features and mean of each eye's rotation templates.
    Ma et al., 2003 (3.4)
    """
    # convert features to float64
    train_features = np.asarray(train_features, dtype=np.float64)
    train_labels = np.asarray(train_labels)
    test_features = np.asarray(test_features, dtype=np.float64)
    # get eye IDs
    classes = np.unique(train_labels)
    # unpack the training features shape
    _, rotations, feature_count = train_features.shape

    # reduce feature dimensions using Fisher/LDA if requested
    if n_components is not None:
        # flatten the training features and repeat labels for each rotation
        flat_train = train_features.reshape(-1, feature_count)
        repeated_labels = np.repeat(train_labels, rotations)

        # fit LDA to the training data and keep the first n_components columns of W
        lda = LinearDiscriminantAnalysis(solver="svd", n_components=n_components)
        lda.fit(flat_train, repeated_labels)
        projection = lda.scalings_[:, :n_components]

        # project the training and test features into the Fisher space
        train_features = train_features @ projection
        test_features = test_features @ projection

    # average training features for each eye
    centers = []
    for label in classes:
        centers.append(train_features[train_labels == label].mean(axis=0))
    centers = np.asarray(centers)

    # create a dictionary to hold the distance matrices for L1, L2, and cosine distances
    distances = {
        metric: np.zeros((len(test_features), len(classes)))
        for metric in ("L1", "L2", "cosine")
    }

    # compare each test image with each eye's mean rotation templates
    for sample_index, feature in enumerate(test_features):
        for class_index, templates in enumerate(centers):
            difference = templates - feature

            # compute L1 and L2 distances for each rotation
            l1 = np.abs(difference).sum(axis=1)
            l2 = (difference**2).sum(axis=1)

            # compute cosine similarity and convert to distance
            similarity = (templates @ feature) / (
                np.linalg.norm(templates, axis=1) * np.linalg.norm(feature)
            )
            # clip to [-1, 1] to handle rounding errors
            cosine = 1 - np.clip(similarity, -1, 1)

            # keep the smallest distance for each metric across all rotations
            distances["L1"][sample_index, class_index] = l1.min()
            distances["L2"][sample_index, class_index] = l2.min()
            distances["cosine"][sample_index, class_index] = cosine.min()

    # return eye IDs and distance matrices
    return classes, distances
