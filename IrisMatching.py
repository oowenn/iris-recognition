"""Fisher dimension reduction and nearest centers (section 3.4)."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Use LinearDiscriminantAnalysis(solver="svd") to fit Fisher directions.
# Write class means, the three distance formulas, and rotation minimization
# with NumPy. LDA's predict() is a different classifier from this pipeline.


def fit_matcher(train_features, train_labels, n_components=107):
    """Fit using session 1 only; return a dictionary describing the matcher.

    train_features has shape (324, 7, 1536), with one axis for rotations.
    train_labels has one eye ID per image. n_components is the Fisher output
    dimension; use None for the original-feature comparison in Table 3.
    With 108 classes, Fisher has at most 107 discriminant directions.

    Return keys: 'projection' (1536, n_components), or None for raw features;
    'classes' (108,) ordered eye IDs; 'centers' (108, 7, feature_dimension).
    """
    # TODO: fit a shared Fisher projection using only training features/labels.
    #       Flatten image/angle axes and repeat each image label seven times.
    #       Fit LDA with solver="svd" and keep scalings_[:, :n_components].
    #       n_components=None means skip LDA in this wrapper, not in sklearn.
    # TODO: project features as in equation 7 (or retain them for the raw run).
    #       Use features @ projection consistently for equation 7. sklearn's
    #       SVD transform() also subtracts a mean, which changes cosine scores.
    # TODO: average the three training images for each eye at each angle,
    #       keeping seven separate class centers rather than merging angles.
    raise NotImplementedError("TODO: fit Fisher projection and class centers")


def iris_matching(test_features, model, metric="cosine"):
    """Return a (test_images, classes) distance matrix, smaller meaning closer.

    test_features has shape (432, 1536); model is returned by fit_matcher.
    metric selects 'L1', 'L2', or 'cosine'. Columns follow model['classes'].
    """
    # TODO: apply the fitted projection to test features.
    # TODO: compare with every class center using equation 8:
    #       L1 = sum(abs(a-b)), L2 = sum((a-b)**2) as printed in the paper,
    #       cosine distance = 1 - dot(a,b)/(norm(a)*norm(b)).
    # TODO: keep the minimum over the seven rotations for each eye class.
    # Identification uses the smallest-distance class; verification uses
    # each claimed class's distance, including all nonmatching claims.
    raise NotImplementedError("TODO: implement nearest-center matching")
