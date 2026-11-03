"""Training-only Fisher projection and nearest center classification."""


def fit_matcher(train_features, train_labels):
    """Return fitted projection, preprocessing state, and labeled class centers.

    Features: (n_images, n_features). Labels: (n_images,).
    TODO: explain Fisher rank, singular-matrix handling, and center computation.
    Never fit these using test images.
    """
    raise NotImplementedError("Implement Fisher projection and class centers")


def score_queries(model, query_features, metric):
    """Return (distances, class_labels); columns follow class_labels ordering.

    metric: 'l1', 'l2', or 'cosine'. Output: (n_queries, n_classes).
    Smaller distances must always indicate closer matches. TODO: document
    exact formulas, cosine conversion, zero-vector and rotation handling.
    """
    raise NotImplementedError("Implement L1, L2, and cosine matching")
