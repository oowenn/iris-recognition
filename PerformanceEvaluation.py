"""Prepare identification and verification results (section 4.3)."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from IrisMatching import fit_matcher, iris_matching

# Write the experiment loops, CRR, and genuine/impostor labels with NumPy.
# Use roc_curve for threshold counts and matplotlib for the required figures.


def performance_evaluation(
    train_features, train_labels, test_features, test_labels,
    dimensions=(20, 40, 60, 80, 100, 107),
):
    """Return a dictionary of result tables and figures once implemented.

    train_features: (324, 7, 1536) session-1 features and rotation templates.
    train_labels: (324,) eye IDs; test_features: (432, 1536) session-2 features;
    test_labels: (432,) true eye IDs. dimensions lists Fisher dimensions for
    the Figure 10 curve; its last value is used for the reduced-feature table.
    """
    # TODO: call fit_matcher with n_components=None and dimensions[-1].
    #       For each model, call iris_matching with L1, L2, and cosine.
    # TODO: Table 3: predict classes[distances.argmin(axis=1)] and calculate
    #       CRR = 100 * mean(predicted_labels == test_labels), for both models.
    #       Assignment target: at least 80% CRR for all three measures.
    # TODO: Figure 10: repeat Fisher fitting/matching for each dimension,
    #       plotting cosine CRR against dimension. Fit on training data only.
    # TODO: for verification, label each test-to-class score as genuine when
    #       test_labels[row] == classes[column], otherwise as an impostor.
    # TODO: sweep distance thresholds; accept scores <= threshold.
    #       roc_curve(genuine.ravel(), -distances.ravel(),
    #                 drop_intermediate=False) returns fpr, tpr, thresholds.
    #       Convert to FMR=fpr, FNMR=1-tpr, distance_thresholds=-thresholds.
    #       FMR = accepted impostor claims / all impostor claims.
    #       FNMR = rejected genuine claims / all genuine claims.
    # TODO: Table 4: report thresholds with FMR and FNMR percentages.
    #       Plot FNMR (%) against FMR (%) for the ROC (Figures 11/13).
    # Bootstrap and confidence intervals are not required for this project.
    raise NotImplementedError("TODO: implement CRR tables and ROC plots")
