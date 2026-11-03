"""Identification CRR and verification ROC/table; no bootstrap required."""


def evaluate_identification(test_labels, distances, class_labels):
    """Return CRR and minimum-distance class predictions.

    TODO: document tie handling; include all 432 test images and any failures.
    Run for L1, L2, and cosine; each required CRR is at least 80%.
    """
    raise NotImplementedError("Implement identification evaluation")


def evaluate_verification(test_labels, distances, class_labels):
    """Return ROC points and verification table for the agreed paper protocol.

    TODO: confirm comparisons and operating points against the paper. Explain
    genuine/impostor labels, threshold direction, and FAR/FRR formulas.
    Record resolution of the assignment's conflicting figure references.
    """
    raise NotImplementedError("Implement verification evaluation")
