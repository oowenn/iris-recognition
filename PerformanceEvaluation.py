"""Paper identification and verification outputs; no bootstrap required."""


def evaluate_identification(test_labels, distances, class_labels):
    """Return CRR and minimum-distance predictions for each of three metrics.

    Inputs: labels (432,), distances (432,108), sorted class_labels (108,).
    CRR = number correct / 432; target >=0.80 for each metric. TODO: implement,
    document ties, and include every test image; never silently discard failures.
    """
    raise NotImplementedError("Implement identification evaluation")


def evaluate_verification(test_labels, distances, class_labels):
    """Return FMR/FNMR threshold curve and selected operating points.

    Team protocol: each test image claims each enrolled class, after rotation
    minimization. Genuine mask: test_label == class_label (432 genuine claims;
    432*107 = 46224 impostor claims). Confirm this interpretation with instructor.
    Accept if distance <= threshold. FMR = accepted impostors / impostors;
    FNMR = rejected genuine / genuine. Sweep all scores and endpoint thresholds.
    Plot x=FMR, y=FNMR as in paper, labeling percentages/log axes explicitly.
    TODO: implement table operating-point selection and save outputs. Do not copy
    paper thresholds as calibrated thresholds for this smaller dataset.
    """
    raise NotImplementedError("Implement verification curve and operating points")
