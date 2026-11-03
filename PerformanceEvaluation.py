"""Evaluate identification and verification results (section 4.3)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from IrisMatching import iris_matching

def performance_evaluation(
    train_features, train_labels, test_features, test_labels,
    dimensions=(20, 40, 60, 80, 100, 107),
):
    """Save the tables and plots in figures/, display them, and return results.

    train_features: (324, 7, 1536) session-1 features and rotation templates.
    train_labels: (324,) eye IDs; test_features: (432, 1536) session-2 features;
    test_labels: (432,) true eye IDs. dimensions lists Fisher dimensions for
    the Figure 10 curve; its last value is used for the reduced-feature table
    and verification. Use a nonempty sequence of dimensions no larger than
    the number of training classes minus one (107 for the full dataset).

    Returns CRR tables, dimensions and cosine CRRs, a threshold table, ROC
    arrays, and the two matplotlib figures. All rates are percentages.
    Tables are saved as CSV and plots as PNG beside this script in figures/.
    Running again replaces the four output files.
    """
    # 1. Match the original features without Fisher dimension reduction.
    # Each distance matrix has one row per test image and one column per eye.
    test_labels = np.asarray(test_labels)
    classes, original_distances = iris_matching(
        train_features, train_labels, test_features, n_components=None
    )

    # 2. Try each Fisher dimension and record the cosine recognition rate.
    # Each call learns its projection from training data only. Keep the last
    # dimension's distances for the reduced-feature table and verification.
    cosine_crr = []
    for dimension in dimensions:
        classes, reduced_distances = iris_matching(
            train_features, train_labels, test_features, n_components=dimension
        )
        predicted = classes[reduced_distances["cosine"].argmin(axis=1)]
        rate = 100 * np.mean(predicted == test_labels)
        cosine_crr.append(rate)
        print(f"Fisher dimension {dimension}: cosine CRR = {rate:.2f}%")

    # 3. Calculate Table 3: CRR = correct predictions / all test images * 100.
    # The predicted eye is the column with the smallest distance in each row.
    crr = {}
    for name, distances in (("original", original_distances),
                            ("reduced", reduced_distances)):
        crr[name] = {}
        for metric in ("L1", "L2", "cosine"):
            predicted = classes[distances[metric].argmin(axis=1)]
            crr[name][metric] = 100 * np.mean(predicted == test_labels)

    print(f"\nTable 3: CRR (%) using original and {dimensions[-1]}-D features")
    print(f"{'Measure':<12} {'Original':>12} {'Reduced':>12}")
    for metric in ("L1", "L2", "cosine"):
        print(f"{metric:<12} {crr['original'][metric]:>12.2f} "
              f"{crr['reduced'][metric]:>12.2f}")

    # 4. Plot Figure 10: cosine CRR against the number of Fisher dimensions.
    dimension_figure, ax = plt.subplots()
    ax.plot(dimensions, cosine_crr, marker="o")
    ax.set_xlabel("Fisher dimensions")
    ax.set_ylabel("Correct recognition rate (%)")
    ax.set_title("Recognition rate vs. dimension (cosine)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    dimension_figure.tight_layout()

    # 5. Treat each test-to-eye comparison as a verification claim.
    # A claim is genuine if the eye ID matches, otherwise it is an impostor.
    # Accept distance <= threshold. roc_curve expects larger scores to mean
    # more likely genuine, so negate distances and then negate its thresholds.
    scores = reduced_distances["cosine"]
    genuine = test_labels[:, None] == classes[None, :]
    fpr, tpr, thresholds = roc_curve(
        genuine.ravel(), -scores.ravel(), drop_intermediate=False
    )
    fmr = 100 * fpr           # accepted impostor claims / all impostor claims
    fnmr = 100 * (1 - tpr)    # rejected genuine claims / all genuine claims
    distance_thresholds = -thresholds

    # 6. Build Table 4 at three chosen FMR limits (percent, not fractions).
    # Pick the largest threshold whose measured FMR stays within each limit.
    # These points describe this test ROC. The actual rate can be below the
    # requested limit because the number of accepted claims changes in steps.
    threshold_table = []
    print(f"\nTable 4: verification using cosine and {dimensions[-1]}-D features")
    print(f"{'FMR limit (%)':>14} {'Threshold':>14} {'FMR (%)':>12} {'FNMR (%)':>12}")
    for target_fmr in (0.1, 1.0, 5.0):
        index = np.where(fmr <= target_fmr)[0][-1]
        threshold = distance_thresholds[index]
        threshold_table.append({
            "target_fmr": target_fmr, "threshold": threshold,
            "fmr": fmr[index], "fnmr": fnmr[index],
        })
        print(f"{target_fmr:>14.1f} {threshold:>14.6f} "
              f"{fmr[index]:>12.3f} {fnmr[index]:>12.3f}")

    # 7. Plot the verification ROC with the paper's FMR/FNMR axes.
    roc_figure, ax = plt.subplots()
    ax.plot(fmr, fnmr)
    ax.set_xlabel("False match rate (%)")
    ax.set_ylabel("False nonmatch rate (%)")
    ax.set_title(f"Verification ROC (cosine, {dimensions[-1]} dimensions)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True)
    roc_figure.tight_layout()

    # 8. Save the two required plots and two tables in the project's figures
    # folder. Use the script location so the folder does not depend on where
    # Python was launched. Save before showing the plot windows.
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)
    dimension_figure.savefig(output_dir / "figure10_crr_vs_dimension.png", dpi=300, bbox_inches="tight")
    roc_figure.savefig(output_dir / "figure11_verification_roc.png", dpi=300, bbox_inches="tight")

    with (output_dir / "table3_crr.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Measure", "Original CRR (%)", f"Reduced CRR (%) ({dimensions[-1]}D)"])
        for metric in ("L1", "L2", "cosine"):
            writer.writerow([metric, crr["original"][metric], crr["reduced"][metric]])

    with (output_dir / "table4_verification.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["FMR limit (%)", "Cosine distance threshold", "FMR (%)", "FNMR (%)"])
        for row in threshold_table:
            writer.writerow([row["target_fmr"], row["threshold"], row["fmr"], row["fnmr"]])

    print(f"\nSaved tables and plots to {output_dir}")
    plt.show()

    return {
        "crr": crr,
        "dimensions": np.asarray(dimensions),
        "cosine_crr": np.asarray(cosine_crr),
        "threshold_table": threshold_table,
        "roc": {"thresholds": distance_thresholds, "fmr": fmr, "fnmr": fnmr},
        "figures": (dimension_figure, roc_figure),
    }
