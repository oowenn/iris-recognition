import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from IrisMatching import iris_matching

def performance_evaluation(
    train_features, train_labels, test_features, test_labels, dimensions=(20, 40, 60, 80, 100, 107)):
    """
    Calculate recognition rates and verification errors, and save tables/plots.
    Ma et al., 2003 (4.3)
    """
    # match original features without Fisher reduction
    test_labels = np.asarray(test_labels)
    classes, original_distances = iris_matching(train_features, train_labels, test_features, n_components=None)

    # get cosine CRR at each Fisher dimension
    cosine_crr = []
    for dimension in dimensions:
        classes, reduced_distances = iris_matching(train_features, train_labels, test_features, n_components=dimension)
        # predict the eye with the smallest distance
        predicted = classes[reduced_distances["cosine"].argmin(axis=1)]
        rate = 100 * np.mean(predicted == test_labels)
        cosine_crr.append(rate)
        print(f"Fisher dimension {dimension}: cosine CRR = {rate:.2f}%")

    # compare original features with the last Fisher dimension for all 3 distances
    crr = {}
    for name, distances in (("original", original_distances), ("reduced", reduced_distances)):
        crr[name] = {}
        for metric in ("L1", "L2", "cosine"):
            predicted = classes[distances[metric].argmin(axis=1)]
            # CRR = % of test images assigned the correct eye ID
            crr[name][metric] = 100 * np.mean(predicted == test_labels)

    # display Table 3: CRR for original and reduced features
    print(f"\nTable 3: CRR (%) using original and {dimensions[-1]}-D features")
    print(f"{'Measure':<12} {'Original':>12} {'Reduced':>12}")
    for metric in ("L1", "L2", "cosine"):
        print(f"{metric:<12} {crr['original'][metric]:>12.2f} "
              f"{crr['reduced'][metric]:>12.2f}")

    # plot cosine CRR against Fisher dimension
    dimension_figure, ax = plt.subplots()
    ax.plot(dimensions, cosine_crr, marker="o")
    ax.set_xlabel("Fisher dimensions")
    ax.set_ylabel("Correct recognition rate (%)")
    ax.set_title("Recognition rate vs. dimension (cosine)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    dimension_figure.tight_layout()

    # use cosine distances for verification scores
    scores = reduced_distances["cosine"]
    # mark test, eye pairs as genuine if the eye IDS match
    genuine = test_labels[:, None] == classes[None, :]

    # evaluate acceptance thresholds across all claims
    # get false positive rate, true positive rate, thresholds
    fpr, tpr, thresholds = roc_curve(genuine.ravel(), -scores.ravel(), drop_intermediate=False)
    # calculate accepted impostor and rejected genuine claim percentages
    fmr = 100 * fpr # false match rate
    fnmr = 100 * (1 - tpr) # false nonmatch rate
    # convert thresholds back to distances
    distance_thresholds = -thresholds

    # display Table 4: verification thresholds and error rates
    threshold_table = []
    print(f"\nTable 4: verification using cosine and {dimensions[-1]}-D features")
    print(f"{'FMR limit (%)':>14} {'Threshold':>14} {'FMR (%)':>12} {'FNMR (%)':>12}")

    # choose thresholds to achieve FMR limits of 0.1%, 1%, and 5%
    for target_fmr in (0.1, 1.0, 5.0):
        # select the largest distance threshold within the FMR limit
        index = np.where(fmr <= target_fmr)[0][-1]
        threshold = distance_thresholds[index]

        # store the threshold and its error rates
        threshold_table.append({
            "target_fmr": target_fmr, 
            "threshold": threshold,
            "fmr": fmr[index], 
            "fnmr": fnmr[index]
        })
        print(f"{target_fmr:>14.1f} {threshold:>14.6f} "
              f"{fmr[index]:>12.3f} {fnmr[index]:>12.3f}")

    # plot ROC curve
    roc_figure, ax = plt.subplots()
    ax.plot(fmr, fnmr)
    ax.set_xlabel("False match rate (%)")
    ax.set_ylabel("False nonmatch rate (%)")
    ax.set_title(f"Verification ROC (cosine, {dimensions[-1]} dimensions)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True)
    roc_figure.tight_layout()

    # create output directory
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)

    # save plots as pngs
    dimension_figure.savefig(output_dir / "figure10_crr_vs_dimension.png", dpi=300, bbox_inches="tight")
    roc_figure.savefig(output_dir / "figure11_verification_roc.png", dpi=300, bbox_inches="tight")

    # save tables as CVs
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

    # display plots
    # plt.show()
    print(f"\nSaved tables and plots to {output_dir}")

    # return results and figures
    return {
        "crr": crr,
        "dimensions": np.asarray(dimensions),
        "cosine_crr": np.asarray(cosine_crr),
        "threshold_table": threshold_table,
        "roc": {"thresholds": distance_thresholds, "fmr": fmr, "fnmr": fnmr},
        "figures": (dimension_figure, roc_figure),
    }
