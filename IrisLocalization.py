"""A simple pupil-thresholding and iris-Hough baseline for CASIA-IrisV1."""

import cv2
import numpy as np


def iris_localization(image, window_size=120):
    """Return pupil, iris as two (center_x, center_y, radius) tuples in pixels.

    image is a uint8 grayscale CASIA-IrisV1 image. x means column and y means
    row. window_size is the side length of the local pupil search window.
    Radius limits and Hough thresholds below are starting choices for 320x280
    images. This baseline can confuse eyelids with the outer iris boundary.
    """
    # 1. Smooth small image variations before circle detection.
    # (5, 5) is the Gaussian kernel size; 0 lets OpenCV choose its sigma.
    smooth = cv2.GaussianBlur(image, (5, 5), 0)
    half_window = window_size // 2

    # 2. Estimate the pupil center from the darkest column and row averages.
    # axis=0 averages down the rows; axis=1 averages across the columns.
    pupil_x = float(np.argmin(image.mean(axis=0)))
    pupil_y = float(np.argmin(image.mean(axis=1)))

    # 3. Threshold a local window and recenter on its largest dark region twice.
    # Otsu chooses the threshold automatically; inverse thresholding marks dark
    # pixels white. Component 0 is background, so search components 1 onward.
    for _ in range(2):
        x0 = max(0, int(round(pupil_x)) - half_window)
        y0 = max(0, int(round(pupil_y)) - half_window)
        window = image[y0:y0 + window_size, x0:x0 + window_size]
        _, mask = cv2.threshold(
            window, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        _, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        pupil_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # Convert the local centroid to image coordinates; area = pi * radius^2.
        pupil_x, pupil_y = centroids[pupil_label] + (x0, y0)
        pupil_radius = np.sqrt(stats[pupil_label, cv2.CC_STAT_AREA] / np.pi)
    pupil = (float(pupil_x), float(pupil_y), float(pupil_radius))

    # 4. Let OpenCV find outer-circle candidates, including edge detection.
    # dp=1 uses full accumulator resolution; minDist=20 separates circle centers.
    # param1=50 is the high Canny threshold; param2=15 is the circle vote threshold.
    # Search radii from 80 to 130 pixels, larger than a typical pupil here.
    circles = cv2.HoughCircles(
        smooth, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=15, minRadius=80, maxRadius=130,
    )

    # 5. Choose the circle centered closest to the pupil and return both circles.
    # If Hough finds nothing, use a rough concentric 105-pixel radius estimate
    # (the midpoint of 80-130). This fallback is a guess, not a detected boundary.
    if circles is None:
        iris = (pupil[0], pupil[1], 105.0)
    else:
        candidates = circles[0]
        distances = np.linalg.norm(candidates[:, :2] - pupil[:2], axis=1)
        iris = tuple(float(value) for value in candidates[np.argmin(distances)])
    return pupil, iris
