"""Unwrap the iris ring into polar coordinates (section 3.2.2, equation 2)."""

import cv2
import numpy as np

def iris_normalization(image, pupil, iris, rows=64, cols=512, angle_deg=0):
    """Return a float32 (rows, cols) unwrapped iris image.

    image is the original grayscale array; pupil and iris are (x, y, radius)
    in pixels. rows samples outward from the pupil; cols samples angles around
    the iris. angle_deg rotates the starting angle counterclockwise from the
    right-hand side. Intensities keep the input scale (normally 0..255).
    """
    # 1. Read the two circles returned by iris_localization.
    # Keep their centers separate because the circles may not be concentric.
    pupil_x, pupil_y, pupil_radius = pupil
    iris_x, iris_y, iris_radius = iris

    # 2. Choose one angle per output column, without repeating the endpoint 2*pi.
    # angle_deg supports the rotated templates used later during matching.
    theta = 2 * np.pi * np.arange(cols) / cols + np.deg2rad(angle_deg)

    # 3. Find a point on each boundary at every angle.
    # Subtract sin(theta) because image y coordinates increase downward.
    inner_x = pupil_x + pupil_radius * np.cos(theta)
    inner_y = pupil_y - pupil_radius * np.sin(theta)
    outer_x = iris_x + iris_radius * np.cos(theta)
    outer_y = iris_y - iris_radius * np.sin(theta)

    # 4. Fill the rows with points between the two boundaries (equation 2).
    # radial is a column vector, so each radial fraction pairs with every angle.
    # Use Y/rows: the first row is at the pupil, the last just inside the iris.
    radial = np.arange(rows)[:, None] / rows
    map_x = (inner_x + radial*(outer_x - inner_x)).astype(np.float32)
    map_y = (inner_y + radial * (outer_y - inner_y)).astype(np.float32)

    # 5. Sample those coordinates using OpenCV's bilinear interpolation.
    # Float input preserves fractional intensities. Replicate edge pixels if
    # an estimated circle extends outside the image; no occlusion mask is used.
    return cv2.remap(
        image.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
