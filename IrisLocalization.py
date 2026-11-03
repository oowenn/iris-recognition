"""Locate pupil and outer iris boundaries; document method and parameters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Circle:
    """Center x/y and radius in original image pixels."""
    x: float
    y: float
    radius: float


def localize_iris(image):
    """Return (pupil_circle, iris_circle) for a 2D grayscale image.

    Paper baseline: projection-profile minima, histogram threshold/centroid in
    a 120x120 neighborhood (repeat twice), then Canny edges and circular Hough
    in a restricted search region. Circles need not be concentric.
    TODO: explain thresholds, search ranges, loops, and failure handling.
    Validate overlays across multiple eyes before downstream integration.
    """
    raise NotImplementedError("Implement pupil and outer iris localization")
