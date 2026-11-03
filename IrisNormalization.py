"""Unwrap the iris ring into polar coordinates (section 3.2.2, equation 2)."""


def iris_normalization(image, pupil, iris, rows=64, cols=512, angle_deg=0):
    """Return a floating-point (rows, cols) normalized grayscale image.

    image is the original grayscale array; pupil and iris are (x, y, radius).
    rows samples the radial direction from pupil toward the outer iris;
    cols samples the angular direction. angle_deg is the initial rotation.
    """
    # TODO: theta = 2*pi*X/cols + radians(angle_deg), X = 0,...,cols-1.
    # TODO: find points on each circle at theta, moving counterclockwise
    #       (image row coordinates increase downward).
    # TODO: interpolate between the two boundaries using r = Y/rows.
    # TODO: sample the source image at those coordinates with interpolation.
    raise NotImplementedError("TODO: implement iris normalization")
