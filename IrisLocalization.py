"""Locate the pupil and outer iris boundary (Ma et al., section 3.2.1)."""


def iris_localization(image, window_size=120):
    """Return pupil, iris as two (center_x, center_y, radius) tuples in pixels.

    image is a grayscale (height, width) array. x is the column coordinate;
    y is the row coordinate. window_size is the side of the pupil search area.
    The two circles can have different centers.
    """
    # TODO: estimate the pupil center from row/column intensity minima.
    # TODO: threshold the local window and refine its centroid twice.
    # TODO: use Canny edges and a circle Hough transform to refine both circles.
    # The assignment also permits a different localization method.
    raise NotImplementedError("TODO: implement iris localization")
