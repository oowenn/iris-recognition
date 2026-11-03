"""Paper section 3.2.2, equation (2): unwrap between nonconcentric boundaries."""
from PipelineConfig import NORMALIZED_SHAPE


def normalize_iris(image, pupil_circle, iris_circle, *, start_angle_deg=0.0,
                   interpolation=None):
    """Return float64 array of shape NORMALIZED_SHAPE = (64, 512).

    Columns sample theta = start_angle + 2*pi*column/512; do not duplicate the
    angular endpoint. Rows sample radial fraction row/64, following equation (2).
    Interpolate between the two boundary points at each theta. x is column,
    y is row; account for downward image y so positive angles unwrap CCW.
    TODO: implement and document interpolation and out-of-image handling.
    No automatic per-image rescaling; preserve source intensity units.
    """
    raise NotImplementedError("Implement paper equation (2) normalization")
