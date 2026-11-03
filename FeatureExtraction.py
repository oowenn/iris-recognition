"""Paper section 3.3, equations (3)-(6): custom filters and block statistics."""
from PipelineConfig import ROI_SHAPE, FILTER_SIGMAS, FEATURE_BLOCK, RAW_FEATURE_DIM


def make_filter(sigma_x, sigma_y, frequency, kernel_radius):
    """Return sampled Gaussian times radial cosine, not an oriented Gabor.

    G(x,y) = exp(-0.5*(x*x/sigma_x**2 + y*y/sigma_y**2))
             * cos(2*pi*frequency*sqrt(x*x+y*y)) / (2*pi*sigma_x*sigma_y).
    x is angular/horizontal; y is radial/vertical. TODO: implement. Do not
    introduce zero-mean/unit-norm normalization without documenting a deviation.
    """
    raise NotImplementedError("Implement custom spatial filter")


def extract_features(enhanced_iris, *, choices):
    """Return float64 (1536,) from enhanced (64,512) input.

    Crop first 48 rows BEFORE filtering. Use both FILTER_SIGMAS channels and
    explicit frequencies/support/padding from choices; reject unresolved values.
    For each 8x8 response block, a=abs(response), m=mean(a), d=mean(abs(a-m)).
    Flatten channel first, then block rows, block columns, then [m,d].
    2 channels * 6 block rows * 64 block columns * 2 statistics = 1536.
    TODO: implement; d is mean absolute deviation, not standard deviation.
    """
    raise NotImplementedError("Implement two-channel block texture features")
