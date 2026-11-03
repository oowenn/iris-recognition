"""Extract Ma's texture features (section 3.3, equations 3-6)."""


def feature_extraction(
    enhanced, roi_rows=48, block_size=8,
    sigmas=((3.0, 1.5), (4.5, 1.5)), frequencies=None,
):
    """Return a 1D vector of 1536 features with the default image dimensions.

    enhanced is the 64x512 enhanced iris. roi_rows keeps the rows nearest
    the pupil. block_size is the side length of each feature block in pixels.
    sigmas gives (sigma_x, sigma_y) in pixels for the two filter channels.
    frequencies supplies two radial carrier frequencies in cycles/pixel;
    None marks this unresolved implementation choice, not a numeric default.
    """
    # TODO: select enhanced[:roi_rows, :].
    # TODO: construct Gaussian envelopes modulated by cos(2*pi*f*sqrt(x*x+y*y)),
    #       including the 1/(2*pi*sigma_x*sigma_y) factor in equation 3.
    #       These use circular carriers, not ordinary oriented Gabor carriers.
    # TODO: choose/document frequencies and finite kernel sizes. The paper
    #       specifies the sigmas above but does not give numeric f values.
    # TODO: convolve the ROI with each kernel and take response magnitudes.
    # TODO: for each channel, traverse nonoverlapping 8x8 blocks in row order;
    #       append mean(magnitude), then mean(abs(magnitude - block_mean)).
    #       2 channels * 6 * 64 blocks * 2 statistics = 1536 features.
    raise NotImplementedError("TODO: implement texture feature extraction")
