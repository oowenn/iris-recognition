"""Correct illumination and improve local contrast (section 3.2.3)."""


def image_enhancement(normalized, background_block=16, histogram_block=32):
    """Return an enhanced grayscale array with the same shape as normalized.

    normalized is the 64x512 unwrapped iris. background_block is the side
    length in pixels for illumination averaging; histogram_block is the side
    length for local histogram equalization. Output intensities span 0..255.
    """
    # TODO: average each nonoverlapping 16x16 block to estimate illumination.
    # TODO: expand that grid using bicubic interpolation and subtract it
    #       from the normalized image in floating point.
    # TODO: equalize the corrected image in each 32x32 region.
    raise NotImplementedError("TODO: implement image enhancement")
