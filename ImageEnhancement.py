"""Paper section 3.2.3: illumination subtraction and local equalization."""
from PipelineConfig import ILLUMINATION_BLOCK, EQUALIZATION_BLOCK


def enhance_image(normalized_iris):
    """Return float64 (64, 512), with a documented fixed 0..255 output scale.

    Compute nonoverlapping 16x16 means, bicubically expand the 4x32 background
    grid to 64x512, subtract in floating point, then equalize each 32x32 region.
    TODO: implement; document histogram quantization, range conversion, clipping,
    and interpolation edges. Do not silently substitute CLAHE or uint8 subtraction.
    Output range is a team interface convention, not an extra paper parameter.
    """
    raise NotImplementedError("Implement background correction and local equalization")
