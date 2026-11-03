"""Shared paper constants and explicit unresolved implementation choices."""
from dataclasses import dataclass

NORMALIZED_SHAPE = (64, 512)  # radial rows, angular columns
ROI_SHAPE = (48, 512)         # first 48 rows, nearest pupil
ILLUMINATION_BLOCK = (16, 16)
EQUALIZATION_BLOCK = (32, 32)
FEATURE_BLOCK = (8, 8)
FILTER_SIGMAS = ((3.0, 1.5), (4.5, 1.5))  # (x/angular, y/radial)
ROTATION_ANGLES = (-9, -6, -3, 0, 3, 6, 9)
RAW_FEATURE_DIM = 1536
METRICS = ("l1", "l2", "cosine")


@dataclass(frozen=True)
class DesignChoices:
    """Unspecified details must be decided and recorded, never silently guessed.

    Frequencies are cycles/pixel, in FILTER_SIGMAS channel order. Kernel radius
    truncates support to [-radius, radius] on each axis. Padding and sampling
    rules need a documented choice. None means unresolved, not a default.
    """
    filter_frequencies: tuple[float, float] | None = None
    kernel_radius: int | None = None
    convolution_padding: str | None = None
    normalization_interpolation: str | None = None
    fisher_solver: str | None = None
    fisher_components: int = 107  # project adaptation: at most 108 - 1
