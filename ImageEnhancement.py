"""Correct illumination and improve local contrast (section 3.2.3)."""

import cv2
import numpy as np

def image_enhancement(normalized, background_block=16, histogram_block=32):
    """Return an enhanced uint8 image with the same shape as normalized.

    normalized is the 64x512 unwrapped grayscale iris. background_block is
    the side length for illumination averaging; histogram_block is the side
    length for local equalization, both in pixels. Input dimensions should be
    multiples of background_block. Output intensities are in the range 0..255.
    """
    # 1. Work in floating point so subtracting brightness can produce negatives.
    image = normalized.astype(np.float32)
    height, width = image.shape

    # 2. Average each 16x16 block to estimate the local background brightness.
    # A 64x512 image produces a small 4x32 grid, with one mean per block.
    background = np.zeros(
        (height // background_block, width // background_block), dtype=np.float32
    )
    for row in range(background.shape[0]):
        for col in range(background.shape[1]):
            y, x = row * background_block, col * background_block
            block = image[y:y + background_block, x:x + background_block]
            background[row, col] = block.mean()

    # 3. Expand the background estimate and subtract it from the iris image.
    # resize takes (width, height); INTER_CUBIC uses bicubic interpolation.
    illumination = cv2.resize(background, (width, height), interpolation=cv2.INTER_CUBIC)
    corrected = image - illumination

    # 4. Convert the corrected intensities to the uint8 format equalizeHist needs.
    # We choose min-max scaling over the whole image: minimum -> 0, maximum -> 255.
    # This is our conversion choice; the paper does not specify the quantization.
    corrected = cv2.normalize(
        corrected, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # 5. Equalize each 32x32 block independently to bring out local texture.
    # The loops visit nonoverlapping blocks and write them to the same positions.
    enhanced = np.zeros_like(corrected)
    for y in range(0, height, histogram_block):
        for x in range(0, width, histogram_block):
            block = corrected[y:y + histogram_block, x:x + histogram_block]
            enhanced[y:y + histogram_block, x:x + histogram_block] = cv2.equalizeHist(block)
    return enhanced
