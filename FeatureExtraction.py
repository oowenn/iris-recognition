"""Extract Ma's texture features (section 3.3, equations 3-6)."""

import cv2
import numpy as np

def feature_extraction(enhanced, roi_rows=48, block_size=8,sigmas=((3.0, 1.5), (4.5, 1.5)), frequencies=None,):
    """Return a float32 vector of 1536 features with the default parameters.

    enhanced is the 64x512 enhanced iris. roi_rows keeps rows near the pupil;
    block_size is the side length of each feature block in pixels. The ROI
    dimensions should be multiples of block_size. sigmas contains Gaussian
    (sigma_x, sigma_y) widths in pixels, one pair per channel. frequencies
    contains one radial carrier frequency per channel in cycles/pixel;
    None uses our starting choices of (1/8, 1/16).
    """
    # 1. Keep the 48 rows nearest the pupil and work in floating point.
    # features will collect each block's mean and mean absolute deviation.
    roi = enhanced[:roi_rows, :].astype(np.float32)
    features = []

    # 2. Choose the two carrier frequencies (periods of 8 and 16 pixels).
    # These are simple starting values, not numeric settings given in the paper.
    if frequencies is None:
        frequencies = (1 / 8, 1 / 16)

    # 3. Process each filter channel using a grid centered at (0, 0).
    # Our finite kernel extends at least three times the larger Gaussian width
    # on each side: 19x19 and 29x29 kernels for the default sigma pairs.
    for (sigma_x, sigma_y), frequency in zip(sigmas, frequencies):
        radius = int(np.ceil(3 * max(sigma_x, sigma_y)))
        coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
        x, y = np.meshgrid(coordinates, coordinates)

        # 4. Build the Gaussian envelope and circular cosine carrier (equation 3).
        # x and y are pixel offsets inside the kernel, not image coordinates.
        gaussian = np.exp(-0.5 * ((x / sigma_x)**2 + (y / sigma_y)**2))
        gaussian /= 2 * np.pi * sigma_x * sigma_y
        carrier = np.cos(2 * np.pi * frequency * np.sqrt(x*x + y*y))
        kernel = gaussian * carrier

        # 5. Filter the ROI, then take the magnitude of the signed response.
        # Reflect pixels at the borders to keep the same image dimensions.
        # filter2D computes correlation, equivalent to convolution here because
        # our centered kernels are symmetric under a 180-degree rotation.
        response = cv2.filter2D(roi, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
        magnitude = np.abs(response)

        # 6. Visit 8x8 blocks left to right, then top to bottom (equations 5-6).
        # Store the mean first, then mean absolute deviation, not standard deviation.
        for row in range(0, roi.shape[0], block_size):
            for col in range(0, roi.shape[1], block_size):
                block = magnitude[row:row + block_size, col:col + block_size]
                mean = block.mean()
                deviation = np.abs(block - mean).mean()
                features.extend((mean, deviation))

    # 7. Return one vector: 2 filters * 6 * 64 blocks * 2 statistics = 1536 values.
    return np.asarray(features, dtype=np.float32)
