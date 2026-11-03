import cv2
import numpy as np

def feature_extraction(enhanced, roi_rows=48, block_size=8,sigmas=((3.0, 1.5), (4.5, 1.5)), frequencies=None,):
    """
    Extract features from enhanced iris image.
    Ma et al., 2003 (3.3)
    """
    # keep rows near the pupil and convert to float
    roi = enhanced[:roi_rows, :].astype(np.float32)
    features = []

    # default frequencies chosen to have periods of 8 and 16 pixels
    if frequencies is None:
        frequencies = (1 / 8, 1 / 16)

    # build and apply each texture filter for the sigma and frequency pairs
    for (sigma_x, sigma_y), frequency in zip(sigmas, frequencies):
        # create a coordinate grid centered on the filter
        radius = int(np.ceil(3 * max(sigma_x, sigma_y)))
        coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
        x, y = np.meshgrid(coordinates, coordinates)

        # build gaussian envelope, reducing weights away from the center
        gaussian = np.exp(-0.5 * ((x / sigma_x)**2 + (y / sigma_y)**2))
        gaussian /= 2 * np.pi * sigma_x * sigma_y

        # combine circular cosine bands with gaussian envelope
        carrier = np.cos(2 * np.pi * frequency * np.sqrt(x*x + y*y))
        kernel = gaussian * carrier

        # filter iris region and keep response magnitude
        response = cv2.filter2D(roi, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
        magnitude = np.abs(response)

        # split response into blocks to compute mean and mean absolute deviation for each block
        for row in range(0, roi.shape[0], block_size):
            for col in range(0, roi.shape[1], block_size):
                block = magnitude[row:row + block_size, col:col + block_size]
                mean = block.mean()
                deviation = np.abs(block - mean).mean()
                # append mean and deviation to the feature vector
                features.extend((mean, deviation))

    # return features as a 1D float32 array
    return np.asarray(features, dtype=np.float32)
