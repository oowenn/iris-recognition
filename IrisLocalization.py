import cv2
import numpy as np


def iris_localization(image, window_size=120):
    """
    Returns pupil, iris as two (center_x, center_y, radius) tuples in pixels.
    Ma et al., 2003 (3.2.1)
    """
    # Gaussian blur to reduce noise and help iris detection
    smooth = cv2.GaussianBlur(image, (5, 5), 0)
    half_window = window_size // 2

    # initial estimate for pupil center using the darkest row/col
    pupil_x = float(np.argmin(image.mean(axis=0)))
    pupil_y = float(np.argmin(image.mean(axis=1)))

    # 2 iterations of refining pupil center and radius using brightness thresholding and connected components
    for _ in range(2):
        # get top left corner of window around the pupil center
        x0 = max(0, int(round(pupil_x)) - half_window)
        y0 = max(0, int(round(pupil_y)) - half_window)
        
        # get window of image around the pupil center
        window = image[y0:y0 + window_size, x0:x0 + window_size]
        
        # separate the pupil by masking the darkest pixels in the window 
        _, mask = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # connect components to find the largest dark region, which should be the pupil
        _, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        pupil_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        # update pupil center and radius based on the connected component
        pupil_x, pupil_y = centroids[pupil_label] + (x0, y0)
        pupil_radius = np.sqrt(stats[pupil_label, cv2.CC_STAT_AREA] / np.pi)

    pupil = (float(pupil_x), float(pupil_y), float(pupil_radius)) # cast numpy values to float

    # find other circles in the image
    circles = cv2.HoughCircles(smooth, cv2.HOUGH_GRADIENT, dp=1, minDist=20,param1=50, param2=15, minRadius=80, maxRadius=130)

    # if no circles are found, use a default iris radius of 105 pixels
    if circles is None:
        iris = (pupil[0], pupil[1], 105.0)
    # if circles are found, choose the one closest to the pupil center
    else:
        candidates = circles[0] # get circle candidates
        distances = np.linalg.norm(candidates[:, :2] - pupil[:2], axis=1) # compute distances from pupil center
        iris = tuple(float(value) for value in candidates[np.argmin(distances)]) # cast numpy values to float

    # (pupil_x, pupil_y, pupil_radius), (iris_x, iris_y, iris_radius)
    return pupil, iris 
