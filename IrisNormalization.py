import cv2
import numpy as np

def iris_normalization(image, pupil, iris, rows=64, cols=512, angle_deg=0):
    """
    Maps the iris ring region to a rectangular grid of size (rows, cols)
    Ma et al., 2003 (3.2.2)
    """
    # unpack inputs
    pupil_x, pupil_y, pupil_radius = pupil
    iris_x, iris_y, iris_radius = iris

    # use number of cols to divide the iris ring into equal angles, adding angle_deg offset
    theta = 2 * np.pi * np.arange(cols) / cols + np.deg2rad(angle_deg)

    # calculate inner and outer boundary points for each angle
    inner_x = pupil_x + pupil_radius * np.cos(theta)
    inner_y = pupil_y - pupil_radius * np.sin(theta)
    outer_x = iris_x + iris_radius * np.cos(theta)
    outer_y = iris_y - iris_radius * np.sin(theta)

    # get evenly spaced fractions from the pupil boundary toward the iris boundary
    radial = np.arange(rows)[:, None] / rows

    # compute x and y coordinates at each fraction for every angle
    map_x = (inner_x + radial*(outer_x - inner_x)).astype(np.float32)
    map_y = (inner_y + radial * (outer_y - inner_y)).astype(np.float32)

    # construct the normalized image by remapping the original image to the new coordinates
    return cv2.remap(image.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
