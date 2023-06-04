#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch disparity computation from left to right image
Last modification: 06/03/2023 by Romain Froger
"""

import sys
import cv2
import numpy as np
from skimage import io
from blockmatching import block_matching, ZSSD, mode_filter
from utils import compute_census


def main(left_image, right_image, output_file):
    """
    Compute disparity from left_image --> right_image with:
    - preprocessing (census)
    - processing (blockmatching)
    - postprocessnig (mode filter)

    Parameters
    ------
    left_image: np.ndarray
        Left grayscale image
    right_image: np.ndarray
        Right grayscale image
    output_file: string
        Name of the output file
    """

    MAXDISP = 64
    N = 7
    N_MODE = 11
    KSIZE = (3,3)
    OCCL_THRESH = 56

    left_image = cv2.imread(left_image, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image, cv2.IMREAD_GRAYSCALE)

    assert (
        left_image.shape == right_image.shape
    ), f"Left and right images dimensions mismatch: {left_image.shape=} vs {right_image.shape=}"

    # Preprocessing
    left_census, right_census = compute_census(left_image, right_image, KSIZE)

    # Processing
    disp = block_matching(left_census, right_census, N, MAXDISP, ZSSD)
    disp = disp.astype(np.uint8)

    # Postprocessing
    disp = mode_filter(disp, N_MODE)
    disp = np.interp(disp, (0, MAXDISP), (0, 255)).astype(np.uint8)
    disp = np.where(disp <= OCCL_THRESH, 0, disp)

    # Save to output_file
    io.imsave(output_file, disp)


if __name__ == "__main__":
    # arguments
    assert (
        len(sys.argv) == 4
    ), "Il faut 3 arguments : stereomatch.py im_gche.png im_dte.png disp_sortie.png"

    main(sys.argv[1], sys.argv[2], sys.argv[3])
