#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch disparity computation from left to right image
Last modification: 06/03/2023 by Romain Froger
"""
import os
import sys
import cv2
from skimage import io
from blockmatching import *
sys.path.append(os.getcwd())
from algoBM.utils import compute_census


def main(Ig, Id, output_file):
    """
    Compute disparity from Ig --> Id with:
    - preprocessing (census)
    - processing (blockmatching)
    - postprocessnig (mode filter)

    Parameters
    ------
    Ig: np.ndarray
        Left grayscale image
    Id: np.ndarray
        Right grayscale image
    output_file: string
        Name of the output file
    """

    MAXDISP = 64
    N = 7
    N_MODE = 11
    KSIZE = (3,3)
    OCCL_THRESH = 56

    Ig = cv2.imread(Ig, cv2.IMREAD_GRAYSCALE)
    Id = cv2.imread(Id, cv2.IMREAD_GRAYSCALE)

    assert (
        Ig.shape == Id.shape
    ), f"Left and right images dimensions must match, got {Ig.shape=} and {Id.shape=}"

    # Preprocessing
    left_census, right_census = compute_census(Ig, Id, KSIZE)

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
