#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:14:24 2023

@author: thomas
"""

import sys
import cv2
from skimage import io
from semiglobalmatching import compute_costs, aggregate_costs, select_disparity, mode_filter, normalize
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

        P1 = 0
        P2 = 4
        maxdisp = 64
        csize = (7, 7)
        msize = 9

        left_image = cv2.imread(left_image, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_image, cv2.IMREAD_GRAYSCALE)

        assert (left_image.shape == right_image.shape
        ), f"Left and right images dimensions mismatch: {left_image.shape=} vs {right_image.shape=}"

        # Preprocessing
        left_census, right_census = compute_census(left_image, right_image, csize)

        # Processing
        left_cost_volume = compute_costs(left_census, right_census, csize, maxdisp)
        left_aggregation_volume = aggregate_costs(left_cost_volume, P1, P2)
        left_disparity_map = select_disparity(left_aggregation_volume)

        # Postprocessing
        left_disparity_map = mode_filter(left_disparity_map, msize)
        left_disparity_map = normalize(left_disparity_map, maxdisp)

        # Save to output_file
        cv2.imwrite(output_file, left_disparity_map)

if __name__ == "__main__":
    # arguments
    assert (
        len(sys.argv) == 4
    ), "Il faut 3 arguments : stereomatch.py im_gche.png im_dte.png disp_sortie.png"

    main(sys.argv[1], sys.argv[2], sys.argv[3])
