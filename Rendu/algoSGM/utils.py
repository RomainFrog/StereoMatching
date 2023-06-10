#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File containing auxiliary functions used by stereomatch.py
Last modification: 06/03/2023 by Romain Froger
"""

import numpy as np


def compute_census(left, right, kernel):
    """
    Computes census bit strings for each pixel in the left and right images.

    Parameters
    -----
    left: np.ndarray
        Left grayscale image.
    right: np.ndarray
        Right grayscale image.
    kernel: int
        Kernel size for the census transform.

    Returns
    -----
    left_census: np.ndarray
        Left image with pixel intensities replaced with census bit strings.
    right_census: np.ndarray
        Right image with pixel intensities replaced with census bit strings.
    """
    k_height, k_width = kernel
    y_offset = k_height // 2
    x_offset = k_width // 2
    height, width = left.shape

    left_census = np.zeros_like(left, dtype=np.uint64)
    right_census = np.zeros_like(left, dtype=np.uint64)

    # offset is used since pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            apex = left[y, x]
            ref = np.full(shape=(k_height, k_width), fill_value=apex, dtype=np.int32)
            image = left[
                y - y_offset : y + y_offset + 1, x - x_offset : x + x_offset + 1
            ]
            diff = image - ref
            # If value is less than center value assign 1 otherwise assign 0
            left_mask_arr = np.where(diff < 0, 1, 0).flatten()
            # Convert census array to an integer by using bit shift operator
            left_mask = np.int32(
                left_mask_arr.dot(1 << np.arange(k_height * k_width)[::-1])
            )
            left_census[y, x] = left_mask

            apex = right[y, x]
            ref = np.full(shape=(k_height, k_width), fill_value=apex, dtype=np.int32)
            image = right[
                y - y_offset : y + y_offset + 1, x - x_offset : x + x_offset + 1
            ]
            diff = image - ref
            # If value is less than center value assign 1 otherwise assign 0
            right_mask_arr = np.where(diff < 0, 1, 0).flatten()
            # Convert census array to an integer by using bit shift operator
            right_mask = np.int32(
                right_mask_arr.dot(1 << np.arange(k_height * k_width)[::-1])
            )
            right_census[y, x] = right_mask

    return left_census, right_census


def add_mirror_padding(image, pad_size):
    """
    Given an image and a padding_size returns the image with a mirror padding on its edges

    Parameters
    -----
    image: np.ndarray
        Grayscale image to pad
    pad_size: int
        Size of the padding on each edge of the image

    Returns
    -----
    pad_img: np.ndarray
        Mirror-padded-image
    """
    # Calculate the new dimensions for the padded image
    height, width = image.shape[:2]
    new_height = height + 2 * pad_size
    new_width = width + 2 * pad_size
    # Create a new blank image with the new dimensions
    pad_img = np.zeros((new_height, new_width), dtype=np.uint8)
    # Copy the original image into the center of the padded image
    pad_img[pad_size:pad_size+height, pad_size:pad_size+width] = image
    # Mirror padding for the top and bottom borders
    pad_img[:pad_size, pad_size:pad_size+width] = np.flipud(image[:pad_size])
    pad_img[pad_size+height:, pad_size:pad_size+width] = np.flipud(image[height-pad_size:])
    # Mirror padding for the left and right borders
    pad_img[:, :pad_size] = np.fliplr(pad_img[:, pad_size:2*pad_size])
    pad_img[:, pad_size+width:] = np.fliplr(pad_img[:, width:width+pad_size])
    return pad_img
