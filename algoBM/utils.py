import numpy as np
import time as t
import sys


def compute_census(left, right, kernel):
    """
    Calculate census bit strings for each pixel in the left and right images.
    Arguments:
        - left: left grayscale image.
        - right: right grayscale image.
        - kernel: kernel size for the census transform.

    Return: Left and right images with pixel intensities replaced with census bit strings.
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
