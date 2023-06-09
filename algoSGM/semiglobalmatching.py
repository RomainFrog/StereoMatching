#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:40:56 2023

@author: thomas
"""

import numpy as np
from numba import njit

def compute_costs(left, right, csize, maxdisp):
    """
    Computes the matching cost volume between two stereo images.
    Args:
        left (ndarray): Left image (ndarray format).
        right (ndarray): Right image (ndarray format).
        csize (int): Window size for Census computation.
        maxdisp (int): Maximum disparity value.

    Returns:
        ndarray: Left cost volume (shape: [height, width, maxdisp]).
    """

    height = left.shape[0]
    width = left.shape[1]
    x_offset = csize // 2

    left_cost_volume = np.zeros(shape=(height, width, maxdisp), dtype=np.uint32)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)
    for d in range(0, maxdisp):
        rcensus[:, (x_offset + d):(width - x_offset)] = right[:, x_offset:(width - d - x_offset)]
        left_xor = np.int64(np.bitwise_xor(np.int64(left), rcensus))
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while np.any(left_xor != 0):
            left_xor &= (left_xor - 1)
            np.add(left_distance, 1, where=(left_xor != 0), out=left_distance)

        left_cost_volume[:, :, d] = left_distance

    return left_cost_volume


def compute_path_cost(slice, offset, p1, p2):
    """"
    Computes the minimum costs in a D x M slice during the aggregation step.
    Args:
        slice (ndarray): M x D array from the cost volume.
        offset (int): Ignore the pixels on the border.
        p1 (float): Parameter value for P1.
        p2 (float): Parameter value for P2.

    Returns:
        ndarray: M x D array of the minimum costs for a given slice in a given direction.
    """
    disparity_dim = slice.shape[1]

    disparities = np.arange(disparity_dim)
    penalties = np.zeros((disparity_dim, disparity_dim), dtype=slice.dtype)
    disparities_diff = np.abs(disparities - disparities[:, np.newaxis])
    penalties[disparities_diff == 1] = p1
    penalties[disparities_diff > 1] = p2

    minimum_cost_path = np.zeros_like(slice)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, slice.shape[0]):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, disparity_dim).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)

    return minimum_cost_path


def get_indices(offset, dim, direction, height):
    """
    Computes the indices for the current slice in diagonal directions (SE, SW, NW, NE).
    Args:
        offset (int): Difference with the main diagonal of the cost volume.
        dim (int): Number of elements along the path.
        direction (tuple): Current aggregation direction.
        height (int): Height of the cost volume.

    Returns:
        ndarray: Array of y (H dimension) indices.
        ndarray: Array of x (W dimension) indices.
    """
    y_indices = []
    x_indices = []

    if direction == (1, 1):
        if offset < 0:
            y_indices = np.arange(-offset, -offset + dim)
            x_indices = np.arange(0, dim)
        else:
            y_indices = np.arange(0, dim)
            x_indices = np.arange(offset, offset + dim)

    if direction == (-1, 1):
        if offset < 0:
            y_indices = np.arange(height + offset, height + offset - dim, -1)
            x_indices = np.arange(0, dim)
        else:
            y_indices = np.arange(height - 1, height - dim - 1, -1)
            x_indices = np.arange(offset, offset + dim)

    return y_indices, x_indices

def aggregate_costs(cost_volume, p1, p2):
    """
    Aggregates matching costs for multiple directions in the SGM algorithm.
    Args:
        cost_volume (ndarray): Array containing the matching costs.
        p1 (float): Parameter value for P1.
        p2 (float): Parameter value for P2.

    Returns:
        ndarray: Aggregated cost volume (shape: [height, width, disparities, num_directions * 2]).
    """
    height, width, disparities = cost_volume.shape
    start = -(height - 1)
    end = width - 1

    directions = [((1, 0), (-1, 0)), ((1, 1), (-1, -1)), ((0, 1), (0, -1)), ((-1, 1), (1, -1))]

    num_directions = len(directions)
    aggregation_volume = np.zeros((height, width, disparities, num_directions * 2), dtype=cost_volume.dtype)

    path_id = 0
    for direction in directions:
        main_aggregation = np.zeros((height, width, disparities), dtype=cost_volume.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        dx, dy = direction[0]
        opposite_dx, opposite_dy = direction[1]

        if dx == 0 and dy == 1:
            for x in range(width):
                south = cost_volume[:, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = compute_path_cost(south, 1, p1, p2)
                opposite_aggregation[:, x, :] = np.flip(compute_path_cost(north, 1, p1, p2), axis=0)

        if dx == 1 and dy == 0:
            for y in range(height):
                east = cost_volume[y, :, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = compute_path_cost(east, 1, p1, p2)
                opposite_aggregation[y, :, :] = np.flip(compute_path_cost(west, 1, p1, p2), axis=0)

        if dx == 1 and dy == 1:
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, (1, 1), None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = compute_path_cost(south_east, 1, p1, p2)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = compute_path_cost(north_west, 1, p1, p2)

        if dx == -1 and dy == 1:
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, (-1, 1), height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = compute_path_cost(south_west, 1, p1, p2)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = compute_path_cost(north_east, 1, p1, p2)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id += 2

    return aggregation_volume

@njit
def mode_filter(img, N=5):
    h, w = img.shape
    filtered_img = np.zeros_like(img)

    margin = N//2
    # Parcourir chaque pixel de la carte de disparités
    for i in range(h):
        for j in range(w):
            # Extraire la fenêtre 5x5 centrée sur le pixel actuel
            block = img[max(0, i - margin):min(h, i + margin+1), max(0, j - margin):min(w, j + margin+1)]

            occ = {val:0 for val in set(block.flatten())}
            for val in block.flatten():
                occ[val]+=1

            max_k=0
            max_val=0
            for key, value in occ.items():
                if value > max_val:
                    max_k=key
                    max_val=value
            filtered_img[i,j]=max_k

    return filtered_img
