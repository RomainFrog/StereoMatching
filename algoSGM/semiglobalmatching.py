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
    x_offset = int(csize[0] / 2)

    left_cost_volume = np.zeros(shape=(height, width, maxdisp), dtype=np.uint32)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)
    for d in range(0, maxdisp):
        rcensus[:, (x_offset + d):(width - x_offset)] = right[:, x_offset:(width - d - x_offset)]
        left_xor = np.int64(np.bitwise_xor(np.int64(left), rcensus))
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        left_cost_volume[:, :, d] = left_distance

    return left_cost_volume


def get_path_cost(slice, offset, P1, P2):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param parameters: structure containing parameters of the algorithm.
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = P1
    penalties[np.abs(disparities - disparities.T) > 1] = P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path

def get_indices(offset, dim, direction, height):
    """
    for the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
    :param offset: difference with the main diagonal of the cost volume.
    :param dim: number of elements along the path.
    :param direction: current aggregation direction.
    :param height: H of the cost volume.
    :return: arrays for the y (H dimension) and x (W dimension) indices.
    """
    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == (1, 1):
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(0 + i)
            else:
                y_indices.append(0 + i)
                x_indices.append(offset + i)

        if direction == (-1, 1):
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(0 + i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)

def aggregate_costs(cost_volume, P1, P2):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1

    effective_directions = [((1, 0),  (-1, 0)), ((1, 1), (-1, -1)), ((0, 1), (0, -1)), ((-1, 1), (1, -1))]

    aggregation_volume = np.zeros(shape=(height, width, disparities, len(effective_directions)*2), dtype=cost_volume.dtype)

    path_id = 0
    for direction in effective_directions:
        main_aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        if direction[0] == (0, 1):
            for x in range(0, width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = get_path_cost(south, 1, P1, P2)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, P1, P2), axis=0)

        if direction[0] == (1,0):
            for y in range(0, height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1, P1, P2)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, P1, P2), axis=0)

        if direction[0] == (1,1):
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, (1,1), None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1, P1, P2)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1, P1, P2)

        if direction[0] == (-1, 1):
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, (-1, 1), height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1, P1, P2)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1, P1, P2)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

    return aggregation_volume

def select_disparity(aggregation_volume):
    """
    last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    :return: disparity image.
    """
    volume = np.sum(aggregation_volume, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return disparity_map


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

def normalize(disp, max_disparity):
    return 255.0 * disp / max_disparity
