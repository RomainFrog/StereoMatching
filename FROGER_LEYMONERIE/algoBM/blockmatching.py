#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File containing all blockmatching functions used by stereomatch.py
Last modification: 06/03/2023 by Romain Froger
"""

import numpy as np
from numba import njit
from scipy.ndimage import sobel


@njit
def SAD(patch1, patch2):
    """Compute the Sum of Absolute Differences (SAD) between two blocks."""
    return np.sum(np.abs(patch1 - patch2))


@njit
def SSD(patch1, patch2):
    """Compute the Sum of Squared Differences (SSD) between two blocks."""
    return (np.abs(patch1 - patch2) ** 2).sum()


@njit
def ZSSD(patch1, patch2):
    """Compute the Zero-Mean Sum of Squared Differences (ZSSD) between two patches."""
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    patch1_normalized = patch1 - mean1
    patch2_normalized = patch2 - mean2
    zssd_score = np.square(patch1_normalized - patch2_normalized).sum()
    return zssd_score


@njit
def ZNSSD(patch1, patch2):
    """Compute the Zero-Mean Normalized Sum of Squared Differences (ZNSSD) between two patches."""
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    patch1_normalized = (patch1 - mean1) / np.linalg.norm(patch1 - mean1)
    patch2_normalized = (patch2 - mean2) / np.linalg.norm(patch1 - mean1)
    nzssd_score = np.square(patch1_normalized - patch2_normalized).sum()
    return nzssd_score


@njit
def NCC(patch1, patch2):
    """Compute the Normalized Cross-Correlation (NCC) between two patches."""
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)

    patch1_normalized = patch1 - mean1
    patch2_normalized = patch2 - mean2

    numerator = np.sum(patch1_normalized * patch2_normalized)
    denominator = np.sqrt(
        np.sum(patch1_normalized**2) * np.sum(patch2_normalized**2)
    )

    ncc_score = numerator / denominator

    return ncc_score


def GC(patch1, patch2):
    """Compute the Gradient Correlation (GC) between two patches."""
    grad_patch1 = np.abs(sobel(patch1, axis=0)) + np.abs(sobel(patch1, axis=1))
    grad_patch2 = np.abs(sobel(patch2, axis=0)) + np.abs(sobel(patch2, axis=1))

    numerator = np.linalg.norm(grad_patch1 - grad_patch2)
    denominator = np.linalg.norm(grad_patch1 + grad_patch2)

    gfc_score = numerator / denominator

    return gfc_score


@njit
def block_matching(Iref, Isearch, N, maxdisp, similarity):
    """
    Perform block matching using a specific similarity metric. Returns the disparity map.
    Disparity is computed from Iref to Isearch (left to right).
    Should not be used with similarity functino such as GC and NCC.

    Parameters
    -----
    Iref: np.ndarray
            Left image
    Isearch: np.ndarray
            Right image
    N: int
            Neigborhood size to compute dissimilarity
    maxdisp: int
            Maximum gap between Isearch pixel and Isearch matching pixel
    similarity: func(patch1, patch2) -> float
            Similarity function that computes the similarity of two given patches of pixels


    Returns
    -----
    disp: np.ndarray
            Disparity map (left to right)
    """
    disp = np.zeros_like(Iref)
    margin = N // 2
    for i in np.arange(margin, Iref.shape[0] - margin):
        for j in np.arange(margin, Iref.shape[1] - margin):
            ref_block = Iref[i - margin : i + margin + 1, j - margin : j + margin + 1]
            min_sad = np.inf
            min_pos = 0
            for x_dec in np.arange(0, maxdisp):
                new_x = j - x_dec
                if new_x >= margin:
                    search_block = Isearch[
                        i - margin : i + margin + 1, new_x - margin : new_x + margin + 1
                    ]
                    sad = similarity(ref_block, search_block)
                    if sad < min_sad:
                        min_sad = sad
                        min_pos = x_dec
            disp[i, j] = min_pos
    return disp


@njit
def block_matching_NCC(Iref, Isearch, N, maxdisp, similarity):
    """
    Perform block matching using a specific similarity metric. Returns the disparity map.
    Disparity is computed from Iref to Isearch (left to right)
    Must be used with NCC similarity function only.

    Parameters
    -----
    Iref: np.ndarray
            Left image
    Isearch: np.ndarray
            Right image
    N: int
            Neigborhood size to compute dissimilarity
    maxdisp: int
            Maximum gap between Isearch pixel and Isearch matching pixel
    similarity: func(patch1, patch2) -> float
            Similarity function that computes the similarity of two given patches of pixels


    Returns
    -----
    disp: np.ndarray
            Disparity map (left to right)
    """
    disp = np.zeros_like(Iref)
    margin = N // 2
    for i in np.arange(margin, Iref.shape[0] - margin):
        for j in np.arange(margin, Iref.shape[1] - margin):
            ref_block = Iref[i - margin : i + margin + 1, j - margin : j + margin + 1]
            min_sad = -np.Inf
            min_pos = 0
            for x_dec in np.arange(0, maxdisp):
                new_x = j - x_dec
                if new_x >= margin:
                    search_block = Isearch[
                        i - margin : i + margin + 1, new_x - margin : new_x + margin + 1
                    ]
                    sad = similarity(ref_block, search_block)
                    if sad > min_sad:
                        min_sad = sad
                        min_pos = x_dec
            disp[i, j] = min_pos
    return disp


# @njit
# def mode_filter(img, N=5):
    """
    Given a grayscale image returns the filtered image using a mode filter.
    Assigns the most frequent value of its neighborhood to the conresponding pixel.

    Parameters
    -----
    img: np.ndarray
            Grayscale image on which to apply the mode_filter
    N: int
            Size of the neighboorhood to consider


    Returns
    -----
    filtered_img: np.ndarray
            Filtered image

    """
    # h, w = img.shape
    # filtered_img = np.zeros_like(img)

    # margin = N // 2
    # # Parcourir chaque pixel de la carte de disparités
    # for i in range(h):
    #     for j in range(w):
    #         # Extraire la fenêtre 5x5 centrée sur le pixel actuel
    #         block = img[
    #             max(0, i - margin) : min(h, i + margin + 1),
    #             max(0, j - margin) : min(w, j + margin + 1),
    #         ]

    #         occ = {val: 0 for val in set(block.flatten())}
    #         for val in block.flatten():
    #             occ[val] += 1
    #         max_k = 0
    #         max_val = 0
    #         for key, value in occ.items():
    #             if value > max_val:
    #                 max_k = key
    #                 max_val = value
    #         filtered_img[i, j] = max_k

    # return filtered_img


def mode_filter(disparity_map, window_size=5):
    h, w = disparity_map.shape
    filtered_map = np.zeros_like(disparity_map)

    pas = window_size // 2
    # Parcourir chaque pixel de la carte de disparités
    for i in range(h):
        for j in range(w):
            # Extraire la fenêtre
            window = disparity_map[
                max(0, i - pas) : min(h, i + pas + 1),
                max(0, j - pas) : min(w, j + pas + 1),
            ]

            # Calculer les valeurs uniques et leurs occurrences dans la fenêtre
            unique_values, counts = np.unique(window, return_counts=True)

            # Trouver l'indice de la valeur la plus fréquente
            max_count_index = np.argmax(counts)

            # Affecter la valeur la plus fréquente au pixel considéré
            filtered_map[i, j] = unique_values[max_count_index]

    return filtered_map


@njit
def block_matching_mirror(Iref, Isearch, N, maxdisp, similarity):
    """
    Perform block matching using a specific similarity metric. Returns the disparity map.
    Disparity is computed from Iref to Isearch (left to right)
    Must be used with NCC similarity function only.
    This specific version of Block matching uses mirror-padded-images to easily apply
    blockmatching on the edges of the considered images.

    Parameters
    -----
    Iref: np.ndarray
            Left image
    Isearch: np.ndarray
            Right image
    N: int
            Neigborhood size to compute dissimilarity
    maxdisp: int
            Maximum gap between Isearch pixel and Isearch matching pixel
    similarity: func(patch1, patch2) -> float
            Similarity function that computes the similarity of two given patches of pixels


    Returns
    -----
    disp: np.ndarray
            Disparity map (left to right)
    """
    disp = np.zeros(shape=(Iref.shape[0] - 2 * N, Iref.shape[1] - 2 * N))
    print(disp.shape)
    margin = N // 2
    for i in np.arange(margin, Iref.shape[0] - margin):
        for j in np.arange(margin, Iref.shape[1] - margin):
            ref_block = Iref[i - margin : i + margin + 1, j - margin : j + margin + 1]
            min_sad = np.inf
            min_pos = 0
            for x_dec in np.arange(0, maxdisp):
                new_x = j - x_dec
                if new_x >= margin:
                    search_block = Isearch[
                        i - margin : i + margin + 1, new_x - margin : new_x + margin + 1
                    ]
                    sad = similarity(ref_block, search_block)
                    if sad < min_sad:
                        min_sad = sad
                        min_pos = x_dec
            disp[i - N, j - N] = min_pos
    return disp
