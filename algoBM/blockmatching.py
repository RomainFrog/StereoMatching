import numpy as np
from skimage import data
import matplotlib . pyplot as plt
from tqdm import tqdm
from numba import njit
import time
from scipy.ndimage import sobel

@njit
def SAD(block1, block2):
    return np.abs(block1 - block2).sum()

@njit
def SSD(block1, block2):
    return (np.abs(block1 - block2)**2).sum()

@njit
def ZSSD(patch1, patch2):
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    patch1_normalized = patch1 - mean1
    patch2_normalized = patch2 - mean2
    zssd_score = np.square(patch1_normalized - patch2_normalized).sum()
    return zssd_score

@njit
def ZNSSD(patch1, patch2):
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    patch1_normalized = (patch1 - mean1) / np.linalg.norm(patch1 - mean1)
    patch2_normalized = (patch2 - mean2) / np.linalg.norm(patch1 - mean1)
    nzssd_score = np.square(patch1_normalized - patch2_normalized).sum()
    return nzssd_score

@njit
def NCC(patch1, patch2):
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)

    patch1_normalized = patch1 - mean1
    patch2_normalized = patch2 - mean2

    numerator = np.sum(patch1_normalized * patch2_normalized)
    denominator = np.sqrt(np.sum(patch1_normalized ** 2) * np.sum(patch2_normalized ** 2))

    ncc_score = numerator / denominator

    return ncc_score



def GC(patch1, patch2):
    grad_patch1 = np.abs(sobel(patch1, axis=0)) + np.abs(sobel(patch1, axis=1))
    grad_patch2 = np.abs(sobel(patch2, axis=0)) + np.abs(sobel(patch2, axis=1))

    numerator = np.linalg.norm(grad_patch1 - grad_patch2)
    denominator = np.linalg.norm(grad_patch1 + grad_patch2)

    gfc_score = numerator / denominator

    return gfc_score





@njit
def block_matching(Iref, Isearch, N, maxdisp, similarity):
    disp = np.zeros_like(Iref)
    margin = N//2
    for i in np.arange(margin,Iref.shape[0] - margin):
        for j in np.arange(margin, Iref.shape[1] - margin):
            ref_block = Iref[i-margin:i+margin+1,j-margin:j+margin+1]
            min_sad= np.inf
            min_pos=0
            for x_dec in np.arange(0, maxdisp):
                new_x = j-x_dec
                if new_x >= margin:
                    search_block = Isearch[i-margin:i+margin+1, new_x-margin:new_x+margin+1]
                    sad=similarity(ref_block, search_block)
                    if sad < min_sad:
                        min_sad=sad
                        min_pos=x_dec
            disp[i,j]=min_pos
    return disp


@njit
def block_matching_NCC(Iref, Isearch, N, maxdisp, similarity):
    disp = np.zeros_like(Iref)
    margin = N//2
    for i in tqdm(np.arange(margin,Iref.shape[0] - margin)):
        for j in np.arange(margin, Iref.shape[1] - margin):
            ref_block = Iref[i-margin:i+margin+1,j-margin:j+margin+1]
            min_sad= -np.Inf
            min_pos=0
            for x_dec in np.arange(0, maxdisp):
                new_x = j-x_dec
                if new_x >= margin:
                    search_block = Isearch[i-margin:i+margin+1, new_x-margin:new_x+margin+1]
                    sad=similarity(ref_block, search_block)
                    if sad > min_sad:
                        min_sad=sad
                        min_pos=x_dec
            disp[i,j]=min_pos
    return disp




@njit
def mode_filter(img, N=5):
    filtered_img = np.zeros_like(img)
    pas = N//2
    for i in np.arange(pas,img.shape [0]-pas):
        for j in np.arange(pas, img.shape [1]-pas):
            block = img[i-pas:i+pas+1,j-pas:j+pas+1]
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


