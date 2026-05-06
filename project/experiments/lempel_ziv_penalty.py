# Essential imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import array
import os
import json
import contextlib
import random

# import for LZP
import ETCPy.ETC.CCMC.pairs as pairs

# LZP
def compute_lzp_values(x, y):
    try:
        lzp_dict = pairs.LZ_causality(x, y)
        lzp_xy = lzp_dict['LZP_x_to_y'] # type: ignore
        lzp_yx = lzp_dict['LZP_y_to_x'] # type: ignore

        return lzp_xy, lzp_yx

    except Exception as e:
        print('LZP error:', e)
        return np.nan, np.nan

def lzp_surrogate_threshold(X, Y, n_surrogates=100, percentile=95):
    """
    Estimate the gap distribution under the null hypothesis of independence
    by randomly shuffling X and measuring the gap.
    """
    gaps = []
    
    # Convert string to list of characters for shuffling
    X_list = list(X)
    Y_list = list(Y)
    
    for _ in range(n_surrogates):
        # Shuffle both to destroy any causal structure
        X_shuffled = np.random.permutation(X_list)
        Y_shuffled = np.random.permutation(Y_list)
        
        # Recompute stats for shuffled pair
        lzp_xy, lzp_yx = compute_lzp_values(X_shuffled, Y_shuffled)
        
        gaps.append(abs(lzp_xy - lzp_yx))
    
    return np.percentile(gaps, percentile)

# LZP identify causality
def lzp_identify_causality(x, y):
    lzp_xy, lzp_yx = compute_lzp_values(x, y)
    gap = lzp_xy - lzp_yx
    abs_gap = abs(gap)

    threshold = lzp_surrogate_threshold(x, y, n_surrogates=100)

    if abs_gap < threshold:
        return 'Independence', abs_gap
    
    if gap < 0:
        return 'X -> Y', abs_gap
    else:
        return 'Y -> X', abs_gap