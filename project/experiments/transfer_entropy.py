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

# import for TE
from pyinform.transferentropy import transfer_entropy

# ==============================
# Old function
# TE
# def compute_te(x, y, k=1):
#     try:
#         te_xy = transfer_entropy(x, y, k=k)
#         te_yx = transfer_entropy(y, x, k=k)

#         if te_xy > te_yx:
#             return 'X -> Y', te_xy, te_yx
#         elif te_yx > te_xy:
#             return 'Y -> X', te_xy, te_yx
#         else:
#             return 'Independent', te_xy, te_yx
#     except:
#         return 'Independent', np.nan, np.nan


# Updated Transfer Entropy
def compute_te(x, y, k=1):
    try:
        te_xy = transfer_entropy(x, y, k=k)
        te_yx = transfer_entropy(y, x, k=k)

        return te_xy, te_yx
    except Exception as e:
        print("TE error:", e)
        return np.nan, np.nan

def te_surrogate_threshold(X, Y, n_surrogates=100, percentile=95):
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
        te_xy, te_yx = compute_te(X_shuffled, Y_shuffled)
        
        gaps.append(abs(te_xy - te_yx))
    
    return np.percentile(gaps, percentile)

# TE identify causality
def te_identify_causality(x, y, k=1):
    te_xy, te_yx = compute_te(x, y)
    gap = te_xy - te_yx
    abs_gap = abs(gap)

    threshold = te_surrogate_threshold(x, y, n_surrogates=100)

    if abs_gap < threshold:
        return 'Independence', abs_gap
    
    if gap < 0:
        return 'Y -> X', abs_gap
    else:
        return 'X -> Y', abs_gap