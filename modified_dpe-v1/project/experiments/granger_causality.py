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

# import for GC
from statsmodels.tsa.stattools import grangercausalitytests

# Old code for GC
# GC
def compute_gc_values(x, y, maxlag=1):
    try:
        data_xy = np.column_stack([x, y])
        data_yx = np.column_stack([y, x])

        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            res_xy = grangercausalitytests(data_xy, maxlag=maxlag)
            res_yx = grangercausalitytests(data_yx, maxlag=maxlag)

        # extract optimal p-value
        p_xy = res_xy[maxlag][0]['ssr_ftest'][1]
        p_yx = res_yx[maxlag][0]['ssr_ftest'][1]

        return p_xy, p_yx
    except Exception as e:
        print("GC error:", e)
        return 1.0, 1.0

def compute_gc_direction(x, y, maxlag=1, alpha=0.05):
    try:

        p_xy, p_yx = compute_gc_values(x, y, maxlag=maxlag)

        # p_xy, p_yx = compute_gc_values(x, y)

        if p_xy < alpha and p_yx >= alpha:
            return 'Y -> X', p_xy, p_yx
        elif p_yx < alpha and p_xy >= alpha:
            return 'X -> Y', p_xy, p_yx
        else:
            return 'Independence', p_xy, p_yx

    except:
        return 'Independence', np.nan, np.nan

# # Updated code for GC
# def compute_gc_values(x, y, maxlag=1):
#     try:
#         data_xy = np.column_stack([x, y])
#         data_yx = np.column_stack([y, x])

#         with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
#             res_xy = grangercausalitytests(data_xy, maxlag=maxlag)
#             res_yx = grangercausalitytests(data_yx, maxlag=maxlag)

#         # extract optimal p-value
#         p_xy = res_xy[maxlag][0]['ssr_ftest'][1]
#         p_yx = res_yx[maxlag][0]['ssr_ftest'][1]

#         return p_xy, p_yx
#     except Exception as e:
#         print("GC error:", e)
#         return 1.0, 1.0

# def gc_surrogate_threshold(X, Y, n_surrogates=100, percentile=95):
#     """
#     Estimate the gap distribution under the null hypothesis of independence
#     by randomly shuffling X and measuring the gap.
#     """
#     gaps = []
    
#     # Convert string to list of characters for shuffling
#     X_list = list(X)
#     Y_list = list(Y)
    
#     for _ in range(n_surrogates):
#         # Shuffle both to destroy any causal structure
#         X_shuffled = np.random.permutation(X_list)
#         Y_shuffled = np.random.permutation(Y_list)
        
#         # Recompute stats for shuffled pair
#         gc_xy, gc_yx = compute_gc_values(X_shuffled, Y_shuffled)
        
#         gaps.append(abs(gc_xy - gc_yx))
    
#     return np.percentile(gaps, percentile)

# # GC identify causality
# def gc_identify_causality(x, y, max_lag=1):
#     gc_xy, gc_yx = compute_gc_values(x, y, max_lag)
#     gap = gc_xy - gc_yx
#     abs_gap = abs(gap)

#     threshold = gc_surrogate_threshold(x, y, n_surrogates=100)

#     if abs_gap < threshold:
#         return 'Independence', abs_gap
    
#     # if the p value from X to Y is less than p value from Y to X, then Y -> X is the predicted direction as p value from Y to X is larger rejecting H_0
#     if gap < 0 :
#         return 'Y -> X', abs_gap
#     elif gap > 0:
#         return 'X -> Y', abs_gap
#     else:
#         return 'Independence', abs_gap
