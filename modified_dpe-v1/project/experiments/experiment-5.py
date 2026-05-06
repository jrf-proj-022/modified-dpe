# Population Dynamics: Predator-Prey 

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
import itertools

# imports for DPE
from model.cy_utils import calculate_causal_history, generate_pattern_dictionary, calculate_contribution_analysis, normalized_weighted_entropy
from model.utils import extract_json_data, discrete

# imports for TE
from pyinform.transferentropy import transfer_entropy

# imports for GC
from statsmodels.tsa.stattools import grangercausalitytests

# imports for LZP
import ETCPy.ETC.CCMC.pairs as pairs

# Transfer entropy
def compute_te(x, y, k=1):
    try:
        te_xy = transfer_entropy(x, y, k=k)
        te_yx = transfer_entropy(y, x, k=k)

        if te_xy > te_yx:
            return 'X -> Y', te_xy, te_yx
        elif te_yx > te_xy:
            return 'Y -> X', te_xy, te_yx
        else:
            return 'Independence', te_xy, te_yx
    except:
        return 'Independence', np.nan, np.nan


# Granger causality
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

def compute_lzp_direction(x, y):
    lzp_xy, lzp_yx = compute_lzp_values(x, y)

    if lzp_xy < lzp_yx:
        return 'X -> Y', lzp_xy, lzp_yx
    elif lzp_xy > lzp_yx:
        return 'Y -> X', lzp_xy, lzp_yx
    else:
        return 'Independence', lzp_xy, lzp_yx


# DPE
def compute_dpe_direction(x, y):
    # 1. Calculate Causal History (Converting array to string if needed)
    x_str = "".join(map(str, x))
    y_str = "".join(map(str, y))
    
    Gx_y = calculate_causal_history(y_str, x_str)
    Gy_x = calculate_causal_history(x_str, y_str)
    
    # 2. Generate Pattern Dictionaries
    Px_y = generate_pattern_dictionary(Gx_y)
    Py_x = generate_pattern_dictionary(Gy_x)
    
    # 3. Contribution Analysis
    stats_xy = calculate_contribution_analysis(Px_y, x_str, y_str)
    stats_yx = calculate_contribution_analysis(Py_x, y_str, x_str)

    avg_h_xy, _ = normalized_weighted_entropy(stats_xy, len(x))
    avg_h_yx, _ = normalized_weighted_entropy(stats_yx, len(y))

    if avg_h_xy < avg_h_yx:
        return 'X -> Y', avg_h_xy, avg_h_yx
    elif avg_h_xy > avg_h_yx:
        return 'Y -> X', avg_h_xy, avg_h_yx
    else:
        return 'Independence', avg_h_xy, avg_h_yx

# extract csv data remove first 9 samples
def extract_csv(file_path):
    data = pd.read_csv(file_path)
    data = data.iloc[9:].reset_index(drop=True)
    return data

# plot timeseries
def plot_data(data, results_folder='results/experiment-4'):
    plt.figure(figsize=(9, 7))
    plt.plot(
    data['Didinium'],
    color='black',
    marker='^',
    linestyle='-',
    linewidth=1,
    markersize=4,
    label='Predator (Didinium)'
    )

    plt.plot(
        data['Paramecium'],
        color='red',
        marker='o',
        linestyle='--',
        linewidth=1,
        markersize=4,
        label='Prey (Paramecium)'
    )

    plt.xlabel("Time (days)", fontsize=21, fontweight='bold')
    plt.ylabel("Abundance (# ml)", fontsize=21, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(fontsize=22, loc='upper left')
    plt.grid(True, linestyle='-', alpha=0.3)

    plt.xlim(-1, 65)
    plt.ylim(0, 400)

    plt.tight_layout()
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    plt.savefig(f'{results_folder}/predator_prey.png', dpi=300)
    plt.show()

# run analysis
def predator_prey_analysis(file_path, results_folder='results/experiment-4'):
    data = extract_csv(file_path)

    results = []

    # predator-prey raw
    predator_raw = data['Didinium'].to_numpy()
    prey_raw = data['Paramecium'].to_numpy()

    # Discretization
    pred_bin = discrete(predator_raw, n_bins=2)
    prey_bin = discrete(prey_raw, n_bins=2)

    # Data conversion for TE 
    te_pred = [int(i) for i in pred_bin]
    te_prey = [int(j) for j in prey_bin]

    # Data conversion for LZP
    lzp_pred = [int(i) + 1 for i in pred_bin]
    lzp_prey = [int(j) + 1 for j in prey_bin]

    # DPE
    verdict_dpe, xy_dpe, yx_dpe = compute_dpe_direction(prey_bin, pred_bin)
    # print(verdict_dpe, xy_dpe, yx_dpe)
    results.append(
        {
            'model': 'DPE',
            'direction': verdict_dpe,
            'x_to_y': xy_dpe,
            'y_to_x': yx_dpe,
            'gap': round(abs(xy_dpe - yx_dpe), 4)
        }
    )

    # GC
    verdict_gc, xy_gc, yx_gc = compute_gc_direction(prey_raw, predator_raw)
    results.append(
        {
            'model': 'GC',
            'direction': verdict_gc,
            'x_to_y': xy_gc,
            'y_to_x': yx_gc,
            'gap': abs(xy_gc - yx_gc)
        }
    )

    # TE
    verdict_te, xy_te, yx_te = compute_te(te_prey, te_pred)
    results.append(
        {
            'model': 'TE',
            'direction': verdict_te,
            'x_to_y': xy_te,
            'y_to_x': yx_te,
            'gap': round(abs(xy_te - yx_te), 4)
        }
    )

    # LZP
    verdict_lzp, lzp_xy, lzp_yx = compute_lzp_direction(lzp_prey, lzp_pred)
    results.append(
        {
            'model': 'LZP',
            'direction': verdict_lzp,
            'x_to_y': lzp_xy,
            'y_to_x': lzp_yx,
            'gap': round(abs(lzp_xy - lzp_yx), 1)
        }
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{results_folder}/results.csv')

    return results_df

# file path
pop_dynamics_filepath = '../dataset/prey_predator_final.csv'

# read csv
pop_dynamics_df = extract_csv(pop_dynamics_filepath)

# plot and analysis
plot_data(pop_dynamics_df)
results_df = predator_prey_analysis(pop_dynamics_filepath)
print(results_df)