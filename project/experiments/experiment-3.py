# 1 D Coupled Skewtent Maps

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
from model.cy_utils import run_causal_analysis, extract_json
from model.utils import discrete

# imports for TE
from transfer_entropy import compute_te, te_surrogate_threshold, te_identify_causality

# imports for GC
from granger_causality import compute_gc_values, compute_gc_direction

# imports for LZP
from lempel_ziv_penalty import compute_lzp_values, lzp_surrogate_threshold, lzp_identify_causality

# run experiment for skewtent maps
def skewtent_experiment(data, results_folder='results/experiment-3'):
    results = []

    for eta, content in data.items():
        eta_val = round(float(eta), 2)

        print(r'Processing $\eta$ ', eta_val)

        driver_trials = content['Y']
        response_trials = content['X']

        number_of_trials = len(driver_trials)
        print(number_of_trials)

        for i in range(number_of_trials):

            # Raw Sequences
            driver_raw = np.array(driver_trials[i], dtype=np.float32)
            response_raw = np.array(response_trials[i], dtype=np.float32)

            # Discretization
            bin_driver = discrete(driver_raw, n_bins=2)
            bin_response = discrete(response_raw, n_bins=2)

            # Data conversion for TE
            list_driver = [int(i) for i in bin_driver]
            list_response = [int(j) for j in bin_response]

            # Data conversion for LZP
            lzp_driver = [int(i) + 1 for i in bin_driver]
            lzp_response = [int(j) + 1 for j in bin_response]

            # DPE
            verdict_dpe, gap_dpe = run_causal_analysis(bin_response, bin_driver)

            results.append(
                {
                'eta': eta_val,
                'trial': i,
                'model': 'DPE',
                'direction': verdict_dpe,
                'gap': gap_dpe
                }
            )

            # GC
            # code updated with the original method from stats.tsa
            verdict_gc, xy_gc, yx_gc = compute_gc_direction(response_raw, driver_raw)
            results.append(
                {
                'eta': eta_val,
                'trial': i,
                'model': 'GC',
                'direction': verdict_gc,
                'gap': abs(xy_gc - yx_gc)
                }
            )

            # TE
            verdict_te, gap_te = te_identify_causality(list_response, list_driver)
            results.append(
                {
                'eta': eta_val,
                'trial': i,
                'model': 'TE',
                'direction': verdict_te,
                'gap': gap_te
                }
            )

            # LZP
            verdict_lzp, gap_lzp = lzp_identify_causality(lzp_response, lzp_driver)
            results.append(
                {
                'eta': eta_val,
                'trial': i,
                'model': 'LZP',
                'direction': verdict_lzp,
                'gap': gap_lzp
                }
            )

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{results_folder}/experiment-3-all_results.csv')
    
    return results_df

# calculating metrics
def calculate_metrics(df: pd.DataFrame):

    df = df.loc[
        :, ~df.columns.str.contains(r'^Unnamed')
    ].copy()

    df['eta'] = df['eta'].astype(str)

    # only fill NaN in the direction column — don't touch numeric columns
    df['direction'] = df['direction'].fillna('Independence')

    accuracy     = []
    distribution = []

    models = df['model'].unique()
    etas = df['eta'].unique()

    for model, eta in itertools.product(models, etas):

        subset = df[
            (df['model'] == model) &
            (df['eta'] == eta)
        ]
        total_trials = len(subset)

        if total_trials == 0:
            continue

        counts = subset['direction'].value_counts()

        xy_count          = int(counts.get('X -> Y',       0))
        yx_count          = int(counts.get('Y -> X',       0))
        independent_count = int(counts.get('Independence', 0))

        accuracy.append({
            'eta':    eta,
            'model':    model,
            'accuracy': round(yx_count / total_trials, 4) if eta != '0.0' else round(independent_count / total_trials, 4)
        })

        distribution.append({
            'eta':             eta,
            'model':             model,
            'total_trials':      total_trials,
            'xy_count':          xy_count,
            'yx_count':          yx_count,
            'independent_count': independent_count
        })

    accuracy_df     = pd.DataFrame(accuracy)
    distribution_df = pd.DataFrame(distribution)
    return accuracy_df, distribution_df

# accuracy plot
def plot_accuracy(df, results_folder='results/experiment-3'):
    plt.figure(figsize=(9, 7))
    markers = ['o', 'v', 's', 'p']
    models = ['DPE', 'GC', 'TC', 'LZP']
    for i, model in enumerate(df['model'].unique()):
        subset = df[df['model'] == model].sort_values('eta')
        plt.plot(subset['eta'], subset['accuracy'], 
                 marker=markers[i % len(markers)], 
                 label=model, linewidth=2, markersize=12)
    
    fs = 26
    
    plt.xlabel(r'Coupling Strength ($\eta$)', fontsize=fs-5, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=fs-5, fontweight='bold')

    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fs)

    leg = plt.legend(
        bbox_to_anchor=(0.98, 0.80), 
        loc='center right',
        frameon=True,
        framealpha=0.8,
        fontsize=fs-5
    )
    

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_folder}/accuracy-plot.png')
    plt.show()

def bar_plots(df, results_folder='results/experiment-3'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    models = df['model'].unique()
    
    rename_map = {
        'xy_count': r'$R_t$ -> $D_t$', 
        'yx_count': r'$D_t$ -> $R_t$', 
        'independent_count': 'Independence'
    }

    for i, model in enumerate(models):
        subset = df[df['model'] == model].copy()
        
        subset = subset.rename(columns=rename_map)
        
        subset.set_index('eta')[[r'$D_t$ -> $R_t$', r'$R_t$ -> $D_t$', 'Independence']].plot(
            kind='bar', 
            ax=axes[i], 
            edgecolor='black',
            width=0.8
        )
        axes[i].set_title(model, fontsize=20)

        axes[i].set_ylabel('Count', fontsize=18)
        axes[i].set_xlabel(r'Coupling Strength ($\eta$)', fontsize=18)
        
        axes[i].tick_params(axis='both', labelsize=18)
        axes[i].tick_params(axis='x', rotation=0)
        
        axes[i].legend(fontsize=13, loc='upper right')
        
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        for container in axes[i].containers:
            axes[i].bar_label(container, padding=3, fontsize=12)
            
        axes[i].margins(y=0.2)

    plt.tight_layout()
    plt.savefig(f'{results_folder}/barplots.png')
    plt.show()

# extract json data
# skewtent_path = '../dataset/skewtent_maps_data.json'
# skewtent_data = extract_json(skewtent_path)

# # run analysis
# summary_df = skewtent_experiment(skewtent_data)
# print(summary_df)

# analysis and plots
summary_df = pd.read_csv('results/experiment-3/experiment-3-all_results.csv')
acc_df, dist_df = calculate_metrics(summary_df)

acc_df.to_csv('results/experiment-3/experiment-3-accuracy.csv')
dist_df.to_csv('results/experiment-3/experiment-3-proportions-count.csv')

plot_accuracy(acc_df)
bar_plots(dist_df)
