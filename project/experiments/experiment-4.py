# Delayed Bitflip

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
from model.cy_utils import run_causal_analysis
from model.utils import extract_json_data, discrete

# imports for TE
from transfer_entropy import compute_te, te_surrogate_threshold, te_identify_causality

# imports for GC
from granger_causality import compute_gc_values, compute_gc_direction

# imports for LZP
from lempel_ziv_penalty import compute_lzp_values, lzp_surrogate_threshold, lzp_identify_causality

# pattern considered for delayed bitflip
PATTERN = '1101'

# generate sequences 'X' that contains the pattern '1101'
def generate_main_pattern(length=1000):
    sequence = ""
    bits = ['0', '1', PATTERN]

    while len(sequence) < length:
        choice = random.choice(bits)
        sequence += choice

    return sequence[:length]

# generate sequence 'Y' based on pattern '1101' in sequence 'X' a delayed bitflip is added to sequence 'Y'
def generate_delayed_flip(main_sequence, delay):
    length = len(main_sequence)
    delayed_seq = ['0'] * length
    
    for i in range(len(PATTERN) - 1, length):
        if main_sequence[i - len(PATTERN) + 1 : i + 1] == PATTERN:
            target_idx = i + delay
            if target_idx < length:
                delayed_seq[target_idx] = '1'
                
    return "".join(delayed_seq)

# generate pairs of sequences ('X' and 'Y') equal to number of trials
def generate_n_sequences(trials=1000, length=1000):
    data = {delay: [] for delay in range(7)}

    for i in range(trials):
        main_seq = generate_main_pattern(length)

        for delay in range(7):
            delayed_seq = generate_delayed_flip(main_seq, delay)
            
            data[delay].append({
                'trial_id': i,
                'X': main_seq,
                'Y': delayed_seq
            })

    with open('../dataset/delayed-bitflip-data.json', 'w') as file:
        json.dump(data, file, indent=4)
        
    return data

def extract_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        return None
    print('Loading data...')
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        print('Load completed.')
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def delayed_bitflip_experiment(data, results_folder='results/experiment-3'):
    results = []

    for delay, trials in data.items():
        print(f'Analyzing Delay: {delay}...')

        trial_counter = 0

        for trial in trials:
            # Raw string sequences
            x_str, y_str = trial['X'], trial['Y']

            # Data conversion for DPE
            x_raw = np.array(list(x_str), dtype=np.int32)
            y_raw = np.array(list(y_str), dtype=np.int32)

            # Data conversion for TE and GC
            list_x = [i for i in x_raw]
            list_y = [j for j in y_raw]

            # Data conversion for lzp
            lzp_x = [i + 1 for i in x_raw]
            lzp_y = [j + 1 for j in y_raw]

            # DPE
            verdict_dpe, gap_dpe = run_causal_analysis(x_raw, y_raw)
            results.append(
                {
                    'delay': delay,
                    'trial': trial_counter,
                    'model': 'DPE',
                    'direction': verdict_dpe,
                    'gap': gap_dpe
                }
            )

            # GC
            # code updated with the original method from stats.tsa
            verdict_gc, xy_gc, yx_gc = compute_gc_direction(x_raw, y_raw)
            results.append(
                {
                'delay': delay,
                'trial': trial_counter,
                'model': 'GC',
                'direction': verdict_gc,
                'gap': abs(xy_gc - yx_gc)
                }
            )

            # TE
            verdict_te, gap_te = te_identify_causality(list_x, list_y)
            results.append(
                {
                    'delay': delay,
                    'trial': trial_counter,
                    'model': 'TE',
                    'direction': verdict_te,
                    'gap': gap_te
                }
            )

            # LZP
            verdict_lzp, gap_lzp = lzp_identify_causality(lzp_x, lzp_y)
            results.append(
                {
                    'delay': delay,
                    'trial': trial_counter,
                    'model': 'LZP',
                    'direction': verdict_lzp,
                    'gap': gap_lzp
                }
            )

            trial_counter += 1
            # if trial_counter == 5:
            #     break

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{results_folder}/experiment-3-all_results.csv')
    
    return results_df

# calculating metrics
def calculate_metrics(delayed_df: pd.DataFrame):

    delayed_df = delayed_df.loc[
        :, ~delayed_df.columns.str.contains(r'^Unnamed')
    ].copy()

    delayed_df['delay'] = delayed_df['delay'].astype(str)

    # only fill NaN in the direction column — don't touch numeric columns
    delayed_df['direction'] = delayed_df['direction'].fillna('Independence')

    accuracy     = []
    distribution = []

    models = delayed_df['model'].unique()
    delays = delayed_df['delay'].unique()

    for model, delay in itertools.product(models, delays):

        subset = delayed_df[
            (delayed_df['model'] == model) &
            (delayed_df['delay'] == delay)
        ]
        total_trials = len(subset)

        if total_trials == 0:
            continue

        counts = subset['direction'].value_counts()

        xy_count          = int(counts.get('X -> Y',       0))
        yx_count          = int(counts.get('Y -> X',       0))
        independent_count = int(counts.get('Independence', 0))

        accuracy.append({
            'delay':    delay,
            'model':    model,
            'accuracy': round(xy_count / total_trials, 4),
        })

        distribution.append({
            'delay':             delay,
            'model':             model,
            'total_trials':      total_trials,
            'xy_count':          xy_count,
            'yx_count':          yx_count,
            'independent_count': independent_count,
        })

    accuracy_df     = pd.DataFrame(accuracy)
    distribution_df = pd.DataFrame(distribution)
    return accuracy_df, distribution_df

# accuracy plot
def plot_accuracy(df, results_folder='results/experiment-3'):
    plt.figure(figsize=(9, 7))
    markers = ['o', 'v', 's', 'p']
    
    for i, model in enumerate(df['model'].unique()):
        subset = df[df['model'] == model].sort_values('delay')
        plt.plot(subset['delay'], subset['accuracy'], 
                 marker=markers[i % len(markers)], 
                 label=model, linewidth=2, markersize=12)
    
    fs = 26
    
    plt.xlabel('Delay', fontsize=fs-5, fontweight='bold')
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
        'xy_count': 'X -> Y', 
        'yx_count': 'Y -> X', 
        'independent_count': 'Independent'
    }

    for i, model in enumerate(models):
        subset = df[df['model'] == model].copy()
        
        subset = subset.rename(columns=rename_map)
        
        subset.set_index('delay')[['X -> Y', 'Y -> X', 'Independent']].plot(
            kind='bar', 
            ax=axes[i], 
            edgecolor='black',
            width=0.8
        )
        axes[i].set_title(model, fontsize=20)

        axes[i].set_ylabel('Count', fontsize=18)
        axes[i].set_xlabel('Delay', fontsize=18)
        axes[i].set_ylim(0, 1050)
        
        axes[i].tick_params(axis='both', labelsize=18)
        axes[i].tick_params(axis='x', rotation=0)
        
        axes[i].legend(fontsize=13, loc='best')
        
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        for container in axes[i].containers:
            axes[i].bar_label(container, padding=3, fontsize=12)
            
        axes[i].margins(y=0.2)

    plt.tight_layout()
    plt.savefig(f'{results_folder}/barplots.png')
    plt.show()

# generate data
generate_n_sequences(trials=1000, length=1000)

# extract data
# dealy_data_path = '../dataset/delayed-bitflip-data.json'
# delay_data = extract_data(dealy_data_path)

# # run analysis
# delay_summary_data = delayed_bitflip_experiment(delay_data)
# print(delay_summary_data)

# # analysis and plots
file_path = 'results/experiment-3/experiment-3-all_results.csv'
delayed_df = pd.read_csv(file_path)

accuracy_df, distribution_df = calculate_metrics(delayed_df)

accuracy_df.to_csv('results/experiment-3/experiment-3-accuracy.csv')
distribution_df.to_csv('results/experiment-3/experiment-3-proportions-count.csv')

plot_accuracy(accuracy_df)
bar_plots(distribution_df)