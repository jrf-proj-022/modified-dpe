# # Random-Delay (Stochastic) Bitflip Benchmark — Spread Sweep

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import json
# import random
# import itertools

# from model.cy_utils import (
#     run_causal_analysis,
#     calculate_causal_history,
#     generate_pattern_dictionary,
#     calculate_contribution_analysis,
#     normalized_weighted_entropy,
# )
# from model.utils import extract_json_data, discrete

# # imports for TE
# from transfer_entropy import compute_te, te_surrogate_threshold, te_identify_causality

# # imports for GC
# from granger_causality import compute_gc_values, compute_gc_direction

# # imports for LZP
# from lempel_ziv_penalty import compute_lzp_values, lzp_surrogate_threshold, lzp_identify_causality

# # imports for CCM (raw)
# from causal_ccm.causal_ccm import ccm

# from ccm_model import ccm_identify_causality

# from etc_penalty import compute_etcp_values, etcp_identify_causality
# from etc_efficacy import compute_etce_values, etce_identify_causality

# PATTERN = '1101'

# def dpe_raw_verdict(x_str, y_str):
#     Gx_y = calculate_causal_history(y_str, x_str)
#     Gy_x = calculate_causal_history(x_str, y_str)

#     Px_y = generate_pattern_dictionary(Gx_y)
#     Py_x = generate_pattern_dictionary(Gy_x)

#     stats_xy = calculate_contribution_analysis(Px_y, x_str, y_str)
#     stats_yx = calculate_contribution_analysis(Py_x, y_str, x_str)

#     avg_h_xy, _ = normalized_weighted_entropy(stats_xy, len(x_str))
#     avg_h_yx, _ = normalized_weighted_entropy(stats_yx, len(y_str))

#     gap = avg_h_xy - avg_h_yx
#     if gap < 0:
#         return 'X -> Y', abs(gap)
#     elif gap > 0:
#         return 'Y -> X', abs(gap)
#     else:
#         return 'Independence', abs(gap)

# def generate_main_pattern(length=1000):
#     sequence = ""
#     bits = ['0', '1', PATTERN]

#     while len(sequence) < length:
#         choice = random.choice(bits)
#         sequence += choice

#     return sequence[:length]


# def generate_random_delayed_flip(main_sequence, delay_choices=range(7), rng=None):
#     if rng is None:
#         rng = random
#     length = len(main_sequence)
#     delayed_seq = ['0'] * length

#     i = len(PATTERN) - 1
#     while i < length:
#         if main_sequence[i - len(PATTERN) + 1 : i + 1] == PATTERN:
#             delay = rng.choice(delay_choices)
#             target_idx = i + delay
#             if target_idx < length:
#                 delayed_seq[target_idx] = '1'
#             i += len(PATTERN)  
#         else:
#             i += 1

#     return "".join(delayed_seq)


# def generate_random_delay_sweep(trials=1000, length=1000, max_spread=6,
#                                  save_dir='../dataset/random-delay-sweep'):
#     os.makedirs(save_dir, exist_ok=True)
#     all_data = {}

#     for spread in range(max_spread + 1):
#         delay_choices = range(spread + 1)
#         data = []

#         for i in range(trials):
#             main_seq = generate_main_pattern(length)
#             delayed_seq = generate_random_delayed_flip(main_seq, delay_choices)
#             data.append({
#                 'trial_id': i,
#                 'X': main_seq,
#                 'Y': delayed_seq,
#                 'spread': spread
#             })

#         with open(f'{save_dir}/spread-{spread}.json', 'w') as file:
#             json.dump(data, file, indent=4)

#         all_data[spread] = data
#         print(f"Generated spread={spread} (delay in 0..{spread}), {trials} trials.")

#     return all_data


# def extract_sweep_data(save_dir='../dataset/random-delay-sweep', max_spread=6):
#     all_data = {}
#     for spread in range(max_spread + 1):
#         file_path = f'{save_dir}/spread-{spread}.json'
#         if not os.path.exists(file_path):
#             print(f"Error: The file at {file_path} does not exist.")
#             continue
#         with open(file_path, 'r') as file:
#             all_data[spread] = json.load(file)
#     return all_data


# def run_all_methods_on_trials(data, results_folder='results/experiment-1-random-delay-sweep'):
#     results = []
#     trial_counter = 0

#     for trial in data:
#         x_str, y_str = trial['X'], trial['Y']

#         x_raw = np.array(list(x_str), dtype=np.int32)
#         y_raw = np.array(list(y_str), dtype=np.int32)

#         list_x = [i for i in x_raw]
#         list_y = [j for j in y_raw]

#         lzp_x = [i + 1 for i in x_raw]
#         lzp_y = [j + 1 for j in y_raw]

#         # DPE (run_causal_analysis -- this IS surrogate-tested internally, confirmed
#         verdict_dpe, gap_dpe = run_causal_analysis(x_str, y_str)
#         results.append({'trial': trial_counter, 'model': 'DPE',
#                          'direction': verdict_dpe, 'gap': gap_dpe})

#         # DPE_raw (same underlying cy_utils primitives, no surrogate threshold --
#         verdict_dpe_raw, gap_dpe_raw = dpe_raw_verdict(x_str, y_str)
#         results.append({'trial': trial_counter, 'model': 'DPE_raw',
#                          'direction': verdict_dpe_raw, 'gap': gap_dpe_raw})

#         # GC (existing, alpha-based significance test -- already a form of
#         verdict_gc, xy_gc, yx_gc = compute_gc_direction(x_raw, y_raw)
#         results.append({'trial': trial_counter, 'model': 'GC',
#                          'direction': verdict_gc, 'gap': abs(xy_gc - yx_gc)})

#         # GC_raw (no alpha threshold -- just whichever p-value is smaller wins;
#         p_xy_gc, p_yx_gc = compute_gc_values(x_raw, y_raw)
#         if np.isnan(p_xy_gc) or np.isnan(p_yx_gc) or p_xy_gc == p_yx_gc:
#             verdict_gc_raw, gap_gc_raw = 'Independence', 0.0
#         elif p_xy_gc < p_yx_gc:
#             # p_xy tests "does Y granger-cause X" (see compute_gc_direction convention)
#             verdict_gc_raw, gap_gc_raw = 'Y -> X', abs(p_xy_gc - p_yx_gc)
#         else:
#             verdict_gc_raw, gap_gc_raw = 'X -> Y', abs(p_xy_gc - p_yx_gc)
#         results.append({'trial': trial_counter, 'model': 'GC_raw',
#                          'direction': verdict_gc_raw, 'gap': gap_gc_raw})

#         # TE (surrogate-tested, unchanged)
#         verdict_te, gap_te = te_identify_causality(list_x, list_y)
#         results.append({'trial': trial_counter, 'model': 'TE',
#                          'direction': verdict_te, 'gap': gap_te})

#         # TE_raw (raw comparison, no surrogate threshold)
#         te_xy, te_yx = compute_te(list_x, list_y)
#         if np.isnan(te_xy) or np.isnan(te_yx) or te_xy == te_yx:
#             verdict_te_raw, gap_te_raw = 'Independence', 0.0
#         elif te_xy < te_yx:
#             verdict_te_raw, gap_te_raw = 'Y -> X', abs(te_xy - te_yx)
#         else:
#             verdict_te_raw, gap_te_raw = 'X -> Y', abs(te_xy - te_yx)
#         results.append({'trial': trial_counter, 'model': 'TE_raw',
#                          'direction': verdict_te_raw, 'gap': gap_te_raw})

#         # LZP (surrogate-tested, unchanged)
#         verdict_lzp, gap_lzp = lzp_identify_causality(lzp_x, lzp_y)
#         results.append({'trial': trial_counter, 'model': 'LZP',
#                          'direction': verdict_lzp, 'gap': gap_lzp})

#         # LZP_raw (raw comparison, no surrogate threshold)
#         lzp_xy, lzp_yx = compute_lzp_values(lzp_x, lzp_y)
#         if np.isnan(lzp_xy) or np.isnan(lzp_yx) or lzp_xy == lzp_yx:
#             verdict_lzp_raw, gap_lzp_raw = 'Independence', 0.0
#         elif lzp_xy < lzp_yx:
#             verdict_lzp_raw, gap_lzp_raw = 'X -> Y', abs(lzp_xy - lzp_yx)
#         else:
#             verdict_lzp_raw, gap_lzp_raw = 'Y -> X', abs(lzp_xy - lzp_yx)
#         results.append({'trial': trial_counter, 'model': 'LZP_raw',
#                          'direction': verdict_lzp_raw, 'gap': gap_lzp_raw})

#         # CCM_raw (raw rho comparison, no significance test -- renamed from
#         try:
#             ccm_obj = ccm(X=x_raw, Y=y_raw, E=2, tau=1)
#             x_to_y_rho, y_to_x_rho = ccm_obj.causality()

#             if x_to_y_rho > y_to_x_rho:
#                 verdict_ccm = 'Y -> X'
#             elif y_to_x_rho > x_to_y_rho:
#                 verdict_ccm = 'X -> Y'
#             else:
#                 verdict_ccm = 'Independence'

#             gap_ccm = abs(x_to_y_rho - y_to_x_rho)
#         except Exception:
#             verdict_ccm = 'Independence'
#             gap_ccm = 0.0

#         results.append({'trial': trial_counter, 'model': 'CCM_raw',
#                          'direction': verdict_ccm, 'gap': gap_ccm})

#         # CCM_surrogate (surrogate-tested significance version from ccm_model.py)
#         verdict_ccm_surr, gap_ccm_surr = ccm_identify_causality(
#             x_raw, y_raw, E=2, tau=1, n_surrogates=100, percentile=95
#         )
#         results.append({'trial': trial_counter, 'model': 'CCM_surrogate',
#                          'direction': verdict_ccm_surr, 'gap': gap_ccm_surr})

#         # ETCP (surrogate-tested, unchanged)
#         verdict_etcp, gap_etcp = etcp_identify_causality(lzp_x, lzp_y)
#         results.append({'trial': trial_counter, 'model': 'ETCP',
#                          'direction': verdict_etcp, 'gap': gap_etcp})

#         # ETCP_raw (raw comparison, no surrogate threshold)
#         etcp_xy, etcp_yx = compute_etcp_values(lzp_x, lzp_y)
#         if np.isnan(etcp_xy) or np.isnan(etcp_yx) or etcp_xy == etcp_yx:
#             verdict_etcp_raw, gap_etcp_raw = 'Independence', 0.0
#         elif etcp_xy < etcp_yx:
#             verdict_etcp_raw, gap_etcp_raw = 'X -> Y', abs(etcp_xy - etcp_yx)
#         else:
#             verdict_etcp_raw, gap_etcp_raw = 'Y -> X', abs(etcp_xy - etcp_yx)
#         results.append({'trial': trial_counter, 'model': 'ETCP_raw',
#                          'direction': verdict_etcp_raw, 'gap': gap_etcp_raw})

#         # ETCE (surrogate-tested, unchanged)
#         verdict_etce, gap_etce = etce_identify_causality(lzp_x, lzp_y)
#         results.append({'trial': trial_counter, 'model': 'ETCE',
#                          'direction': verdict_etce, 'gap': gap_etce})

#         # ETCE_raw (raw comparison, no surrogate threshold)
#         etce_xy, etce_yx = compute_etce_values(lzp_x, lzp_y)
#         if np.isnan(etce_xy) or np.isnan(etce_yx) or etce_xy == etce_yx:
#             verdict_etce_raw, gap_etce_raw = 'Independence', 0.0
#         elif etce_xy < etce_yx:
#             verdict_etce_raw, gap_etce_raw = 'X -> Y', abs(etce_xy - etce_yx)
#         else:
#             verdict_etce_raw, gap_etce_raw = 'Y -> X', abs(etce_xy - etce_yx)
#         results.append({'trial': trial_counter, 'model': 'ETCE_raw',
#                          'direction': verdict_etce_raw, 'gap': gap_etce_raw})

#         trial_counter += 1

#     return pd.DataFrame(results)


# def random_delay_sweep_experiment(all_data, results_folder='results/experiment-1-random-delay-sweep'):
#     os.makedirs(results_folder, exist_ok=True)
#     all_results = []

#     for spread, data in all_data.items():
#         print(f"Running methods on spread={spread} ({len(data)} trials)...")
#         df = run_all_methods_on_trials(data, results_folder=results_folder)
#         df['delay'] = spread
#         all_results.append(df)

#     results_df = pd.concat(all_results, ignore_index=True)
#     results_df.to_csv(f'{results_folder}/experiment-1-random-delay-sweep-all_results.csv')
#     return results_df


# def calculate_metrics(delayed_df: pd.DataFrame):

#     delayed_df = delayed_df.loc[
#         :, ~delayed_df.columns.str.contains(r'^Unnamed')
#     ].copy()

#     delayed_df['delay'] = delayed_df['delay'].astype(str)
#     delayed_df['direction'] = delayed_df['direction'].fillna('Independence')

#     accuracy     = []
#     distribution = []

#     models = delayed_df['model'].unique()
#     delays = delayed_df['delay'].unique()

#     for model, delay in itertools.product(models, delays):

#         subset = delayed_df[
#             (delayed_df['model'] == model) &
#             (delayed_df['delay'] == delay)
#         ]
#         total_trials = len(subset)

#         if total_trials == 0:
#             continue

#         counts = subset['direction'].value_counts()

#         xy_count          = int(counts.get('X -> Y',       0))
#         yx_count          = int(counts.get('Y -> X',       0))
#         independent_count = int(counts.get('Independence', 0))

#         accuracy.append({
#             'delay':    delay,
#             'model':    model,
#             'accuracy': round(xy_count / total_trials, 4),
#         })

#         distribution.append({
#             'delay':             delay,
#             'model':             model,
#             'total_trials':      total_trials,
#             'xy_count':          xy_count,
#             'yx_count':          yx_count,
#             'independent_count': independent_count,
#         })

#     accuracy_df     = pd.DataFrame(accuracy)
#     distribution_df = pd.DataFrame(distribution)
#     return accuracy_df, distribution_df


# def wilson_ci(k, n, z=1.96):
#     k = np.asarray(k, dtype=float)
#     n = np.asarray(n, dtype=float)

#     p_hat = k / n
#     z2    = z ** 2

#     center     = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
#     half_width = (z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n**2))) / (1 + z2 / n)

#     lower = np.clip(center - half_width, 0, 1)
#     upper = np.clip(center + half_width, 0, 1)
#     return lower, upper


# def _plot_accuracy_subset(accuracy_df, distribution_df, model_order, title,
#                            filename, results_folder='results/experiment-1-random-delay-sweep'):
#     plot_df = accuracy_df.merge(
#         distribution_df[['delay', 'model', 'xy_count', 'total_trials']],
#         on=['delay', 'model'],
#         how='left'
#     )
#     plot_df['delay'] = plot_df['delay'].astype(int)
#     plot_df = plot_df.sort_values(['model', 'delay'])
 
#     markers = ['s', 'o', '^', 'v', 'D', 'P', 'X', '*', 'h', '<', '>', 'p']
 
#     fs = 20
#     fig, ax = plt.subplots(figsize=(10, 6))
 
#     for i, model in enumerate(model_order):
#         sub = plot_df[plot_df['model'] == model]
#         if sub.empty:
#             continue
#         k = sub['xy_count'].values
#         n = sub['total_trials'].values
#         lower, upper = wilson_ci(k, n)
 
#         marker = markers[i % len(markers)]
#         ax.plot(sub['delay'], sub['accuracy'], marker=marker, markersize=8,
#                  label=model, color=f'C{i}')
#         ax.fill_between(sub['delay'], lower, upper, alpha=0.15, color=f'C{i}')
 
#     ax.set_xlabel('Max delay spread  (delay ~ Uniform{0, ..., spread})',
#                   fontsize=fs - 4, fontweight='bold')
#     ax.set_ylabel('Accuracy', fontsize=fs - 2, fontweight='bold')
#     ax.set_ylim(0, 1.05)
#     ax.set_title(title, fontsize=fs - 4)
#     ax.legend(fontsize=fs - 8, ncol=2)
#     ax.grid(alpha=0.3)
 
#     plt.tight_layout()
#     os.makedirs(results_folder, exist_ok=True)
#     save_path = f'{results_folder}/{filename}'
#     plt.savefig(save_path, dpi=150)
#     plt.show()
#     print(f"Saved at {save_path}")



# def plot_accuracy_without_surrogate(accuracy_df, distribution_df,
#                                      results_folder='results/experiment-1-random-delay-sweep'):
#     model_order = ['DPE_raw', 'GC_raw', 'TE_raw', 'LZP_raw', 'CCM_raw', 'ETCP_raw', 'ETCE_raw']
#     _plot_accuracy_subset(
#         accuracy_df, distribution_df, model_order,
#         title='Random-Delay Bit-Flip: accuracy WITHOUT surrogate testing',
#         filename='random-delay-sweep-accuracy-without-surrogate.png',
#         results_folder=results_folder
#     )


# def plot_accuracy_with_surrogate(accuracy_df, distribution_df,
#                                   results_folder='results/experiment-1-random-delay-sweep'):
#     model_order = ['DPE', 'GC', 'TE', 'LZP']
#     _plot_accuracy_subset(
#         accuracy_df, distribution_df, model_order,
#         title='Random-Delay Bit-Flip: accuracy WITH surrogate testing',
#         filename='random-delay-sweep-accuracy-with-surrogate.png',
#         results_folder=results_folder
#     )


# # def plot_random_delay_sweep_distribution_bars(distribution_df, model_order=None,
# #                                                suptitle='Random-Delay Bit-Flip: verdict distribution by model and spread',
# #                                                filename='random-delay-sweep-distribution-bars.png',
# #                                                results_folder='results/experiment-1-random-delay-sweep'):
# #     dist = distribution_df.copy()
# #     dist['delay'] = dist['delay'].astype(int)
# #     dist = dist.sort_values('delay')

# #     if model_order is None:
# #         # model_order = ['DPE', 'GC', 'TE', 'LZP', 'CCM_raw', 'CCM_surrogate', 'ETCP', 'ETCE']
# #         model_order = ['DPE', 'GC', 'TE', 'LZP']

# #     present_models = [m for m in model_order if m in dist['model'].unique()]
# #     n_models = len(present_models)
# #     n_cols = 4
# #     n_rows = int(np.ceil(n_models / n_cols))

# #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
# #     axes = np.atleast_1d(axes).flatten()

# #     rename_map = {
# #         'xy_count': 'X -> Y',
# #         'yx_count': 'Y -> X',
# #         'independent_count': 'Independence'
# #     }

# #     for i, model in enumerate(present_models):
# #         sub = dist[dist['model'] == model].sort_values('delay')
# #         plot_data = sub.rename(columns=rename_map).set_index('delay')[
# #             ['X -> Y', 'Y -> X', 'Independence']
# #         ]
# #         plot_data.plot(kind='bar', ax=axes[i], edgecolor='black', width=0.8,
# #                         color=['C0', 'C3', 'C7'])
# #         axes[i].set_title(model, fontsize=16, fontweight='bold')
# #         axes[i].set_xlabel('Max delay spread', fontsize=11)
# #         axes[i].set_ylabel('Trial count', fontsize=11)
# #         axes[i].tick_params(axis='x', rotation=0, labelsize=10)
# #         axes[i].tick_params(axis='y', labelsize=10)
# #         axes[i].legend(fontsize=9, loc='upper right')
# #         axes[i].grid(axis='y', linestyle='--', alpha=0.4)
# #         for container in axes[i].containers:
# #             axes[i].bar_label(container, padding=2, fontsize=7)

# #     for j in range(n_models, len(axes)):
# #         axes[j].axis('off')

# #     fig.suptitle(suptitle, fontsize=18, fontweight='bold', y=1.02)
# #     plt.tight_layout()
# #     os.makedirs(results_folder, exist_ok=True)
# #     save_path = f'{results_folder}/{filename}'
# #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
# #     plt.show()
# #     print(f"Saved at {save_path}")

# def plot_random_delay_sweep_distribution_bars(distribution_df, model_order=None,
#                                                n_cols=2,  # Set to 2 for a 2x2 grid layout
#                                                suptitle='Random-Delay Bit-Flip: verdict distribution by model and spread',
#                                                filename='random-delay-sweep-distribution-bars.png',
#                                                results_folder='results/experiment-1-random-delay-sweep'):
#     dist = distribution_df.copy()
#     dist['delay'] = dist['delay'].astype(int)
#     dist = dist.sort_values('delay')

#     if model_order is None:
#         model_order = ['DPE', 'GC', 'TE', 'LZP']

#     present_models = [m for m in model_order if m in dist['model'].unique()]
#     n_models = len(present_models)
#     n_rows = int(np.ceil(n_models / n_cols))

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
#     axes = np.atleast_1d(axes).flatten()

#     rename_map = {
#         'xy_count': 'X -> Y',
#         'yx_count': 'Y -> X',
#         'independent_count': 'Independence'
#     }

#     for i, model in enumerate(present_models):
#         sub = dist[dist['model'] == model].sort_values('delay')
#         plot_data = sub.rename(columns=rename_map).set_index('delay')[
#             ['X -> Y', 'Y -> X', 'Independence']
#         ]
#         plot_data.plot(kind='bar', ax=axes[i], edgecolor='black', width=0.8,
#                         color=['tab:blue', 'tab:orange', 'tab:green'])
#         axes[i].set_title(model, fontsize=16, fontweight='bold')
#         axes[i].set_xlabel('Max delay spread', fontsize=16)
#         axes[i].set_ylabel('Trial count', fontsize=16)
#         axes[i].tick_params(axis='x', rotation=0, labelsize=16)
#         axes[i].tick_params(axis='y', labelsize=16)
#         axes[i].legend(fontsize=9, loc='upper right')
#         axes[i].grid(axis='y', linestyle='--', alpha=0.4)
#         for container in axes[i].containers:
#             axes[i].bar_label(container, padding=2, fontsize=7)

#     # Hide any unused subplots
#     for j in range(n_models, len(axes)):
#         axes[j].axis('off')

#     # fig.suptitle(suptitle, fontsize=18, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     os.makedirs(results_folder, exist_ok=True)
#     save_path = f'{results_folder}/{filename}'
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"Saved at {save_path}")

# def plot_distribution_bars_without_surrogate(distribution_df,
#                                               results_folder='results/experiment-1-random-delay-sweep'):
#     plot_random_delay_sweep_distribution_bars(
#         distribution_df,
#         model_order=['DPE_raw', 'GC_raw', 'TE_raw', 'LZP_raw', 'CCM_raw', 'ETCP_raw', 'ETCE_raw'],
#         suptitle='Verdict distribution WITHOUT surrogate testing',
#         filename='random-delay-sweep-distribution-bars-without-surrogate.png',
#         results_folder=results_folder
#     )


# def plot_distribution_bars_with_surrogate(distribution_df,
#                                            results_folder='results/experiment-1-random-delay-sweep'):
#     plot_random_delay_sweep_distribution_bars(
#         distribution_df,
#         model_order=['DPE', 'GC', 'TE', 'LZP'],
#         suptitle='Verdict distribution WITH surrogate testing',
#         filename='random-delay-sweep-distribution-bars-with-surrogate.png',
#         results_folder=results_folder
#     )


# # Main
# MAX_SPREAD = 6
# TRIALS     = 1000
# LENGTH     = 1000

# # sweep_data = generate_random_delay_sweep(
# #     trials=TRIALS, length=LENGTH, max_spread=MAX_SPREAD
# # )


# # sweep_results_df = random_delay_sweep_experiment(sweep_data)
# sweep_results_df = pd.read_csv('results/experiment-1-random-delay-sweep/experiment-1-random-delay-sweep-all_results.csv')

# sweep_accuracy_df, sweep_distribution_df = calculate_metrics(sweep_results_df)

# sweep_accuracy_df.to_csv('results/experiment-1-random-delay-sweep/experiment-1-accuracy-random-delay-sweep-all_results.csv')
# print(sweep_accuracy_df)

# plot_accuracy_without_surrogate(sweep_accuracy_df, sweep_distribution_df)
# plot_accuracy_with_surrogate(sweep_accuracy_df, sweep_distribution_df)
# plot_distribution_bars_without_surrogate(sweep_distribution_df)
# plot_distribution_bars_with_surrogate(sweep_distribution_df)



# Random-Delay (Stochastic) Bitflip Benchmark — Spread Sweep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import random
import itertools

from model.cy_utils import (
    run_causal_analysis,
    calculate_causal_history,
    generate_pattern_dictionary,
    calculate_contribution_analysis,
    normalized_weighted_entropy,
)
from model.utils import extract_json_data, discrete

# imports for TE
from transfer_entropy import compute_te, te_surrogate_threshold, te_identify_causality

# imports for GC
from granger_causality import compute_gc_values, compute_gc_direction

# imports for LZP
from lempel_ziv_penalty import compute_lzp_values, lzp_surrogate_threshold, lzp_identify_causality

# imports for CCM (raw)
from causal_ccm.causal_ccm import ccm

from ccm_model import ccm_identify_causality

from etc_penalty import compute_etcp_values, etcp_identify_causality
from etc_efficacy import compute_etce_values, etce_identify_causality

PATTERN = '1101'

# ---------------------------------------------------------------------------
# The four "core" models that were used in the original delayed-bitflip
# experiment. Every plotting function below is restricted to this set so the
# figures line up 1:1 with the earlier script, regardless of how many extra
# methods (raw / CCM / ETCP / ETCE) were computed into the results CSV.
# ---------------------------------------------------------------------------
CORE_MODELS = ['DPE', 'GC', 'TE', 'LZP']


def dpe_raw_verdict(x_str, y_str):
    Gx_y = calculate_causal_history(y_str, x_str)
    Gy_x = calculate_causal_history(x_str, y_str)

    Px_y = generate_pattern_dictionary(Gx_y)
    Py_x = generate_pattern_dictionary(Gy_x)

    stats_xy = calculate_contribution_analysis(Px_y, x_str, y_str)
    stats_yx = calculate_contribution_analysis(Py_x, y_str, x_str)

    avg_h_xy, _ = normalized_weighted_entropy(stats_xy, len(x_str))
    avg_h_yx, _ = normalized_weighted_entropy(stats_yx, len(y_str))

    gap = avg_h_xy - avg_h_yx
    if gap < 0:
        return 'X -> Y', abs(gap)
    elif gap > 0:
        return 'Y -> X', abs(gap)
    else:
        return 'Independence', abs(gap)


def generate_main_pattern(length=1000):
    sequence = ""
    bits = ['0', '1', PATTERN]

    while len(sequence) < length:
        choice = random.choice(bits)
        sequence += choice

    return sequence[:length]


def generate_random_delayed_flip(main_sequence, delay_choices=range(7), rng=None):
    if rng is None:
        rng = random
    length = len(main_sequence)
    delayed_seq = ['0'] * length

    i = len(PATTERN) - 1
    while i < length:
        if main_sequence[i - len(PATTERN) + 1 : i + 1] == PATTERN:
            delay = rng.choice(delay_choices)
            target_idx = i + delay
            if target_idx < length:
                delayed_seq[target_idx] = '1'
            i += len(PATTERN)
        else:
            i += 1

    return "".join(delayed_seq)


def generate_random_delay_sweep(trials=1000, length=1000, max_spread=6,
                                 save_dir='../dataset/random-delay-sweep'):
    os.makedirs(save_dir, exist_ok=True)
    all_data = {}

    for spread in range(max_spread + 1):
        delay_choices = range(spread + 1)
        data = []

        for i in range(trials):
            main_seq = generate_main_pattern(length)
            delayed_seq = generate_random_delayed_flip(main_seq, delay_choices)
            data.append({
                'trial_id': i,
                'X': main_seq,
                'Y': delayed_seq,
                'spread': spread
            })

        with open(f'{save_dir}/spread-{spread}.json', 'w') as file:
            json.dump(data, file, indent=4)

        all_data[spread] = data
        print(f"Generated spread={spread} (delay in 0..{spread}), {trials} trials.")

    return all_data


def extract_sweep_data(save_dir='../dataset/random-delay-sweep', max_spread=6):
    all_data = {}
    for spread in range(max_spread + 1):
        file_path = f'{save_dir}/spread-{spread}.json'
        if not os.path.exists(file_path):
            print(f"Error: The file at {file_path} does not exist.")
            continue
        with open(file_path, 'r') as file:
            all_data[spread] = json.load(file)
    return all_data


def run_all_methods_on_trials(data, results_folder='results/experiment-1-random-delay-sweep'):
    # NOTE: this still computes every method (core + raw + CCM + ETCP/ETCE) so
    # the underlying CSV keeps full provenance. Only the *plots* further down
    # are restricted to CORE_MODELS.
    results = []
    trial_counter = 0

    for trial in data:
        x_str, y_str = trial['X'], trial['Y']

        x_raw = np.array(list(x_str), dtype=np.int32)
        y_raw = np.array(list(y_str), dtype=np.int32)

        list_x = [i for i in x_raw]
        list_y = [j for j in y_raw]

        lzp_x = [i + 1 for i in x_raw]
        lzp_y = [j + 1 for j in y_raw]

        # DPE (run_causal_analysis -- this IS surrogate-tested internally, confirmed
        verdict_dpe, gap_dpe = run_causal_analysis(x_str, y_str)
        results.append({'trial': trial_counter, 'model': 'DPE',
                         'direction': verdict_dpe, 'gap': gap_dpe})

        # DPE_raw (same underlying cy_utils primitives, no surrogate threshold --
        verdict_dpe_raw, gap_dpe_raw = dpe_raw_verdict(x_str, y_str)
        results.append({'trial': trial_counter, 'model': 'DPE_raw',
                         'direction': verdict_dpe_raw, 'gap': gap_dpe_raw})

        # GC (existing, alpha-based significance test -- already a form of
        verdict_gc, xy_gc, yx_gc = compute_gc_direction(x_raw, y_raw)
        results.append({'trial': trial_counter, 'model': 'GC',
                         'direction': verdict_gc, 'gap': abs(xy_gc - yx_gc)})

        # GC_raw (no alpha threshold -- just whichever p-value is smaller wins;
        p_xy_gc, p_yx_gc = compute_gc_values(x_raw, y_raw)
        if np.isnan(p_xy_gc) or np.isnan(p_yx_gc) or p_xy_gc == p_yx_gc:
            verdict_gc_raw, gap_gc_raw = 'Independence', 0.0
        elif p_xy_gc < p_yx_gc:
            verdict_gc_raw, gap_gc_raw = 'Y -> X', abs(p_xy_gc - p_yx_gc)
        else:
            verdict_gc_raw, gap_gc_raw = 'X -> Y', abs(p_xy_gc - p_yx_gc)
        results.append({'trial': trial_counter, 'model': 'GC_raw',
                         'direction': verdict_gc_raw, 'gap': gap_gc_raw})

        # TE (surrogate-tested, unchanged)
        verdict_te, gap_te = te_identify_causality(list_x, list_y)
        results.append({'trial': trial_counter, 'model': 'TE',
                         'direction': verdict_te, 'gap': gap_te})

        # TE_raw (raw comparison, no surrogate threshold)
        te_xy, te_yx = compute_te(list_x, list_y)
        if np.isnan(te_xy) or np.isnan(te_yx) or te_xy == te_yx:
            verdict_te_raw, gap_te_raw = 'Independence', 0.0
        elif te_xy < te_yx:
            verdict_te_raw, gap_te_raw = 'Y -> X', abs(te_xy - te_yx)
        else:
            verdict_te_raw, gap_te_raw = 'X -> Y', abs(te_xy - te_yx)
        results.append({'trial': trial_counter, 'model': 'TE_raw',
                         'direction': verdict_te_raw, 'gap': gap_te_raw})

        # LZP (surrogate-tested, unchanged)
        verdict_lzp, gap_lzp = lzp_identify_causality(lzp_x, lzp_y)
        results.append({'trial': trial_counter, 'model': 'LZP',
                         'direction': verdict_lzp, 'gap': gap_lzp})

        # LZP_raw (raw comparison, no surrogate threshold)
        lzp_xy, lzp_yx = compute_lzp_values(lzp_x, lzp_y)
        if np.isnan(lzp_xy) or np.isnan(lzp_yx) or lzp_xy == lzp_yx:
            verdict_lzp_raw, gap_lzp_raw = 'Independence', 0.0
        elif lzp_xy < lzp_yx:
            verdict_lzp_raw, gap_lzp_raw = 'X -> Y', abs(lzp_xy - lzp_yx)
        else:
            verdict_lzp_raw, gap_lzp_raw = 'Y -> X', abs(lzp_xy - lzp_yx)
        results.append({'trial': trial_counter, 'model': 'LZP_raw',
                         'direction': verdict_lzp_raw, 'gap': gap_lzp_raw})

        # CCM_raw (raw rho comparison, no significance test -- renamed from
        try:
            ccm_obj = ccm(X=x_raw, Y=y_raw, E=2, tau=1)
            x_to_y_rho, y_to_x_rho = ccm_obj.causality()

            if x_to_y_rho > y_to_x_rho:
                verdict_ccm = 'Y -> X'
            elif y_to_x_rho > x_to_y_rho:
                verdict_ccm = 'X -> Y'
            else:
                verdict_ccm = 'Independence'

            gap_ccm = abs(x_to_y_rho - y_to_x_rho)
        except Exception:
            verdict_ccm = 'Independence'
            gap_ccm = 0.0

        results.append({'trial': trial_counter, 'model': 'CCM_raw',
                         'direction': verdict_ccm, 'gap': gap_ccm})

        # CCM_surrogate (surrogate-tested significance version from ccm_model.py)
        verdict_ccm_surr, gap_ccm_surr = ccm_identify_causality(
            x_raw, y_raw, E=2, tau=1, n_surrogates=100, percentile=95
        )
        results.append({'trial': trial_counter, 'model': 'CCM_surrogate',
                         'direction': verdict_ccm_surr, 'gap': gap_ccm_surr})

        # ETCP (surrogate-tested, unchanged)
        verdict_etcp, gap_etcp = etcp_identify_causality(lzp_x, lzp_y)
        results.append({'trial': trial_counter, 'model': 'ETCP',
                         'direction': verdict_etcp, 'gap': gap_etcp})

        # ETCP_raw (raw comparison, no surrogate threshold)
        etcp_xy, etcp_yx = compute_etcp_values(lzp_x, lzp_y)
        if np.isnan(etcp_xy) or np.isnan(etcp_yx) or etcp_xy == etcp_yx:
            verdict_etcp_raw, gap_etcp_raw = 'Independence', 0.0
        elif etcp_xy < etcp_yx:
            verdict_etcp_raw, gap_etcp_raw = 'X -> Y', abs(etcp_xy - etcp_yx)
        else:
            verdict_etcp_raw, gap_etcp_raw = 'Y -> X', abs(etcp_xy - etcp_yx)
        results.append({'trial': trial_counter, 'model': 'ETCP_raw',
                         'direction': verdict_etcp_raw, 'gap': gap_etcp_raw})

        # ETCE (surrogate-tested, unchanged)
        verdict_etce, gap_etce = etce_identify_causality(lzp_x, lzp_y)
        results.append({'trial': trial_counter, 'model': 'ETCE',
                         'direction': verdict_etce, 'gap': gap_etce})

        # ETCE_raw (raw comparison, no surrogate threshold)
        etce_xy, etce_yx = compute_etce_values(lzp_x, lzp_y)
        if np.isnan(etce_xy) or np.isnan(etce_yx) or etce_xy == etce_yx:
            verdict_etce_raw, gap_etce_raw = 'Independence', 0.0
        elif etce_xy < etce_yx:
            verdict_etce_raw, gap_etce_raw = 'X -> Y', abs(etce_xy - etce_yx)
        else:
            verdict_etce_raw, gap_etce_raw = 'Y -> X', abs(etce_xy - etce_yx)
        results.append({'trial': trial_counter, 'model': 'ETCE_raw',
                         'direction': verdict_etce_raw, 'gap': gap_etce_raw})

        trial_counter += 1

    return pd.DataFrame(results)


def random_delay_sweep_experiment(all_data, results_folder='results/experiment-1-random-delay-sweep'):
    os.makedirs(results_folder, exist_ok=True)
    all_results = []

    for spread, data in all_data.items():
        print(f"Running methods on spread={spread} ({len(data)} trials)...")
        df = run_all_methods_on_trials(data, results_folder=results_folder)
        df['delay'] = spread
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(f'{results_folder}/experiment-1-random-delay-sweep-all_results.csv')
    return results_df


def calculate_metrics(delayed_df: pd.DataFrame):

    delayed_df = delayed_df.loc[
        :, ~delayed_df.columns.str.contains(r'^Unnamed')
    ].copy()

    delayed_df['delay'] = delayed_df['delay'].astype(str)
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


def wilson_ci(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)

    p_hat = k / n
    z2    = z ** 2

    center     = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    half_width = (z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n**2))) / (1 + z2 / n)

    lower = np.clip(center - half_width, 0, 1)
    upper = np.clip(center + half_width, 0, 1)
    return lower, upper


# ---------------------------------------------------------------------------
# Plot style below mirrors delayed_bitflip_experiment's plot_accuracy_with_ci
# and bar_plots exactly (same markers, font sizes, figure size, colors,
# legend placement), restricted to CORE_MODELS = ['DPE', 'GC', 'TE', 'LZP'].
# ---------------------------------------------------------------------------

def plot_accuracy_with_ci(accuracy_df, distribution_df,
                           results_folder='results/experiment-1-random-delay-sweep'):
    plot_df = accuracy_df.merge(
        distribution_df[['delay', 'model', 'xy_count', 'total_trials']],
        on=['delay', 'model'],
        how='left'
    )

    # restrict to the same models used in the original delayed-bitflip experiment
    plot_df = plot_df[plot_df['model'].isin(CORE_MODELS)].copy()
    plot_df['delay'] = pd.to_numeric(plot_df['delay'])

    model_order = CORE_MODELS
    plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values(['model', 'delay'])

    markers = ['o', 'v', 'p', 's']
    fs      = 26
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (model, grp) in enumerate(plot_df.groupby('model', observed=True, sort=True)):
        delays   = grp['delay'].values
        acc      = grp['accuracy'].values
        k        = grp['xy_count'].values
        n        = grp['total_trials'].values
        lower, upper = wilson_ci(k, n)
        color = f'C{i}'
        ax.fill_between(
            delays, lower, upper,
            alpha=0.18,
            color=color,
            label='_nolegend_'
        )
        yerr_low  = acc - lower
        yerr_high = upper - acc
        ax.errorbar(
            delays, acc,
            yerr=[yerr_low, yerr_high],
            fmt=markers[i % len(markers)],
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=5,
            capthick=1.5,
            linewidth=2,
            markersize=10,
            label=model
        )

    ax.set_xlabel('Max Delay Spread', fontsize=fs - 2, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=fs - 2, fontweight='bold')
    ax.set_xticks(sorted(plot_df['delay'].unique()))
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(axis='both', labelsize=fs - 4)
    ax.legend(
        loc='center right',
        fontsize=fs - 5,
        framealpha=0.85
    )
    ax.grid(True, alpha=0.3)
    ax.annotate(
        'Shaded bands = Wilson 95% CI',
        xy=(0.02, 0.04), xycoords='axes fraction',
        fontsize=fs - 8, color='grey', style='italic'
    )
    plt.tight_layout()

    os.makedirs(results_folder, exist_ok=True)
    save_path = f'{results_folder}/random-delay-sweep-accuracy-with-wilson-ci.png'
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved at {save_path}")


def bar_plots(distribution_df, results_folder='results/experiment-1-random-delay-sweep'):
    dist = distribution_df.copy()
    dist = dist[dist['model'].isin(CORE_MODELS)].copy()
    dist['delay'] = dist['delay'].astype(int)
    dist = dist.sort_values('delay')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    rename_map = {
        'xy_count': 'X -> Y',
        'yx_count': 'Y -> X',
        'independent_count': 'Independent'
    }

    for i, model in enumerate(CORE_MODELS):
        subset = dist[dist['model'] == model].copy()
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
        axes[i].set_ylim(0, subset[['X -> Y', 'Y -> X', 'Independent']].values.max() * 1.05)

        axes[i].tick_params(axis='both', labelsize=18)
        axes[i].tick_params(axis='x', rotation=0)

        axes[i].legend(fontsize=13, loc='best')

        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        for container in axes[i].containers:
            axes[i].bar_label(container, padding=3, fontsize=12)

        axes[i].margins(y=0.2)

    plt.tight_layout()
    os.makedirs(results_folder, exist_ok=True)
    save_path = f'{results_folder}/random-delay-sweep-barplots.png'
    plt.savefig(save_path)
    plt.show()
    print(f"Saved at {save_path}")


# Main
MAX_SPREAD = 6
TRIALS     = 1000
LENGTH     = 1000

# sweep_data = generate_random_delay_sweep(
#     trials=TRIALS, length=LENGTH, max_spread=MAX_SPREAD
# )

# sweep_results_df = random_delay_sweep_experiment(sweep_data)
sweep_results_df = pd.read_csv('results/experiment-1-random-delay-sweep/experiment-1-random-delay-sweep-all_results.csv')

sweep_accuracy_df, sweep_distribution_df = calculate_metrics(sweep_results_df)

sweep_accuracy_df.to_csv('results/experiment-1-random-delay-sweep/experiment-1-accuracy-random-delay-sweep-all_results.csv')
print(sweep_accuracy_df[sweep_accuracy_df['model'].isin(CORE_MODELS)])

plot_accuracy_with_ci(sweep_accuracy_df, sweep_distribution_df)
bar_plots(sweep_distribution_df)