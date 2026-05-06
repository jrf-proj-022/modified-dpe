#  cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
import os
import re
import cython
from typing import List, Set, Any

cnp.import_array()

def _ensure_uint8(seq: Any):
    """Internal helper to get a read-only byte view."""
    if isinstance(seq, str):
        return seq.encode('ascii')
    elif isinstance(seq, list):
        return np.array(seq, dtype=np.uint8)
    return np.asanyarray(seq, dtype=np.uint8)


def calculate_causal_history(seq_1, seq_2):
    '''
    Constructs the dictionary GX→Y by scanning Y from left to right and recording 
    the corresponding sub-pattern in X each time a bit flip occurs in Y.
    
    Args:
        seq_1 (str/list/np.ndarray): Sequence_Y (Target)
        seq_2 (str/list/np.ndarray): Sequence_X (Source)
    
    Returns:
        set: A set of unique segments from Sequence_X associated with flips in Y.
    '''
    cdef const cnp.uint8_t[:] s1 = _ensure_uint8(seq_1)
    cdef const cnp.uint8_t[:] s2 = _ensure_uint8(seq_2)
    
    cdef int len_1 = s1.shape[0]
    cdef int last_position = 0
    cdef int k = 1
    cdef list G = []
    
    # if len_1 != s2.shape[0]:
    #     raise ValueError(f"Sequences must be of equal length.")
    
    while k < len_1:
        if s1[k] != s1[k-1]:
            # byte slicing
            G.append(bytes(s2[last_position : k+1]).decode('ascii'))
            last_position = k + 1
            k += 1
        k += 1
    
    return set(G)


def sliding_xnor_comparison(p1, p2):
    '''
    Identifies common subsequences between two patterns using XNOR-based sliding.
    Operates on NumPy arrays for bitwise comparison.

    Args:
        p1 (str/list/np.ndarray): The first binary subpattern.
        p2 (str/list/np.ndarray): The second binary subpattern.

    Returns:
        set: Common subsequences (length >= 2).
    '''
    cdef const cnp.uint8_t[:] v1 = _ensure_uint8(p1)
    cdef const cnp.uint8_t[:] v2 = _ensure_uint8(p2)
    cdef int n1 = v1.shape[0]
    cdef int n2 = v2.shape[0]
    
    if n1 > n2:
        v1, v2 = v2, v1
        n1, n2 = n2, n1

    found_in_pair = set()
    cdef int shift, i
    cdef list current_match
    
    for shift in range(n2 - n1 + 1):
        current_match = []
        for i in range(n1):
            if v1[i] == v2[shift + i]:
                current_match.append(chr(v1[i]))
            else:
                if len(current_match) >= 2:
                    found_in_pair.add("".join(current_match))
                current_match = []
        
        if len(current_match) >= 2:
            found_in_pair.add("".join(current_match))
    
    return found_in_pair


def generate_pattern_dictionary(G_input):
    '''
    Constructs the overall pattern dictionary PX->Y from a causal history.
    Iterates through all distinct pairs in G and extracts common sub-patterns.

    Args:
        G (set/list): The causal history dictionary GX->Y.

    Returns:
        set: The final PX->Y pattern dictionary.
    '''
    G = list(G_input)
    patterns = set()
    cdef int n = len(G)
    cdef int i, j
    
    for i in range(n):
        for j in range(i + 1, n):
            extracted = sliding_xnor_comparison(G[i], G[j])
            patterns.update(extracted)
    
    # if there is no pattern in patterns set.
    if not patterns:
        return G_input

    return patterns


def calculate_contribution_analysis(patterns, X_in, Y_in):
    '''
    Performs frequency and contribution analysis for extracted causal patterns.
    Quantifies the likelihood that a pattern in X induces a variation in Y.

    Args:
        patterns (set/list): Dictionary of extracted patterns (PX->Y).
        X (str/list/np.ndarray): The causal input sequence.
        Y (str/list/np.ndarray): The output sequence containing bit flips.

    Returns:
        pd.DataFrame: Mapping of patterns to Change, No-change, and Ratio (R_flip).
    '''
    cdef const cnp.uint8_t[:] X = _ensure_uint8(X_in)
    cdef const cnp.uint8_t[:] Y = _ensure_uint8(Y_in)
    cdef const cnp.uint8_t[:] p
    
    cdef int x_len = X.shape[0]
    cdef int y_len = Y.shape[0]
    cdef int p_len, i, j, k, change_cnt, match_count
    cdef bint has_flip, is_match
    
    results = {}

    for pattern in patterns:
        p = _ensure_uint8(pattern)
        p_len = p.shape[0]
        change_cnt = 0
        match_count = 0
        
        # searching for pattern
        for i in range(x_len - p_len + 1):
            is_match = True
            for j in range(p_len):
                if X[i + j] != p[j]:
                    is_match = False
                    break
            
            if is_match:
                match_count += 1
                has_flip = False
                for k in range(i + 1, i + p_len):
                    if k < y_len and Y[k] != Y[k-1]:
                        has_flip = True
                        break
                
                if has_flip:
                    change_cnt += 1

        ratio = <float>change_cnt / match_count if match_count > 0 else 0.0
        results[pattern] = {
            'Change': change_cnt,
            'No-change': match_count - change_cnt,
            'Ratio': round(ratio, 4)
        }

    df = pd.DataFrame(results).T
    df.index.name = 'Pattern'
    return df


def discrete(sequence: np.ndarray, n_bins: int = 2):
    '''
    Discretizes a continuous signal into symbolic integer bins.

    This function uses uniform binning to map a range of continuous values 
    to a set of discrete states. This transformation is a prerequisite for 
    symbolic information-theoretic analysis (like Entropy or Mutual Information).

    

    Args:
        sequence (np.ndarray): The input numerical time-series data.
        n_bins (int): The number of discrete levels to create.

    Returns:
        np.ndarray: An array of integers in the range [0, n_bins - 1].
    '''
    cdef float seq_min, seq_max, bin_width, data_range

    if len(sequence) == 0:
        return np.empty(0, dtype=int)

    seq_min = sequence.min()
    seq_max = sequence.max()
    data_range = seq_max - seq_min

    if data_range == 0:
        return np.zeros(len(sequence), dtype=int)

    bin_width = data_range / n_bins
    indices = ((sequence - seq_min) / bin_width).astype(int)
    
    return np.clip(indices, 0, n_bins - 1)


# Plot causal network graph
def draw_network(W_p_x, W_p_y, file_name='causal_network_dual'):
    G = nx.DiGraph()

    hub_size = 14000     
    pattern_size = 8500  
    font_size_nodes = 24
    font_size_edges = 22
    font_size_title = 42

    G.add_node("X", color='skyblue', size=hub_size)
    G.add_node("Y", color='#e67e22', size=hub_size)

    all_patterns = []
    for p in list(W_p_x.index) + list(W_p_y.index):
        if p not in all_patterns:
            all_patterns.append(p)
    
    for pattern in all_patterns:
        p_str = str(pattern).strip()
        G.add_node(p_str, size=pattern_size)
        if pattern in W_p_x.index:
            ent = W_p_x[pattern]
            G.add_edge(p_str, "X", entropy=ent, label=f"{ent:.2f}")
        if pattern in W_p_y.index:
            ent = W_p_y[pattern]
            G.add_edge(p_str, "Y", entropy=ent, label=f"{ent:.2f}")

    fig, ax = plt.subplots(figsize=(32, 18)) 
    
    hub_sep = 200 
    hub_x_pos = np.array([-hub_sep, 0])
    hub_y_pos = np.array([hub_sep, 0])
    pos = {"X": hub_x_pos, "Y": hub_y_pos}

    x_only, y_only, shared = [], [], []
    for node in all_patterns:
        succs = list(G.successors(node))
        if "X" in succs and "Y" in succs: shared.append(node)
        elif "X" in succs: x_only.append(node)
        else: y_only.append(node)


    def position_outward_arc(nodes, hub_pos, direction, hub_key):
        if not nodes: return
        count = len(nodes)
        angles = np.linspace(-1.2, 1.2, count) if count > 1 else [0]
        for i, node in enumerate(nodes):
            ent = G.edges[(node, hub_key)]['entropy']
            dist = 400 + (ent * 250) 
            offset = np.array([direction * np.cos(angles[i]), np.sin(angles[i])]) * dist
            pos[node] = hub_pos + offset
            G.nodes[node]['color'] = '#aed6f1' if hub_key == "X" else '#f9e79f'

    position_outward_arc(x_only, hub_x_pos, -1, "X") 
    position_outward_arc(y_only, hub_y_pos, 1, "Y")  

    if shared:
        v_spacing = 350 
        total_height = (len(shared) - 1) * v_spacing
        y_space = np.linspace(-total_height/2, total_height/2, len(shared)) if len(shared) > 1 else [0]
        for i, node in enumerate(shared):
            pos[node] = np.array([0, y_space[i]])
            G.nodes[node]['color'] = '#9b59b6'

    nodes_list = G.nodes(data=True)
    nx.draw_networkx_nodes(G, pos, 
                           node_color=[d['color'] for n, d in nodes_list],
                           node_size=[d['size'] for n, d in nodes_list], 
                           edgecolors='black', linewidths=4, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=font_size_nodes, font_weight='bold', ax=ax)

    nx.draw_networkx_edges(
        G, pos, arrowstyle='-|>', arrowsize=70, width=5.0, 
        edge_color='black', alpha=0.3, 
        min_source_margin=60, min_target_margin=140, ax=ax
    )

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size_edges, 
                                 font_color='red', font_weight='bold', label_pos=0.5, 
                                 rotate=True, ax=ax)
    
    plt.title(r'$\mathbf{Distance = 1 + H(P)}$', fontsize=font_size_title, pad=180)
    plt.axis('off')
    plt.savefig(f'results/{file_name}.png', bbox_inches='tight', dpi=300)
    plt.show()


# # ===================================================
# Calculation part after pattern statistics

def weighted_binary_entropy(ratio, count, seq_len, pattern_len):
    """
    Calculates the binary entropy of a pattern weighted by its frequency.
    
    Args:
        ratio (float): The success probability (R_flip).
        count (int): Total occurrences of this pattern (change + no change).
        seq_len (int): Total length of the sequence.
        pattern_len (int): The length of the specific pattern.
        
    Returns:
        float: The frequency-weighted entropy in bits.
    """
    cdef float entropy, weight, total_windows

    # 1. Calculate standard binary entropy Hb(r)
    if ratio <= 0 or ratio >= 1:
        entropy = 0.0
    else:
        entropy = -(ratio * np.log2(ratio) + (1 - ratio) * np.log2(1 - ratio))
    
    # 2. Calculate weight: Occurrences / Total possible windows
    total_windows = seq_len - pattern_len + 1
    weight = count / total_windows

    # 3. Return weighted entropy
    return weight * entropy, weight


def compute_scale_entropy(stats_df, seq_len):
    """
    Computes average weighted entropy grouped by pattern length.

    Args:
        stats_df (pd.DataFrame): Output from calculate_contribution_analysis
                                 Must contain ['Change', 'No-change', 'Ratio']
        seq_len (int): Length of the sequence (X or Y)

    Returns:
        pd.DataFrame: Columns:
            - Scale (pattern length)
            - Patterns (set of patterns)
            - Avg Weighted Entropy
    """
    results = {}

    for pattern, row in stats_df.iterrows():
        pattern_len = len(pattern)
        change = row['Change']
        nochange = row['No-change']
        ratio = row['Ratio']
        count = change + nochange

        w_entropy, weight = weighted_binary_entropy(
            ratio=ratio,
            count=count,
            seq_len=seq_len,
            pattern_len=pattern_len
        )

        if pattern_len not in results:
            results[pattern_len] = {
                "patterns": [],
                "entropy_sum": 0.0,
                "weight_sum": 0.0
            }

        results[pattern_len]["patterns"].append(pattern)
        results[pattern_len]["entropy_sum"] += w_entropy
        results[pattern_len]["weight_sum"] += weight

    final_data = []
    for length, data in sorted(results.items()):
        num_patterns = len(data["patterns"])
        avg_entropy = data["entropy_sum"] / num_patterns if num_patterns > 0 else 0
        normalized_weighted_entropy = data["entropy_sum"] / data["weight_sum"] if num_patterns > 0 else 0

        final_data.append({
            "Scale": length,
            "Pattern Set": set(data["patterns"]),
            "Avg Weighted Entropy": round(avg_entropy, 6),
            "Normalized Weighted Entropy": round(normalized_weighted_entropy, 6),
        })

    return pd.DataFrame(final_data)


def normalized_weighted_entropy(stats, seq_len):
    """
    Computes the normalized weighted binary entropy across a pattern dictionary.
    
    Args:
        stats (pd.DataFrame): DataFrame with columns ['Pattern', 'Ratio', 'Count'].
            'Count' should be the total occurrences of that pattern.
        seq_len (int): The length of the original sequence used.
    Returns:
        float: Sum of weighted entropies divided by Cardinality.
                = ∑(Wp * Hb(rp)) / |P|
                → 1.0 under independence
                → 0.0 under strong causation
    """
    if stats is None or stats.empty:
        return 0.0
    
    # Calculate (weighted_entropy, weight) for each row
    results = stats.apply(
        lambda row: weighted_binary_entropy(
            row['Ratio'], 
            row['Change'] + row['No-change'], 
            seq_len, 
            len(str(row.name).strip())
        ), 
        axis=1
    )
    
    # Unpack the (weighted_entropy, weight) tuples
    weighted_entropies = results.apply(lambda x: x[0])
    weights = results.apply(lambda x: x[1])
    
    sum_weights = weights.sum()
    
    # Avoid division by zero
    if sum_weights == 0:
        return 0.0

    # using cardinality
    return weighted_entropies.mean(), weighted_entropies


def surrogate_threshold(X, Y, n_surrogates=100, percentile=95):
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
        Gx_y = calculate_causal_history(Y_shuffled, X_shuffled)
        Gy_x = calculate_causal_history(X_shuffled, Y_shuffled)
        Px_y = generate_pattern_dictionary(Gx_y)
        Py_x = generate_pattern_dictionary(Gy_x)
        stats_xy_s = calculate_contribution_analysis(Px_y, X_shuffled, Y_shuffled)
        stats_yx_s = calculate_contribution_analysis(Py_x, Y_shuffled, X_shuffled)
        
#         stats_xy_s, stats_yx_s = compute_dpe_stats(X_shuffled, Y_shuffled)
        
        h_xy, _ = normalized_weighted_entropy(stats_xy_s, len(X_shuffled))
        h_yx, _ = normalized_weighted_entropy(stats_yx_s, len(Y_shuffled))
        
        gaps.append(abs(h_xy - h_yx))
    
    return np.percentile(gaps, percentile)


def identify_causality(stats_x_y, stats_y_x, X, Y, n_surrogates=100):
    
    avg_h_xy, _ = normalized_weighted_entropy(stats_x_y, len(X))
    avg_h_yx, _ = normalized_weighted_entropy(stats_y_x, len(Y))
    
    # print("Normalized Weighted Entropy X to Y =", avg_h_xy)
    # print("Normalized Weighted Entropy Y to X =", avg_h_yx)
    
    gap = avg_h_xy - avg_h_yx
    abs_gap = abs(gap)
    
    # Data-driven threshold from surrogates
    threshold = surrogate_threshold(X, Y, n_surrogates)
    # print(f"Surrogate threshold (95th percentile) = {threshold:.4f}")
    
    if abs_gap < threshold:
        return 'Independence', abs_gap
    
    if gap < 0:
        return 'X -> Y', abs_gap
    else:
        return 'Y -> X', abs_gap


def run_causal_analysis(seq_x_bin: np.ndarray, seq_y_bin:np.ndarray):
    '''
    Executes the full causal discovery pipeline on a single pair of binary sequences.
    
    This function converts binary arrays to strings, calculates causal history 
    metrics (likelihood of one sequence given the other's history), generates 
    pattern dictionaries for compression/contribution analysis, and provides 
     a final directionality verdict.

    Args:
        seq_x_bin (numpy.ndarray): Binary array representing the first time series (X).
        seq_y_bin (numpy.ndarray): Binary array representing the second time series (Y).

    Returns:
        str: A string indicating the causal verdict (e.g., "Y -> X", "X -> Y", 
             "Independence", or "Undecided").
    '''

    # 1. Calculate Causal History (Converting array to string if needed)
    # x_str = "".join(map(str, seq_x_bin))
    # y_str = "".join(map(str, seq_y_bin))
    
    Gx_y = calculate_causal_history(seq_y_bin, seq_x_bin)
    Gy_x = calculate_causal_history(seq_x_bin, seq_y_bin)
    
    # 2. Generate Pattern Dictionaries
    Px_y = generate_pattern_dictionary(Gx_y)
    Py_x = generate_pattern_dictionary(Gy_x)
    
    # 3. Contribution Analysis
    stats1 = calculate_contribution_analysis(Px_y, seq_x_bin, seq_y_bin)
    stats2 = calculate_contribution_analysis(Py_x, seq_y_bin, seq_x_bin)
    
    # 4. Final Verdict
    return identify_causality(stats1, stats2, seq_x_bin, seq_y_bin, n_surrogates=100)

# # ===================================================
# AR(1) process (Synthetic Unidirectional Coupling) data generation functions

def ar1_pair_generator(length:int=1000, noise_intensity:float=0.01, a:float=0.8, b:float=0.8, phi:float=0):
    '''
    Generates synthetic bivariate time series using a coupled Autoregressive (AR) model.

    The model follows the structure:
    X[t] = a*X[t-1] + phi*Y[t-1] + noise
    Y[t] = b*Y[t-1] + noise

    When phi > 0, Y exerts a causal influence on X. This generator is used to 
    produce ground-truth data for validating causal discovery algorithms.

    

    Args:
        length (int): The number of time steps to generate.
        noise_intensity (float): Scaling factor for the Gaussian noise (e_x, e_y).
        a (float): Self-dependence coefficient for sequence X.
        b (float): Self-dependence coefficient for sequence Y.
        phi (float): Coupling strength representing the causal influence of Y on X.

    Returns:
        tuple: A pair of numpy arrays (X, Y) representing the coupled time series.
    '''
    X = np.zeros(length)
    Y = np.zeros(length)

    e_x = noise_intensity * np.random.normal(0, 1, length)
    e_y = noise_intensity * np.random.normal(0, 1, length)

    for t in range(1, length):
        X[t] = (a * X[t-1]) + (phi * Y[t-1]) + e_x[t]
        Y[t] = (b * Y[t-1]) + e_y[t]

    return X, Y

def ar1_data_generator(num_sequences: int = 2000, sequence_length: int = 2000, transients: int =500):
    '''
    Executes a large-scale simulation to analyze causal detection sensitivity.

    This method sweeps through various coupling strengths (phi) from 0 to 1. 
    For each phi, it generates multiple pairs of sequences, discretizes them 
    into binary signals, and organizes them for ensemble analysis. This is 
    designed to evaluate at which threshold of phi the causal method accurately 
    identifies 'Y -> X'.

    Args:
        num_sequences (int): Number of independent realizations to generate per phi value.
        sequence_length (int): The number of data points in each individual sequence.
        transients (int): starting values to be removed.

    Returns:
        dict: A nested dictionary where keys are phi values and values are 
            dictionaries containing the discretized 'X' and 'Y' sequence matrices.
    '''
    phis = np.arange(0, 1, 0.1)
    result_dict = {str(key): {'X':[], 'Y':[]} for key in phis}

    for phi in phis:

        for i in range(num_sequences):
            x_raw, y_raw = ar1_pair_generator(length=sequence_length + transients, phi=phi)

            x_seq = x_raw.tolist()
            y_seq = y_raw.tolist()

            result_dict[str(phi)]['X'].append(x_seq[transients:])
            result_dict[str(phi)]['Y'].append(y_seq[transients:])

    # Save to JSON
    output_path = 'dataset/ar1_data.json'
    with open(output_path, 'w') as f:
        json.dump(result_dict, f)
    
    print(f"\nData generation complete. Saved to: {output_path}")


# # ===================================================
# 1D Coupled Skewtent Maps data generation functions

def skew_tent_map(x, b):
    '''
    Computes the next iteration of a 1D Skew-Tent Map.
    
    The Skew-Tent map is a piecewise linear chaotic map defined by the 
    skewness parameter 'b'. It maps the unit interval [0, 1] onto itself.
    
    Args:
        x (float): The current state value in the interval [0, 1].
        b (float): The skewness parameter (peak position) in the interval (0, 1).
        
    Returns:
        float: The mapped value T(x).
    '''
    if 0 <= x < b:
        return x / b
    elif b <= x < 1:
        return (1 - x) / (1 - b)
    else:
        return 0 

def skewtent_data_generator(b1=0.65, b2=0.47, initial_values=None):
    '''
    Generates time series data for a Master-Slave system of coupled skew-tent maps.
    
    This function simulates a unidirectional coupling where the Master system (Y) 
    evolves independently, and the Slave system (X) is influenced by the Master 
    based on a coupling coefficient 'eta'. Data is generated for multiple 
    coupling strengths, with initial transients removed to ensure the system 
    is on a chaotic attractor.
    
    The simulation follows these governing equations:
    M(t) = T1(M(t-1))
    S(t) = (1 - eta) * T2(S(t-1)) + eta * M(t)
    
    Args:
        b1 (float): Skewness parameter for the Master map (T1). Defaults to 0.65.
        b2 (float): Skewness parameter for the Slave map (T2). Defaults to 0.47.
        initial_values (dict, optional): Manual starting points for specific 
            coupling strengths. Format: {"0.1": {"M": val, "S": val}}. 
            If None, values are chosen randomly from U(0, 1).
            
    Returns:
        None: Saves a nested dictionary to 'dataset/coupled_map_data.json' 
            containing 'X' (Slave), 'Y' (Master), and 'initial_conditions'.
            
    Note:
        - Number of trials per eta: 2000
        - Time series length: 2000 (after discarding 500 transient steps).
        - Coupling (eta) range: [0.0, 0.1, ..., 0.9].
    '''
    cdef int num_trials = 2000
    cdef int sequence_length = 2000
    cdef int transient_cutoff = 500
    cdef int total_steps = sequence_length + transient_cutoff

    coupling_coefficients = np.arange(0, 1.0, 0.1)
    results = {}

    print(f"{'Coupling (η)':<15} | {'Initial M':<15} | {'Initial S':<15}")

    for eta in coupling_coefficients:
        eta_key = f"{eta:.1f}"
        results[eta_key] = {"X": [], "Y": []}
        
        for trial in range(num_trials):
            if initial_values and eta_key in initial_values:
                m_val = initial_values[eta_key]['M']
                s_val = initial_values[eta_key]['S']
            else:
                m_val = np.random.uniform(0, 1)
                s_val = np.random.uniform(0, 1)

            if trial == 0:
                print(f"{eta_key:<15} | {m_val:<15.6f} | {s_val:<15.6f}")

            m_series = []
            s_series = []

            for t in range(total_steps):
                m_next = skew_tent_map(m_val, b1)
                s_next = (1 - eta) * skew_tent_map(s_val, b2) + eta * m_next
                
                m_val = m_next
                s_val = s_next

                if t >= transient_cutoff:
                    m_series.append(m_val)
                    s_series.append(s_val)

            results[eta_key]["Y"].append(m_series)
            results[eta_key]["X"].append(s_series)

    # Save to JSON
    output_path = 'dataset/skewtent_maps_data.json'
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"\nData generation complete. Saved to: {output_path}")


# # ===================================================
# Data extraction function

def extract_json(file_path='dataset/skewtent_maps_data.json'):
    '''
    Loads the skew-tent map time series data from a JSON file.
    
    Args:
        file_path (str): The path to the generated JSON file.
        
    Returns:
        dict: The parsed JSON data containing Master (Y) and Slave (X) 
              series for each coupling coefficient.
              
    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
    '''
    print(f"Loading data")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data found at {file_path}. Please run the generator first.")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully extracted data from {file_path}")
        return data
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return None