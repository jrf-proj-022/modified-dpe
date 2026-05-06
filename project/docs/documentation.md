# DPE Causal Discovery — Full API Documentation

> **Dictionary-Based Pattern Entropy (DPE) Method for Causal Inference**
> A framework for extracting binary patterns from time-series data and quantifying their predictive power using the R_flip ratio, with Cython-optimized kernels and NetworkX visualizations.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Mathematical Background](#2-mathematical-background)
3. [Module: `cy_utils.pyx` — Cython Computational Kernels](#3-module-cy_utilspyx--cython-computational-kernels)
4. [Module: `utils.py` — Pure Python Reference Implementation](#4-module-utilspy--pure-python-reference-implementation)
5. [Module: `main.py` — CLI Entry Point](#5-module-mainpy--cli-entry-point)
6. [The Full Analysis Pipeline](#6-the-full-analysis-pipeline)
7. [Synthetic Data Generators](#7-synthetic-data-generators)
8. [Output Reference](#8-output-reference)
9. [Notes on `cy_utils` vs `utils`](#9-notes-on-cy_utils-vs-utils)

---

## 1. Architecture Overview

The package is split into three layers:

```
main.py              ← CLI entry point; orchestrates the pipeline
cy_utils.pyx         ← Cython-optimized kernels (production speed)
utils.py             ← Pure-Python mirror (readable reference / fallback)
```

Both `cy_utils` and `utils` expose the same public API. `cy_utils` uses typed
C-level memory views (`cdef const cnp.uint8_t[:]`) and is compiled with
`boundscheck=False` for maximum throughput on long sequences. `utils.py` uses
standard NumPy and is easier to read, debug, and extend.

---

## 2. Mathematical Background

### The R_flip Ratio

The central quantity is **R_flip**, the fraction of times a pattern P observed
in sequence X is followed by a bit-flip in sequence Y:

```
R_flip(P) = |{occurrences of P in X where Y flips}| / |{all occurrences of P in X}|
```

- R_flip → 1.0: P is a **perfect causal predictor** (every occurrence precedes a change in Y).
- R_flip → 0.0: P has **no predictive power** (Y never changes when P is observed).
- R_flip ≈ 0.5: P is consistent with **statistical independence**.

### Weighted Binary Entropy

For each pattern P with ratio r and occurrence count c in a sequence of length N:

```
H_b(r)  = -(r·log₂r + (1-r)·log₂(1-r))      [standard binary entropy]
w(P)    = c / (N - |P| + 1)                   [frequency weight]
WH(P)   = w(P) · H_b(r)                       [weighted entropy]
```

### Normalized Weighted Entropy (W)

```
W = mean( WH(P) ) over all P in the pattern dictionary
```

- W → 1.0 under independence (R_flip ≈ 0.5 for all patterns).
- W → 0.0 under strong causation (R_flip → 0 or 1 for all patterns).

### Causal Direction Decision

Given two directions X→Y and Y→X:

```
gap = W(X→Y) - W(Y→X)
```

The direction with the **lower** W is the dominant causal direction. A surrogate
permutation test (default 100 surrogates, 95th percentile) provides a
data-driven significance threshold.

---

## 3. Module: `cy_utils.pyx` — Cython Computational Kernels

Compiled with:
```
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
```

---

### `_ensure_uint8(seq)`

```python
_ensure_uint8(seq: Any) -> bytes | np.ndarray
```

**Internal helper.** Converts an input sequence into a contiguous `uint8` byte
buffer suitable for C-level memory views.

| Input type | Output |
|---|---|
| `str` | `bytes` (ASCII encoded) |
| `list` | `np.ndarray(dtype=uint8)` |
| anything else | `np.asanyarray(..., dtype=uint8)` |

Not intended for direct use; called internally by all kernel functions.

---

### `calculate_causal_history(seq_1, seq_2)`

```python
calculate_causal_history(seq_1, seq_2) -> set[str]
```

Constructs the **causal history dictionary G(X→Y)** by scanning the target
sequence (Y = `seq_1`) from left to right and recording the corresponding
sub-segment of the source sequence (X = `seq_2`) at each bit-flip position.

**Parameters**

| Name | Type | Role |
|---|---|---|
| `seq_1` | `str / list / np.ndarray` | Target sequence **Y** |
| `seq_2` | `str / list / np.ndarray` | Source sequence **X** |

**Returns** `set[str]` — unique X-segments aligned with Y's transition points.

**Algorithm**

```
last_position = 0
for k in 1 .. len(Y)-1:
    if Y[k] != Y[k-1]:          # bit flip detected
        G.add( X[last_position : k+1] )
        last_position = k + 1
```

**Example**

```python
from model.cy_utils import calculate_causal_history

Y = "000100"
X = "011010"
G = calculate_causal_history(Y, X)
# G contains X-segments captured at each flip in Y
```

---

### `sliding_xnor_comparison(p1, p2)`

```python
sliding_xnor_comparison(p1, p2) -> set[str]
```

Finds all common sub-sequences shared between two binary strings by sliding
the shorter one across the longer and performing element-wise equality
(conceptually XNOR) comparison. Only matches of **length ≥ 2** are returned.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `p1` | `str / list / np.ndarray` | First binary sub-pattern |
| `p2` | `str / list / np.ndarray` | Second binary sub-pattern |

**Returns** `set[str]` — common subsequences of length ≥ 2.

**Note:** The function automatically swaps inputs so the shorter sequence is
always the sliding window, ensuring all possible alignments are covered.

---

### `generate_pattern_dictionary(G_input)`

```python
generate_pattern_dictionary(G_input) -> set[str]
```

Builds the **pattern dictionary P(X→Y)** from a causal history set G by
running `sliding_xnor_comparison` on every distinct pair of elements in G and
collecting all extracted common sub-patterns.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `G_input` | `set / list` | Output of `calculate_causal_history` |

**Returns** `set[str]` — the pattern dictionary. Falls back to returning
`G_input` unchanged if no cross-pair patterns can be extracted (e.g., G
contains only one element, or all pairs are entirely dissimilar).

---

### `calculate_contribution_analysis(patterns, X_in, Y_in)`

```python
calculate_contribution_analysis(patterns, X_in, Y_in) -> pd.DataFrame
```

The core **contribution analysis** function. For every pattern P in the
dictionary it:

1. Scans sequence X for all occurrences of P (overlapping allowed).
2. For each occurrence, checks whether any bit-flip exists in the aligned
   window of Y.
3. Computes R_flip = Change / (Change + No-change).

**Parameters**

| Name | Type | Description |
|---|---|---|
| `patterns` | `set / list` | Pattern dictionary P(X→Y) |
| `X_in` | `str / list / np.ndarray` | Source sequence X |
| `Y_in` | `str / list / np.ndarray` | Target sequence Y |

**Returns** `pd.DataFrame` with index `Pattern` and columns:

| Column | Type | Description |
|---|---|---|
| `Change` | `int` | Occurrences of P in X where Y flips |
| `No-change` | `int` | Occurrences of P in X where Y does not flip |
| `Ratio` | `float` | R_flip score, rounded to 4 decimal places |

---

### `discrete(sequence, n_bins=2)`

```python
discrete(sequence: np.ndarray, n_bins: int = 2) -> np.ndarray
```

Discretizes a continuous time series into uniform integer bins. Used as a
pre-processing step before applying the DPE method to real-valued data.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `sequence` | `np.ndarray` | — | Input continuous time series |
| `n_bins` | `int` | `2` | Number of discrete levels |

**Returns** `np.ndarray` of integers in `[0, n_bins - 1]`.

**Edge cases:** Returns an empty array for empty input; returns all-zeros for
a constant (zero-range) sequence.

---

### `weighted_binary_entropy(ratio, count, seq_len, pattern_len)`

```python
weighted_binary_entropy(ratio, count, seq_len, pattern_len) -> tuple[float, float]
```

Computes the frequency-weighted binary entropy for a single pattern.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `ratio` | `float` | R_flip for this pattern |
| `count` | `int` | Total occurrences (Change + No-change) |
| `seq_len` | `int` | Length of the original sequence |
| `pattern_len` | `int` | Length of the pattern string |

**Returns** `(weighted_entropy, weight)` where:
- `weight = count / (seq_len - pattern_len + 1)`
- `weighted_entropy = weight * H_b(ratio)`

**Special cases:** `H_b(0) = H_b(1) = 0` (no entropy at deterministic extremes).

---

### `compute_scale_entropy(stats_df, seq_len)`

```python
compute_scale_entropy(stats_df: pd.DataFrame, seq_len: int) -> pd.DataFrame
```

Groups patterns by length ("scale") and computes the average and normalized
weighted entropy at each scale. Useful for multi-scale causal analysis.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `stats_df` | `pd.DataFrame` | Output of `calculate_contribution_analysis` |
| `seq_len` | `int` | Length of the sequence (X or Y) |

**Returns** `pd.DataFrame` with columns:

| Column | Description |
|---|---|
| `Scale` | Pattern length |
| `Pattern Set` | Set of patterns at this length |
| `Avg Weighted Entropy` | Mean WH across patterns of this length |
| `Normalized Weighted Entropy` | entropy_sum / weight_sum at this scale |

---

### `normalized_weighted_entropy(stats, seq_len)`

```python
normalized_weighted_entropy(stats: pd.DataFrame, seq_len: int) -> tuple[float, pd.Series]
```

Computes the scalar **W** statistic used for the final causal decision, plus
the per-pattern weighted entropy series.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `stats` | `pd.DataFrame` | Output of `calculate_contribution_analysis` |
| `seq_len` | `int` | Length of the sequence used |

**Returns** `(W, weighted_entropies_series)` where:
- `W` = `mean(WH(P))` over all patterns (mean of per-pattern weighted entropies).
- `weighted_entropies_series` is a `pd.Series` of individual `WH(P)` values.

**Note:** Returns `0.0` for an empty or `None` stats DataFrame.

> **Divergence between `cy_utils` and `utils`:** The Cython version divides by
> cardinality (`mean`). The pure-Python version also uses `mean`. Both are
> consistent as of the current codebase.

---

### `surrogate_threshold(X, Y, n_surrogates=100, percentile=95)`

```python
surrogate_threshold(X, Y, n_surrogates=100, percentile=95) -> float
```

Estimates a significance threshold for the entropy gap under the null
hypothesis of independence, using a permutation (surrogate) test.

**Algorithm:** Randomly shuffles both X and Y `n_surrogates` times, recomputes
the full DPE pipeline for each shuffle, and records `|W_XY - W_YX|`. Returns
the `percentile`-th value of the resulting gap distribution.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `X` | `str / list` | — | First binary sequence |
| `Y` | `str / list` | — | Second binary sequence |
| `n_surrogates` | `int` | `100` | Number of random permutations |
| `percentile` | `int` | `95` | Percentile of gap distribution to use as threshold |

**Returns** `float` — the surrogate threshold value.

---

### `identify_causality(stats_x_y, stats_y_x, X, Y, n_surrogates=100)`

```python
identify_causality(stats_x_y, stats_y_x, X, Y, n_surrogates=100) -> tuple[str, float]
```

The **final decision function**. Compares the normalized weighted entropy in
both directions and applies the surrogate threshold to declare a result.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `stats_x_y` | `pd.DataFrame` | Contribution analysis for X→Y direction |
| `stats_y_x` | `pd.DataFrame` | Contribution analysis for Y→X direction |
| `X` | `str` | Source binary sequence |
| `Y` | `str` | Target binary sequence |
| `n_surrogates` | `int` | Surrogate permutations for threshold estimation |

**Returns** `(direction, abs_gap)`:

| `direction` | Condition |
|---|---|
| `'X -> Y'` | `gap < 0` and `abs_gap >= threshold` |
| `'Y -> X'` | `gap > 0` and `abs_gap >= threshold` |
| `'Independence'` | `abs_gap < threshold` |

---

### `run_causal_analysis(seq_x_bin, seq_y_bin)`

```python
run_causal_analysis(seq_x_bin: np.ndarray, seq_y_bin: np.ndarray) -> tuple[str, float]
```

**Convenience wrapper** that executes the entire DPE pipeline in one call,
designed for programmatic / batch use (e.g., looping over many sequence pairs).

**Parameters**

| Name | Type | Description |
|---|---|---|
| `seq_x_bin` | `np.ndarray` | Binary array for sequence X |
| `seq_y_bin` | `np.ndarray` | Binary array for sequence Y |

**Returns** `(direction, abs_gap)` — same as `identify_causality`.

**Pipeline executed internally:**

```
calculate_causal_history  (both directions)
    ↓
generate_pattern_dictionary  (both directions)
    ↓
calculate_contribution_analysis  (both directions)
    ↓
identify_causality  (with surrogate test)
```

---

### `draw_network(W_p_x, W_p_y, file_name='causal_network_dual')`

```python
draw_network(W_p_x: pd.Series, W_p_y: pd.Series, file_name: str = 'causal_network_dual') -> None
```

Renders the **Causal Proximity Network** graph and saves it as a 300 DPI PNG.

**Layout logic:**

- Nodes **X** and **Y** are placed as hubs at fixed positions.
- Pattern nodes exclusive to X are fanned out to the left.
- Pattern nodes exclusive to Y are fanned out to the right.
- Shared patterns are positioned between the hubs using a distance-weighted
  trilateration formula: `Distance = 1 + H(P)`.
- Edge labels show the weighted entropy value for each pattern→hub link.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `W_p_x` | `pd.Series` | Weighted entropies indexed by pattern (X→Y direction) |
| `W_p_y` | `pd.Series` | Weighted entropies indexed by pattern (Y→X direction) |
| `file_name` | `str` | Output file name (no extension); saved under `results/` |

**Output:** `results/{file_name}.png` at 300 DPI.

---

## 4. Module: `utils.py` — Pure Python Reference Implementation

`utils.py` mirrors every function in `cy_utils.pyx` without Cython syntax. It
additionally contains:

---

### `barchart_analysis(stats, file_name, direction)`

```python
barchart_analysis(stats: pd.DataFrame, file_name: str = 'causal_analysis_chart', direction: str = 'x->y') -> None
```

Generates and saves a bar chart of R_flip scores for all extracted patterns.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `stats` | `pd.DataFrame` | — | Output of `calculate_contribution_analysis` |
| `file_name` | `str` | `'causal_analysis_chart'` | Output filename (no extension) |
| `direction` | `str` | `'x->y'` | Axis label string; used in axis title |

**Behaviour:**
- Skips and prints a warning if `stats` is empty or lacks a `Ratio` column.
- Creates the `results/` directory if it does not exist.
- Annotates each bar with its exact ratio value.
- Saves to `results/{file_name}.png` at 300 DPI.

---

### `extract_json_data(file_path, as_numpy=True)`

```python
extract_json_data(file_path: str, as_numpy: bool = True) -> dict
```

Loads a generated dataset JSON file and optionally converts the lists back to
NumPy arrays.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str` | — | Path to the JSON dataset file |
| `as_numpy` | `bool` | `True` | Convert inner lists to `np.ndarray(float64)` |

**Returns** a nested dictionary:
```
{
  "0.1": { "X": <array (n_trials, seq_len)>, "Y": <array (n_trials, seq_len)> },
  "0.2": { ... },
  ...
}
```

**Raises**
- `FileNotFoundError` if the path does not exist.
- `json.JSONDecodeError` if the file is not valid JSON.

---

## 5. Module: `main.py` — CLI Entry Point

```python
# Invoked via:
analyze-causality        # after pip install -e .
# or
python -m model.main
```

### `main()`

Runs the interactive command-line analysis pipeline.

**Steps performed:**

| Step | Function called | Output |
|---|---|---|
| 1. Read input | `input()` | Sequences X and Y from stdin |
| 2. Causal history | `calculate_causal_history` (both directions) | G(X→Y), G(Y→X) |
| 3. Pattern extraction | `generate_pattern_dictionary` | P(X→Y), P(Y→X) |
| 4. Contribution analysis | `calculate_contribution_analysis` | stats DataFrames |
| 5. Weighted entropy | `normalized_weighted_entropy` | W_xy, W_yx scalars |
| 6. Bar charts | `barchart_analysis` | PNGs in `results/` |
| 7. Network graph | `draw_network` | PNG in `results/` |
| 8. Final verdict | `identify_causality` | Direction string + gap |

**Input format:** Both X and Y must be binary strings of equal length (composed
of `'0'` and `'1'` characters only).

**Example session:**

```
$ analyze-causality
Causal Pattern Analysis Pipeline
Enter sequence Y
000001000010000000000100001000
Enter sequence X
011101111010011001110101101001

Final Dictionaries
    Gx->y: {'011101', '11010', ...}
    Gy->x: {'000001', '10000', ...}
...
Detected Causal Direction: X -> Y
Gap: 0.0842
```

---

## 6. The Full Analysis Pipeline

```
Binary sequences X, Y (equal length)
            │
            ▼
  calculate_causal_history(Y, X)  ──→  G(X→Y)  [X-segments at Y's flips]
  calculate_causal_history(X, Y)  ──→  G(Y→X)
            │
            ▼
  generate_pattern_dictionary(G)  ──→  P(X→Y), P(Y→X)
  [sliding XNOR comparison on all pairs in G]
            │
            ▼
  calculate_contribution_analysis(P, X, Y)
  [R_flip per pattern]  ──→  stats DataFrame
            │
            ▼
  normalized_weighted_entropy(stats, len)
  [W scalar per direction]  ──→  W_xy, W_yx
            │
            ▼
  identify_causality(stats, stats, X, Y)
  [surrogate test + gap comparison]
            │
            ▼
  Output: "X -> Y" | "Y -> X" | "Independence"
```

---

## 7. Synthetic Data Generators

Both modules include generators for producing ground-truth datasets.

---

### AR(1) Coupled Process

#### `ar1_pair_generator(length, noise_intensity, a, b, phi)`

```python
ar1_pair_generator(length=1000, noise_intensity=0.01, a=0.8, b=0.8, phi=0) -> tuple[np.ndarray, np.ndarray]
```

Generates a bivariate coupled AR(1) process:

```
X[t] = a·X[t-1] + φ·Y[t-1] + ε_x
Y[t] = b·Y[t-1]             + ε_y
```

When `phi > 0`, Y causally drives X. Used to validate the DPE method against
known ground truth.

| Parameter | Default | Description |
|---|---|---|
| `length` | `1000` | Number of time steps |
| `noise_intensity` | `0.01` | Gaussian noise scaling factor |
| `a` | `0.8` | Self-dependence of X |
| `b` | `0.8` | Self-dependence of Y |
| `phi` | `0` | Coupling strength (Y→X) |

---

#### `ar1_data_generator(num_sequences, sequence_length, [transients])`

```python
# cy_utils version:
ar1_data_generator(num_sequences=2000, sequence_length=2000, transients=500) -> None

# utils version:
ar1_data_generator(num_sequences=1000, sequence_length=1000) -> None
```

Sweeps phi from 0 to 1 in steps of 0.1 (cy_utils) / 0.05 (utils), generates
`num_sequences` realizations per phi, and saves the result to
`dataset/ar1_data.json`.

The cy_utils version discards `transients` initial steps per realization
to ensure the system is past its warm-up period.

---

### Coupled Skew-Tent Maps

#### `skew_tent_map(x, b)`

```python
skew_tent_map(x: float, b: float) -> float
```

Computes one iteration of the piecewise-linear chaotic Skew-Tent map:

```
T(x) = x / b          if 0 ≤ x < b
T(x) = (1-x) / (1-b)  if b ≤ x < 1
```

`b` controls the position of the peak (skewness). The map is ergodic and
chaotic for all `b ∈ (0, 1)`.

---

#### `skewtent_data_generator` / `generate_coupled_data`

```python
# cy_utils:
skewtent_data_generator(b1=0.65, b2=0.47, initial_values=None) -> None

# utils:
generate_coupled_data(b1=0.65, b2=0.47, initial_values=None) -> None
```

Simulates a **Master–Slave** coupled system:

```
M(t) = T1( M(t-1) )
S(t) = (1 - η)·T2( S(t-1) ) + η·M(t)
```

Y = Master (evolves independently), X = Slave (influenced by Master via η).
Data is generated for η ∈ {0.0, 0.1, ..., 0.9} with 500 transient steps
discarded per trial.

Saves to `dataset/coupled_map_data.json` (utils) or
`dataset/skewtent_maps_data.json` (cy_utils).

| Parameter | Description |
|---|---|
| `b1` | Skewness of Master map T1 |
| `b2` | Skewness of Slave map T2 |
| `initial_values` | Optional dict of manual starting conditions per eta |

---

#### `extract_json` / `extract_json_data`

```python
# cy_utils:
extract_json(file_path='dataset/skewtent_maps_data.json') -> dict

# utils:
extract_json_data(file_path: str, as_numpy: bool = True) -> dict
```

Loads a previously generated JSON dataset. The `utils` version can
auto-convert lists to NumPy arrays (`as_numpy=True`).

---

## 8. Output Reference

| File | Location | Description |
|---|---|---|
| `causal_analysis_chart_xy.png` | `results/` | R_flip bar chart for X→Y patterns |
| `causal_analysis_chart_yx.png` | `results/` | R_flip bar chart for Y→X patterns |
| `causal_network_graph.png` | `results/` | Causal proximity network (300 DPI) |
| `ar1_data.json` | `dataset/` | AR(1) simulation dataset |
| `coupled_map_data.json` | `dataset/` | Skew-tent map simulation dataset |

All PNGs are saved at **300 DPI** with `bbox_inches='tight'`.

---

## 9. Notes on `cy_utils` vs `utils`

| Aspect | `cy_utils.pyx` | `utils.py` |
|---|---|---|
| Speed | Compiled C extensions; fastest | Pure Python / NumPy |
| Portability | Requires C compiler + Cython build | Runs anywhere with NumPy |
| `calculate_causal_history` | Raises no length error (commented out) | Raises `ValueError` on length mismatch |
| `ar1_data_generator` phi step | 0.1 | 0.05 |
| `ar1_data_generator` transients | Explicit `transients=500` arg | Not exposed; no warm-up |
| `normalized_weighted_entropy` return | `(mean, Series)` tuple | Scalar `mean` only |
| `identify_causality` calls `normalized_weighted_entropy` | Unpacks tuple `(val, _)` | Calls scalar version directly |
| Dataset JSON path (skew-tent) | `skewtent_maps_data.json` | `akewtent_maps_data.json` *(typo)* |

> **Recommendation:** Use `cy_utils` for production runs. Use `utils.py` for
> debugging, testing, and understanding the algorithm. Ensure both remain in
> sync when modifying the core logic.