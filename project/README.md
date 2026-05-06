# Dictionary-Based Pattern Entropy (DPE) for Causal Inference

A causal inference framework that extracts binary patterns from time-series data and quantifies their predictive power using the **R_flip ratio**. The method includes Cython-optimized computational kernels for high-throughput analysis and aesthetic NetworkX visualizations to communicate causal strength.

---

## How It Works

The DPE method identifies causality between two binary sequences X and Y in five stages:

```
1. Causal History Extraction   — Scan Y for bit-flips; record the aligned X-segment at each flip → G(X→Y)
2. Pattern Dictionary          — Find common sub-patterns across all pairs in G → P(X→Y)
3. Contribution Analysis       — For each pattern P, compute R_flip = flips triggered / total occurrences
4. Weighted Entropy            — Summarize the pattern dictionary into a single scalar W per direction
5. Causal Decision             — Compare W(X→Y) vs W(Y→X) against a surrogate significance threshold
```

A lower W indicates stronger causal structure. The direction with the lower W is the dominant cause.

---

## Features

- **Causal History Extraction** — identifies sub-patterns in X associated with bit-flips in Y
- **Cython Kernels** — boundscheck-free C-level memory views for fast sequence scanning
- **Surrogate Testing** — permutation-based null distribution for statistically grounded decisions
- **Multi-scale Entropy** — entropy analysis grouped by pattern length
- **Automated Visualization** — 300 DPI bar charts and radial network graphs saved to `results/`
- **Synthetic Benchmarks** — AR(1) coupled processes and skew-tent chaotic map generators for ground-truth validation

---

## Project Structure

```
DPE/
└── causal-method/
    ├── environment.yml        # Conda environment
    ├── pyproject.toml         # Build system & CLI entry point
    ├── setup.py               # Cython compilation script
    ├── requirements.txt       # Python dependencies
    ├── README.md
    ├── docs/                  # Extended documentation with worked examples
    └── src/
        └── model/
            ├── __init__.py
            ├── main.py        # CLI entry point (analyze-causality command)
            ├── cy_utils.pyx   # Cython-optimized computational kernels
            ├── utils.py       # Pure-Python reference implementation
            └── demo.py        # Programmatic usage examples
    └── results/               # Auto-generated PNG outputs
└── experiments/
    └── ETCPy/                 # ETCPy package fork (comparative baseline)
    └── results/
```

---

## Installation

### Prerequisites

A C compiler is required to build the Cython extension:
- **Linux / macOS**: GCC (usually pre-installed; `sudo apt install gcc` if missing)
- **Windows**: MSVC via [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### 1. Get the Code

**Option A — Clone from the repository:**
```bash
git clone <repository-url>
```

### 2. Create the Environment and Install

```bash
# Navigate to the project directory
cd causal-method

# Create and activate the conda environment
conda env create -f environment.yml
conda activate dpe

# Install the package in editable mode (registers the analyze-causality CLI)
pip install -e .

# Compile the Cython extension
python setup.py build_ext --inplace
```

### 3. Install the ETCPy Baseline (optional)
**Clone from the repository:**
## References
Pranay SY, Nithin Nagaraj, Causal discovery using compression-complexity measures, Journal of Biomedical Informatics, Volume 117,
2021,103724, ISSN 1532-0464, [Paper](https://www.sciencedirect.com/science/article/pii/S1532046421000538?via%3Dihub), [GitHub](https://github.com/pranaysy/ETCPy/)
```bash
cd ../experiments/ETCPy/
pip install -e . --no-build-isolation
cd ..
```

---

## Usage

### Command Line (Recommended)

After installation, run from any directory:

```bash
analyze-causality
```

You will be prompted to enter two binary sequences interactively:

```
Causal Pattern Analysis Pipeline
Enter sequence Y
000001000010000000000100001000
Enter sequence X
011101111010011001110101101001

Detected Causal Direction: X -> Y
Gap: 0.0842
```

Three output files are saved automatically to `results/`:
- `causal_analysis_chart_xy.png` — R_flip bar chart for X→Y patterns
- `causal_analysis_chart_yx.png` — R_flip bar chart for Y→X patterns
- `causal_network_graph.png` — Causal proximity network

### Programmatic Usage

For scripted or batch analysis, use `run_causal_analysis` directly (see `demo.py` for full examples):

```python
import numpy as np
from model.cy_utils import run_causal_analysis
from model.utils import discrete

# Example: continuous time series → discretize → infer causality
X_raw = np.random.randn(500)
Y_raw = 0.8 * X_raw + np.random.randn(500) * 0.1   # Y is driven by X

X_bin = discrete(X_raw, n_bins=2)
Y_bin = discrete(Y_raw, n_bins=2)

direction, gap = run_causal_analysis(X_bin, Y_bin)
print(f"Direction: {direction}, Gap: {gap:.4f}")
# Direction: X -> Y, Gap: 0.xxxx
```

For a step-by-step breakdown of the pipeline:

```python
from model.cy_utils import (
    calculate_causal_history,
    generate_pattern_dictionary,
    calculate_contribution_analysis,
    normalized_weighted_entropy,
    identify_causality,
)

X = "011101111010011001110101101001"
Y = "000001000010000000000100001000"

# Step 1: Causal histories
Gx_y = calculate_causal_history(Y, X)
Gy_x = calculate_causal_history(X, Y)

# Step 2: Pattern dictionaries
Px_y = generate_pattern_dictionary(Gx_y)
Py_x = generate_pattern_dictionary(Gy_x)

# Step 3: Contribution analysis
stats_xy = calculate_contribution_analysis(Px_y, X, Y)
stats_yx = calculate_contribution_analysis(Py_x, Y, X)

# Step 4 & 5: Final verdict
direction, gap = identify_causality(stats_xy, stats_yx, X, Y)
print(f"Direction: {direction}, Gap: {gap:.4f}")
```

---

## Visualizations

### R_flip Bar Chart

Plots the R_flip ratio for every extracted pattern in the dictionary.

- **Y-axis**: R_flip score (0.0 – 1.0)
- **Score of 1.0**: "Perfect Predictor" — every time this pattern appears in X, a flip occurs in Y
- **Score ≈ 0.5**: Pattern is consistent with independence
- **Score of 0.0**: Pattern actively suppresses change

### Causal Proximity Network

A radial directed graph where patterns are positioned at a distance proportional to their weighted entropy (`Distance = 1 + H(P)`) from their target hub (X or Y).

- **Orange node**: Hub X
- **Purple node**: Hub Y
- **Blue nodes**: Patterns exclusive to the X→Y direction
- **Yellow nodes**: Patterns exclusive to the Y→X direction
- **Violet nodes**: Shared patterns influencing both directions
- **Edge labels** (red): Weighted entropy score for that pattern–hub link

---

## Synthetic Benchmarks

Two ground-truth generators are included for validating the method.

### AR(1) Coupled Process

```python
from model.utils import ar1_pair_generator, discrete

# Y causes X when phi > 0
X_raw, Y_raw = ar1_pair_generator(length=1000, phi=0.5, a=0.8, b=0.8)
X_bin = discrete(X_raw, n_bins=2)
Y_bin = discrete(Y_raw, n_bins=2)
```

Model equations:
```
X[t] = a·X[t-1] + φ·Y[t-1] + ε_x
Y[t] = b·Y[t-1]             + ε_y
```

### Coupled Skew-Tent Maps

```python
from model.utils import generate_coupled_data

# Generates data for eta ∈ [0.0, 0.9], saves to dataset/
generate_coupled_data(b1=0.65, b2=0.47)
```

Master–Slave equations:
```
M(t) = T1( M(t-1) )
S(t) = (1 - η)·T2( S(t-1) ) + η·M(t)
```

---

## Cleaning Build Artifacts

```bash
rm -rf build/
find . -name "*.so" -delete
find . -name "*.c" -delete
```

---

## Full API Reference

See `causal-method/docs/documentation.md` for the complete function-level reference including parameter tables, return types, algorithm notes, and differences between `cy_utils` and `utils`.
