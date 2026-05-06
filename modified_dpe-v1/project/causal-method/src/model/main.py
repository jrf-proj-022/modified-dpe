# src/model/main.py
import sys
import os

# Import from siblings in the same package
from model.cy_utils import (
    calculate_causal_history, 
    generate_pattern_dictionary, 
    calculate_contribution_analysis,
    identify_causality,
    normalized_weighted_entropy
)
from model.utils import ( 
    barchart_analysis, 
    draw_network
)

def main():
    """Main entry point for the analyze-causality command."""
    print("Causal Pattern Analysis Pipeline")
    
    # # Example data
    # y = "000001000010000000000100001000"
    # x = "011101111010011001110101101001"

    y = input('Enter sequence Y\n')
    x = input('Enter sequence X\n')

    # 1. Evaluate Direction
    gx_y = calculate_causal_history(y, x)
    gy_x = calculate_causal_history(x, y)

    print(f'''Final Dictionaries
    Gx->y: {gx_y}
    Gy->x: {gy_x}
    ''')

    # 2. Pattern Extraction
    px_y = generate_pattern_dictionary(gx_y)
    py_x = generate_pattern_dictionary(gy_x)

    print("\nFinal Pattern Dictionary PX->Y:")
    print(sorted(list(px_y), key=len))
    print("\nFinal Pattern Dictionary PY->X:")
    print(sorted(list(py_x), key=len))
    
    # 3. Contribution Analysis
    stats1 = calculate_contribution_analysis(px_y, x, y)
    stats2 = calculate_contribution_analysis(py_x, y, x)
    print("\nPattern Statistics:")
    print("X -> Y\n ")
    print(stats1)
    print("Y -> X\n")
    print(stats2)

    # 4. Weighted entropies
    _, W_xy = normalized_weighted_entropy(stats1, len(x))
    _, W_yx = normalized_weighted_entropy(stats2, len(y))

    # 5. Visualization
    # Ensure results directory exists relative to current working directory
    if not os.path.exists('results'):
        os.makedirs('results')
    
    barchart_analysis(stats1, 'causal_analysis_chart_xy', direction='x->y')

    barchart_analysis(stats2, 'causal_analysis_chart_yx', direction='y->x')
    draw_network(W_xy, W_yx, 'causal_network_graph')

    direction, gap= identify_causality(stats1, stats2, x, y)
    print(f"Detected Causal Direction: {direction}\nGap: {gap}")
    
if __name__ == "__main__":
    main()