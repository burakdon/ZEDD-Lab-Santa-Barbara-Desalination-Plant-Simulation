#!/usr/bin/env python3
"""
Overlay saved Pareto fronts for multiple cases without rerunning simulations.

Usage examples:
  python overlay_pareto.py --drought pers87_sev0.83n_4 --cases 14 43 \
      --labels "Case 14 (peaky)" "Case 43 (neg)" \
      --out result/plots/pareto/overlay_pers87_14_vs_43.png
  
  python overlay_pareto.py --drought pers87_sev0.83n_4 \
      --cases basetariff_baseline/3mpd_30vessels basetariff_flexible/3mpd_30vessels \
      --out result/plots/pareto/overlay_baseline_vs_flexible.png

If labels are omitted, labels default to case numbers.
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

# Add src to path and import directly to avoid triggering src/__init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cost_curve_loader import CostCurveLoader


def parse_case_identifier(case_identifier):
    """
    Parse case identifier to extract folder and curve name.
    Returns (folder, curve_name) where folder can be None for root-level cases.
    """
    case_str = str(case_identifier).strip()
    if '/' in case_str:
        parts = case_str.split('/', 1)
        folder = parts[0]
        curve_name = parts[1]
        # Extract just the folder name (e.g., "basetariff_baseline" -> "baseline")
        folder_short = folder.split('_')[-1] if '_' in folder else folder
        return folder_short, curve_name
    return None, case_str


def format_case_for_filename(case_identifier):
    """Format case identifier for use in filenames."""
    folder, curve_name = parse_case_identifier(case_identifier)
    if folder:
        return f"{folder}_{curve_name}"
    return str(case_identifier)


def load_pareto_csv(drought: str, case_id):
    """Load Pareto CSV, handling both old and new filename formats."""
    case_filename = format_case_for_filename(case_id)
    path = f"result/data/pareto/pareto_{drought}_case_{case_filename}.csv"
    if not os.path.exists(path):
        # Try old format as fallback
        old_path = f"result/data/pareto/pareto_{drought}_case_{case_id}.csv"
        if os.path.exists(old_path):
            path = old_path
        else:
            raise FileNotFoundError(f"Pareto CSV not found: {path} or {old_path}. Run main.py for this case first.")
    df = pd.read_csv(path)
    return df


def get_capacity_for_case(case_identifier, loader):
    """
    Get the maximum production capacity (AF/month) for a case from its cost curve.
    
    Returns:
        Maximum capacity in AF/month
    """
    try:
        max_summer = loader.get_max_production(case_identifier, is_summer=True)
        max_winter = loader.get_max_production(case_identifier, is_summer=False)
        return max(max_summer, max_winter)
    except Exception as e:
        print(f"Warning: Could not load capacity for case {case_identifier}: {e}")
        return None


def get_colors_by_capacity(cases):
    """
    Generate colors for curves based on their capacity values.
    Uses a gradient from yellow (low capacity) to dark blue (high capacity).
    
    Returns:
        Dictionary mapping case_identifier to (color_rgb, capacity_value)
    """
    loader = CostCurveLoader()
    
    # Get capacities for all cases
    capacities = {}
    for case in cases:
        cap = get_capacity_for_case(case, loader)
        if cap is not None:
            capacities[case] = cap
    
    if not capacities:
        # Fallback: return gray for all if no capacities found
        return {case: (mcolors.hex2color('#808080'), 0) for case in cases}
    
    # Create colormap: yellow (low) to dark blue (high)
    # Yellow: #FFFF00, Dark Blue: #000080
    min_cap = min(capacities.values())
    max_cap = max(capacities.values())
    
    # Normalize capacities to [0, 1] range
    normalized_caps = {}
    if max_cap > min_cap:
        for case, cap in capacities.items():
            normalized_caps[case] = (cap - min_cap) / (max_cap - min_cap)
    else:
        # All capacities are the same
        normalized_caps = {case: 0.5 for case in capacities.keys()}
    
    # Create custom colormap from yellow to dark blue
    colors_yellow_to_blue = ['#FFFF00', '#0080FF', '#000080']  # Yellow, Medium Blue, Dark Blue
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('yellow_to_blue', colors_yellow_to_blue, N=n_bins)
    
    # Map normalized capacities to colors
    color_map = {}
    for case in cases:
        if case in normalized_caps:
            norm_val = normalized_caps[case]
            color_rgba = cmap(norm_val)
            color_map[case] = (color_rgba[:3], capacities[case])  # RGB tuple and capacity value
        else:
            color_map[case] = (mcolors.hex2color('#808080'), 0)  # Gray for missing data
    
    return color_map


def overlay(drought: str, cases: list, labels: list = None, out: str = None, title: str = None):
    plt.figure(figsize=(10, 8))  # Larger figure size for bigger plot area
    labs = labels if labels and len(labels) == len(cases) else [f"case {c}" for c in cases]

    # Get colors based on capacity (yellow to dark blue gradient)
    color_map = get_colors_by_capacity(cases)

    for case, lab in zip(cases, labs):
        df = load_pareto_csv(drought, case)
        color_rgb, capacity = color_map.get(case, (mcolors.hex2color('#808080'), 0))
        
        # Convert RGB tuple to hex for matplotlib
        color_hex = mcolors.rgb2hex(color_rgb)
        
        # All curves use the same style now, colored by capacity
        plt.scatter(df['cost'], df['risk_months_supply'], s=22, 
                   c=color_hex, label=f"{lab} ({capacity:.0f} AF/mo)", 
                   marker='o', alpha=0.7, edgecolors='black', linewidths=0.5)

    plt.xlabel('cost')
    plt.ylabel('# demand months left in storage (risk)')
    if title:
        plt.title(title)
    
    # Place legend at the bottom of the plot with multiple columns
    # Use smaller font and 3 columns for better horizontal layout
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
               fontsize=8, ncol=3, framealpha=0.9, columnspacing=1.0)
    plt.tight_layout(rect=[0, 0.15, 1, 1])  # Leave space at bottom for legend

    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        # Use bbox_inches='tight' to ensure legend is included in saved image
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved overlay plot to: {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--drought', required=True, help='Drought type string used in saved filenames')
    p.add_argument('--cases', nargs='+', required=True, help='List of case identifiers to overlay')
    p.add_argument('--labels', nargs='*', help='Optional list of labels same length as cases')
    p.add_argument('--out', help='Output PNG path for the overlay plot')
    p.add_argument('--title', help='Custom title for the plot')
    args = p.parse_args()

    cases = [c.strip() for c in args.cases if c.strip()]
    labels = args.labels
    if labels:
        labels = [lab.strip() for lab in labels]
    overlay(args.drought, cases, labels, args.out, args.title)


if __name__ == '__main__':
    main()


