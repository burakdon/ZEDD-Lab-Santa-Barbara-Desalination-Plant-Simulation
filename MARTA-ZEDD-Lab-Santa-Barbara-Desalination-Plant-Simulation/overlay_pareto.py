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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


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


# Color palette for 6 MPD/vessel combinations
CURVE_COLORS = {
    '3mpd_30vessels': '#d62728',  # Red
    '3mpd_36vessels': '#ff7f0e',  # Orange
    '4mpd_30vessels': '#2ca02c',  # Green
    '4mpd_36vessels': '#1f77b4',  # Blue
    '6mpd_36vessels': '#9467bd',  # Purple
    '8mpd_36vessels': '#17becf',  # Teal
}


def get_base_curve_name(case_identifier):
    """Extract the base curve name (MPD/vessel identifier) from case identifier."""
    folder, curve_name = parse_case_identifier(case_identifier)
    return curve_name


def mute_color(color_hex, saturation_reduction=0.5, brightness_increase=0.1):
    """
    Convert a hex color to a muted version.
    
    Args:
        color_hex: Hex color string (e.g., '#d62728')
        saturation_reduction: Amount to reduce saturation (0-1)
        brightness_increase: Amount to increase brightness (0-1)
    
    Returns:
        Muted hex color string
    """
    # Convert hex to RGB (0-1 range)
    rgb = mcolors.hex2color(color_hex)
    
    # Convert RGB to HSV
    hsv = mcolors.rgb_to_hsv(rgb)
    
    # Reduce saturation and increase brightness
    hsv[1] = max(0, hsv[1] - saturation_reduction)
    hsv[2] = min(1, hsv[2] + brightness_increase)
    
    # Convert back to RGB then hex
    rgb_muted = mcolors.hsv_to_rgb(hsv)
    return mcolors.rgb2hex(rgb_muted)


def get_color_for_curve(case_identifier):
    """
    Get color for a curve based on its type (baseline/flexible) and MPD/vessel identifier.
    
    Returns:
        (color, is_baseline): Color hex string and boolean indicating if it's baseline
    """
    folder, curve_name = parse_case_identifier(case_identifier)
    
    # Get base color for this curve type
    base_color = CURVE_COLORS.get(curve_name, '#808080')  # Default to gray if not found
    
    # Determine if it's baseline or flexible
    is_baseline = folder is not None and 'baseline' in folder.lower()
    
    if is_baseline:
        return base_color, True
    else:
        # Mute the color for flexible curves
        muted_color = mute_color(base_color)
        return muted_color, False


def overlay(drought: str, cases: list, labels: list = None, out: str = None, title: str = None):
    plt.figure(figsize=(7, 5))
    labs = labels if labels and len(labels) == len(cases) else [f"case {c}" for c in cases]

    for case, lab in zip(cases, labs):
        df = load_pareto_csv(drought, case)
        color, is_baseline = get_color_for_curve(case)
        
        # Use filled markers for baseline, outlined markers for flexible
        if is_baseline:
            plt.scatter(df['cost'], df['risk_months_supply'], s=22, 
                       c=color, label=lab, marker='o', alpha=0.7, edgecolors='black', linewidths=0.5)
        else:
            plt.scatter(df['cost'], df['risk_months_supply'], s=22,
                       c=color, label=lab, marker='o', alpha=0.7, 
                       edgecolors=color, linewidths=1.5, facecolors='none')

    plt.xlabel('cost')
    plt.ylabel('# demand months left in storage (risk)')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=150)
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


