#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run multiple cost-curve cases with fixed desal utilization.

This script mirrors run_all_cases.py, but replaces the flex operations policy
with a fixed desal production fraction (e.g., 1.0, 0.8, 0.6, 0.4, 0.2).

Usage (from repo root):
  python fixed_desal_experiment/run_all_cases_fixed.py     --drought pers87_sev0.83n_4     --cases all     --fractions 1.0,0.8,0.6,0.4,0.2     --nsim 20     --save-timeseries

Notes:
- Uses the repo's CostCurveLoader to discover the same cases.
- Uses the same cost accounting path as simulation.py / sim_individual.py.
"""

import os
import sys
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is importable when executed as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import OptimizationParameters
from cost_curve_loader import CostCurveLoader
from plot_optimization import plot_timeseries

from fixed_desal_experiment.fixed_sb import SBFixed, SBsimFixed


def parse_case_identifier(case_identifier):
    """Parse case identifiers that may include folders (e.g., 'basetariff_baseline/12')."""
    case_str = str(case_identifier).strip()
    if '/' in case_str:
        parts = case_str.split('/', 1)
        folder = parts[0]
        curve_name = parts[1]
        folder_short = folder.split('_')[-1] if '_' in folder else folder
        return folder_short, curve_name
    return None, case_str


def format_case_for_filename(case_identifier) -> str:
    folder, curve_name = parse_case_identifier(case_identifier)
    if folder:
        return f"{folder}_{curve_name}"
    return str(case_identifier)


def _parse_fractions(fractions_arg: str) -> List[float]:
    vals = []
    for s in fractions_arg.split(','):
        s = s.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError('No fractions provided')
    return vals


def _plot_case_curves(df_case: pd.DataFrame, case_label: str, outdir: str) -> None:
    os.makedirs(os.path.join(outdir, 'plots', 'case_curves'), exist_ok=True)
    df = df_case.sort_values('fraction').reset_index(drop=True)
    x = df['fraction'].to_numpy()

    # cost vs fraction
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x, df['cost_mean'].to_numpy(), marker='o')
    if 'cost_p10' in df.columns and 'cost_p90' in df.columns:
        ax.fill_between(x, df['cost_p10'].to_numpy(), df['cost_p90'].to_numpy(), alpha=0.2)
    ax.set_xlabel('Fixed desal utilization (fraction of capacity)')
    ax.set_ylabel('Mean total cost')
    ax.set_title(f'Cost vs fraction — {case_label}')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'plots', 'case_curves', f'cost_vs_fraction__{case_label}.png'), dpi=150)
    plt.close(fig)

    # risk vs fraction
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x, df['risk_months_mean'].to_numpy(), marker='o')
    if 'risk_months_p10' in df.columns and 'risk_months_p90' in df.columns:
        ax.fill_between(x, df['risk_months_p10'].to_numpy(), df['risk_months_p90'].to_numpy(), alpha=0.2)
    ax.set_xlabel('Fixed desal utilization (fraction of capacity)')
    ax.set_ylabel('Months of supply (25th percentile; higher is better)')
    ax.set_title(f'Risk vs fraction — {case_label}')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'plots', 'case_curves', f'risk_vs_fraction__{case_label}.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Batch-run fixed desal fractions over cost curve cases')
    parser.add_argument('--drought', required=True, help='Drought type, e.g., pers87_sev0.83n_4')
    parser.add_argument('--cases', nargs='+', required=True, help='"all" or list of case identifiers')
    parser.add_argument('--fractions', default='1.0,0.8,0.6,0.4,0.2', help='Comma-separated fractions')
    parser.add_argument('--nsim', type=int, default=20, help='Number of hydrological scenarios')
    parser.add_argument('--save-timeseries', action='store_true', help='Save scenario-0 timeseries for each fraction')
    parser.add_argument('--list-cases', action='store_true', help='List available cost curve cases and exit')
    parser.add_argument('--outdir', default='result/fixed_desal', help='Output directory')
    args = parser.parse_args()

    loader = CostCurveLoader()
    if args.list_cases:
        print('Available cost curve cases:')
        for case in loader.get_available_cases():
            info = loader.get_case_info(case)
            scenario = info.get('scenario') if isinstance(info, dict) else None
            if scenario:
                print(f"  {info['case_number']} (scenario: {scenario})")
            else:
                print(f"  {case}")
        return

    # Resolve case list
    if len(args.cases) == 1 and args.cases[0].lower() == 'all':
        cases = loader.get_available_cases()
    else:
        cases = []
        for c in args.cases:
            c = c.strip()
            if not c:
                continue
            if c.isdigit():
                cases.append(int(c))
            else:
                cases.append(c)

    fractions = _parse_fractions(args.fractions)

    # Set up parameters
    opt_par = OptimizationParameters()
    opt_par.nsim = int(args.nsim)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'plots'), exist_ok=True)
    if args.save_timeseries:
        os.makedirs(os.path.join(outdir, 'plots', 'timeseries'), exist_ok=True)

    drought_type = args.drought

    rows = []
    per_scenario_rows = []

    for case_identifier in cases:
        print(f"Running fixed fractions for case {case_identifier} (drought={drought_type})")

        model = SBFixed(opt_par, case_identifier, drought_type)

        for frac in fractions:
            Jcost_mean, Jrisk_mean = model.simulate_fixed(frac)

            # Convert risk to positive months-of-supply for reporting
            risk_months_mean = -Jrisk_mean

            # Also compute scenario-level stats for uncertainty bands
            scenario_costs, scenario_risks = model.simulate_fixed_per_scenario(frac)

            row = {
                'case': case_identifier,
                'case_label': format_case_for_filename(case_identifier),
                'fraction': frac,
                'cost_mean': Jcost_mean,
                'cost_p10': float(np.percentile(scenario_costs, 10)),
                'cost_p50': float(np.percentile(scenario_costs, 50)),
                'cost_p90': float(np.percentile(scenario_costs, 90)),
                'risk_months_mean': risk_months_mean,
                'risk_months_p10': float(np.percentile(scenario_risks, 10)),
                'risk_months_p50': float(np.percentile(scenario_risks, 50)),
                'risk_months_p90': float(np.percentile(scenario_risks, 90)),
            }
            rows.append(row)

            for i, (cst, rsk) in enumerate(zip(scenario_costs, scenario_risks)):
                per_scenario_rows.append({
                    'case': case_identifier,
                    'case_label': format_case_for_filename(case_identifier),
                    'fraction': frac,
                    'scenario': i,
                    'cost': float(cst),
                    'risk_months': float(rsk),
                })

            if args.save_timeseries:
                ts_model = SBsimFixed(opt_par, case_identifier, drought_type)
                log = ts_model.simulate_fixed(frac, 0)
                ts_path = os.path.join(
                    outdir,
                    'plots',
                    'timeseries',
                    f"timeseries_fraction_{int(round(frac*100))}_{drought_type}_case_{format_case_for_filename(case_identifier)}.png",
                )
                plot_timeseries(
                    log,
                    title=f"Fixed fraction {frac:.2f} — {drought_type}, case {case_identifier}",
                    save_path=ts_path,
                )

        # Per-case response curves
        df_case = pd.DataFrame([r for r in rows if r['case'] == case_identifier])
        _plot_case_curves(df_case, format_case_for_filename(case_identifier), outdir)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(outdir, 'summary.csv'), index=False)

    df_by_scenario = pd.DataFrame(per_scenario_rows)
    df_by_scenario.to_csv(os.path.join(outdir, 'by_scenario.csv'), index=False)

    # Global scatter plot
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(df_summary['cost_mean'], df_summary['risk_months_mean'], s=18)
    ax.set_xlabel('Mean total cost')
    ax.set_ylabel('Months of supply (25th pct; higher is better)')
    ax.set_title('Fixed desal experiment — all cases')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'plots', 'cost_vs_risk_all_cases.png'), dpi=150)
    plt.close(fig)

    print(f"Done. Wrote: {os.path.join(outdir, 'summary.csv')}")


if __name__ == '__main__':
    main()
