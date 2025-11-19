#!/usr/bin/env python3
"""
Run multiple cost-curve cases back-to-back and save outputs (Pareto PNG/CSV and time-series PNGs).

Example:
  python run_all_cases.py --drought pers87_sev0.83n_4 --cases all
  python run_all_cases.py --drought pers87_sev0.83n_4 --cases 1 14 25 43
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

# local imports
from main import OptimizationParameters, run
from plot_optimization import plot_timeseries, plot_pareto, is_pareto_efficient
from sim_individual import SBsim
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


def save_outputs_for_case(opt_par, case_identifier, drought_type: str, results):
    # Collect solutions
    objs = []
    param = []
    for seed in range(opt_par.nseeds):
        solutions = results['NSGAII']['SB_Problem'][seed]
        objs.extend([[s.objectives[0], s.objectives[1]] for s in solutions])
        param.extend([s.variables for s in solutions])

    mask = is_pareto_efficient(np.array(objs))
    objs_eff = []
    param_eff = []
    for m, o, p in zip(mask, objs, param):
        if m:
            objs_eff.append(o)
            param_eff.append(p)

    # Save Pareto
    os.makedirs('result/plots/pareto', exist_ok=True)
    os.makedirs('result/data/pareto', exist_ok=True)
    case_filename = format_case_for_filename(case_identifier)
    pareto_png = f"result/plots/pareto/pareto_{drought_type}_case_{case_filename}.png"
    pareto_csv = f"result/data/pareto/pareto_{drought_type}_case_{case_filename}.csv"

    eff = plot_pareto(
        objs_eff,
        title=f"Pareto — {drought_type}, case {case_identifier}",
        save_path=pareto_png,
    )
    df = pd.DataFrame({
        'cost': [o[0] for o in eff],
        'risk_months_supply': [-o[1] for o in eff],
    })
    df.to_csv(pareto_csv, index=False)

    # Time-series plots for representative solutions
    if not param_eff:
        return

    idx_nodeficit = int(np.argmin(np.array(objs_eff)[:, 1]))
    idx_maxdeficit = int(np.argmax(np.array(objs_eff)[:, 1]))

    sim_model = SBsim(opt_par, case_identifier, drought_type)

    # Save one scenario timeseries for nodeficit
    log = sim_model.simulate(param_eff[idx_nodeficit], 0)
    os.makedirs('result/plots/timeseries', exist_ok=True)
    case_filename = format_case_for_filename(case_identifier)
    ts1_path = f"result/plots/timeseries/timeseries_nodeficit_{drought_type}_case_{case_filename}.png"
    plot_timeseries(
        log,
        title=f"No-deficit solution — {drought_type}, case {case_identifier}",
        save_path=ts1_path,
    )

    # Save one scenario timeseries for maxdeficit
    log = sim_model.simulate(param_eff[idx_maxdeficit], 0)
    ts2_path = f"result/plots/timeseries/timeseries_maxdeficit_{drought_type}_case_{case_filename}.png"
    plot_timeseries(
        log,
        title=f"Max-deficit solution — {drought_type}, case {case_identifier}",
        save_path=ts2_path,
    )


def main():
    parser = argparse.ArgumentParser(description='Batch-run cost curves and save outputs')
    parser.add_argument('--drought', required=True, help='Drought type, e.g., pers87_sev0.83n_4')
    parser.add_argument('--cases', nargs='+', required=True, help='"all" or list of case identifiers')
    parser.add_argument('--list-cases', action='store_true', help='List available cost curve cases and exit')
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
        sys.exit(0)

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

    drought_type = args.drought
    opt_par = OptimizationParameters()

    # Loop over cases sequentially
    for case_identifier in cases:
        print(f"Running case {case_identifier} for drought {drought_type}...")
        results = run(opt_par, case_identifier, drought_type)
        save_outputs_for_case(opt_par, case_identifier, drought_type, results)
        print(f"Completed case {case_identifier}")


if __name__ == '__main__':
    main()


