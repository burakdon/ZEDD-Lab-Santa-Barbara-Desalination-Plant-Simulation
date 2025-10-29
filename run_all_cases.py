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


def save_outputs_for_case(opt_par, case_number: int, drought_type: str, results):
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
    pareto_png = f"result/plots/pareto/pareto_{drought_type}_case_{case_number}.png"
    pareto_csv = f"result/data/pareto/pareto_{drought_type}_case_{case_number}.csv"

    eff = plot_pareto(
        objs_eff,
        title=f"Pareto — {drought_type}, case {case_number}",
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

    sim_model = SBsim(opt_par, case_number, drought_type)

    # Save one scenario timeseries for nodeficit
    log = sim_model.simulate(param_eff[idx_nodeficit], 0)
    os.makedirs('result/plots/timeseries', exist_ok=True)
    ts1_path = f"result/plots/timeseries/timeseries_nodeficit_{drought_type}_case_{case_number}.png"
    plot_timeseries(
        log,
        title=f"No-deficit solution — {drought_type}, case {case_number}",
        save_path=ts1_path,
    )

    # Save one scenario timeseries for maxdeficit
    log = sim_model.simulate(param_eff[idx_maxdeficit], 0)
    ts2_path = f"result/plots/timeseries/timeseries_maxdeficit_{drought_type}_case_{case_number}.png"
    plot_timeseries(
        log,
        title=f"Max-deficit solution — {drought_type}, case {case_number}",
        save_path=ts2_path,
    )


def main():
    parser = argparse.ArgumentParser(description='Batch-run cost curves and save outputs')
    parser.add_argument('--drought', required=True, help='Drought type, e.g., pers87_sev0.83n_4')
    parser.add_argument('--cases', nargs='+', required=True, help='"all" or list of case numbers')
    args = parser.parse_args()

    # Resolve case list
    if len(args.cases) == 1 and args.cases[0].lower() == 'all':
        cases = list(range(1, 46))
    else:
        cases = [int(c) for c in args.cases]

    drought_type = args.drought
    opt_par = OptimizationParameters()

    # Loop over cases sequentially
    for case_number in cases:
        print(f"Running case {case_number} for drought {drought_type}...")
        results = run(opt_par, case_number, drought_type)
        save_outputs_for_case(opt_par, case_number, drought_type, results)
        print(f"Completed case {case_number}")


if __name__ == '__main__':
    main()


