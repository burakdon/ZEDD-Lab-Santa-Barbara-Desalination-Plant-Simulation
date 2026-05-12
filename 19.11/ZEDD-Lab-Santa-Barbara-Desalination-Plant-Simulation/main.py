#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:36:20 2021

@author: martazaniolo
"""
import sys
import argparse
sys.path.append('src')
sys.path.append('ptreeopt')
import simulation
from simulation import SB
from sim_individual import SBsim
from src import *
import numpy as np
import numpy.matlib as mat
from platypus import *
from platypus.experimenter import experiment, calculate, display
#from concurrent.futures import ProcessPoolExecutor
from sb_problem import SB_Problem
from plot_optimization import (
    is_pareto_efficient,
    plot_pareto,
    plot_timeseries,
    select_pareto_timeseries_indices,
    save_timeseries_log_csv,
    companion_timeseries_data_csv_path,
)
import matplotlib.pyplot as plt
import pickle
from platypus import ProcessPoolEvaluator
import pandas as pd
import logging
import os
from cost_curve_loader import CostCurveLoader
from capacity_tiers import get_capacity_tier


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
    """Format case identifier for use in filenames (no path separators)."""
    s = str(case_identifier).strip()
    if "/" not in s:
        return s
    n = s.count("/")
    if n == 1:
        folder, curve_name = s.split("/", 1)
        folder_short = folder.split("_")[-1] if "_" in folder else folder
        return f"{folder_short}_{curve_name}"
    return s.replace("/", "_")


def describe_capacity(best_solution, case_identifier):
    """Return a short string describing the desal expansion tier."""
    if not best_solution:
        return "(no solution)"

    p0 = float(best_solution[0])
    montecito_annual = 1430.0
    
    tier_info = get_capacity_tier(p0)
    gross_annual = tier_info["gross_annual"]
    net_annual = gross_annual - montecito_annual
    return f"Desal expansion tier: {tier_info['label']} ({gross_annual:.0f} AF/yr gross, {net_annual:.0f} AF/yr net)"

class OptimizationParameters(object):
    def __init__(self):
        ###DEMO OPTIMIZATION, use higher values of max_gen and npop if results are not converged
        self.max_gen  = 100 
        self.npop     = 50
        self.nfe      = self.max_gen*self.npop
        self.cores    = 50
        self.nseeds   = 1
        self.N        = 3 #hidden nodes
        self.M        = 2 #inputs
        self.K        = 1 #outputs
        self.nparam   = 1 + self.N*(2*self.M + self.K) + self.K #first param is desal size, others are desal operations
        self.nobjs    = 2
        self.lb       = np.concatenate( (mat.repmat([-1,0], 1, self.M ), np.zeros([1,self.K])), axis = 1 )
        self.LB       = np.concatenate( (np.zeros([1,self.K + 1]), mat.repmat( self.lb, 1, self.N ) ), axis = 1 ) #optimization params lower bound
        self.ub       = np.concatenate( (mat.repmat([1,1], 1, self.M ), np.ones([1,self.K])), axis = 1 ) 
        self.UB       = np.concatenate( (np.ones([1,self.K + 1]), mat.repmat( self.ub, 1, self.N ) ), axis = 1 ) #optimization params upper bound
        self.log_freq = 2000
        self.nsim     = 20

class Solution():
    pass

def run(opt_par, case_number, drought_type):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    if 1: #switch to 1 for parallelizing computation across number of available cores
        with ProcessPoolEvaluator(opt_par.cores) as pool:
            algorithm = [(NSGAII, { "population_size":opt_par.npop, "offspring_size":opt_par.npop,  "evaluator":pool, "log_frequency":opt_par.log_freq, "verbose":3})]
            problem   = [SB_Problem(opt_par, case_number, drought_type)]
            results = experiment(algorithm, problem, seeds = opt_par.nseeds, nfe=opt_par.nfe)
    else:
        algorithm = [(NSGAII, { "population_size":opt_par.npop, "offspring_size":opt_par.npop})]
        problem   = [SB_Problem(opt_par, case_number, drought_type)]
        results = experiment(algorithm, problem, seeds = opt_par.nseeds, nfe=opt_par.nfe)


    return results 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run desalination optimization for a given cost curve case.")
    parser.add_argument(
        "--case",
        default="4mpd_36vessels",
        help="Case identifier (matches <case>_overall.csv, e.g. '1' or '3mpd_30vessels').",
    )
    parser.add_argument(
        "--drought",
        default="pers87_sev0.83n_4",
        help="Drought scenario identifier.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available cost curve cases and exit.",
    )
    parser.add_argument(
        "--timeseries-scenario",
        type=int,
        default=0,
        help="Hydrological scenario index (0..nsim-1) for timeseries PNGs (default: 0).",
    )

    args = parser.parse_args()

    loader = CostCurveLoader()
    if args.list_cases:
        print("Available cost curve cases:")
        for case in loader.get_available_cases():
            info = loader.get_case_info(case)
            scenario = info.get('scenario') if isinstance(info, dict) else None
            if scenario:
                print(f"  {info['case_number']} (scenario: {scenario})")
            else:
                print(f"  {case}")
        sys.exit(0)

    case_identifier = args.case
    drought_type = args.drought

    opt_par = OptimizationParameters()

    # run optimization
    result = run(opt_par, case_identifier, drought_type)

    objs  = []
    param = []

    for seed in range(opt_par.nseeds):
        solutions = result['NSGAII']['SB_Problem'][seed]

        objs.extend([ [s.objectives[0], s.objectives[1]] for s in solutions]) 
        param.extend([s.variables for s in solutions])

    mask = is_pareto_efficient(np.array(objs))

    objs_eff   = []
    param_eff  = []
    for m, o, p in zip(mask, objs, param):
        if m:
            objs_eff.append(o)
            param_eff.append(p)


    #find no deficit solution
    idx_nodeficit = np.argmin(np.array(objs_eff)[:,1] )
    
    idx_maxdeficit = np.argmax(np.array(objs_eff)[:,1] )

    idx_ts_a, idx_ts_b = select_pareto_timeseries_indices(objs_eff)
    ts_s = int(args.timeseries_scenario)
    if ts_s < 0 or ts_s >= opt_par.nsim:
        raise ValueError(
            f"--timeseries-scenario must be in [0, {opt_par.nsim - 1}], got {ts_s}"
        )

    #creat solution structure
    solution = Solution()
    solution.best_score = objs_eff[idx_nodeficit]
    solution.best_solution = param_eff[idx_nodeficit]
    solution.all_solutions = param_eff
    solution.objs = objs_eff
    solution.log = []
    

    #simulate solution
    sim_model = SBsim(opt_par, case_identifier, drought_type)
    sim = SB(opt_par, case_identifier, drought_type)
    print(describe_capacity(solution.best_solution, case_identifier))

    log = sim_model.simulate(param_eff[idx_ts_a], ts_s)
    solution.sim_model = sim_model
    solution.log.append(log)

    # save time-series (mid-front Pareto policy; filename kept for compatibility)
    os.makedirs('result/plots/timeseries', exist_ok=True)
    case_filename = format_case_for_filename(case_identifier)
    ts1_path = f"result/plots/timeseries/timeseries_nodeficit_{drought_type}_case_{case_filename}.png"
    plot_timeseries(
        log,
        title=(
            f"Pareto mid-front (objective-space center) — {drought_type}, "
            f"case {case_identifier}, scenario {ts_s}"
        ),
        save_path=ts1_path,
    )
    save_timeseries_log_csv(
        log,
        companion_timeseries_data_csv_path(ts1_path),
        metadata={
            "drought": drought_type,
            "case": str(case_identifier),
            "hydrologic_scenario_index": ts_s,
            "pareto_policy_index": idx_ts_a,
            "timeseries_slot": "midfront_primary",
            "source": "main.py",
        },
    )

    ##creat solution structure
    solution = Solution()
    solution.best_score = objs_eff[idx_maxdeficit]
    solution.best_solution = param_eff[idx_maxdeficit]
    solution.all_solutions = param_eff
    solution.objs = objs_eff
    solution.log = []
    

    #simulate solution
    sim_model = SBsim(opt_par, case_identifier, drought_type)
    sim = SB(opt_par, case_identifier, drought_type)
    log = sim_model.simulate(param_eff[idx_ts_b], ts_s)
    solution.sim_model = sim_model
    solution.log.append(log)

    # save time-series (second interior policy)
    ts2_path = f"result/plots/timeseries/timeseries_maxdeficit_{drought_type}_case_{case_filename}.png"
    plot_timeseries(
        log,
        title=(
            f"Pareto mid-front (second interior policy) — {drought_type}, "
            f"case {case_identifier}, scenario {ts_s}"
        ),
        save_path=ts2_path,
    )
    save_timeseries_log_csv(
        log,
        companion_timeseries_data_csv_path(ts2_path),
        metadata={
            "drought": drought_type,
            "case": str(case_identifier),
            "hydrologic_scenario_index": ts_s,
            "pareto_policy_index": idx_ts_b,
            "timeseries_slot": "midfront_secondary",
            "source": "main.py",
        },
    )

    # creating a Dataframe object

    string = f'result/results_drought{drought_type}case_{case_filename}.dat'
    with open(string, 'wb') as f:
        pickle.dump(solution, f)

    # save pareto data and plot
    os.makedirs('result/plots/pareto', exist_ok=True)
    os.makedirs('result/data/pareto', exist_ok=True)
    pareto_png = f"result/plots/pareto/pareto_{drought_type}_case_{case_filename}.png"
    pareto_csv = f"result/data/pareto/pareto_{drought_type}_case_{case_filename}.csv"

    eff = plot_pareto(objs_eff, title=f"Pareto — {drought_type}, case {case_identifier}", save_path=pareto_png)
    # save efficient points to CSV for overlaying later
    df = pd.DataFrame({
        'cost': [o[0] for o in eff],
        'risk_months_supply': [-o[1] for o in eff]
    })
    df.to_csv(pareto_csv, index=False)
    

