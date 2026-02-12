#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fixed-desal variants of the Santa Barbara simulators.

Drop this file into the repo (it relies on the same local modules as the
original model).

Core behavior:
- desal capacity comes from CostCurveLoader.get_max_production(case)
- desal release is fixed at fraction * capacity for every month
- desal cost accounting matches simulation.py / sim_individual.py, including
  seasonal fixed charges and capital amortization
"""

from __future__ import annotations

import numpy as np
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        # Case 1: used as @njit (no parentheses)
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        # Case 2: used as @njit(...) with options/signature
        def decorator(func):
            return func
        return decorator


from cachuma_lake import Cachuma
from gibraltar_lake import Gibraltar
from swp_lake import SWP

from cost_curve_loader import CostCurveLoader


class log_results:
    """Container with the same attribute names used in plot_optimization.plot_timeseries."""

    pass


@njit
def nsim_equiv_res(s: float, u: float, n: float, s_max: float):
    """Fast reservoir mass balance used in the original model."""
    r = max(0.0, min(u, s))
    s_ = s + n - r
    s_ = max(0.0, min(s_, s_max))
    return s_, r


class SBFixed(object):
    """Multi-scenario simulator returning [mean(cost), mean(risk)] for a fixed desal fraction."""

    def __init__(self, opt_par, case_number, drought_type: str):
        self.T = 12
        self.gibraltar = Gibraltar(drought_type)
        self.cachuma = Cachuma(drought_type)
        self.swp = SWP(drought_type)
        self.H = self.gibraltar.H
        self.Ny = int(self.H / self.T)

        annual_dem = np.loadtxt('data/d12_predrought.txt')
        self.demand = annual_dem

        self.mds = np.loadtxt(f'data/mission_{drought_type}.txt')
        self.nsim = int(getattr(opt_par, 'nsim', 20))

        # Cost curves / capacity
        self.cost_curve_loader = CostCurveLoader()
        self.case_number = case_number
        self.cost_curve_data = self.cost_curve_loader.load_cost_curve(case_number)

        self.capital_amortization_years = 30
        self.capital_monthly_cost = self.cost_curve_loader.get_capital_cost_amortized(
            case_number,
            amortization_years=self.capital_amortization_years,
            period='monthly',
        )


    def simulate_fixed(self, fraction: float):
        if fraction < 0 or fraction > 1:
            raise ValueError(f'fraction must be in [0,1], got {fraction}')

        ncs = self.cachuma.inflow
        ngis = self.gibraltar.inflow
        nswps = self.swp.inflow
        smax_gi = self.gibraltar.smax
        smax_ca = self.cachuma.smax
        smax_sw = self.swp.smax

        sustainable_yield = 1250 / 12

        desal_capacity = float(self.cost_curve_loader.get_max_production(self.case_number))
        fixed_release = fraction * desal_capacity

        H = self.gibraltar.H

        Jrisk = []
        Jcost = []

        for s in range(self.nsim):
            nc = ncs[s, :]
            ngi = ngis[s, :]
            nswp = nswps[s, :]
            md = self.mds[s, :]

            sc = np.zeros(H + 1)
            sgi = np.zeros(H + 1)
            sswp = np.zeros(H + 1)
            jrisk = np.zeros(H)
            desal_cost = np.zeros(H)
            deficit = np.zeros(H + 1)

            sc[0] = self.cachuma.s0
            sgi[0] = self.gibraltar.s0
            sswp[0] = self.swp.s0

            def_penalty = 0
            demand = self.demand

            for t in range(H):
                desal_release = fixed_release

                dem = demand[t % 12]
                d = max(0.0, dem - desal_release - md[t] - sustainable_yield)

                SS = 0.0001 + (sc[t] + sgi[t] + sswp[t])
                uc = sc[t] / SS
                ugi = sgi[t] / SS
                uswp = sswp[t] / SS

                if uswp * d > self.swp.max_release:
                    while uswp * d > self.swp.max_release:
                        uswp -= 0.05
                        uc += 0.04
                        ugi += 0.01

                # annual allocations: Cachuma in Oct, SWP in May
                if (t % 12) == 9:
                    sc[t] = sc[t] * self.cachuma.carryover + (1 - self.cachuma.carryover) * 0.3
                    nc_ = nc[int((t - 9) / self.T)]
                else:
                    nc_ = 0.0

                if (t % 12) == 4:
                    nswp_ = nswp[int((t - 4) / self.T)]
                else:
                    nswp_ = 0.0

                s_, r_c = nsim_equiv_res(sc[t], uc * d, nc_, smax_ca)
                sc[t + 1] = s_

                s_, r_gi = nsim_equiv_res(sgi[t], ugi * d, ngi[t], smax_gi)
                sgi[t + 1] = s_

                s_, r_swp = nsim_equiv_res(sswp[t], uswp * d, nswp_, smax_sw)
                sswp[t + 1] = s_

                # seasonal cost handling
                month = t % 12
                is_summer = month >= 4 and month <= 9

                if desal_release > 1e-6:
                    elec_cost, fixed_cost = self.cost_curve_loader.get_cost_for_production(
                        self.case_number, desal_release, is_summer
                    )
                    labor_cost = self.cost_curve_loader.get_labor_cost(self.case_number)
                    base_cost = fixed_cost + elec_cost + labor_cost
                else:
                    if is_summer:
                        fixed_cost = self.cost_curve_data['summer']['fixed_cost'][0]
                    else:
                        fixed_cost = self.cost_curve_data['winter']['fixed_cost'][0]
                    base_cost = fixed_cost

                desal_cost[t] = base_cost + self.capital_monthly_cost

                deficit[t + 1] = max(
                    0.0,
                    demand[t % 12]
                    - r_swp
                    - r_c
                    - r_gi
                    - md[t]
                    - desal_release
                    - sustainable_yield,
                )
                if deficit[t + 1] < 1e-4:
                    deficit[t + 1] = 0.0
                if deficit[t + 1] > 0:
                    def_penalty += 10000

                jrisk[t] = (sc[t + 1] + sgi[t + 1] + sswp[t + 1]) / (
                    np.mean(self.demand) - desal_capacity - sustainable_yield
                )

            Jrisk.append(-np.percentile(jrisk, 25))
            Jcost.append(float(np.sum(desal_cost) + def_penalty))

        return [float(np.mean(Jcost)), float(np.mean(Jrisk))]

    def simulate_fixed_per_scenario(self, fraction: float):
        """Return (scenario_costs, scenario_risk_months) for this fixed desal fraction."""
        if fraction < 0 or fraction > 1:
            raise ValueError(f'fraction must be in [0,1], got {fraction}')

        ncs = self.cachuma.inflow
        ngis = self.gibraltar.inflow
        nswps = self.swp.inflow
        smax_gi = self.gibraltar.smax
        smax_ca = self.cachuma.smax
        smax_sw = self.swp.smax

        sustainable_yield = 1250 / 12

        desal_capacity = float(self.cost_curve_loader.get_max_production(self.case_number))
        fixed_release = fraction * desal_capacity

        H = self.gibraltar.H

        scenario_costs = np.zeros(self.nsim, dtype=float)
        scenario_risk_months = np.zeros(self.nsim, dtype=float)

        denom = (np.mean(self.demand) - desal_capacity - sustainable_yield)

        for s in range(self.nsim):
            nc = ncs[s, :]
            ngi = ngis[s, :]
            nswp = nswps[s, :]
            md = self.mds[s, :]

            sc = np.zeros(H + 1)
            sgi = np.zeros(H + 1)
            sswp = np.zeros(H + 1)
            jrisk = np.zeros(H)
            desal_cost = np.zeros(H)
            deficit = np.zeros(H + 1)

            sc[0] = self.cachuma.s0
            sgi[0] = self.gibraltar.s0
            sswp[0] = self.swp.s0

            def_penalty = 0
            demand = self.demand

            for t in range(H):
                desal_release = fixed_release

                dem = demand[t % 12]
                d = max(0.0, dem - desal_release - md[t] - sustainable_yield)

                SS = 0.0001 + (sc[t] + sgi[t] + sswp[t])
                uc = sc[t] / SS
                ugi = sgi[t] / SS
                uswp = sswp[t] / SS

                if uswp * d > self.swp.max_release:
                    while uswp * d > self.swp.max_release:
                        uswp -= 0.05
                        uc += 0.04
                        ugi += 0.01

                # annual allocations
                if (t % 12) == 9:
                    sc[t] = sc[t] * self.cachuma.carryover + (1 - self.cachuma.carryover) * 0.3
                    nc_ = nc[int((t - 9) / self.T)]
                else:
                    nc_ = 0.0

                if (t % 12) == 4:
                    nswp_ = nswp[int((t - 4) / self.T)]
                else:
                    nswp_ = 0.0

                s_, r_c = nsim_equiv_res(sc[t], uc * d, nc_, smax_ca)
                sc[t + 1] = s_

                s_, r_gi = nsim_equiv_res(sgi[t], ugi * d, ngi[t], smax_gi)
                sgi[t + 1] = s_

                s_, r_swp = nsim_equiv_res(sswp[t], uswp * d, nswp_, smax_sw)
                sswp[t + 1] = s_

                # seasonal costs
                month = t % 12
                is_summer = (month >= 4 and month <= 9)

                if desal_release > 1e-6:
                    elec_cost, fixed_cost = self.cost_curve_loader.get_cost_for_production(
                        self.case_number, desal_release, is_summer
                    )
                    labor_cost = self.cost_curve_loader.get_labor_cost(self.case_number)
                    base_cost = fixed_cost + elec_cost + labor_cost
                else:
                    if is_summer:
                        fixed_cost = self.cost_curve_data['summer']['fixed_cost'][0]
                    else:
                        fixed_cost = self.cost_curve_data['winter']['fixed_cost'][0]
                    base_cost = fixed_cost

                desal_cost[t] = base_cost + self.capital_monthly_cost

                deficit[t + 1] = max(
                    0.0,
                    demand[t % 12]
                    - r_swp
                    - r_c
                    - r_gi
                    - md[t]
                    - desal_release
                    - sustainable_yield,
                )

                if deficit[t + 1] < 1e-4:
                    deficit[t + 1] = 0.0
                if deficit[t + 1] > 0:
                    def_penalty += 10000

                jrisk[t] = (sc[t + 1] + sgi[t + 1] + sswp[t + 1]) / denom

            scenario_risk_months[s] = float(np.percentile(jrisk, 25))   # positive months-of-supply metric
            scenario_costs[s] = float(np.sum(desal_cost) + def_penalty)

        return scenario_costs, scenario_risk_months


class SBsimFixed(object):
    """Single-scenario simulator returning trajectories for a fixed desal fraction."""

    def __init__(self, opt_par, case_number, drought_type: str):
        self.T = 12
        self.gibraltar = Gibraltar(drought_type)
        self.cachuma = Cachuma(drought_type)
        self.swp = SWP(drought_type)
        self.H = self.gibraltar.H
        self.Ny = int(self.H / self.T)

        annual_dem = np.loadtxt('data/d12_predrought.txt')
        self.demand = annual_dem

        self.mds = np.loadtxt(f'data/mission_{drought_type}.txt')

        # Cost curves / capacity
        self.cost_curve_loader = CostCurveLoader()
        self.case_number = case_number
        self.cost_curve_data = self.cost_curve_loader.load_cost_curve(case_number)

        self.capital_amortization_years = 30
        self.capital_monthly_cost = self.cost_curve_loader.get_capital_cost_amortized(
            case_number,
            amortization_years=self.capital_amortization_years,
            period='monthly',
        )

    def simulate_fixed(self, fraction: float, s: int):
        if fraction < 0 or fraction > 1:
            raise ValueError(f'fraction must be in [0,1], got {fraction}')

        ncs = self.cachuma.inflow
        ngis = self.gibraltar.inflow
        nswps = self.swp.inflow
        smax_gi = self.gibraltar.smax
        smax_ca = self.cachuma.smax
        smax_sw = self.swp.smax

        sustainable_yield = 1250 / 12

        desal_capacity = float(self.cost_curve_loader.get_max_production(self.case_number))
        fixed_release = fraction * desal_capacity

        H = self.gibraltar.H

        # outputs
        log = log_results()
        log.desal_capac = desal_capacity
        log.sc = []
        log.rc = []
        log.sgi = []
        log.rgi = []
        log.sswp = []
        log.rswp = []
        log.deficit = []
        log.demand = []
        log.desal_production = []
        log.desal_cost = []
        log.jrisk = []

        # scenario inputs
        nc = ncs[s, :]
        ngi = ngis[s, :]
        nswp = nswps[s, :]
        md = self.mds[s, :]

        sc = np.zeros(H + 1)
        sgi = np.zeros(H + 1)
        sswp = np.zeros(H + 1)
        rc = np.zeros(H + 1)
        rgi = np.zeros(H + 1)
        rswp = np.zeros(H + 1)
        jrisk = np.zeros(H)
        desal_cost = np.zeros(H)
        deficit = np.zeros(H + 1)

        sc[0] = self.cachuma.s0
        sgi[0] = self.gibraltar.s0
        sswp[0] = self.swp.s0

        demand = self.demand

        for t in range(H):
            desal_release = fixed_release

            dem = demand[t % 12]
            d = max(0.0, dem - desal_release - md[t] - sustainable_yield)

            SS = 0.0001 + (sc[t] + sgi[t] + sswp[t])
            uc = sc[t] / SS
            ugi = sgi[t] / SS
            uswp = sswp[t] / SS

            if uswp * d > self.swp.max_release:
                while uswp * d > self.swp.max_release:
                    uswp -= 0.05
                    uc += 0.04
                    ugi += 0.01

            if (t % 12) == 9:
                sc[t] = sc[t] * self.cachuma.carryover + (1 - self.cachuma.carryover) * 0.3
                nc_ = nc[int((t - 9) / self.T)]
            else:
                nc_ = 0.0

            if (t % 12) == 4:
                nswp_ = nswp[int((t - 4) / self.T)]
            else:
                nswp_ = 0.0

            s_, r_c = nsim_equiv_res(sc[t], uc * d, nc_, smax_ca)
            sc[t + 1] = s_
            rc[t + 1] = r_c

            s_, r_gi = nsim_equiv_res(sgi[t], ugi * d, ngi[t], smax_gi)
            sgi[t + 1] = s_
            rgi[t + 1] = r_gi

            s_, r_swp = nsim_equiv_res(sswp[t], uswp * d, nswp_, smax_sw)
            sswp[t + 1] = s_
            rswp[t + 1] = r_swp

            month = t % 12
            is_summer = month >= 4 and month <= 9

            if desal_release > 1e-6:
                elec_cost, fixed_cost = self.cost_curve_loader.get_cost_for_production(
                    self.case_number, desal_release, is_summer
                )
                labor_cost = self.cost_curve_loader.get_labor_cost(self.case_number)
                base_cost = fixed_cost + elec_cost + labor_cost
            else:
                if is_summer:
                    fixed_cost = self.cost_curve_data['summer']['fixed_cost'][0]
                else:
                    fixed_cost = self.cost_curve_data['winter']['fixed_cost'][0]
                base_cost = fixed_cost

            desal_cost[t] = base_cost + self.capital_monthly_cost

            deficit[t + 1] = max(
                0.0,
                demand[t % 12]
                - r_swp
                - r_c
                - r_gi
                - md[t]
                - desal_release
                - sustainable_yield,
            )
            if deficit[t + 1] < 1e-4:
                deficit[t + 1] = 0.0

            jrisk[t] = (sc[t + 1] + sgi[t + 1] + sswp[t + 1]) / (
                np.mean(self.demand) - desal_capacity - sustainable_yield
            )

            log.desal_production.append(desal_release)

        log.sc = sc[:-1]
        log.rc = rc[1:]
        log.sgi = sgi[:-1]
        log.rgi = rgi[1:]
        log.sswp = sswp[:-1]
        log.rswp = rswp[1:]
        log.desal_cost = desal_cost
        log.deficit = deficit[1:]
        log.demand = demand
        log.jrisk = jrisk

        return log
