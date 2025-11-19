#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: martazaniolo
simulation.py simulates the Santa Barbara water system given a desal expansion
decision policy and a set of hydrological scenarios and returns objective values

"""

import numpy as np
from cachuma_lake import Cachuma
from gibraltar_lake import Gibraltar
from swp_lake import SWP
from policy import *
import numpy.matlib as mat
import numba
from numba import njit
import random
from cost_curve_loader import CostCurveLoader
from capacity_tiers import get_capacity_tier


class log_results:
    pass
    class traj:
        pass
    class cost:
        pass



@njit
def nsim_equiv_res(s, u, n, s_max):
    # extremely fast reservoir mass balance computation
    r         = max(0, min(u, s))
    s_        = s + n - r
    s_        = max(0, min(s_, s_max ))

    return s_, r



class SB(object):
   ############# define relevant class parameters
    def __init__(self, opt_par, case_number, drought_type):
        self.T           = 12 # period
        self.gibraltar   = Gibraltar(drought_type)
        self.cachuma     = Cachuma(drought_type)
        self.swp         = SWP(drought_type)
        self.H           = self.gibraltar.H # length of time horizon
        self.Ny          = int(self.H/self.T) #number of years

        annual_dem       = np.loadtxt('data/d12_predrought.txt')
        self.demand      = annual_dem 

        self.nom_cost_sw = 100
        self.nom_cost_rs = 420
        self.mds   = np.loadtxt('data/mission_'+str(drought_type) + '.txt')
        self.nsim        = opt_par.nsim
        self.N           = opt_par.N #hidden nodes
        self.M           = opt_par.M #inputs
        self.K           = opt_par.K #outputs


        self.max_swp_market = 275

        # desal
        self.water_sold  = 1430 
        self.efficiency  = 1.0 
        self.time_exp    = 24
        
        # Load custom cost curve data
        self.cost_curve_loader = CostCurveLoader()
        self.case_number = case_number
        self.cost_curve_data = self.cost_curve_loader.load_cost_curve(case_number)
        # Capital cost amortized annually over a default horizon
        self.capital_amortization_years = 30
        self.capital_monthly_cost = self.cost_curve_loader.get_capital_cost_amortized(
            case_number,
            amortization_years=self.capital_amortization_years,
            period="monthly",
        )


    def simulate(self, P):


        ncs     = self.cachuma.inflow
        ngis    = self.gibraltar.inflow
        nswps   = self.swp.inflow
        smax_gi = self.gibraltar.smax
        smax_ca = self.cachuma.smax
        smax_sw = self.swp.smax

        montecito_agreement = 1430/12 # SB transfers desal water to Montecito
        sustainable_yield = 1250/12 #contant yield from groundwater

        tier_info = get_capacity_tier(P[0])
        gross_capacity = tier_info["gross_month"]
        desal_capacity = gross_capacity #- montecito_agreement
        
        # other policy parameters relate to monthly operations. Extract and interpret RBF paramters from param list P
        param, lin_param = set_param(P[1:], self.N, self.M, self.K)

        self.H  = self.gibraltar.H
        H       = self.H
        self.Ny = H/self.T

        Jrisk = []
        Jcost = []

        for s in range(self.nsim): # repeat simulation for the number of hydrological scenarios

            #initialize variables
            nc    = ncs[s,:]
            ngi   = ngis[s,:]
            nswp  = nswps[s,:]
            md    = self.mds[s,:]


            sc   = np.zeros(H+1)
            sgi  = np.zeros(H+1)
            sswp = np.zeros(H+1)
            jrisk= np.zeros(H)
            desal_cost = np.zeros(H+1)
            deficit = np.zeros(H+1)
            rgi  = np.zeros(H+1)
            sc[0]      = self.cachuma.s0
            sgi[0]     = self.gibraltar.s0
            sswp[0]    = self.swp.s0
            u_ = []


            def_penalty       = 0
            demand = self.demand
            smax_gi = self.gibraltar.smax
            storage_normalization = 35000


            for t in range(H): #simulate for the duration of the time horizon
   ############# compute value of indicators at time T

                storage_t    = self.compute_stor(sc, t)
                storage_t   += self.compute_stor(sswp, t)
                storage_t   += self.compute_stor(sgi, t)

                moy = t%12/12 # month of the year
                inputs = [storage_t/storage_normalization, moy]

   ############## extract action from policy
                u        = get_output(inputs, param, lin_param, self.N, self.M, self.K)
                desal_release = u[0]* desal_capacity
                u_.append(desal_release)


   ############## simulation of surface water reservoirs
                dem = demand[t%12]
                # demand from surface water = total demand - tech installed - mission tunnel inflow - gw sustainable yield
                d = max( 0, dem - desal_release - md[t] - sustainable_yield)

                # release decision is proportional to the storage in each reservoir
                SS = 0.0001+(sc[t] + sgi[t] + sswp[t])
                uc  = sc[t]/SS
                ugi = sgi[t]/SS
                uswp = sswp[t]/SS

                if uswp*d > self.swp.max_release:
                    while uswp*d > self.swp.max_release:
                        uswp -= 0.05
                        uc += 0.04
                        ugi += 0.01

                # surface water allocation in cachuma swp and comes in the form of an annual allocation
                # distributed in the month of October for Cachuma and May for SWP
                if (t%12)==9: # October
                    sc[t]  = sc[t]*self.cachuma.carryover + (1- self.cachuma.carryover)*0.3 #a third of curtailed allocation is redistributed to SB
                    nc_    = nc[int((t-9)/self.T)] 
                else:
                    nc_ = 0

                if (t%12)==4: #May
                    nswp_ = nswp[int((t-4)/self.T)]
                else:
                    nswp_ = 0

                # mass balance of water reservoirs
                s_, r_c  = nsim_equiv_res(sc[t], uc*d, nc_, smax_ca) #self.cachuma.integration(sc[t], uc, nc_, d)
                sc[t+1] = s_

                s_, r_gi  = nsim_equiv_res(sgi[t], ugi*d, ngi[t], smax_gi) #self.gibraltar.integration(sgi[t], ugi, ngi[t], d)
                sgi[t+1] = s_
                rgi[t+1] = r_gi

                s_, r_swp  = nsim_equiv_res(sswp[t], uswp*d, nswp_, smax_sw) #self.swp.integration(sswp[t], uswp, nswp_, d)
                sswp[t+1] = s_

                # Determine if current month is summer (May-October) or winter (November-April)
                month = t % 12
                is_summer = month >= 4 and month <= 9  # May (4) to October (9)
                
                # Get costs from custom cost curve based on production level and season
                # Use a small epsilon to treat near-zero production as zero to avoid spurious jumps
                if desal_release > 1e-6:
                    elec_cost, fixed_cost = self.cost_curve_loader.get_cost_for_production(
                        self.case_number, desal_release, is_summer
                    )
                    # Add labor cost when producing
                    labor_cost = self.cost_curve_loader.get_labor_cost(self.case_number)
                    base_cost = fixed_cost + elec_cost + labor_cost
                else:
                    # No production, use base fixed cost
                    if is_summer:
                        fixed_cost = self.cost_curve_data['summer']['fixed_cost'][0]
                        elec_cost = 0
                    else:
                        fixed_cost = self.cost_curve_data['winter']['fixed_cost'][0]
                        elec_cost = 0
                    # Option A: No labor when idle
                    base_cost = fixed_cost + elec_cost
                    # Option B: Reduced labor when idle (uncomment to use)
                    # labor_cost = self.cost_curve_loader.get_labor_cost(self.case_number) * 0.5
                    # base_cost = fixed_cost + elec_cost + labor_cost
                    # Option C: Full labor always (uncomment to use)
                    # labor_cost = self.cost_curve_loader.get_labor_cost(self.case_number)
                    # base_cost = fixed_cost + elec_cost + labor_cost

                desal_cost[t] = base_cost + self.capital_monthly_cost
                # Option: apply capital charge annually instead of monthly
                # if t % 12 == 0:
                #     desal_cost[t] += self.capital_monthly_cost * 12
                
                
                # calculation of deficit for penalty
                deficit[t+1] = max( 0, demand[t%12] - r_swp - r_c - r_gi - md[t] - desal_release - sustainable_yield)
                if deficit[t+1] < 1e-4:
                    deficit[t+1] = 0
                # constraint on deficit: any deficit is penalized
                if deficit[t+1] > 0:
                    def_penalty += 10000

                #calculate current risk based on months of supply left: current storage / (avg monthly demand - desal capacity)
                jrisk[t] = ( sc[t+1] + sgi[t+1] + sswp[t+1] ) / (np.mean(self.demand) - desal_capacity - sustainable_yield)  # months of supply left 

            Jrisk.append( - np.percentile(jrisk, 25) ) # risk objective formulated as number of months of demand left in storage. Lower number of months higher risk of supply deficit. 
            Jcost.append( np.sum(desal_cost) + def_penalty )
            
            
            
        return [np.mean(Jcost), np.mean(Jrisk)]


    def compute_stor(self, sc, t):
        if t== 0:
            st = sc[0]
        elif t < 12:
            st = np.mean(sc[:t])
        else:
            st = np.mean(sc[t-11:t])
        return st
