#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:34:40 2021

@author: martazaniolo
"""

import matplotlib
import numpy as np
from simulation import SB
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.matlib as mat

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    

def plot_pareto(obj, nseeds = 1):
#    plt.style.use('seaborn-darkgrid')
    mask = is_pareto_efficient(np.array(obj))
    #print(mask)
    
    objs = []
    for m, o in zip(mask, obj):
        if m:
            objs.append(o)
        
    
    if nseeds == 1:
        
        plt.scatter([o[0] for o in objs],[-o[1] for o in objs])
        plt.xlabel( 'cost' )
        plt.ylabel( '# demand months left in storage (risk)' )
        
        #plt.xlim([0, 1500])
        #plt.ylim([0, 20])
    else:        
        for s in range(nseeds): #assuming multiple runs    
            plt.scatter([o[0] for o in objs[s]],[-o[1] for o in objs[s]])
            #plt.xlim([0, 1500])
            #plt.ylim([0, 1500])
            
    plt.show()
            
    
    
def plot_timeseries(log):
    n_months = len(log.sc)
    time_years = np.arange(n_months) / 12  # convert months to fractional years

    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Top panel: Reservoir Storage
    axs[0].stackplot(
        time_years, log.sc, log.sgi, log.sswp,
        labels=['Cachuma', 'Gibraltar', 'SWP'],
        colors=['#08519c', '#3182bd', '#9ecae1'] 
    )
    axs[0].set_ylabel('AF')
    axs[0].set_title('Reservoir Storage')
    axs[0].legend(loc='upper left')

    # Middle panel: Release and Production
    axs[1].stackplot(
        time_years, log.desal_production, log.rc, log.rgi, log.rswp,
        labels=['desal production', 'Cachuma', 'Gibraltar', 'SWP'],
        colors=['#ff7f0e', '#006d2c', '#31a354', '#a1d99b']
    )
    axs[1].plot(
    time_years, 
    np.full_like(time_years, log.desal_capac), 
    linestyle=':', color='#ff7f0e', linewidth=2, 
    label='max desal capacity')
    
    
    axs[1].set_ylabel('AF/month')
    axs[1].set_title('Reservoir Release and Production')
    axs[1].legend(loc='upper left')

    # Bottom panel: Objectives
    ax3 = axs[2]
    ax4 = ax3.twinx()
    ax3.plot(time_years, log.desal_cost, label='desal_cost ($)', color='black')
    ax4.plot(time_years, -log.jrisk, label='risk (months of supply)', color='red')
    #fill_between(time_years, -log.jrisk, 0,  # negated to flip below x-axis
    #color='red', alpha=0.3, label='risk (AF/month)')
    
    ax3.set_ylabel('($)')
    ax4.set_ylabel('months to day 0')
    ax3.set_title('Objectives')
    
    ax4.set_ylim(-100, 0)

    # Combine legends from both y-axes
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper left')

    # Format x-axis to show years
    axs[2].set_xlabel('Year')
    axs[2].set_xlim(time_years[0], time_years[-1])
    axs[2].set_xticks(np.arange(int(time_years[0]), int(time_years[-1]) + 1, 1))

    plt.tight_layout()
    plt.show()
