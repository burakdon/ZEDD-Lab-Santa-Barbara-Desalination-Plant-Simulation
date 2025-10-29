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
import os

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
    

def plot_pareto(obj, nseeds = 1, title=None, save_path=None):
    # compute efficient set
    mask = is_pareto_efficient(np.array(obj))
    objs = []
    for m, o in zip(mask, obj):
        if m:
            objs.append(o)

    plt.figure(figsize=(7, 5))
    if nseeds == 1:
        plt.scatter([o[0] for o in objs], [-o[1] for o in objs])
    else:
        for s in range(nseeds):
            plt.scatter([o[0] for o in objs[s]], [-o[1] for o in objs[s]])

    plt.xlabel('cost')
    plt.ylabel('# demand months left in storage (risk)')
    if title:
        plt.title(title)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
    return objs
            
    
    
def plot_timeseries(log, title=None, save_path=None):
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

    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_pareto_overlay(curves, labels=None, title=None, save_path=None):
    """
    Overlay multiple Pareto fronts on the same axes.

    Args:
        curves: list of sequences of [cost, risk] pairs (risk stored as -months in objs, convert to months here)
        labels: list of legend labels, same length as curves (optional)
        title: optional plot title
        save_path: optional file path to save the figure
    """
    plt.figure(figsize=(7, 5))
    for i, objs in enumerate(curves):
        x = [o[0] for o in objs]
        y = [-o[1] for o in objs]
        lab = labels[i] if labels and i < len(labels) else f"curve_{i+1}"
        plt.scatter(x, y, s=20, label=lab)

    plt.xlabel('cost')
    plt.ylabel('# demand months left in storage (risk)')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
