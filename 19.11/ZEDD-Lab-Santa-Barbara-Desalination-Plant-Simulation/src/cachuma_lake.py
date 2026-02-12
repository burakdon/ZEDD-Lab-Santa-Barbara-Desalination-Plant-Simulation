#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cachuma class is a subclass of the Lake class and implements geomorphological
characteristics of the Cachuma reservoir along with methods needed for its simulation.
"""

from lake import Lake
import numpy as np

class Cachuma(Lake):
    def __init__(self, drought_type):

        self.MEF     = 0
        self.integration_step = 1
        self.deltaH  = 1
        self.T       = 12
        self.max_release = 1500 #AF/month
        self.A  = 1

        self.inflow  = np.loadtxt('data/all_cachuma_'+drought_type+'.txt')


        self.Ny      = 53 #number of simulation years
        self.H       = 636
        self.smax    = 20000.0 #AF
        self.smin    = 0
        self.s0      = 15000.00 #initial storage
        self.carryover = 1 # carryover allowed

    def max_rel(self,s):
        if s < self.smin:
            q = 0
        else:
            q = self.max_release
        return q

    def min_rel(self,s):
        if s > self.smax:
            q = self.max_release
        else:
            q = 0
        return q

    def storage_to_level(self, s): #overriding the general lsv formulation in lake.py
        return s / self.A

    def level_to_storage(self, l): #overriding the general lsv formulation in lake.py
        return l*self.A # AF

    def storage_to_area(self, s):
        return float(self.A)
