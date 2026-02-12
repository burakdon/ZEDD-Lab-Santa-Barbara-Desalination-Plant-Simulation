#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gibraltar class is a subclass of the Lake class and implements geomorphological
characteristics of the Gibraltar reservoir along with methods needed for its simulation.
"""

from lake import Lake
import numpy as np

class Gibraltar(Lake):
    def __init__(self, drought_type):
        self.MEF     = 0
        self.integration_step = 1
        self.deltaH  = 1
        self.T       = 12
        self.max_release = 900 # AF
        self.A  = 1

        self.inflow  = np.loadtxt('data/gibr_'+drought_type+'.txt')

        #self.Ny      = 53 #number of simulation years
        self.H       = len(self.inflow[0]) 

        self.smax    = 4550#AF
        self.smin    = 0
        self.s0      = 2000 #initial storage
        self.max_city = 379.16 #maximum volume of water that can be delivered to the city through the mission tunnel


    def max_rel(self,s):
        if s < self.smax:
            q = self.max_release
            if s < self.smin:
                q = 0
        else:
            q = 0.4*s
        return q


    def min_rel(self,s):
        if s > self.smax:
            q = 0.4*s
        else:
            q = 0
        return q


    def storage_to_level(self, s): #overriding the general lsv formulation in lake.py
        return s / self.A


    def level_to_storage(self, l): #overriding the general lsv formulation in lake.py
        return l*self.A # m3

    def storage_to_area(self, s):
        return self.A
