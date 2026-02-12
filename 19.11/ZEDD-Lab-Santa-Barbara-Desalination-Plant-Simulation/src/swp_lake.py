#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWP class is a subclass of the Lake class and implements geomorphological
characteristics of the San Luis reservoir where Santa Barbara's SWP allocation
is stored.
"""

from lake import Lake
import numpy as np

class SWP(Lake):
    def __init__(self, drought_type):
        self.A       = 1
        self.MEF     = 0
        self.integration_step = 1
        self.deltaH  = 1
        self.T       = 12
        self.max_release = 275
        self.inflow  = np.loadtxt('data/all_swp_'+ drought_type + '.txt')


        self.Ny      = 53 #int(np.size(self.inflow)/self.T)
        self.H       = 636 #int(np.size(self.inflow))
        self.smax    = 7500
        self.smin    = 0
        self.s0      = 4500


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
        return self.A
