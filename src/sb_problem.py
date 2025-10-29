#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:02:47 2021

@author: martazaniolo
"""


from __future__ import absolute_import, division, print_function

import math
import numpy as np
import random
import operator
import functools
from platypus.core import Problem, Solution, EPSILON
from platypus.types import Real, Binary
from abc import ABCMeta
import simulation


class SB_Problem(Problem):

    def __init__(self, opt_param, case_number, drought_type):
        super(SB_Problem, self).__init__(opt_param.nparam, opt_param.nobjs, nconstrs = 0, function=None)
        self.types[:] = [Real(lb, ub) for lb, ub in zip(opt_param.LB[0], opt_param.UB[0])]
        self.model = simulation.SB(opt_param, case_number, drought_type)

        
    def evaluate(self, individual): #individual is a string of param, sim contains all methods and objects
        str_param = individual.variables
        J1, J2 = self.model.simulate(str_param)

        individual.objectives = [J1, J2]

    def random(self):
        solution = Solution(self)
        solution.variables[:self.nobjs-1] = [random.uniform(0.0, 1.0) for _ in range(self.nobjs-1)]
        solution.variables[self.nobjs-1:] = 0.5
        solution.evaluate()
        return solution
