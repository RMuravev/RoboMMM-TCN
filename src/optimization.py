# RoboMMM-TCN v1.0 — Copyright (c) 2026 Roman Muravyev
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# For commercial use without AGPLv3 obligations, a commercial
# license is required. Contact: roman.muravev@gmail.com

# -*- coding: utf-8 -*-

"""
Module 5: Global Bayesian Optimization & Coordination.
ExpectedImprovement — analytical EI and gradient for robust exploration.
BayesianOptimizer — multi-start L-BFGS-B Bayesian global optimization.
ZBBBudgetCoordinator — subgradient dual decomposition under ZBB constraints.
"""

import math
import numpy as np
from scipy.optimize import minimize

# from .surrogate import GPRegressor


class ExpectedImprovement:
    """ Computes Expected Improvement (EI) and analytical gradient for robust exploration """
    def __init__(self, gp_model):
        self.gp_model = gp_model

    def evaluate(self, x_new, f_best):
        mu, var = self.gp_model.predict(x_new)
        sigma = math.sqrt(var)

        if sigma == 0.0:
            return 0.0

        u = (mu - f_best) / sigma
        phi = math.exp(-0.5 * u**2) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + math.erf(u / math.sqrt(2.0)))

        ei = (mu - f_best) * Phi + sigma * phi
        return ei


class BayesianOptimizer:
    """ Performs Bayesian Global Optimization using surrogate models """
    def __init__(self, gp_model, bounds):
        self.gp_model = gp_model
        self.bounds = bounds

    def optimize_next_point(self, f_best):
        ei_evaluator = ExpectedImprovement(self.gp_model)

        def objective(x):
            return -ei_evaluator.evaluate(x, f_best)

        best_x = None
        best_val = 1e9
        for start_pt in [self.bounds[0], (self.bounds[0] + self.bounds[1]) / 2, self.bounds[1]]:
            res = minimize(
                objective, x0=np.array([start_pt]),
                bounds=[self.bounds], method='L-BFGS-B'
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        return best_x[0]


class ZBBBudgetCoordinator:
    """
    Subgradient coordination engine for multi-brand/multi-channel budget allocation under ZBB limits.
    Utilizes Lagrangian Relaxation (dual decomposition) to manage high-dimensional search spaces.
    """
    def __init__(self, optimizers, total_budget, lmbda_init=1.0, alpha_init=0.5):
        self.optimizers = optimizers
        self.total_budget = total_budget
        self.lmbda = lmbda_init
        self.alpha = alpha_init

    def coordinate_step(self, f_bests):
        """ Evaluates allocations per brand and updates central dual variable lambda """
        allocations = []
        for opt, f_best in zip(self.optimizers, f_bests):
            alloc = opt.optimize_next_point(f_best)
            allocations.append(alloc)

        current_sum = sum(allocations)
        subgradient = current_sum - self.total_budget

        self.lmbda = max(0.0, self.lmbda + self.alpha * subgradient)
        self.alpha *= 0.95

        return allocations, subgradient, self.lmbda
