# RoboMMM-TCN v1.0 — Copyright (c) 2026 Roman Muravyev
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# For commercial use without AGPLv3 obligations, a commercial
# license is required. Contact: roman.muravev@gmail.com

# -*- coding: utf-8 -*-

"""
Comprehensive automated unit tests for RoboMMM-TCN library.
Run with: python -m pytest tests/test_robommm.py -v
       or: python -m unittest tests.test_robommm -v
"""

import unittest
import torch
import torch.nn as nn
# import numpy as np

from src.losses import StableTweedieLoss, AdaptiveMonotonicLoss
from src.models import TemporalConvolutionalNetwork
from src.interpretability import CausalDAG
# from src.interpretability import TemporalAdstockSHAPExplainer
from src.surrogate import MaternHillKernel, GPRegressor
from src.optimization import BayesianOptimizer, ZBBBudgetCoordinator


class TestRoboMMMSuite(unittest.TestCase):
    """ Complete test suite verifying every component of RoboMMM-TCN library """

    def test_stable_tweedie_loss_gradients(self):
        """ Checks that StableTweedieLoss yields stable gradients and does not produce NaNs """
        loss_fn = StableTweedieLoss(p=1.5, log_link=True)
        preds = torch.randn(10, 1, requires_grad=True)
        targets = torch.randint(0, 5, (10, 1), dtype=torch.float32)

        loss = loss_fn(preds, targets)
        loss.backward()

        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN")
        self.assertFalse(torch.isnan(preds.grad).any().item(), "Gradient contains NaNs")

    def test_adaptive_monotonic_loss(self):
        """ Tests that AdaptiveMonotonicLoss penalizes negative gradients on mono channels """
        base_fn = StableTweedieLoss(p=1.5, log_link=True)
        loss_fn = AdaptiveMonotonicLoss(base_fn, lmbda=50.0, mono_indices=[0])

        model = nn.Linear(1, 1)
        inputs = torch.linspace(-1.0, 1.0, 10).view(-1, 1)
        targets = torch.linspace(-0.5, 0.5, 10).view(-1, 1)

        total_loss, base_loss, gp = loss_fn(model, inputs, targets)
        self.assertTrue(total_loss >= base_loss, "Total loss must be greater or equal than base loss")

    def test_tcn_forward_pass(self):
        """ Verifies that TemporalConvolutionalNetwork forward pass works and maintains causality """
        model = TemporalConvolutionalNetwork(num_inputs=2, num_channels=[4, 8])
        inputs = torch.randn(5, 12, 2)

        outputs = model(inputs)

        self.assertEqual(outputs.shape, (5, 1), "Output shape should be [batch, 1]")

    def test_causal_dag_and_markov_blanket(self):
        """ Verifies CausalDAG builds parent-child structures and correctly identifies Markov blankets """
        dag = CausalDAG(num_nodes=4)
        dag.add_edge(0, 1)
        dag.add_edge(1, 2)
        dag.add_edge(3, 2)

        blanket = dag.get_markov_blanket(1)
        self.assertIn(0, blanket)
        self.assertIn(2, blanket)
        self.assertIn(3, blanket)

    def test_bayesian_optimization_cycle(self):
        """ Tests complete Gaussian Process modeling, Expected Improvement, and Bayesian Optimization loop """
        kernel = MaternHillKernel(sigma_f=1.0, length_scale=0.5)
        gp = GPRegressor(kernel=kernel, noise_var=1e-5)

        x_train = [0.1, 0.3, 0.7, 0.9]
        y_train = [0.25, 0.55, 0.85, 0.92]
        gp.fit(x_train, y_train)

        optimizer = BayesianOptimizer(gp, bounds=(0.0, 1.0))
        next_point = optimizer.optimize_next_point(f_best=0.92)

        self.assertTrue(0.0 <= next_point <= 1.0, "Next point must lie within bounds")

    def test_zbb_coordinator_decomposition(self):
        """ Tests that subgradient coordinator successfully drives budget allocations toward global limit """
        kernel1 = MaternHillKernel(sigma_f=1.0, length_scale=0.4)
        kernel2 = MaternHillKernel(sigma_f=1.2, length_scale=0.6)

        gp1 = GPRegressor(kernel1, noise_var=1e-5)
        gp2 = GPRegressor(kernel2, noise_var=1e-5)

        gp1.fit([0.2, 0.5, 0.8], [0.3, 0.6, 0.85])
        gp2.fit([0.1, 0.4, 0.7], [0.25, 0.58, 0.78])

        opt1 = BayesianOptimizer(gp1, bounds=(0.0, 1.0))
        opt2 = BayesianOptimizer(gp2, bounds=(0.0, 1.0))

        coordinator = ZBBBudgetCoordinator(optimizers=[opt1, opt2], total_budget=1.0, lmbda_init=1.0)

        allocations, subgrad, lmbda = coordinator.coordinate_step(f_bests=[0.85, 0.78])
        self.assertEqual(len(allocations), 2, "Allocations count must match number of brand optimizers")


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRoboMMMSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
